from PIL import Image
import torch.nn as nn
import torch
import os
from pyshtools.rotate import SHRotateRealCoef, djpi2
import numpy as np

try:
    from termcolor import cprint
except ImportError:
    print("termcolor not found, using print instead")

    def cprint(text, color=None, on_color=None, attrs=None, end="\n"):
        print(text, end=end)


def perror(text: str, end="\n"):
    cprint("[ERROR] ", "red", end="")
    cprint(text, attrs=["bold"], end=end)


def pwarning(text: str, end="\n"):
    cprint("[WARNING] ", "yellow", end="")
    cprint(text, attrs=["bold"], end=end)


def pinfo(text: str, end="\n"):
    cprint("[INFO] ", "green", end="")
    print(text, end=end)


def pextra(text: str, end="\n"):
    cprint("[DEBUG] ", "blue", end="")
    cprint(text, attrs=["bold"], end=end)


def try_mkdir(path):
    if not os.path.exists(path):
        pinfo(f"Creating {path}")
        os.makedirs(path)


def toggle_grad(model, flag=True, verbose=False):
    if verbose:
        pinfo(f"Turning {model.__class__.__name__} grad to {flag}")
    if isinstance(model, nn.Module):
        for p in model.parameters():
            p.requires_grad = flag
    elif isinstance(model, torch.Tensor):
        model.requires_grad = flag
    else:
        raise NotImplementedError


def check_grad(model):
    all_true = True
    all_false = True
    if isinstance(model, nn.Module):
        for p in model.parameters():
            all_true = all_true and p.requires_grad
            all_false = all_false and (not p.requires_grad)
    elif isinstance(model, torch.Tensor):
        all_true = model.requires_grad
        all_false = not model.requires_grad
    else:
        raise NotImplementedError
    if all_true:
        return "True"
    elif all_false:
        return "False"
    else:
        return "Mixed"


def render_tensor(img, as_pil=True):
    """
    img: torch.Tensor, shape (1, 3, H, W) in range [-1,1]
    """
    img = (img + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    if as_pil:
        img = (img * 255).astype("uint8")
        img_pil = Image.fromarray(img[0])
        return img_pil
    else:
        return img


def render_shading(img, siz=512.0, as_pil=True):
    """
    img: torch.Tensor, shape (1, 1, h,w)
    """
    img = nn.functional.interpolate(
        img,
        scale_factor=512 // img.shape[2],
    ).repeat(1, 3, 1, 1)
    img = (img - img.min()) / (img.max() - img.min())
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    if as_pil:
        img = (img * 255).astype("uint8")
        img_pil = Image.fromarray(img[0])
        return img_pil
    else:
        return img


def warp(x, flo):
    # print(x.min(),x.max(),flo.min(),flo.max())
    # interpolate flo if flo is not in the same size as x
    if flo.size()[2:] != x.size()[2:]:
        scale_factor = flo.shape[2] / x.shape[2]
        flo = torch.nn.functional.interpolate(
            flo, x.size()[2:], mode="bilinear", align_corners=False
        )
        flo = flo / scale_factor

    x_min, x_max = x.min(), x.max()
    # normalize x to [-1,1]
    x = (x - x_min) / (x_max - x_min)
    x = x * 2 - 1

    B, C, H, W = x.size()
    flo = flo.permute(0, 2, 3, 1)  # [B,2,H,W] -> [B,H,W,2]
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

    xx = xx.view(1, H, W, 1).repeat(B, 1, 1, 1)
    yy = yy.view(1, H, W, 1).repeat(B, 1, 1, 1)

    grid = torch.cat((xx, yy), 3).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = grid + flo

    ## scale grid to [-1,1]
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1].clone() / max(H - 1, 1) - 1.0

    # x = x.permute(0, 3, 1, 2)

    output = torch.nn.functional.grid_sample(x, vgrid)
    # mask = torch.autograd.Variable(torch.ones(x.size()))
    # mask = torch.ones(x.size()).cuda()
    # mask = torch.nn.functional.grid_sample(mask, vgrid)

    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1

    # re normalize to [x_min,x_max]
    output = (output + 1) / 2
    output = output * (x_max - x_min) + x_min
    return output  # * mask


def get_normals(img_size):
    """
    https://github.com/zhhoper/DPR/blob/master/testNetwork_demo_512.py

    """
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    return normal, valid


def get_shading(normal, SH):
    """
    https://github.com/zhhoper/DPR/blob/master/utils/utils_SH.py
    get shading based on normals and SH
    normal is Nx3 matrix
    SH: 9 x m vector
    return Nxm vector, where m is the number of returned images
    """
    sh_basis = SH_basis(normal)
    shading = np.matmul(sh_basis, SH)
    # shading = np.matmul(np.reshape(sh_basis, (-1, 9)), SH)
    # shading = np.reshape(shading, normal.shape[0:2])
    return shading


def SH_basis(normal):
    """
    https://github.com/zhhoper/DPR/blob/master/utils/utils_SH.py
    get SH basis based on normal
    normal is a Nx3 matrix
    return a Nx9 matrix
    The order of SH here is:
    1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    """
    numElem = normal.shape[0]

    norm_X = normal[:, 0]
    norm_Y = normal[:, 1]
    norm_Z = normal[:, 2]

    sh_basis = np.zeros((numElem, 9))
    att = np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])
    sh_basis[:, 0] = 0.5 / np.sqrt(np.pi) * att[0]

    sh_basis[:, 1] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_Y * att[1]
    sh_basis[:, 2] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_Z * att[1]
    sh_basis[:, 3] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_X * att[1]

    sh_basis[:, 4] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_Y * norm_X * att[2]
    sh_basis[:, 5] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_Y * norm_Z * att[2]
    sh_basis[:, 6] = np.sqrt(5) / 4 / np.sqrt(np.pi) * (3 * norm_Z**2 - 1) * att[2]
    sh_basis[:, 7] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_X * norm_Z * att[2]
    sh_basis[:, 8] = (
        np.sqrt(15) / 4 / np.sqrt(np.pi) * (norm_X**2 - norm_Y**2) * att[2]
    )
    return sh_basis


def render_half_sphere(sh, img_size):
    """
    sh: np.array (9x3)
    https://github.com/zhhoper/DPR/blob/master/testNetwork_demo_512.py
    """
    # sh = rotate_SH_coeffs(sh, np.array([np.pi/2, 0, 0])) # DONTDOTHIS

    normal, valid = get_normals(img_size)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    shading = (shading * 255.0).astype(np.uint8)
    shading = np.reshape(shading, (img_size, img_size, 3))
    shading = shading * valid[:, :, None]
    return shading, valid


def shtools_matrix2vec(SH_matrix):
    """
    for the sh matrix created by sh tools,
    we create the vector of the sh
    """
    numOrder = SH_matrix.shape[1]
    vec_SH = np.zeros(numOrder**2)
    count = 0
    for i in range(numOrder):
        for j in range(i, 0, -1):
            vec_SH[count] = SH_matrix[1, i, j]
            count = count + 1
        for j in range(0, i + 1):
            vec_SH[count] = SH_matrix[0, i, j]
            count = count + 1
    return vec_SH


def shtools_sh2matrix(coefficients, degree):
    """
    convert vector of sh to matrix
    """
    coeffs_matrix = np.zeros((2, degree + 1, degree + 1))
    current_zero_index = 0
    for l in range(0, degree + 1):
        coeffs_matrix[0, l, 0] = coefficients[current_zero_index]
        for m in range(1, l + 1):
            coeffs_matrix[0, l, m] = coefficients[current_zero_index + m]
            coeffs_matrix[1, l, m] = coefficients[current_zero_index - m]
        current_zero_index += 2 * (l + 1)
    return coeffs_matrix


def rotate_SH_coeffs(sh, angles, dj=None):
    if dj is None:
        dj = djpi2(2)
    rotated = np.zeros(sh.shape)
    for i in range(sh.shape[1]):
        rotmat = SHRotateRealCoef(shtools_sh2matrix(sh[:, i], 2), angles, dj)
        rotated[:, i] = shtools_matrix2vec(rotmat)
    return rotated


def paste_light_on_img_tensor(sphere_size, light_coeff, img):
    """
    sphere_size: int, denoting SxS sized half sphere
    light_coeff: 9x3(colored) or 1x9(white) tensor of sh coefficient
    img: BxCxHxW batched images
    """
    # print(light_coeff.shape)
    if light_coeff.shape[0] == 1:
        light_coeff = light_coeff.repeat(3, 1).permute(1, 0)
    device = img.device
    sphere_img, alpha_mask = render_half_sphere(light_coeff.cpu().numpy(), sphere_size)
    sphere_img = torch.Tensor(sphere_img).permute(2, 0, 1).to(device)
    sphere_img = (sphere_img - sphere_img.min()) / (
        sphere_img.max() - sphere_img.min()
    ) * 2 - 1

    alpha_mask = torch.Tensor(alpha_mask).to(device)
    img[:, :, -sphere_size:, -sphere_size:] = (1 - alpha_mask[None, None, :, :]) * img[
        :, :, -sphere_size:, -sphere_size:
    ] + alpha_mask[None, None, :, :] * sphere_img.unsqueeze(0)
    return img


def angle_in_a_circle(param, axis="z"):
    assert 0 <= param <= 1

    if axis == "x":
        return np.array([np.pi / 2, 0, param * 2 * np.pi])
    if axis == "y":
        return np.array([np.pi / 2, param * 2 * np.pi, 0])
    if axis == "z":
        return np.array([param * 2 * np.pi, np.pi / 2, 0])
    if axis == "p":
        return np.array([param * 2 * np.pi, 0, 0])
