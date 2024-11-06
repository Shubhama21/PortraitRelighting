# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as Fv
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

import random as r
from lpips import LPIPS
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from external_dependencies.DPR.DPR import DPR
from external_dependencies.facemesh.facemesh import FaceMesh

from kornia.color import rgb_to_lab
from torchmetrics.functional import image_gradients

# ----------------------------------------------------------------------------
# Face-parsing

from external_dependencies.face_parsing.model import BiSeNet

seg_mapping = {
    0: 0,  # Background
    1: 1,  # Skin
    2: 2,  # Brow (L)
    3: 2,  # Brow (R)
    4: 3,  # Eye (L)
    5: 3,  # Eye (R)
    6: 4,  # Glasses
    7: 5,  # Ear (L)
    8: 5,  # Ear (R)
    9: 6,  # Ear-ring
    10: 7,  # Nose
    11: 8,  # Mouth
    12: 9,  # Lip (U)
    13: 9,  # Lip (D)
    14: 10,  # Neck
    15: 11,  # Neck-lace
    16: 12,  # Cloth
    17: 13,  # Hair
    18: 14,  # Hat
}


def remap_seg(seg: torch.Tensor):
    for key, value in seg_mapping.items():
        seg[seg == key] = value
    return seg


# ----------------------------------------------------------------------------
# Histogram Color Loss
class RGBuvHistBlock(torch.nn.Module):
    def __init__(
        self,
        h=64,
        method="inverse-quadratic",
        sigma=0.02,
        intensity_scale=True,
    ):
        """Computes the RGB-uv histogram feature of a given image.
        Args:
        h: histogram dimension size (scalar). The default value is 64.
        method: the method used to count the number of pixels for each bin in the
        histogram feature. Options are: 'thresholding', 'RBF' (radial basis
        function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
        sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
        the sigma parameter of the kernel function. The default value is 0.02.
        intensity_scale: boolean variable to use the intensity scale (I_y in
        Equation 2). Default value is True.

        Methods:
        forward: accepts input image and returns its histogram feature. Note that
        unless the method is 'thresholding', this is a differentiable function
        and can be easily integrated with the loss function. As mentioned in the
            paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
            training.
        """
        super(RGBuvHistBlock, self).__init__()
        self.EPS = 1e-6
        self.h = h
        self.method = method
        self.intensity_scale = intensity_scale
        if self.method == "thresholding":
            self.eps_ = 6.0 / h
        else:
            self.sigma = sigma

    def forward(self, x: torch.Tensor):
        # (B, 3, N)
        x = torch.clamp(x / 2.0 + 0.5, 0, 1)  # Convert (-1, 1) to (0, 1)

        I = x.permute(0, 2, 1)  # (B, N, 3)
        II = I.square()
        if self.intensity_scale:
            Iy = torch.sqrt(torch.sum(II, dim=-1, keepdim=True) + self.EPS)  # (B, N, 1)
        else:
            Iy = torch.ones(I.size(0), I.size(1), 1, dtype=I.dtype, device=I.device)
        linspace = torch.linspace(-3, 3, steps=self.h, device=x.device)[
            None, None, None, ...
        ]  # (1, 1, 1, self.h)
        Iu = (torch.log(I + self.EPS) - torch.log(I[..., [1, 0, 0]] + self.EPS))[
            ..., None
        ]  # (B, N, 3, 1)
        Iv = (torch.log(I + self.EPS) - torch.log(I[..., [2, 2, 1]] + self.EPS))[
            ..., None
        ]  # (B, N, 3, 1)
        diff_u = torch.abs(Iu - linspace)  # (B, N, 3, self.h)
        diff_v = torch.abs(Iv - linspace)  # (B, N, 3, self.h)

        # 'inverse-quadratic'
        diff_u = 1 / (1 + diff_u.square() / self.sigma**2)  # (B, N, 3, self.h)
        diff_v = 1 / (1 + diff_v.square() / self.sigma**2)  # (B, N, 3, self.h)

        diff_u = (Iy[..., None] * diff_u).permute(0, 2, 3, 1)  # (B, 3, self.h, N)
        diff_v = diff_v.permute(0, 2, 1, 3)  # (B, 3, N, self.h)
        hists = torch.matmul(diff_u, diff_v)  # (B, 3, self.h, self.h)

        # normalization
        hists_normalized = hists / (
            torch.sum(hists, dim=(1, 2, 3), keepdim=True) + self.EPS
        )
        return hists_normalized


def compute_hist_dist(target_hist: torch.Tensor, input_hist: torch.Tensor):
    return (
        (1 / 2**0.5)
        * (
            torch.sqrt(
                torch.sum(
                    torch.square(torch.sqrt(target_hist) - torch.sqrt(input_hist))
                )
            )
        )
        / max(input_hist.shape[0], target_hist.shape[0])
    )


seg2weight = {
    0: 1 / 15,
    1: 3 / 15,
    2: 1 / 75,
    4: 1 / 75,
    5: 1 / 75,
    7: 1 / 15,
    8: 1 / 75,
    9: 1 / 15,
    10: 1 / 15,
    12: 1 / 15,
    13: 5 / 15,
    14: 1 / 15,
}


def compute_seg_hist_dist(
    HistExtractor: RGBuvHistBlock,
    gen_img: torch.Tensor,
    gen_albedo: torch.Tensor,
    gen_seg: torch.Tensor,
):
    assert gen_img.shape[1:] == gen_albedo.shape[1:]
    assert gen_albedo.size(0) == 1 and gen_seg.size(0) == 1

    loss = 0.0
    for i, weight in seg2weight.items():
        mask = gen_seg == i  # (1, 1, H, W)

        colors_img = torch.cat(
            [
                HistExtractor(
                    gen_img[j : j + 1][mask.expand(-1, 3, -1, -1)].reshape(1, 3, -1)
                )
                for j in range(gen_img.size(0))
            ],
            dim=0,
        )
        colors_albedo = HistExtractor(
            gen_albedo[mask.expand(-1, 3, -1, -1)].reshape(1, 3, -1)
        )  # (1, 3, h, h)
        loss = loss + weight * compute_hist_dist(colors_img, colors_albedo)
    return loss


# ----------------------------------------------------------------------------
# Explicit Shading Model
att = [1 * np.pi, (2.0 / 3.0) * np.pi, (1.0 / 4.0) * np.pi]
coeffs = [
    0.5 / np.sqrt(np.pi),
    np.sqrt(3) / 2 / np.sqrt(np.pi),
    np.sqrt(3) / 2 / np.sqrt(np.pi),
    np.sqrt(3) / 2 / np.sqrt(np.pi),
    np.sqrt(15) / 2 / np.sqrt(np.pi),
    np.sqrt(15) / 2 / np.sqrt(np.pi),
    np.sqrt(5) / 4 / np.sqrt(np.pi),
    np.sqrt(15) / 2 / np.sqrt(np.pi),
    np.sqrt(15) / 4 / np.sqrt(np.pi),
]


def SH_basis(normal):
    norm_X = normal[:, 0]
    norm_Y = normal[:, 1]
    norm_Z = normal[:, 2]

    sh_basis = torch.stack(
        [
            torch.ones_like(norm_X) * coeffs[0] * att[0],
            coeffs[1] * norm_Y * att[1],
            coeffs[2] * norm_Z * att[1],
            coeffs[3] * norm_X * att[1],
            coeffs[4] * norm_Y * norm_X * att[2],
            coeffs[5] * norm_Y * norm_Z * att[2],
            coeffs[6] * (3 * norm_Z**2 - 1) * att[2],
            coeffs[7] * norm_X * norm_Z * att[2],
            coeffs[8] * (norm_X**2 - norm_Y**2) * att[2],
        ],
        dim=1,
    )
    return sh_basis


@torch.no_grad()
def get_shading_from_normal(normals, lighting_sh) -> torch.Tensor:
    normals_basis = SH_basis(normals)  # (B, 9, H, W)
    shadings = torch.sum(
        normals_basis * lighting_sh[:, :, None, None], dim=1, keepdims=True
    )
    return shadings


def sample_lighting_coeffs(batch_size, device, dtype) -> torch.Tensor:
    mean = (
        torch.from_numpy(
            np.array(
                [
                    0.7859381031677419,
                    -0.12334039774885386,
                    0.23426547924177285,
                    -0.06139476676951558,
                    -0.045562432399563424,
                    -0.01241622716945287,
                    -0.1687575493496713,
                    -0.0129877446787183,
                    0.06899898604344021,
                ]
            )
        )
        .to(device)
        .to(dtype)
    )
    std = (
        torch.from_numpy(
            np.array(
                [
                    0.12635883481143398,
                    0.10553028240569193,
                    0.1192847029131568,
                    0.125616270742281,
                    0.08427672322342594,
                    0.08394754616864866,
                    0.09460516675625726,
                    0.07872756138726367,
                    0.13098047330669854,
                ]
            )
        )
        .to(device)
        .to(dtype)
    )

    coeffs = (
        torch.randn(batch_size, 1, dtype=dtype, device=device) * std[None, :]
        + mean[None, :]
    )
    coeffs = torch.clamp(coeffs, min=0.0)
    return coeffs


# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(
        self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg
    ):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------


class StyleGAN2Loss(Loss):
    def __init__(
        self,
        device,
        G,
        D,
        augment_pipe=None,
        r1_gamma=10,
        style_mixing_prob=0,
        pl_weight=0,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_no_weight_grad=False,
        blur_init_sigma=0,
        blur_fade_kimg=0,
        r1_gamma_init=0,
        r1_gamma_fade_kimg=0,
        neural_rendering_resolution_initial=64,
        neural_rendering_resolution_final=None,
        neural_rendering_resolution_fade_kimg=0,
        gpc_reg_fade_kimg=1000,
        gpc_reg_prob=None,
        dual_discrimination=False,
        filter_mode="antialiased",
        lighting_weight=0,
        smooth_weight=0,
        hist_weight=0,
        symmetry_weight=0,
        disambiguation_weight=0,
        sh_std=None,
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.r1_gamma_init = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = (
            neural_rendering_resolution_fade_kimg
        )
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

        # self.dpr = DPR(device)
        self.lighting_weight = lighting_weight
        self.smooth_weight = smooth_weight
        self.hist_weight = hist_weight

        self.lpips_loss = LPIPS(net="alex").to(device)
        self.symmetry_weight = symmetry_weight
        self.disambiguation_weight = disambiguation_weight

        self.sh_std = torch.from_numpy(sh_std).to(device)

        # self.blur = transforms.GaussianBlur(7, 2)

        # Face-parsing
        # self.face2seg_ = BiSeNet(n_classes=19).to(device).eval().requires_grad_(False)
        # self.face2seg_.load_state_dict(torch.load("./external_dependencies/face_parsing/79999_iter.pth", map_location=device))
        # self.face2seg = lambda x: remap_seg(torch.argmax(self.face2seg_(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x))[0], dim=1, keepdim=True))

        # Histogram Color Loss
        # self.HistExtractor = RGBuvHistBlock()

        # FaceMesh
        # self.facemesh = FaceMesh().to(device)
        # self.facemesh.load_weights("./external_dependencies/facemesh/facemesh.pth")
        # self.flip_indices = np.array([0, 1, 2, 248, 4, 5, 6, 390, 8, 9, 10, 11, 12, 12, 14, 15, 16, 17, 18, 19, 462, 251, 252, 253, 254, 339, 256, 257, 258, 259, 260, 448, 262, 249, 264, 265, 266, 267, 312, 269, 304, 311, 310, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 291, 288, 305, 328, 292, 407, 293, 439, 295, 296, 297, 298, 299, 300, 301, 302, 311, 310, 305, 308, 324, 407, 459, 271, 311, 312, 313, 314, 315, 316, 317, 402, 318, 318, 320, 322, 323, 141, 318, 324, 326, 460, 326, 329, 330, 278, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 467, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 264, 357, 278, 263, 360, 433, 463, 363, 364, 365, 366, 367, 368, 369, 94, 371, 372, 373, 374, 325, 376, 377, 378, 379, 151, 152, 380, 382, 362, 383, 384, 385, 386, 387, 388, 368, 373, 164, 391, 392, 393, 168, 394, 395, 396, 397, 362, 399, 175, 400, 401, 402, 402, 404, 405, 406, 415, 415, 407, 410, 411, 412, 413, 414, 272, 416, 417, 418, 195, 419, 197, 420, 199, 200, 421, 422, 423, 424, 425, 426, 427, 428, 360, 430, 431, 432, 433, 434, 435, 436, 437, 309, 392, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 289, 456, 457, 461, 459, 305, 354, 370, 464, 465, 465, 388, 466, 3, 33, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 130, 127, 35, 36, 37, 38, 39, 185, 80, 191, 43, 44, 45, 46, 47, 102, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 235, 60, 57, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 80, 40, 240, 61, 146, 61, 218, 74, 73, 38, 83, 84, 85, 86, 87, 95, 95, 77, 91, 92, 93, 77, 146, 99, 98, 60, 100, 101, 129, 103, 104, 105, 106, 107, 108, 109, 25, 111, 155, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 241, 126, 127, 128, 129, 130, 209, 132, 155, 134, 135, 136, 137, 138, 162, 140, 242, 142, 143, 163, 145, 146, 147, 148, 149, 150, 153, 154, 154, 156, 157, 158, 159, 160, 246, 162, 7, 165, 219, 167, 169, 170, 171, 172, 173, 174, 176, 177, 179, 88, 180, 181, 182, 62, 185, 61, 186, 187, 188, 189, 173, 184, 192, 193, 194, 196, 198, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 64, 220, 221, 222, 223, 224, 225, 226, 227, 31, 229, 230, 231, 232, 233, 234, 64, 236, 79, 238, 79, 98, 238, 20, 133, 243, 244, 33, 113])
        # self.flip_indices = torch.from_numpy(self.flip_indices).to(device)
        # X, Y = torch.meshgrid(torch.arange(512, dtype=int, device=device), torch.arange(512, dtype=int, device=device), indexing='ij')
        # self.indices = torch.stack((X, Y), axis=-1).to(torch.float)/512

    # @torch.no_grad()
    # def parse_face(self, img: torch.Tensor):
    #     seg = torch.argmax(self.face2seg(img), dim=1, keepdim=True)
    #     return {
    #         "background": (seg == 0),
    #     }
    # @torch.no_grad()
    # def get_flip_face(self, img: torch.Tensor, return_mask: bool=True):
    #     assert img.shape[0] == 1
    #     detections = self.facemesh.predict_on_image(F.interpolate(img.clamp(-1, 1), (192, 192), mode='bilinear', align_corners=False, antialias=True)[0])[:, :2]*(1/192)
    #     detect_matrix = torch.topk(torch.sum((detections[:, None, :] - detections[None, :, :])**2, dim=-1), k=5, dim=-1, largest=False, sorted=True).values[:, -1]
    #     flip_detections = detections[self.flip_indices]
    #     dist_matrix = torch.sum((self.indices[:, :, None, :] - detections[None, None, :, :])**2, dim=-1)
    #     dist_matrix = (-dist_matrix) / detect_matrix[None, None, :]
    #     dist_weight = F.softmax(dist_matrix, dim=-1)
    #     dist_weight = torch.nan_to_num(dist_weight)
    #     warped_indices = torch.matmul(dist_weight, flip_detections)*2-1
    #     warped = torch.flip(Fv.rotate(F.grid_sample(img, warped_indices[None, ...], mode='bilinear', align_corners=False), 90), dims=(-2, ))

    #     if return_mask:
    #         hull = ConvexHull(detections.cpu().numpy())
    #         polygons = np.round(detections.cpu().numpy()[hull.vertices]*512).astype(np.int64).tolist()
    #         polygons = [tuple(e) for e in polygons]
    #         mask = Image.new("L", (512, 512), 0)
    #         draw = ImageDraw.Draw(mask)
    #         draw.polygon(polygons, fill=255, outline=None)
    #         mask = torch.from_numpy(np.array(mask).astype(np.float32)/255.)
    #         mask = mask[None, None, ...].to(img.device)
    #     else:
    #         mask = None

    #     return mask, warped

    def run_G(
        self,
        z,
        c,
        l,
        swapping_prob,
        neural_rendering_resolution,
        update_emas=False,
        **kwargs,
    ):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(
                torch.rand((c.shape[0], 1), device=c.device) < swapping_prob,
                c_swapped,
                c,
            )
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws, ws_lit = self.G.mapping(z, c_gen_conditioning, l, update_emas=update_emas)
        # if self.style_mixing_prob > 0:
        #     with torch.autograd.profiler.record_function('style_mixing'):
        #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(
            ws,
            ws_lit,
            c,
            l,
            neural_rendering_resolution=neural_rendering_resolution,
            update_emas=update_emas,
            **kwargs,
        )
        return gen_output, ws, ws_lit

    def run_D(self, img, c, l, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function("blur"):
                f = (
                    torch.arange(-blur_size, blur_size + 1, device=img["image"].device)
                    .div(blur_sigma)
                    .square()
                    .neg()
                    .exp2()
                )
                img["image"] = upfirdn2d.filter2d(img["image"], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(
                torch.cat(
                    [
                        img["image"],
                        torch.nn.functional.interpolate(
                            img["image_raw"],
                            size=img["image"].shape[2:],
                            mode="bilinear",
                            antialias=True,
                        ),
                    ],
                    dim=1,
                )
            )
            img["image"] = augmented_pair[:, : img["image"].shape[1]]
            img["image_raw"] = torch.nn.functional.interpolate(
                augmented_pair[:, img["image"].shape[1] :],
                size=img["image_raw"].shape[2:],
                mode="bilinear",
                antialias=True,
            )

        logits = self.D(img, c, l, update_emas=update_emas)
        return logits

    def accumulate_gradients(
        self, phase, real_img, real_c, real_l, gen_z, gen_c, gen_l, gain, cur_nimg
    ):
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth"]
        # if self.G.rendering_kwargs.get('density_reg', 0) == 0:
        #     phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {"Dreg": "none", "Dboth": "Dmain"}.get(phase, phase)
        blur_sigma = (
            max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
            if self.blur_fade_kimg > 0
            else 0
        )
        r1_gamma = self.r1_gamma

        alpha = (
            min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1)
            if self.gpc_reg_fade_kimg > 0
            else 1
        )
        swapping_prob = (
            (1 - alpha) * 1 + alpha * self.gpc_reg_prob
            if self.gpc_reg_prob is not None
            else None
        )

        if self.neural_rendering_resolution_final is not None:
            alpha = min(
                cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1
            )
            neural_rendering_resolution = int(
                np.rint(
                    self.neural_rendering_resolution_initial * (1 - alpha)
                    + self.neural_rendering_resolution_final * alpha
                )
            )
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(
            real_img,
            size=neural_rendering_resolution,
            f=self.resample_filter,
            filter_mode=self.filter_mode,
        )

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = (
                    torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device)
                    .div(blur_sigma)
                    .square()
                    .neg()
                    .exp2()
                )
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {"image": real_img, "image_raw": real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ["Gmain", "Gboth"]:
            with torch.autograd.profiler.record_function("Gmain_forward"):
                gen_img, _gen_ws, _gen_ws_lit = self.run_G(
                    gen_z,
                    gen_c,
                    gen_l,
                    swapping_prob=swapping_prob,
                    neural_rendering_resolution=neural_rendering_resolution,
                )
                gen_logits = self.run_D(gen_img, gen_c, gen_l, blur_sigma=blur_sigma)
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report("Loss/G/loss", loss_Gmain)
                training_stats.report(
                    "Stat/shade/mean",
                    gen_img["image_shading"].mean(dim=(-1, -2)).squeeze(),
                )
                # Add lighting condition
                # loss_Glighting = (torch.nn.MSELoss()(gen_l[:, :, None, None], self.dpr.extract_lighting(gen_img['image'])) + torch.nn.MSELoss()(gen_l[:, :, None, None], self.dpr.extract_lighting(gen_img['image_raw']))) * .5
                # training_stats.report('Loss/G/loss_lighting', loss_Glighting)
            with torch.autograd.profiler.record_function("Gmain_backward"):
                (loss_Gmain).mean().mul(
                    gain
                ).backward()  #  + loss_Glighting * self.lighting_weight

        # Lighting Regularization
        if phase in ["Greg", "Gboth"]:
            # Same one. (Method I)
            # ws, ws_lit = self.G.mapping(gen_z[:1].expand_as(gen_z), gen_c[:1].expand_as(gen_c), gen_z_l, gen_l, update_emas=False)
            # gen_img = self.G.synthesis(ws, ws_lit, gen_c, gen_l, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            # loss_Glighting = (torch.nn.MSELoss()(gen_l[:, :, None, None], self.dpr.extract_lighting(gen_img['image'])) + torch.nn.MSELoss()(gen_l[:, :, None, None], self.dpr.extract_lighting(gen_img['image_raw']))) * .5
            # training_stats.report('Loss/G/reg_lighting', loss_Glighting)

            # gen_albedo = gen_img["image_albedo"][:1]
            # # face_segs = self.parse_face(gen_albedo)
            # face_mask, gen_flip_albedo = self.get_flip_face(gen_albedo, True)
            # loss_Gsym = self.lpips_loss(gen_albedo, gen_flip_albedo * face_mask + gen_albedo * (1 - face_mask))
            # training_stats.report('Loss/G/reg_symmetry', loss_Gsym)

            # Same one. (Method II)
            # cut_size = gen_z.shape[0] // 1
            # gen_z, gen_c, gen_z_l, gen_l = gen_z[:cut_size], gen_c[:cut_size], gen_z_l[:cut_size], gen_l[:cut_size]
            # ws, ws_lit = self.G.mapping(gen_z[:1].expand_as(gen_z), gen_c[:1].expand_as(gen_c), gen_z_l, gen_l, update_emas=False)
            # gen_img = self.G.synthesis(ws, ws_lit, gen_c, gen_l, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            # gen_logits = self.run_D(gen_img, gen_c, gen_l, blur_sigma=blur_sigma)
            # training_stats.report('Loss/scores/lighting', gen_logits)
            # training_stats.report('Loss/signs/lighting', gen_logits.sign())
            # loss_Glighting = torch.nn.functional.softplus(-gen_logits)
            # training_stats.report('Loss/G/reg_lighting', loss_Glighting)
            # loss_Gsym.mean() * self.symmetry_weight

            # Albedo & Image - Color Regularization.
            # cut_size = gen_z.shape[0] // 2
            # gen_z, gen_c, gen_z_l, gen_l = gen_z[:cut_size], gen_c[:cut_size], gen_z_l[:cut_size], gen_l[:cut_size]
            # ws, ws_lit = self.G.mapping(gen_z[:1].expand_as(gen_z), gen_c[:1].expand_as(gen_c), gen_z_l, gen_l, update_emas=False)
            # gen_img = self.G.synthesis(ws, ws_lit, gen_c, gen_l, neural_rendering_resolution=neural_rendering_resolution, update_emas=False)
            # gen_albedo = gen_img["image_albedo"][:1].detach()
            # gen_seg = self.face2seg(gen_albedo)
            # gen_portrait = gen_img["image"]
            # loss_Ghist = compute_seg_hist_dist(self.HistExtractor, gen_portrait, gen_albedo, gen_seg)
            # training_stats.report('Loss/G/albedo_portrait_hist', loss_Ghist)
            # loss_Ghist * self.hist_weight

            # Explicit Lighting Guidance
            # reg_type = r.sample(['in', 'out'], k=1)[0]
            # if reg_type == 'in':
            #     gen_img, _gen_ws, _gen_ws_lit = self.run_G(gen_z, gen_c, gen_c_l, gen_l, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, verbose=True)
            #     stat_label = 'reg_lighting_in'
            # elif reg_type == 'out':
            #     sample_l = sample_lighting_coeffs(gen_c_l.size(0), gen_c_l.device, gen_c_l.dtype)
            #     gen_img, _gen_ws, _gen_ws_lit = self.run_G(gen_z, gen_c, gen_c_l, sample_l, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, verbose=True)
            #     stat_label = 'reg_lighting_out'
            # else:
            #     raise NotImplementedError("")

            # Shading Guidance
            # implicit_shading = gen_img["image_shading"]
            # explicit_shading = get_shading_from_normal(gen_img["image_normal"], gen_l).clamp(min=0.)
            # loss_Glighting = (implicit_shading - explicit_shading).abs().mean(dim=(-1, -2)).squeeze(1) # (B, )
            # loss_Glighting = torch.clamp(loss_Glighting, min=0.1) - 0.1 # Allow 0.1 Margin
            # training_stats.report(f'Loss/G/{stat_label}', loss_Glighting)
            # loss_Glighting.mean() * self.G.rendering_kwargs.get('lighting_reg', 0) +

            # gen_img, _gen_ws, _gen_ws_lit = self.run_G(gen_z[:1].expand_as(gen_z), gen_c, gen_z_l, gen_l, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, c_cond=gen_c[:1].expand_as(gen_c))

            # Shrink due to Memory Constraint
            cut_size = gen_z.shape[0] // 4
            gen_z, gen_c, gen_l = gen_z[:cut_size], gen_c[:cut_size], gen_l[:cut_size]

            gen_c_swapped = torch.roll(gen_c.clone(), 1, 0)
            c_gen_conditioning = torch.where(
                torch.rand((gen_c.shape[0], 1), device=gen_c.device) < swapping_prob,
                gen_c_swapped,
                gen_c,
            )

            ws, ws_lit = self.G.mapping(gen_z, c_gen_conditioning, gen_l)

            gen_img_s0 = self.G.synthesis(
                ws,
                ws_lit,
                gen_c,
                gen_l,
                neural_rendering_resolution=neural_rendering_resolution,
            )

            # Mixing Another Latent Codes
            _ws = self.G.backbone.mapping(
                torch.randn_like(gen_z),
                c_gen_conditioning,
                truncation_psi=1.0,
                truncation_cutoff=None,
                update_emas=False,
            )
            _ws[:, :7] = ws[:, :7]  # Expect the geometry is similar

            _gen_img_s0 = self.G.synthesis(
                _ws,
                ws_lit,
                gen_c,
                gen_l,
                neural_rendering_resolution=neural_rendering_resolution,
                return_planes=True,
            )

            gen_img_s0_hat = self.G.synthesis(
                ws,
                ws_lit,
                gen_c,
                gen_l,
                neural_rendering_resolution=neural_rendering_resolution,
                _planes_lit=_gen_img_s0["planes_lit"],
            )  # Replace the Shading Tri-plane

            # Same Shading, Cross Texture
            loss_Gcross_T = torch.nn.L1Loss()(
                gen_img_s0["image"], gen_img_s0_hat["image"]
            )
            training_stats.report(f"Loss/G/crossTexture", loss_Gcross_T)

            # Same Texture, Different Shading
            # _gen_l = gen_l + self.sh_std * torch.randn_like(gen_l) * 0.1
            # _ws_lit = self.G.backbone.mapping_lit(None, _gen_l, truncation_psi=1., truncation_cutoff=None, update_emas=False)
            # gen_img_s1 = self.G.synthesis(ws, _ws_lit, gen_c, _gen_l, neural_rendering_resolution=neural_rendering_resolution)

            # loss_Gcross_S = torch.nn.L1Loss()(gen_img_s0["image"], gen_img_s1["image"])
            # training_stats.report(f'Loss/G/crossShading', loss_Gcross_S)

            # loss_Gcomposite = self.lpips_loss(gen_img_s0["image_albedo_raw"], ((gen_img_s0["image_raw"]/2+.5)/(gen_img_s0["image_shading"]+1e-5))*2-1)
            # training_stats.report(f'Loss/G/composition', loss_Gcomposite)

            # Disambiguation Regularization
            # loss_Gdis = torch.nn.L1Loss()(gen_img["image_raw"], (gen_img["image_albedo_raw"]+1) * gen_img["image_shading"]-1)
            # training_stats.report(f'Loss/G/disambiguation', loss_Gdis)
            # loss_Gdis * self.disambiguation_weight

            # Another Shading Latent Codes
            _gen_l = gen_l + gen_l.std(0) * torch.randn_like(gen_l) * 0.1
            _ws_lit = self.G.backbone.mapping_lit(
                None,
                _gen_l,
                truncation_psi=1.0,
                truncation_cutoff=None,
                update_emas=False,
            )
            gen_img_s1 = self.G.synthesis(
                ws,
                _ws_lit,
                gen_c,
                _gen_l,
                neural_rendering_resolution=neural_rendering_resolution,
            )

            # Shrink again due to Memory Constraint :'-|
            # gen_img_s1 = self.G.synthesis(ws[:cut_size//2], _ws_lit[:cut_size//2], gen_c[:cut_size//2], _gen_l[:cut_size//2], neural_rendering_resolution=neural_rendering_resolution)
            # _gen_img_s1 = self.G.synthesis(_ws[:cut_size//2], _ws_lit[:cut_size//2], gen_c[:cut_size//2], _gen_l[:cut_size//2], neural_rendering_resolution=neural_rendering_resolution)

            # loss_Gcross_S = torch.nn.L1Loss()(
            #     rgb_to_lab( gen_img_s1["image"])[:, :1] / 127. - rgb_to_lab( gen_img_s0["image"][:cut_size//2])[:, :1] / 127.,
            #     rgb_to_lab(_gen_img_s1["image"])[:, :1] / 127. - rgb_to_lab(_gen_img_s0["image"][:cut_size//2])[:, :1] / 127.
            # )
            # training_stats.report(f'Loss/G/crossShading', loss_Gcross_S)

            # Same Texture, Cross Shading
            loss_Gcross_S = self.lpips_loss(
                gen_img_s1["image_albedo_raw"],
                (
                    (gen_img_s1["image_raw"] / 2 + 0.5)
                    / (gen_img_s0["image_shading"] + 1e-5)
                )
                * 2
                - 1,
            )
            training_stats.report(f"Loss/G/crossShading", loss_Gcross_S)

            # Chromaticity-based Smoothness
            # foreground_mask = torch.logical_not(self.parse_face(gen_img["image_albedo"])["background"])
            # My, Mx = image_gradients(gen_img["image_albedo"])
            # AB = rgb_to_lab(gen_img["image"].detach()/2.+.5)[:, 1:] / 127. # (B, 2, H, W) [-1, 1]
            # ABy, ABx = image_gradients(AB)
            # loss_Gsmooth = (-torch.linalg.vector_norm(ABy, ord=2, dim=1, keepdim=True).square() / 10.).exp() * My.abs() + \
            #                (-torch.linalg.vector_norm(ABx, ord=2, dim=1, keepdim=True).square() / 10.).exp() * Mx.abs()
            # loss_Gsmooth = torch.mean(loss_Gsmooth, dim=(1, 2, 3)) # (B, )
            # training_stats.report(f'Loss/G/smoothness', loss_Gsmooth)
            # loss_Gsmooth.mean() * self.smooth_weight

            (loss_Gcross_T + loss_Gcross_S.mean()).mul(gain).backward()

        # Density Regularization
        if (
            phase in ["Greg", "Gboth"]
            and self.G.rendering_kwargs.get("density_reg", 0) > 0
            and self.G.rendering_kwargs["reg_type"] == "l1"
        ):
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand((gen_c.shape[0], 1), device=gen_c.device)
                    < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.backbone.mapping(gen_z, c_gen_conditioning, update_emas=False)
            # if self.style_mixing_prob > 0:
            #     with torch.autograd.profiler.record_function('style_mixing'):
            #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = (
                torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            )
            perturbed_coordinates = (
                initial_coordinates
                + torch.randn_like(initial_coordinates)
                * self.G.rendering_kwargs["density_reg_p_dist"]
            )
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1
            )
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            TVloss = (
                torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed)
                * self.G.rendering_kwargs["density_reg"]
            )
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if (
            phase in ["Greg", "Gboth"]
            and self.G.rendering_kwargs.get("density_reg", 0) > 0
            and self.G.rendering_kwargs["reg_type"] == "monotonic-detach"
        ):
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand((gen_c.shape[0], 1), device=gen_c.device)
                    < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.backbone.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = (
                torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1
            )  # Front

            perturbed_coordinates = (
                initial_coordinates
                + torch.tensor([0, 0, -1], device=ws.device)
                * (1 / 256)
                * self.G.rendering_kwargs["box_warp"]
            )  # Behind
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1
            )
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            monotonic_loss = (
                torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            )
            monotonic_loss.mul(gain).backward()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand((gen_c.shape[0], 1), device=gen_c.device)
                    < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.backbone.mapping(gen_z, c_gen_conditioning, update_emas=False)

            # if self.style_mixing_prob > 0:
            #     with torch.autograd.profiler.record_function('style_mixing'):
            #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = (
                torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            )
            perturbed_coordinates = (
                initial_coordinates
                + torch.randn_like(initial_coordinates)
                * (1 / 256)
                * self.G.rendering_kwargs["box_warp"]
            )
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1
            )
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            TVloss = (
                torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed)
                * self.G.rendering_kwargs["density_reg"]
            )
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if (
            phase in ["Greg", "Gboth"]
            and self.G.rendering_kwargs.get("density_reg", 0) > 0
            and self.G.rendering_kwargs["reg_type"] == "monotonic-fixed"
        ):
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand((gen_c.shape[0], 1), device=gen_c.device)
                    < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.backbone.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = (
                torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1
            )  # Front

            perturbed_coordinates = (
                initial_coordinates
                + torch.tensor([0, 0, -1], device=ws.device)
                * (1 / 256)
                * self.G.rendering_kwargs["box_warp"]
            )  # Behind
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1
            )
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand((gen_c.shape[0], 1), device=gen_c.device)
                    < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.backbone.mapping(gen_z, c_gen_conditioning, update_emas=False)

            # if self.style_mixing_prob > 0:
            #     with torch.autograd.profiler.record_function('style_mixing'):
            #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = (
                torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            )
            perturbed_coordinates = (
                initial_coordinates
                + torch.randn_like(initial_coordinates)
                * (1 / 256)
                * self.G.rendering_kwargs["box_warp"]
            )
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1
            )
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            TVloss = (
                torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed)
                * self.G.rendering_kwargs["density_reg"]
            )
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ["Dmain", "Dboth"]:
            with torch.autograd.profiler.record_function("Dgen_forward"):
                gen_img, _gen_ws, _gen_ws_lit = self.run_G(
                    gen_z,
                    gen_c,
                    gen_l,
                    swapping_prob=swapping_prob,
                    neural_rendering_resolution=neural_rendering_resolution,
                    update_emas=True,
                )
                gen_logits = self.run_D(
                    gen_img, gen_c, gen_l, blur_sigma=blur_sigma, update_emas=True
                )
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function("Dgen_backward"):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ["Dmain", "Dreg", "Dboth"]:
            name = (
                "Dreal"
                if phase == "Dmain"
                else "Dr1"
                if phase == "Dreg"
                else "Dreal_Dr1"
            )
            with torch.autograd.profiler.record_function(name + "_forward"):
                real_img_tmp_image = (
                    real_img["image"]
                    .detach()
                    .requires_grad_(phase in ["Dreg", "Dboth"])
                )
                real_img_tmp_image_raw = (
                    real_img["image_raw"]
                    .detach()
                    .requires_grad_(phase in ["Dreg", "Dboth"])
                )
                real_img_tmp = {
                    "image": real_img_tmp_image,
                    "image_raw": real_img_tmp_image_raw,
                }

                real_logits = self.run_D(
                    real_img_tmp, real_c, real_l, blur_sigma=blur_sigma
                )
                training_stats.report("Loss/scores/real", real_logits)
                training_stats.report("Loss/signs/real", real_logits.sign())

                loss_Dreal = 0
                if phase in ["Dmain", "Dboth"]:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report("Loss/D/loss", loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ["Dreg", "Dboth"]:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function(
                            "r1_grads"
                        ), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(
                                outputs=[real_logits.sum()],
                                inputs=[
                                    real_img_tmp["image"],
                                    real_img_tmp["image_raw"],
                                ],
                                create_graph=True,
                                only_inputs=True,
                            )
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum(
                            [1, 2, 3]
                        ) + r1_grads_image_raw.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function(
                            "r1_grads"
                        ), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(
                                outputs=[real_logits.sum()],
                                inputs=[real_img_tmp["image"]],
                                create_graph=True,
                                only_inputs=True,
                            )
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report("Loss/r1_penalty", r1_penalty)
                    training_stats.report("Loss/D/reg", loss_Dr1)

            with torch.autograd.profiler.record_function(name + "_backward"):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()


# ----------------------------------------------------------------------------
