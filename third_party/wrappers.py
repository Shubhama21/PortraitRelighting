import pickle
import torch
import sys

sys.path.append("third_party/NeRFFaceLighting")
from third_party.NeRFFaceLighting.training.networks_stylegan_lit import (
    TriPlaneGenerator,
    TriPlaneGenerator_Fast,
)
from third_party.NeRFFaceLighting.torch_utils import misc
from third_party.NeRFFaceLighting.camera_utils import (
    FOV_to_intrinsics,
    LookAtPoseSampler,
)
from third_party.NeRFFaceLighting.shape_utils import convert_sdf_samples_to_ply
import numpy as np
from rich import print
import trimesh
import mrcfile
from skimage import measure
from PIL import Image
from PIL import ImageFilter


class WrapperNeRFFaceLighting(torch.nn.Module):
    def __init__(
        self,
        device,
        eg3d_pkl="checkpoints/NeRFFaceLighting.pkl",
        verbose=False,
    ):
        self.device = device
        super(WrapperNeRFFaceLighting, self).__init__()
        with open(eg3d_pkl, "rb") as f:
            _G = pickle.load(f)["G_ema"].cpu().eval()
        if verbose:
            print("Loading NeRFFaceLighting from", eg3d_pkl)
        self.G = (
            TriPlaneGenerator(*_G.init_args, **_G.init_kwargs)
            .requires_grad_(False)
            .eval()
            .to(self.device)
        )
        misc.copy_params_and_buffers(_G, self.G)
        self.G.rendering_kwargs["depth_resolution"] = 48
        self.G.rendering_kwargs["depth_resolution_importance"] = 48
        if verbose:
            print(
                f'Setting up NFL: {self.G.neural_rendering_resolution=}, {self.G.rendering_kwargs["depth_resolution"]=}, {self.G.rendering_kwargs["depth_resolution_importance"]=}'
            )
        self.fov_deg = 18.837
        self.intrinsics = FOV_to_intrinsics(self.fov_deg, device=self.device)
        self.cam_pivot = torch.tensor(
            self.G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]),
            device=self.device,
        )
        self.cam_radius = self.G.rendering_kwargs.get("avg_camera_radius", 2.7)
        self.conditioning_cam2world_pose = LookAtPoseSampler.sample(
            np.pi / 2,
            np.pi / 2,
            self.cam_pivot,
            radius=self.cam_radius,
            device=self.device,
        )
        self.conditioning_params = torch.cat(
            [
                self.conditioning_cam2world_pose.reshape(-1, 16),
                self.intrinsics.reshape(-1, 9),
            ],
            1,
        )

    def args2cam(
        self,
        cam_pivot=[0, 0, 0],
        radius=2.7,
        focal_length=18.837,
        roll=0,
        pitch=0,
        yaw=0,
    ):
        cam_pivot = torch.tensor(cam_pivot, device=self.device).float()
        cam2world_pose = LookAtPoseSampler.sample(
            np.pi / 2 + yaw,
            np.pi / 2 + pitch,
            cam_pivot,
            radius=radius,
            device=self.device,
            z=roll,
        )
        intrinsics = FOV_to_intrinsics(focal_length, device=self.device)
        camera_params = torch.cat(
            [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
        )
        return camera_params

    def get_ref_cam(self, batch_size):
        focal_length = 18.837 + np.random.randn() * 1  # N(18.837, 1)
        radius = 2.7 + np.random.randn() * 0.1  # N(2.7, 0.1)
        cam_pivot = [0, 0, 0] + np.random.randn(3) * 14 / 256  # N([0,0,0], 14/256)
        roll = np.random.randn() * 2 / 180 * np.pi  # N(0, 2deg)
        pitch = np.random.uniform(
            -26 / 180 * np.pi, 26 / 180 * np.pi
        )  # U(-26deg, 26deg)
        yaw = np.random.uniform(-49 / 180 * np.pi, 49 / 180 * np.pi)  # U(-49deg, 49deg)
        return torch.cat(
            [
                self.args2cam(cam_pivot, radius, focal_length, roll, pitch, yaw)
                for _ in range(batch_size)
            ],
            dim=0,
        ).contiguous()

    def get_mv_cam(self, batch_size):
        focal_length = 18.837
        radius = 2.7
        cam_pivot = [0, 0, 0]
        roll = 0
        pitch = np.random.uniform(
            -26 / 180 * np.pi, 26 / 180 * np.pi
        )  # U(-26deg, 26deg)
        yaw = np.random.uniform(-36 / 180 * np.pi, 36 / 180 * np.pi)  # U(-36deg, 36deg)
        return torch.cat(
            [
                self.args2cam(cam_pivot, radius, focal_length, roll, pitch, yaw)
                for _ in range(batch_size)
            ],
            dim=0,
        ).contiguous()

    def seed2z(self, seed, batch_size=1):
        """
        input: seed int, batch_size int
        output: z [B,512]
        """
        z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, 512)).to(
            self.device
        )
        return z

    def z2ws(self, z, truncation_psi=1, truncation_cutoff=14):
        """
        input: z [B,512]
        output: ws [B,14,512]
        """
        ws = self.G.backbone.mapping(
            z,
            self.conditioning_params.repeat(z.shape[0], 1),
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        return ws

    def sh2wslit(self, sh):
        return self.G.backbone.mapping_lit(None, sh, update_emas=False)

    def seed2ws(self, seed, batch_size=1, truncation_psi=1, truncation_cutoff=14):
        """
        input: seed int, batch_size int
        output: ws [B,14,512]
        """
        z = self.seed2z(seed, batch_size=batch_size)
        ws = self.z2ws(
            z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )
        return ws

    def ws2all(self, ws, ws_lit, cam, superres_ws) -> dict:
        """
        input: ws [B,14,512], ws_lit [B,6,512], cam [B,25]
        output: dict:{image,image_raw,image_depth}
        """
        planes, planes_lit = self.ws2planes(ws, ws_lit)
        tmp = self.planes2all(planes, planes_lit, cam, superres_ws)
        tmp["superres_ws"] = superres_ws
        return tmp

    def ws2planes(self, ws, ws_lit) -> dict:
        x, planes = self.G.backbone.synthesis(ws, noise_mode="const")
        planes_lit = self.G.backbone.synthesis_lit(
            x, planes, ws_lit, noise_mode="const"
        )

        return planes, planes_lit

    def planes2all(self, planes_geo, planes_lit, cam, superres_ws) -> dict:
        """
        input: planes_geo [B,96,256,256],planes_lit [B,6,256,256], cam [B,25]
        output: dict:{image,image_raw,image_depth,image_normal,image_albedo
        """
        tmp = self.G.synthesis_triplane(
            planes_geo,
            planes_lit,
            cam,
            superres_ws,
            noise_mode="const",
        )
        tmp["superres_ws"] = superres_ws
        return tmp

    def visualize_planes(self, planes, planes_lit, superres_ws):
        return self.G.synthesis_ortho_3plane(planes, planes_lit, superres_ws)

    def create_samples(self, N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = np.array(voxel_origin) - cube_length / 2
        voxel_size = cube_length / (N - 1)

        overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
        samples = torch.zeros(N**3, 3)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.float() / N) % N
        samples[:, 0] = ((overall_index.float() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        num_samples = N**3

        return samples.unsqueeze(0), voxel_origin, voxel_size

    @torch.no_grad()
    def extract_mesh(self, filename, planes, shape_res=512):
        # w_lit = self.G.backbone.mapping_lit(self.G, sh)
        # w_lit = self.sh2wslit(sh)
        max_batch = 1000000
        samples, voxel_origin, voxel_size = self.create_samples(
            N=shape_res,
            voxel_origin=[0, 0, 0],
            cube_length=self.G.rendering_kwargs["box_warp"] * 1,
        )  # .reshape(1, -1, 3)
        samples = samples.to(planes.device)
        sigmas = torch.zeros(
            (samples.shape[0], samples.shape[1], 1), device=planes.device
        )
        transformed_ray_directions_expanded = torch.zeros(
            (samples.shape[0], max_batch, 3), device=planes.device
        )
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        while head < samples.shape[1]:
            torch.manual_seed(0)
            sigma = self.G.sample_mixed_planes(
                samples[:, head : head + max_batch],
                transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                planes,
                noise_mode="const",
            )["sigma"]

            sigmas[:, head : head + max_batch] = sigma
            head += max_batch

        sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)

        # Trim the border of the extracted cube
        pad = int(30 * shape_res / 256)
        pad_value = -1000
        sigmas[:pad] = pad_value
        sigmas[-pad:] = pad_value
        sigmas[:, :pad] = pad_value
        sigmas[:, -pad:] = pad_value
        sigmas[:, :, :pad] = pad_value
        sigmas[:, :, -pad:] = pad_value

        sigmas[sigmas < 0] = 0.0
        if filename.endswith(".obj"):
            verts, faces, normals, values = measure.marching_cubes(sigmas, level=10)
            # verts, faces, normals, values = measure.marching_cubes_lewiner(sigmas, level=10 if self.type_ == "default" else 5)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            trimesh.exchange.export.export_mesh(mesh, filename)
        elif filename.endswith(".ply"):
            convert_sdf_samples_to_ply(
                np.transpose(sigmas, (2, 1, 0)),
                [0, 0, 0],
                1,
                filename,
                level=10,
            )
        elif filename.endswith(".mrc"):
            with mrcfile.new_mmap(
                filename,
                overwrite=True,
                shape=sigmas.shape,
                mrc_mode=2,
            ) as mrc:
                mrc.data[:] = sigmas


from third_party.CropPose.crop import (
    Cropper,
    eg3d_detect_keypoints,
)
from third_party.CropPose.pose import Poser
import PIL.Image as Image
import numpy as np
import math

sys.path.append("third_party/CropPose")
sys.path.append("third_party/CropPose/models")


class ImageCropPoser(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.cropper = Cropper(device=device)
        self.poser = Poser(device=device)
        self.device = device
        self.intrinsics = FOV_to_intrinsics(18.837).reshape(-1, 9)

    def FOV_to_intrinsics(fov_degrees, device="cuda"):
        focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
        intrinsics = torch.tensor(
            [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device
        )
        return intrinsics

    def image2pil(self, image):
        if isinstance(image, Image.Image):
            img_pil = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            image = ((image + 1) / 2 * 255).astype(np.uint8)
            img_pil = Image.fromarray(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            image = ((image + 1) * 127.5).to(torch.uint8)[0].cpu().numpy()
            img_pil = Image.fromarray(image.transpose(1, 2, 0))  # .convert("RGB")
        else:
            raise TypeError("Image can only be Image.Image, np.array or torch.tensor")
        return img_pil

    def wild2all(self, image):
        """
        Estimate camera from any image (aligned or wild)
        """

        img_pil = self.image2pil(image)
        keypoints = eg3d_detect_keypoints(img_pil)
        img_cropped = self.cropper.final_crop(img_pil, keypoints)
        pred_coeffs = self.cropper.get_deep3d_coeffs(img_pil, keypoints)
        # pred_coeffs, img_cropped = self.cropper.merged_process(img_pil, keypoints)
        pose = self.poser.get_pose(pred_coeffs)
        angle = pred_coeffs["angle"]
        cam = torch.cat(
            [torch.from_numpy(pose["pose"]).cpu().reshape(-1, 16), self.intrinsics],
            dim=1,
        ).float()
        img_cropped = (
            torch.from_numpy(np.array(img_cropped)).float().permute(2, 0, 1)[None, ...]
            / 127.5
            - 1
        )
        return {
            "img_cropped": img_cropped,
            "cam": cam,
            "angle": angle,
            "keypoints": keypoints,
            "pred_coeffs": pred_coeffs,
        }


from third_party.DPR import dpr


class WrapperLightingEstimator(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dpr = dpr.DPR(device)

    @torch.no_grad()
    def forward(self, img):
        """
        img: [B,3,H,W] in [-1,1]
        return: [B,9]
        """
        lighting = self.dpr.extract_lighting(img).squeeze()[None, ...]
        return lighting
