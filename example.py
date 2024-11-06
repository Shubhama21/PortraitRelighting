import sys
import torch
from networks.relighting import Relighting
from third_party.wrappers import ImageCropPoser, WrapperLightingEstimator
from PIL import Image
import numpy as np
import imageio
from tqdm import tqdm
import glob
from utils import render_tensor, render_half_sphere, paste_light_on_img_tensor

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def main():
    # 1. Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cropposer, relighting, dpr = initialize_models(device)
    example_lightings = np.load("examples/example_lightings.npy")

    # 2. Warm up models
    warm_up_models(cropposer, relighting, dpr, device)

    # 3. Reconstruct image (i.e. render under original camera and lighting)
    cropped_image, cam, planes = reconstruct_image(cropposer, relighting, dpr, device)

    # 4. Relighting images under novel views and lightings
    perform_relighting(cropped_image, planes, relighting, device, example_lightings)

    # 5. Relighting video under novel lightings
    perform_video_relighting(relighting, device, example_lightings)


def initialize_models(device):
    cropposer = ImageCropPoser(device).to(device)
    relighting = Relighting(device).to(device)
    relighting.load_state_dict(torch.load("checkpoints/model.pth"))
    dpr = WrapperLightingEstimator(device).to(device)
    return cropposer, relighting, dpr


def warm_up_models(cropposer, relighting, dpr, device):
    with torch.no_grad():
        example_img = Image.open("examples/example.png")
        example_img = preprocess_image(example_img, device)
        cropposer.wild2all(example_img)
        relighting.image_forward(
            example_img,
            torch.rand(1, 25, device=device),
            torch.rand(1, 9, device=device),
        )
        dpr.dpr.extract_lighting(example_img)
    print("Models loaded")


def preprocess_image(image, device):
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
    image = image.float().to(device) / 255 * 2 - 1
    return image


def reconstruct_image(cropposer, relighting, dpr, device):
    with torch.no_grad():
        input_image = Image.open("examples/example.png")
        input_image = preprocess_image(input_image, device)
        ret = cropposer.wild2all(input_image)
        cropped_image = ret["img_cropped"].to(device)
        cam = ret["cam"].to(device)
        sh = dpr.dpr.extract_lighting(cropped_image).squeeze().unsqueeze(0)
        ret, planes, _ = relighting.image_forward(cropped_image, cam, sh)
        recon_image = ret["image"]
        render_tensor(recon_image).save("examples/recon_image.png")
        render_tensor(cropped_image).save("examples/cropped_image.png")
    return cropped_image, cam, planes


def perform_relighting(cropped_image, planes, relighting, device, example_lightings):
    fps = 24
    frames_per_step = 36
    steps = 3
    total_frames = steps * frames_per_step

    with imageio.get_writer("examples/relighting.mp4", fps=fps) as video_writer:
        for step in tqdm(range(total_frames)):
            with torch.inference_mode():
                idx = (step // frames_per_step) % len(example_lightings)
                angle = (step % frames_per_step) / frames_per_step * 2 * np.pi
                pitch = 0.3 * np.sin(angle)
                yaw = 0.3 * np.cos(angle)
                cam = relighting.encoders.eg3d.args2cam(pitch=pitch, yaw=yaw).to(device)
                sh = torch.from_numpy(example_lightings[idx]).unsqueeze(0).to(device)

                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        ret, _, _ = relighting.image_forward(
                            cropped_image, cam, sh, gt_planes=planes
                        )
                else:
                    ret, _, _ = relighting.image_forward(
                        cropped_image, cam, sh, gt_planes=planes
                    )

                recon_image = ret["image"]
                frame = render_tensor(paste_light_on_img_tensor(64, sh, recon_image))
                video_writer.append_data(np.array(frame))


def perform_video_relighting(relighting, device, example_lightings):
    frames = sorted(glob.glob("examples/video/cropped/*.jpg"))
    cams = sorted(glob.glob("examples/video/camera/*.npy"))
    frames_per_lighting = 20
    relighting.reset()
    prev_cam = []

    with imageio.get_writer("examples/video_relighting.mp4", fps=24) as video_writer:
        pbar = tqdm(total=len(frames))
        for idx, (frame_path, cam_path) in enumerate(zip(frames, cams)):
            with torch.inference_mode():
                frame = Image.open(frame_path)
                cam = np.load(cam_path)

                prev_cam.append(cam)
                if len(prev_cam) > 5:
                    prev_cam.pop(0)
                cam_avg = np.mean(prev_cam, axis=0)

                frame = preprocess_image(frame, device)
                cam_tensor = torch.from_numpy(cam_avg).to(device)

                cur_idx = (idx // frames_per_lighting) % len(example_lightings)
                next_idx = (cur_idx + 1) % len(example_lightings)
                alpha = (idx % frames_per_lighting) / frames_per_lighting
                cur_sh = example_lightings[cur_idx]
                next_sh = example_lightings[next_idx]
                sh = (
                    torch.from_numpy((1 - alpha) * cur_sh + alpha * next_sh)
                    .unsqueeze(0)
                    .to(device)
                )

                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        ret = relighting.video_forward(frame, cam_tensor, sh)
                else:
                    ret = relighting.video_forward(frame, cam_tensor, sh)

                recon_image = ret["image"]
                frame_output = render_tensor(
                    paste_light_on_img_tensor(64, sh, recon_image)
                )
                video_writer.append_data(np.array(frame_output))
                pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    main()
