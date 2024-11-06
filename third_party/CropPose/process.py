# from crop import Cropper, eg3d_detect_keypoints
# from pose import Poser


# class CroPoser:
#     def __init__(self):
#         self.cropper = Cropper()
#         self.poser = Poser()
#         self.intrinsics = FOV_to_intrinsics(18.837).reshape(-1, 9)

#     def FOV_to_intrinsics(fov_degrees, device="cpu"):
#         """
#         Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
#         Note the intrinsics are returned as normalized by image size, rather than in pixel units.
#         Assumes principal point is at image center.
#         """

#         focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
#         intrinsics = torch.tensor(
#             [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device
#         )
#         return intrinsics

#     def wild2align(self, img: torch.tensor):
#         """
#         img     - torch.tensor [-1,1] float [B,3,H,W]
#         return  - img [B,3,512,512]

#         Align input image and return cropped image.
#         """
#         img_pil = Image.fromarray(img.cpu().numpy()).convert("RGB")
#         keypoints = eg3d_detect_keypoints(img_pil)
#         pred_coeffs = cropper.get_deep3d_coeffs(img_pil, keypoints)
#         img_cropped = cropper.final_crop(img_pil, keypoints)
#         cropped_img = (
#             torch.from_numpy(np.array(im)).float().permute(2, 0, 1)[None, ...] / 127.5
#             - 1
#         ).cpu()
#         pose = poser.get_pose(pred_coeffs)
#         cam = torch.cat(
#             [torch.from_numpy(pose["pose"]).cpu().reshape(-1, 16), intrinsics], dim=1
#         ).float()
#         return cropped_img

#     def img2cam(self, img: torch.tensor):
#         """

#         img     - torch.tensor [-1,1] float [B,3,H,W]
#         return  - cam [B,25]

#         Compute camera from input image (wild or aligned).
#         """
#         img_pil = Image.fromarray(img.cpu().numpy()).convert("RGB")
#         keypoints = eg3d_detect_keypoints(img_pil)
#         pose = poser.get_pose(pred_coeffs)
#         cam = torch.cat(
#             [torch.from_numpy(pose["pose"]).cpu().reshape(-1, 16), intrinsics], dim=1
#         ).float()
#         return cam
