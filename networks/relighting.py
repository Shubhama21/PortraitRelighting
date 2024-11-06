import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from third_party.wrappers import (
    WrapperNeRFFaceLighting,
)
from networks.mix_transformer import OverlapPatchEmbed, Block as TransformerBlock
from third_party.NeRFFaceLighting.torch_utils import misc
from third_party.NeRFFaceLighting.training.networks_stylegan2 import (
    FullyConnectedLayer,
    SynthesisBlock,
)


def remove_batch_norm(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, nn.Identity())
        else:
            remove_batch_norm(child)


class DeepLabV3(nn.Module):
    """
    Modified DeepLabV3 from segmentation_models_pytorch
    Input: [B,5,H,W].       Image concat with coord
    Output: [B,256,64,64]   feature map -- f_low
    """

    def __init__(self):
        super().__init__()
        self.model = smp.DeepLabV3().eval()
        self.model.encoder.conv1 = torch.nn.Conv2d(
            5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        remove_batch_norm(self.model)
        self.model.segmentation_head = torch.nn.Identity()

    def forward(self, x):
        return self.model(x)


class Decoder_f_low(nn.Module):
    """
    ViT decoder for f_low
    Input:  [B,256,64,64]   feature map -- f_low
    Output: [B,96,256,256]  feature map -- F(conv), for concat with f_high
    """

    def __init__(self):
        super().__init__()
        self.input_layer = OverlapPatchEmbed(
            img_size=64, patch_size=3, stride=2, in_chans=256, embed_dim=1024
        )
        self.vit1 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.vit2 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.vit3 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.vit4 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.vit5 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.convs = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),  # [1,128,256,256]
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, f_low):
        f_low = self.input_layer(f_low)
        H, W = f_low[1], f_low[2]
        f_low = f_low[0]  # [1,1024,1024]
        f_low = self.vit1(f_low, H, W)
        f_low = self.vit2(f_low, H, W)
        f_low = self.vit3(f_low, H, W)
        f_low = self.vit4(f_low, H, W)
        f_low = self.vit5(f_low, H, W)  # [1,1024,1024]
        f_low = self.convs(f_low.view(-1, 1024, H, W).contiguous())
        return f_low


class Decoder_f_low_fast(nn.Module):
    """
    ViT decoder for f_low
    Input:  [B,256,64,64]   feature map -- f_low
    Output: [B,96,256,256]  feature map -- F(conv), for concat with f_high
    """

    def __init__(self):
        super().__init__()
        self.input_layer = OverlapPatchEmbed(
            img_size=64, patch_size=3, stride=2, in_chans=256, embed_dim=1024
        )
        self.vit1 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.vit2 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.vit3 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.vit4 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.vit5 = TransformerBlock(
            dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1, qkv_bias=True
        )
        self.convs = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),  # [1,128,256,256]
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, f_low):
        with torch.cuda.amp.autocast():
            with torch.cuda.amp.autocast(enabled=False):
                f_low = self.input_layer(f_low)
                H, W = f_low[1], f_low[2]
                f_low = f_low[0]  # [1,1024,1024]

            f_low = self.vit1(f_low, H, W)
            f_low = self.vit2(f_low, H, W)
            f_low = self.vit3(f_low, H, W)
            f_low = self.vit4(f_low, H, W)
            f_low = self.vit5(f_low, H, W)  # [1,1024,1024]
            f_low = self.convs(f_low.view(-1, 1024, H, W).contiguous())
        return f_low


class Encoder_f_high(nn.Module):
    """
    Encoder for f_high
    Input:  [B,3,256,256]   image
    Output: [B,96,256,256]   feature map -- f_high
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
            ),  # [B,64,128,128]
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),  # [B,96,128,128]
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),  # [B,96,128,128]
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),  # [B,96,128,128]
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),  # [B,96,128,128]
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, img):
        return self.layers(img)


class Decoder_f_hybrid(nn.Module):
    """
    Decoder for f_low and f_high, output triplane
    Input:  f_low [B,96,256,256]
            f_high [B,96,256,256]
    Output: [B,96,256,256]

    """

    def __init__(self):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.embed = OverlapPatchEmbed(
            img_size=256, patch_size=3, stride=2, in_chans=128, embed_dim=1024
        )
        self.vit = TransformerBlock(
            dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2, qkv_bias=True
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        # here concat the f_low to f_high
        self.layers2 = nn.Sequential(
            nn.Conv2d(352, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, f_low, f_high):
        f_high = self.layers1(torch.cat([f_low, f_high], dim=1))
        f_high = self.embed(f_high)
        H, W = f_high[1], f_high[2]
        f_high = f_high[0]
        f_high = self.vit(f_high, H, W)
        f_high = self.pixel_shuffle(f_high.view(-1, 1024, H, W).contiguous())
        f_high = torch.cat([f_low, f_high], dim=1)
        f_high = self.layers2(f_high)
        return f_high


class Planes2WS(nn.Module):
    def __init__(self):
        super().__init__()
        # [B,96,256,256] to [B,512]
        self.layers = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
        )

    def forward(self, planes):
        return self.layers(planes).unsqueeze(1).repeat(1, 14, 1)  # [B,14,512]


class DualEncoders(nn.Module):
    def __init__(self, device, verbose=False):
        super().__init__()
        self.device = device
        self.deeplabv3 = DeepLabV3().eval()
        self.decoder_f_low = Decoder_f_low().eval()
        self.encoder_f_high = Encoder_f_high().eval()
        self.decoder_f_hybrid = Decoder_f_hybrid().eval()
        self.shading_encoder = ShadingEncoder().eval()
        self.eg3d = WrapperNeRFFaceLighting(device, verbose=verbose)
        self.shading_encoder.load_state_dict(
            self.eg3d.G.backbone.synthesis_lit.state_dict(), strict=False
        )
        if verbose:
            print("shading encoder is inited")
        self.mean_ws = self.eg3d.seed2ws(0, 1, 0).requires_grad_(False)  # [1,14,512]
        self.planes2ws = Planes2WS().eval()
        self.cood_cache = None

    def get_coord(self, img):
        """
        Get coordinate maps of images
        Input: img [B,3,H,W]
        Output: coord [B,2,H,W]
        """
        # img [B,3,H,W]
        if self.cood_cache is not None:
            return self.cood_cache
        B, _, H, W = img.shape
        x = torch.linspace(-1, 1, W).repeat(B, H, 1).to(img.device)
        y = torch.linspace(-1, 1, H).repeat(B, W, 1).transpose(1, 2).to(img.device)
        coord = torch.stack([x, y], dim=1)  # [B,2,H,W]
        self.cood_cache = coord
        return coord

    def forward(
        self,
        img,
        cam,
        sh,
        gt_planes=None,
        gt_planes_lit=None,
        gt_superres_ws=None,
        no_render=False,
    ):
        """
        Input:
            img: [B,3,512,512] in range [-1,1]
            cam: [B,25]
            sh:  [B,9]
            gt_planes: [B,96,256,256] optional
            gt_planes_lit: [B,96,256,256] optional
        Return:
            output: dict, contains:
                image,image_raw,image_abledo,image_shading
            planes: [B,96,256,256]
            planes_lit: [B,96,256,256]

        """
        if gt_planes is not None:
            planes = gt_planes
        else:
            with torch.cuda.amp.autocast(enabled=False):
                img_with_coord = torch.cat(
                    [img, self.get_coord(img)], dim=1
                )  # [B,5,H,W]
                f_low = self.deeplabv3(img_with_coord)  # [B,256,64,64]
                f_low = self.decoder_f_low(f_low)  # [B,96,256,256]
            f_high = self.encoder_f_high(img_with_coord)  # [B, 96, 256, 256]
            planes = self.decoder_f_hybrid(f_low, f_high)  # [B,96,256,256]
            planes = planes.float()  # otherwise will cause black image when using amp

        with torch.cuda.amp.autocast(enabled=False):
            if gt_planes_lit is not None:
                planes_lit = gt_planes_lit
            else:
                ws_lit = self.eg3d.sh2wslit(sh)
                planes_lit = self.shading_encoder(planes, ws_lit)

        if gt_superres_ws is not None:
            superres_ws = gt_superres_ws
        else:
            superres_ws = self.mean_ws.repeat(planes.shape[0], 1, 1).to(planes.device)
            superres_ws += self.planes2ws(planes)

        if no_render:
            output = {}
        else:
            with torch.cuda.amp.autocast(enabled=False):
                output = self.eg3d.planes2all(planes, planes_lit, cam, superres_ws)

        output["superres_ws"] = superres_ws  # merge superres_ws into output
        return output, planes, planes_lit


class LittleMapping(torch.nn.Module):
    def __init__(self, img_channels=96, channels=128, w_dim=512):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(img_channels, channels, 1)
        self.act1 = torch.nn.LeakyReLU(0.2)
        self.conv2 = torch.nn.Conv2d(channels, channels, 1)
        self.act2 = torch.nn.LeakyReLU(0.2)
        self.conv3 = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        return x


class ShadingEncoder(torch.nn.Module):
    def __init__(
        self,
        w_dim=512,  # Intermediate latent (W) dimensionality.
        img_resolution=256,  # Output image resolution.
        img_channels=96,  # Number of color channels.
        channels=128,  # Fixed Number of channels.
        num_blocks=2,  # Number of synthesis blocks.
        num_fp16_blk=0,  # Use FP16 for the N highest blocks.
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        assert num_fp16_blk <= num_blocks
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.num_fp16_blk = num_fp16_blk

        self.num_ws = 0
        for num_block in range(num_blocks):
            use_fp16 = num_block >= num_blocks - num_fp16_blk
            is_last = num_block == num_blocks - 1
            block = SynthesisBlock(
                channels,
                channels,
                w_dim=w_dim,
                resolution=img_resolution,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                disable_upsample=True,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f"b{num_block}", block)

        self.mapping_x = LittleMapping()

    def forward(self, plane, ws, **block_kwargs):
        # plane [B,96,256,256]
        # ws [B,5,512]
        block_ws = []
        with torch.autograd.profiler.record_function("split_ws"):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for num_block in range(self.num_blocks):
                block = getattr(self, f"b{num_block}")
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = self.mapping_x(plane)
        for num_block, cur_ws in zip(range(self.num_blocks), block_ws):
            block = getattr(self, f"b{num_block}")
            x, plane = block(x, plane, cur_ws, **block_kwargs)
        return plane

    def extra_repr(self):
        return " ".join(
            [
                f"w_dim={self.w_dim:d}, num_ws={self.num_ws:d},",
                f"img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},",
                f"num_fp16_blk={self.num_fp16_blk:d}",
            ]
        )


import torch
import torch.nn as nn


class ImageTransformer(nn.Module):
    def __init__(self, input_channels=96, num_heads=8, hidden_size=512, num_layers=6):
        super(ImageTransformer, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_channels, num_heads=num_heads
        )

        self.feedforward = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_channels),
        )

        self.norm1 = nn.LayerNorm(input_channels)
        self.norm2 = nn.LayerNorm(input_channels)
        self.dropout = nn.Dropout(0.1)
        self.num_layers = num_layers

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, input_channels=96, num_heads=8, hidden_size=512):
        super(CrossAttentionBlock, self).__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_channels, num_heads=num_heads
        )
        self.feedforward = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_channels),
        )
        self.norm1 = nn.LayerNorm(input_channels)
        self.norm2 = nn.LayerNorm(input_channels)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        attn_output, _ = self.cross_attention(x, y, y)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class ImageTransformerModel(nn.Module):
    def __init__(
        self,
        input_channels=96,
        num_heads=8,
        hidden_size=512,
        num_layers=4,
        num_images=5,
    ):
        super(ImageTransformerModel, self).__init__()
        self.num_images = num_images
        self.combine_channels = nn.Conv2d(
            input_channels * num_images, input_channels, kernel_size=1
        )
        self.input_convs1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.input_convs2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.output_conv1 = nn.ConvTranspose2d(
            input_channels, input_channels * num_images, kernel_size=1
        )
        self.output_conv2 = nn.ConvTranspose2d(
            input_channels, input_channels * num_images, kernel_size=1
        )
        self.output_conv1_2 = nn.ConvTranspose2d(
            input_channels, input_channels, kernel_size=1
        )
        self.output_conv2_2 = nn.ConvTranspose2d(
            input_channels, input_channels, kernel_size=1
        )

        self.transformer_blocks_stream1 = nn.ModuleList(
            [
                ImageTransformer(input_channels, num_heads, hidden_size, num_layers)
                for _ in range(num_layers)
            ]
        )

        self.transformer_blocks_stream2 = nn.ModuleList(
            [
                ImageTransformer(input_channels, num_heads, hidden_size, num_layers)
                for _ in range(num_layers)
            ]
        )

        self.cross_attention_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(input_channels, num_heads, hidden_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y):
        x_orig = x.clone().view(self.num_images, 96, 256, 256)
        y_orig = y.clone().view(self.num_images, 96, 256, 256)
        x = self.combine_channels(x)
        y = self.combine_channels(y)
        x = nn.ReLU(inplace=True)(x)
        y = nn.ReLU(inplace=True)(y)
        x = self.input_convs1(x)
        y = self.input_convs2(y)

        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)

        B, H, W, C = x.size()
        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        for transformer_block1, transformer_block2, cross_attention_block in zip(
            self.transformer_blocks_stream1,
            self.transformer_blocks_stream2,
            self.cross_attention_blocks,
        ):
            x = transformer_block1(x)
            y = transformer_block2(y)
            y = cross_attention_block(y, x)
            x = cross_attention_block(x, y)

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        y = y.view(B, H, W, C)
        y = y.permute(0, 3, 1, 2)

        x = self.output_conv1(x).view(self.num_images, 96, 256, 256)
        y = self.output_conv2(y).view(self.num_images, 96, 256, 256)
        x = nn.ReLU(inplace=True)(x)
        y = nn.ReLU(inplace=True)(y)
        x = self.output_conv1_2(x)
        y = self.output_conv2_2(y)

        x = x * x_orig
        y = y * y_orig

        x = torch.sum(x, dim=0).unsqueeze(0)
        y = torch.sum(y, dim=0).unsqueeze(0)

        return x, y


class Relighting(nn.Module):
    def __init__(self, device, verbose=False):
        super().__init__()
        self.encoders = DualEncoders(device, verbose=verbose)
        self.attention = ImageTransformerModel()
        self.device = device
        self.steps = 5
        self.cnt = 0
        self.prev_planes, self.prev_planes_lit = [], []

    def reset(self):
        self.prev_planes, self.prev_planes_lit = [], []
        self.cnt = 0

    def forward(self):
        raise NotImplementedError

    def video_forward(
        self,
        img,
        cam,
        sh,
        gt_planes=None,
        gt_planes_lit=None,
        gt_superres_ws=None,
        no_render=False,
    ):
        self.cnt += 1
        raw_output, planes, planes_lit = self.encoders(img, cam, sh)
        if gt_planes is not None:
            planes = gt_planes
        if gt_planes_lit is not None:
            planes_lit = gt_planes_lit
        superres_ws = (
            raw_output["superres_ws"] if gt_superres_ws is None else gt_superres_ws
        )
        if self.cnt <= self.steps:
            self.prev_planes.append(planes.clone().detach().unsqueeze(0))
            self.prev_planes_lit.append(planes_lit.clone().detach().unsqueeze(0))
            if no_render:
                return planes, planes_lit
            return raw_output
        else:
            planes_input = torch.cat(self.prev_planes, dim=0)
            planes_lit_input = torch.cat(self.prev_planes_lit, dim=0)
            planes_residual, planes_lit_residual = self.attention(
                planes_input.view(1, self.steps * 96, 256, 256),
                planes_lit_input.view(1, self.steps * 96, 256, 256),
            )
            planes_pred = planes + planes_residual if gt_planes is None else gt_planes
            planes_lit_pred = (
                planes_lit + planes_lit_residual
                if gt_planes_lit is None
                else gt_planes_lit
            )
            with torch.cuda.amp.autocast(enabled=False):
                output = self.encoders.eg3d.planes2all(
                    planes_pred, planes_lit_pred, cam, superres_ws
                )
            self.prev_planes = self.prev_planes[-self.steps :]
            self.prev_planes_lit = self.prev_planes_lit[-self.steps :]
            if no_render:
                return planes_pred, planes_lit_pred
            return output

    def image_forward(
        self,
        img,
        cam,
        sh,
        gt_planes=None,
        gt_planes_lit=None,
        gt_superres_ws=None,
        no_render=False,
    ):
        return self.encoders(
            img, cam, sh, gt_planes, gt_planes_lit, gt_superres_ws, no_render
        )
