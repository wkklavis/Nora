import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
import torch.nn.functional as F

from segment_anything.modeling.image_encoder import PatchEmbed

device = "cuda" if torch.cuda.is_available() else "cpu"
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

# class SpatialPriorModule(nn.Module):
#     def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
#         super().__init__()
#         self.with_cp = with_cp
#
#         self.stem = nn.Sequential(*[
#             nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         ])
#         self.conv2 = nn.Sequential(*[
#             nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(2 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv3 = nn.Sequential(*[
#             nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv4 = nn.Sequential(*[
#             nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#
#     def forward(self, x):
#
#         def _inner_forward(x):
#             c1 = self.stem(x)
#             c2 = self.conv2(c1)
#             c3 = self.conv3(c2)
#             c4 = self.conv4(c3)
#             c1 = self.fc1(c1)
#             c2 = self.fc2(c2)
#             c3 = self.fc3(c3)
#             c4 = self.fc4(c4)
#
#             bs, dim, _, _ = c1.shape
#             # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
#             c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
#             c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
#             c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
#
#             return c1, c2, c3, c4
#
#         if self.with_cp and x.requires_grad:
#             outs = cp.checkpoint(_inner_forward, x)
#         else:
#             outs = _inner_forward(x)
#         return outs

class LearnableFrequencyFilter(nn.Module):
    def __init__(self, channel):
        super(LearnableFrequencyFilter, self).__init__()

        self.complex_weight = nn.Parameter(torch.randn(channel, 9, 16,  2, dtype=torch.float32) * 0.02)#Learnable filter

    def forward(self, x):
        N, C, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, dim=(1, 2), norm='ortho')

        return x
class FrequencyPromptGenerator(nn.Module):
    def __init__(self, embed_dim=256, image_embedding_size=16):
        super(FrequencyPromptGenerator, self).__init__()
        self.lap_pyramid = Lap_Pyramid_Conv(3)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.image_embedding_size = (image_embedding_size, image_embedding_size)

        self.proj = nn.Conv2d(3, embed_dim, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0))
        self.filter = LearnableFrequencyFilter(256)
    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    def fft(self, x, rate, prompt_type):
        mask = torch.zeros(x.shape).to(device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

        if prompt_type == 'highpass':
            fft = fft * (1 - mask)
        elif prompt_type == 'lowpass':
            fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv

    def forward(self, x):
        N, C, H, W = x.shape
        # if self.input_type == 'laplacian':
        #     pyr_A = self.lap_pyramid.pyramid_decom(img=x)
        #     x = pyr_A[:-1]
        #     laplacian = x[0]
        #     for x_i in x[1:]:
        #         x_i = F.interpolate(x_i, size=(laplacian.size(2), laplacian.size(3)), mode='bilinear', align_corners=True)
        #         laplacian = torch.cat([laplacian, x_i], dim=1)
        #     x = laplacian

        # x = self.fft(x, 4, 'highpass')
        x = self.proj(x)
        prompt = self.filter(x)

        sparse_embeddings = torch.empty((1, 0, 256), device=device)
        return sparse_embeddings, prompt
