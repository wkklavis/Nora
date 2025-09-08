import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d

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



# class CrossImageConsistency(nn.Module):
#     def __init__(self, dim=256, num_heads=8, image_embedding_size=16):
#         super(CrossImageConsistency, self).__init__()
#         self.pe_layer = PositionEmbeddingRandom(dim // 2)
#         self.image_embedding_size = (image_embedding_size, image_embedding_size)
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5
#
#         self.layer_norm = nn.LayerNorm(dim)
#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.proj = nn.Linear(dim, 1)
#
#         self.mask_downscaling = nn.Sequential(
#             nn.Conv2d(1, 16 // 4, kernel_size=2, stride=2),
#             LayerNorm2d(16 // 4),
#             nn.GELU(),
#             nn.Conv2d(16 // 4, 16, kernel_size=2, stride=2),
#             LayerNorm2d(16),
#             nn.GELU(),
#             nn.Conv2d(16, dim, kernel_size=1),
#         )  # downsample to 1/4
#
#     def get_dense_pe(self) -> torch.Tensor:
#         return self.pe_layer(self.image_embedding_size).unsqueeze(0)
#
#     # def forward(self, x):
#     #     B, C, H, W = x.shape
#     #
#     #     # Reshape to (B, H*W, C) for processing
#     #     x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
#     #
#     #     # Layer normalization
#     #     x = self.layer_norm(x)
#     #
#     #     # Linear projections for Q, K, V
#     #     qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
#     #     qkv = qkv.permute(2, 0, 3, 1, 4)
#     #     q, k, v = qkv[0], qkv[1], qkv[2]
#     #
#     #     result_list = []
#     #     for qi in q:
#     #         matmul_result = qi @ k.transpose(-2, -1)
#     #         sum_result = torch.sum(matmul_result, dim=0)
#     #         result_list.append(sum_result)
#     #     attn = torch.stack(result_list, dim=0)
#     #
#     #     attn = torch.softmax(attn / self.scale, dim=-1)
#     #     out = torch.einsum('bhij,bhjd->bhid', attn, v)
#     #     out = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
#     #     out = self.proj(out)
#     #
#     #     # Reshape back to (B, C, H, W)
#     #     consistent_prompt = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
#     #
#     #     sparse_embeddings = torch.empty((1, 0, C), device=device)
#     #
#     #     return sparse_embeddings, consistent_prompt
#     # def forward(self, x):
#     #     B, C, H, W = x.shape
#     #
#     #     # Reshape to (B, H*W, C) for processing
#     #     x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
#     #
#     #     # Layer normalization
#     #     x = self.layer_norm(x)
#     #
#     #     # Linear projections for Q, K, V
#     #     qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
#     #     qkv = qkv.permute(2, 0, 3, 1, 4)
#     #     q, k, v = qkv[0], qkv[1], qkv[2]
#     #
#     #     attn = q @ k.transpose(-2, -1)
#     #
#     #     attn = torch.softmax(attn / self.scale, dim=-1)
#     #     out = torch.einsum('bhij,bhjd->bhid', attn, v)
#     #     out = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
#     #     out = self.proj(out)
#     #
#     #     # Reshape back to (B, C, H, W)
#     #     consistent_prompt = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
#     #
#     #     sparse_embeddings = torch.empty((1, 0, C), device=device)
#     #
#     #     return sparse_embeddings, consistent_prompt

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x

class FFN(nn.Module):
    def __init__(self, dim=768):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(dim*4, dim//4)
        self.dwconv = DWConv(dim//4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim//4, dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class CrossImageAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super(CrossImageAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias = True)
        self.proj_k = nn.Linear(dim, dim, bias = True)
        self.proj_v = nn.Linear(dim, dim, bias = True)

        self.proj = nn.Linear(dim, dim)

    def forward(self, query, support, B, C, H, W):
        q = self.proj_q(query).reshape(B, H * W, self.num_heads, C // self.num_heads)
        k = self.proj_k(support).reshape(B, H * W, self.num_heads, C // self.num_heads)
        v = self.proj_v(support).reshape(B, H * W, self.num_heads, C // self.num_heads)
        result_list = []
        for qi in q:
            matmul_result = qi @ k.transpose(-2, -1)
            sum_result = torch.sum(matmul_result, dim=0)
            result_list.append(sum_result)
        attn = torch.stack(result_list, dim=0)
        attn = torch.softmax(attn / self.scale, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        out = self.proj(out)
        return out


class CrossImageConsistency(nn.Module):
    def __init__(self, dim=256, num_heads=8, image_embedding_size=16):
        super(CrossImageConsistency, self).__init__()
        self.pe_layer = PositionEmbeddingRandom(dim // 2)
        self.image_embedding_size = (image_embedding_size, image_embedding_size)

        self.instance_norm = nn.InstanceNorm2d(768)

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.cross_image_attention = CrossImageAttention(dim=dim, num_heads = num_heads)


    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape

        # Reshape to (B, H*W, C) for processing
        x_ = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        out = self.cross_image_attention(x_, x_, B, C, H, W)

        out = self.layer_norm(out)

        # Reshape back to (B, C, H, W)
        consistent_prompt = out.reshape(B, H, W, C).permute(0, 3, 1, 2) + x


        sparse_embeddings = torch.empty((1, 0, C), device=device)

        return sparse_embeddings, consistent_prompt