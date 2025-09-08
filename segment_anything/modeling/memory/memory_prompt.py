import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import random
from sklearn.manifold import TSNE
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type

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


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim=256, fea_dim=256):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.reset_parameters()
        self.att_weight = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if len(input.shape) == 1:  # 出现训练最后一个step时，出现一维的情况
            input = torch.unsqueeze(input, 0)
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)
        self.att_weight = att_weight.squeeze(0)
        # TxM 距离权重
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return output # output
        # proto, cos_sim_map = self.domain_invariant_info_guide(embedding_feature,feature)
        # feature = torch.cat([feature, proto, cos_sim_map],dim=1)
        # channel_compress = getattr(self, 'channel_compress_{}'.format(str(i)))
        # feature = channel_compress(feature)

class PrototypePromptGenerate(nn.Module):
    def __init__(self, mem_dim=256, embed_dim=256, image_embedding_size=24):
        super(PrototypePromptGenerate, self).__init__()
        self.memory_bank = MemoryUnit(mem_dim, embed_dim)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.image_embedding_size = (image_embedding_size, image_embedding_size)
        self.fuse_conv = nn.Conv2d(513, 256, 1)
        # self.prototype = None
        # self.di_prototype = None

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, feature):
        N, C, H, W = feature.shape
        feature_proto_avg = F.avg_pool2d(input=feature, kernel_size=feature.shape[-2:])#b x c x 1 x 1
        feature_proto_max = F.max_pool2d(input=feature, kernel_size=feature.shape[-2:])#b x c x 1 x 1
        feature_proto = (feature_proto_avg + feature_proto_max)

        feature_proto = feature_proto.squeeze()
        di_proto = self.memory_bank(feature_proto)#b x c

        # prototype = feature_proto.cpu().detach().numpy()
        # memory_bank = self.memory_bank.weight.cpu().detach().numpy()
        # di_prototype = di_proto.cpu().detach().numpy()
        # feature = feature.cpu().detach().numpy().reshape(-1,256 )
        # # 将所有特征合并成一个数组
        # all_features = np.vstack((memory_bank, prototype, di_prototype, feature))
        # # 使用 t-SNE 进行降维
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_result = tsne.fit_transform(all_features)
        # # 提取每个类别的 t-SNE 结果
        # tsne_memory_bank = tsne_result[:256, :]
        # tsne_prototype = tsne_result[256:264, :]
        # tsne_di_prototype = tsne_result[264:272, :]
        # tsne_feature = tsne_result[272:, :]
        # # 可视化结果
        # plt.figure(figsize=(50, 50))
        # # 画出每个类别的散点图，使用不同颜色
        # plt.scatter(tsne_feature[:, 0], tsne_feature[:, 1], label='feature', alpha=0.4,
        #             c='gray')
        # plt.scatter(tsne_memory_bank[:, 0], tsne_memory_bank[:, 1], label='memory_bank', alpha=0.7,
        #             c='orange')
        # plt.scatter(tsne_prototype[:, 0], tsne_prototype[:, 1], label='prototype', alpha=0.7,
        #             c='blue')
        # plt.scatter(tsne_di_prototype[:, 0], tsne_di_prototype[:, 1], label='di_prototype', alpha=0.7,
        #             c='green')
        #
        # # 设置标题和图例
        # plt.title('t-SNE Visualization for training')
        # plt.legend()
        # plt.savefig('./result/train_tsne.png')
        # plt.show()
        # self.prototype = feature_proto
        # self.di_prototype = di_proto.squeeze(0)

        di_proto = di_proto.unsqueeze(2).unsqueeze(2)
        info_proto = di_proto.expand_as(feature)

        cos_sim_map = F.cosine_similarity(info_proto, feature, dim=1, eps=1e-7)  # b x h x w
        cos_sim_map = cos_sim_map.unsqueeze(1)# b x 1 x h x w

        prompt = self.fuse_conv(torch.concat([feature, info_proto, cos_sim_map], dim=1))
        # prompt = cos_sim_map * info_proto

        sparse_embeddings = torch.empty((1, 0, C), device=device)
        return sparse_embeddings, prompt

class FeatureReconstruct(nn.Module):
    def __init__(self, mem_dim=19, fea_dim=256):
        super(FeatureReconstruct, self).__init__()
        self.memory = MemoryUnit(mem_dim, fea_dim)
        self.compress = nn.Conv2d(512,256,1)
    def forward(self, feature):
        N, C, H, W = feature.shape
        x = feature.permute(0, 2, 3, 1)
        x = x.contiguous()
        x = x.view(-1, C)
        y = self.memory(x)
        y = y.view(N, H, W, C)
        y = y.permute(0, 3, 1, 2)
        feature = self.compress(torch.concat([feature, y],1))
        return feature
