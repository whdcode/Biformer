"""
BiFormer-STL (Swin-Tiny-Layout) model
From
"author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com"

“
Model dimensions are extended to 3D
The author of the re-creation: Huidong Wu
”
"""
import math
from typing import Tuple, Union
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torch import Tensor, LongTensor
from typing import Optional, Tuple


class LayerNorm3D(nn.Module):
    r"""LayerNorm that supports three-dimensional inputs.

    Args:
        normalized_shape (int or tuple): Input shape from an expected input. If it is a single integer,
            it is treated as a singleton tuple. Default: 1
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            a = self.weight.shape
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
    """
    Args:
        x: BCHW tensor
        region size: int
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col/row
    """
    B, C, H, W, D = x.size()
    region_h, region_w, region_d = H // region_size[0], W // region_size[1], D // region_size[2]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1], region_d,
               region_size[2])
    x = torch.einsum('bmdhpwqtu->bmhwtpqud', x).flatten(2, 3).flatten(2, 3).flatten(-3, -2).flatten(-3,
                                                                                                    -2)  # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w, region_d


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_d: int, region_size: Tuple[int]):
    """
    Args:
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, C, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_d, region_size[0], region_size[1], region_size[2], head_dim)
    x = torch.einsum('bmhwtpqud->bmdhpwqtu', x).reshape(bs, nhead * head_dim,
                                                        region_h * region_size[0], region_w * region_size[1],
                                                        region_d * region_size[2])
    return x


class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


# class InvertedResidualsBlock(nn.Module):
#     def __init__(self, in_channels, expansion, stride):
#         super(InvertedResidualsBlock, self).__init__()
#         channels = expansion * in_channels
#         self.stride = stride
#         self.basic_block = nn.Sequential(
#             nn.Conv3d(in_channels, channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm3d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
#             nn.BatchNorm3d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(channels, in_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm3d(in_channels),
#         )
#
#     def forward(self, x):
#         x_res = x
#         x = self.basic_block(x) + x_res
#         return x


class InvertedResidualsBlock(nn.Module):
    def __init__(self, in_channels):
        super(InvertedResidualsBlock, self).__init__()
        self.dan = DualAdaptiveNeuralBlock(embed_dim=in_channels)

    def forward(self, x):
        x_res = x
        x = self.dan(x) + x_res
        return x


class SpatialAttention_Up(nn.Module):
    def __init__(self):
        super(SpatialAttention_Up, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout = torch.max(x, dim=1, keepdim=True).values
        out = torch.cat([avgout, maxout], 1)
        out = self.sigmoid(self.conv(out))
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=True)

        return out


class Conv_BasicLayer(nn.Module):
    """
    Stack several InvertCONV Blocks
    """

    def __init__(self, dim, depth, mlp_ratio, stride=1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.spa_att = SpatialAttention_Up()
        self.sigmoid = nn.Sigmoid()

        self.blocks = nn.ModuleList([
            # InvertedResidualsBlock(self.dim, expansion=self.mlp_ratio, stride=stride)
            InvertedResidualsBlock(self.dim)
            for _ in range(self.depth)
        ])

    def forward(self, x_bf, x_co):
        """
        :param x_bf: b, c, h, w, d
        :param x_co: b, 2c, 2h, 2w, 2d
        :return: b, c, h, w, d
        """
        # b, 1, h, w, d
        score = self.spa_att(x_bf)
        for blk in self.blocks:
            x_co = blk(x_co)
        return x_co * score

# class MultiScaleDWConv(nn.Module):
#     def __init__(self, dim, scale=(1, 3, 5, 7)):
#         super().__init__()
#         self.scale = scale
#         self.channels = []
#         self.proj = nn.ModuleList()
#         for i in range(len(scale)):
#             if i == 0:
#                 channels = dim - dim // len(scale) * (len(scale) - 1)
#             else:
#                 channels = dim // len(scale)
#             conv = nn.Conv3d(channels, channels,
#                              kernel_size=scale[i],
#                              padding=scale[i] // 2,
#                              groups=channels)
#             self.channels.append(channels)
#             self.proj.append(conv)
#
#     def forward(self, x):
#         x = torch.split(x, split_size_or_sections=self.channels, dim=1)
#         out = []
#         for i, feat in enumerate(x):
#             out.append(self.proj[i](feat))
#         x = torch.cat(out, dim=1)
#         return x
class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, di_kernel=3, scale=(1, 3, 5, 7), di_rate=(1, 2, 3)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
                conv = nn.Conv3d(channels, channels,
                                 kernel_size=scale[i],
                                 padding=scale[i] // 2,
                                 groups=channels)
                self.channels.append(channels)
                self.proj.append(conv)
            else:
                channels = dim // len(scale)
                conv = nn.Sequential(nn.Conv3d(channels, channels,
                                               kernel_size=scale[i],
                                               padding=scale[i] // 2,
                                               groups=channels),
                                     nn.BatchNorm3d(channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv3d(channels, channels,
                                               kernel_size=di_kernel,
                                               padding=(di_kernel + (di_kernel - 1) * (di_rate[i-1] - 1)) // 2,
                                               groups=channels,
                                               dilation=di_rate[i - 1]),
                                     nn.Conv3d(channels, channels,
                                               kernel_size=1),
                                     nn.BatchNorm3d(channels)
                                     )
                self.channels.append(channels)
                self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class DualAdaptiveNeuralBlock(nn.Module):
    def __init__(self, embed_dim):
        super(DualAdaptiveNeuralBlock, self).__init__()
        self.embed_dim = embed_dim
        self.c_split_conv = nn.Sequential(nn.Conv3d(embed_dim, 2 * embed_dim, 1),
                                          nn.BatchNorm3d(2 * embed_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(2 * embed_dim, 2 * embed_dim, 3, 1, 1, groups=embed_dim),
                                          )
        self.px_score_conv = nn.Conv3d(2 * embed_dim, 2 * embed_dim, 1)
        self.post_conv = nn.Sequential(nn.Conv3d(2 * embed_dim, embed_dim, 1),
                                       nn.BatchNorm3d(embed_dim))
        # self.eca = ECA_block(embed_dim)

    def forward(self, x):
        B, C, H, W, D = x.size()
        x = self.c_split_conv(x)
        x = torch.sigmoid(self.px_score_conv(x)) * F.gelu(x)
        x = self.post_conv(x)
        # x = self.eca(x.squeeze(2))
        return x


class MSABlock(nn.Module):  ### MS-FFN
    """
    Mlp implemented by with 1x1 convolutions.

    Input: Tensor with shape [B, C, H, W, D].
    Output: Tensor with shape [B, C, H, W, D].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop_rate=0.,
                 scale=(1, 3, 5, 7)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(in_features, scale=scale)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm3d(in_features)
        self.fc2 = nn.Sequential(
            nn.Conv3d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_features),
        )
        self.drop = nn.Dropout(p=drop_rate)

    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.dwconv(x) + x
    #     x = self.norm(self.act(x))
    #     x = self.fc2(x)
    #     return x

    def forward(self, x):
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.fc1(x)
        return x


def regional_routing_attention_torch(
        query: Tensor, key: Tensor, value: Tensor, scale: float,
        region_graph: LongTensor, region_size: Tuple[int],
        kv_region_size: Optional[Tuple[int]] = None,
        auto_pad=True) -> Tensor:
    """
    Args:
        query, key, value: (B, C, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
        region_size: region/window size for queries, (rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
        auto_pad: required to be true if the input sizes are not divisible by the region_size
    Return:
        output: (B, C, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # Auto pad to deal with any input size
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq, Dq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        q_pad_u = (region_size[2] - Dq % region_size[2]) % region_size[2]
        if q_pad_b > 0 or q_pad_r > 0 or q_pad_u > 0:
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b, 0, q_pad_u))  # zero padding

        _, _, Hk, Wk, Dk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        kv_pad_u = (kv_region_size[2] - Wk % kv_region_size[2]) % kv_region_size[2]
        if kv_pad_r > 0 or kv_pad_b > 0 or kv_pad_b > 0:
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b, 0, kv_pad_u))  # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b, 0, kv_pad_u))  # zero padding

    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_h, q_region_w, q_region_d = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values.
    # TODO: is seperate gathering slower than fused one (our old version) ?
    # torch.gather does not support broadcasting, hence we do it manually
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1). \
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    # 此处的gather操作首先注意做了一次维度扩增后再进行kv_nregion大小的复制，然后在按照索引在第3维进行聚合，如index=（1,4,64,2,343,16）,key=(1,4,64,64,343,16)，out = gather(key,index,dim=3)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                         expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                         index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                           expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                           index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)

    # token-to-token attention
    # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
    # TODO: mask padding region
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    # (bs, nhead, q_nregion, reg_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
    # -> (bs, nhead, q_nregion, reg_size, head_dim)
    output = attn @ value_g.flatten(-3, -2)

    # to BCHW format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_d=q_region_d, region_size=region_size)

    # remove paddings if needed
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0 or q_pad_u > 0):
        output = output[:, :, :Hq, :Wq, Dq]

    return output, attn


class nchwBRA(nn.Module):
    """Bi-Level Routing Attention that takes nchw input

    Compared to legacy version, this implementation:
    * removes unused args and components
    * uses nchw input format to avoid frequent permutation

    When the size of inputs is not divisible by the region size, there is also a numerical difference
    than legacy implementation, due to:
    * different way to pad the input feature map (padding after linear projection)
    * different pooling behavior (count_include_pad=False)

    Current implementation is more reasonable, hence we do not keep backward numerical compatiability
    """

    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4, side_dwconv=3, auto_pad=False,
                 attn_backend='torch', scale=(1, 3, 5, 7)):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5  # NOTE: to be consistent with old models.

        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = nn.Conv3d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ regional routing setting #################
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col

        ##########################################

        self.qkv_linear = nn.Conv3d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv3d(self.dim, self.dim, kernel_size=1)
        self.ms = MSABlock(dim, scale=scale)

        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x: Tensor, ret_attn_mask=False):
        """
        Args:
            x: NCHW tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NCHW tensor
        """
        N, C, H, W, D = x.size()
        region_size = (H // self.n_win, W // self.n_win, D // self.n_win)

        # STEP 1: linear projection
        x = self.ms(x)
        qkv = self.qkv_linear.forward(x)  # ncHW
        q, k, v = qkv.chunk(3, dim=1)  # ncHW

        # STEP 2: region-to-region routing
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        q_r = F.avg_pool3d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool3d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchw
        q_r: Tensor = q_r.permute(0, 2, 3, 4, 1).flatten(1, 2).flatten(1, 2)  # n(hwd)c
        k_r: Tensor = k_r.flatten(2, 3).flatten(2, 3)  # nc(hwd)
        a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        # STEP 3: token to token attention (non-parametric function)
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size
                                        )

        output = output + self.lepe(v)  # ncHW
        output = self.output_linear(output)  # ncHW

        if ret_attn_mask:
            return output, attn_mat

        return output


class BiFormerBlock(nn.Module):
    """
    Attention + FFN
    """

    def __init__(self, dim, drop_path=0., num_heads=8, n_win=7,
                 qk_scale=None, topk=4, mlp_ratio=4, side_dwconv=5, scale=(1, 3, 5, 7)
                 ):

        super().__init__()
        self.norm1 = LayerNorm3D(dim)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = nchwBRA(dim=dim, num_heads=num_heads, n_win=n_win,
                                qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv, scale=scale)
        else:
            raise ValueError('topk should >0!')

        self.norm2 = LayerNorm3D(dim)
        self.mlp = nn.Sequential(nn.Conv3d(dim, int(mlp_ratio * dim), kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio * dim), dim, kernel_size=1)
                                 )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # attention & mlp
        x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, C, H, W, D)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, C, H, W, D)
        return x


class Bf_BasicLayer(nn.Module):
    """
    Stack several BiFormer Blocks
    """

    def __init__(self, dim, depth, num_heads, n_win, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5, scale=None):
        super().__init__()
        if scale is None:
            scale = (1, 3, 5, 7)
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            BiFormerBlock(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                num_heads=num_heads,
                n_win=n_win,
                topk=topk,
                mlp_ratio=mlp_ratio,
                side_dwconv=side_dwconv,
                scale=scale
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: NCHWD tensor
        Return:
            NCHWD tensor
        """
        for blk in self.blocks:
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class MSBiFormer(nn.Module):
    """
    Replace WindowAttn-ShiftWindowAttn in Swin-T model with Bi-Level Routing Attention
    """

    def __init__(self, in_chans=1, num_classes=1,
                 bf_depth=None,
                 co_depth=None,
                 embed_dim=None,
                 head_dim=16, qk_scale=None,
                 drop_path_rate=0.15, drop_rate=0.,
                 mlp_ratios: list = None,
                 invert_conv_mlp_ratios: list = None,
                 norm_layer=None,
                 pre_head_norm_layer=None,
                 n_wins: Tuple[int, int, int] = (),
                 topks: Tuple[int, int, int] = (),
                 side_dwconv: int = 5,
                 data_flows: str = 'conv_bf',  # only_conv
                 scale=None
                 ):
        super().__init__()
        self.co_depth = co_depth
        if scale is None:
            scale = [(1, 3, 5, 7), (1, 3, 5), (1, 3)]
        if mlp_ratios is None:
            mlp_ratios = []
        if bf_depth is None:
            bf_depth = []
        if embed_dim is None:
            embed_dim = []

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.data_flows = data_flows
        self.stage_nums = len(bf_depth)
        ############ downsample layers (patch embeddings) ######################
        # patch embedding: conv-norm
        self.stem = nn.Sequential(nn.Conv3d(in_chans, embed_dim[0], kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=1),
                                  norm_layer(embed_dim[0]),
                                  )
        # biformer ds
        self.downsample_layers = nn.ModuleList()
        for i in range(self.stage_nums):
            # patch merging: norm-conv
            downsample_layer = nn.Sequential(
                norm_layer(embed_dim[i]),
                nn.Conv3d(embed_dim[i], embed_dim[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        # conv_branch ds
        self.downsample_layers_co = nn.ModuleList()
        for i in range(self.stage_nums):
            # patch merging: norm-conv
            downsample_layer_co = nn.Sequential(
                nn.Conv3d(embed_dim[i], embed_dim[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.BatchNorm3d(embed_dim[i + 1]),
            )
            self.downsample_layers_co.append(downsample_layer_co)

        self.invertConvbranch = nn.ModuleList()

        for i in range(self.stage_nums):
            invertConv = Conv_BasicLayer(dim=embed_dim[i], depth=co_depth[i], mlp_ratio=invert_conv_mlp_ratios[i])
            self.invertConvbranch.append(invertConv)

        ##########################################################################
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in embed_dim]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(bf_depth))]

        for i in range(self.stage_nums):
            stage = Bf_BasicLayer(dim=embed_dim[i + 1],
                                  depth=bf_depth[i],
                                  num_heads=nheads[i + 1],
                                  mlp_ratio=mlp_ratios[i],
                                  drop_path=dp_rates[sum(bf_depth[:i]):sum(bf_depth[:i + 1])],
                                  ####### biformer specific ########
                                  n_win=n_wins[i], topk=topks[i], side_dwconv=side_dwconv,
                                  scale=scale[i]
                                  )
            self.stages.append(stage)

        ##########################################################################
        pre_head_norm = pre_head_norm_layer or norm_layer
        self.norm = pre_head_norm(embed_dim[-1])
        # Classifier head
        self.head = nn.Linear(2 * embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        # self.head = nn.Sequential(nn.Linear(2 * embed_dim[-1], embed_dim[-1]),
        #                           nn.ReLU(),
        #                           nn.Linear(embed_dim[-1], num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm3D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor):
        bf_out_per_stage = []
        x_bf = self.stem(x)
        x_cov = x_bf
        for i in range(self.stage_nums):
            x_bf = self.downsample_layers[i](x_bf)
            x_bf = self.stages[i](x_bf)
            bf_out_per_stage.append(x_bf)
        x_bf = self.norm(x_bf)

        #  Conv_branch
        for i in range(self.stage_nums):
            x_cov = self.invertConvbranch[i](bf_out_per_stage[i], x_cov)
            x_cov = self.downsample_layers_co[i](x_cov)
        return x_bf, x_cov

    def forward(self, x: torch.Tensor):
        if self.data_flows == 'conv_bf':
            x_bf, x_cov = self.forward_features(x)
            x = torch.cat([x_bf, x_cov], dim=1).mean([2, 3, 4])
            # x = x_bf.mean([2, 3, 4]) + x_cov.mean([2, 3, 4])
            x = self.head(x)
        elif self.data_flows == 'only_conv':
            _, x_cov = self.forward_features(x)
            x = x_cov.mean([2, 3, 4])
            x = self.head(x)
        return x


@register_model
def MS_biformer():
    model = MSBiFormer(in_chans=1, num_classes=1,
                       bf_depth=[2, 2, 3],  # 1, 1, 3
                       co_depth=[2, 2, 2],  # 1, 1, 1
                       embed_dim=[32, 64, 128, 256],  # 32, 64, 128, 576
                       mlp_ratios=[2, 2, 2],  # 1, 1, 1
                       invert_conv_mlp_ratios=[2, 2, 2],
                       head_dim=16,  # 16
                       drop_path_rate=0.2,
                       norm_layer=nn.BatchNorm3d,
                       pre_head_norm_layer=LayerNorm3D,
                       n_wins=(4, 2, 1),  # win_size:7, 7, 7
                       topks=(2, 2, 1),
                       side_dwconv=5,  # 5
                       data_flows='conv_bf')  # conv_bf or only_conv

    return model


if __name__ == "__main__":
    input = torch.rand((1, 1, 112, 112, 112)).cuda()
    net = MS_biformer()
    net.cuda()
    flops, params = profile(net, inputs=(input,))
    print('FLOPs = ' + str(flops / (1000 ** 3)) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    out = net(input)
    print(out.shape)
    # print(net)
