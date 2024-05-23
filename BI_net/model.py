from typing import Tuple, Union
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torch import Tensor, LongTensor
from typing import Optional, Tuple
from Pconv import MLPBlock


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
                 attn_backend='torch'):
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
        qkv = self.qkv_linear.forward(x)  # ncHW
        q, k, v = qkv.chunk(3, dim=1)  # ncHW

        # STEP 2: region-to-region routing
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        q_r = F.avg_pool3d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool3d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchw
        q_r: Tensor = q_r.permute(0, 2, 3, 4, 1).flatten(1, 2).flatten(1, 2)  # n(hw)c
        k_r: Tensor = k_r.flatten(2, 3).flatten(2, 3)  # nc(hw)
        a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        # STEP 3: token to token attention (non-parametric function)
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size
                                        )

        output = output + self.lepe(v)  # ncHW,
        output = self.output_linear(output)  # ncHW

        if ret_attn_mask:
            return output, attn_mat

        return output


class BiFormerBlock(nn.Module):
    """
    Attention + FFN
    """

    def __init__(self, dim, drop_path=0., num_heads=8, n_win=7,
                 qk_scale=None, topk=4, mlp_ratio=4, side_dwconv=5,
                 ):

        super().__init__()

        self.norm1 = LayerNorm3D(dim)  # important to avoid attention collapsing

        if topk > 0:
            self.attn = nchwBRA(dim=dim, num_heads=num_heads, n_win=n_win,
                                qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        else:
            raise ValueError('topk should >0!')

        self.norm2 = LayerNorm3D(dim)
        self.mlp = nn.Sequential(nn.Conv3d(dim, int(mlp_ratio * dim), kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio * dim), dim, kernel_size=1)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, xc):
        """
        Args:
            x: NCHWD tensor
        Return:
            NCHWD tensor

        """
        # attention & mlp
        x = x + self.drop_path(self.attn(self.norm1(x + xc)))  # (N, C, H, W)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, C, H, W)
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout = torch.max(x, dim=1, keepdim=True).values
        out = torch.cat([avgout, maxout], 1)
        out = self.sigmoid(self.conv(out))
        return out

class FusionNet(nn.Module):
    """
    Stack several BiFormer Blocks
    """

    def __init__(self, n_win=None, topk=None,
                 mlp_ratio=4., drop_path=0.1, side_dwconv=5, dim=None, num_heads=None):
        super().__init__()
        if topk is None:
            topk = [2, 2, 1]
        if n_win is None:
            n_win = [4, 2, 1]
        self.n_win = n_win
        self.topk = topk
        self.drop_path = drop_path
        if num_heads is None:
            num_heads = [4, 8, 32]
        self.num_heads = num_heads
        if dim is None:
            dim = [1, 64, 128, 576]
        self.dim = dim
        self.norm_layer = nn.BatchNorm3d

        self.stem = nn.Sequential(nn.Conv3d(dim[0], dim[1], kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=1),
                                  self.norm_layer(dim[1]),
                                  )

        self.convdownsample_layers = nn.ModuleList()
        for i in range(2):
            # patch merging: norm-conv
            downsample_layer = nn.Sequential(
                self.norm_layer(dim[i + 1]),
                nn.Conv3d(dim[i + 1], dim[i + 2], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
            self.convdownsample_layers.append(downsample_layer)

        self.bifordownsample_layers = nn.ModuleList()
        for i in range(2):
            # patch merging: norm-conv
            downsample_layer = nn.Sequential(
                self.norm_layer(dim[i + 1]),
                nn.Conv3d(dim[i + 1], dim[i + 2], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
            self.bifordownsample_layers.append(downsample_layer)

        self.biformerblock1 = BiFormerBlock(
            dim=dim[1],
            num_heads=self.num_heads[0],
            n_win=self.n_win[0],
            topk=self.topk[0],
            mlp_ratio=mlp_ratio,
            side_dwconv=side_dwconv,
        )

        self.biformerblock2 = BiFormerBlock(
            dim=dim[2],
            num_heads=self.num_heads[1],
            n_win=self.n_win[1],
            topk=self.topk[1],
            mlp_ratio=mlp_ratio,
            side_dwconv=side_dwconv,
        )
        self.biformerblock3 = BiFormerBlock(
            dim=dim[3],
            num_heads=self.num_heads[2],
            n_win=self.n_win[2],
            topk=self.topk[2],
            mlp_ratio=mlp_ratio,
            side_dwconv=side_dwconv,
        )
        self.biformerblock4 = BiFormerBlock(
            dim=dim[3],
            num_heads=self.num_heads[2],
            n_win=self.n_win[2],
            topk=self.topk[2],
            mlp_ratio=mlp_ratio,
            side_dwconv=side_dwconv,
        )
        self.biformerblock5 = BiFormerBlock(
            dim=dim[3],
            num_heads=self.num_heads[2],
            n_win=self.n_win[2],
            topk=self.topk[2],
            mlp_ratio=mlp_ratio,
            side_dwconv=side_dwconv,
        )
        self.Pconvblock = nn.ModuleList()
        for i in range(5):
            # patch merging: norm-conv
            if i == 0:
                pconvblock = MLPBlock(dim[i + 1], 4, mlp_ratio=2, drop_path=self.drop_path, layer_scale_init_value=0,
                                      act_layer=nn.ReLU,
                                      norm_layer=nn.BatchNorm3d, pconv_fw_type='split_cat')
            elif i == 1:
                pconvblock = MLPBlock(dim[i + 1], 4, mlp_ratio=2, drop_path=self.drop_path, layer_scale_init_value=0,
                                      act_layer=nn.ReLU,
                                      norm_layer=nn.BatchNorm3d, pconv_fw_type='split_cat', double_input=True)
            else:
                pconvblock = MLPBlock(dim[3], 4, mlp_ratio=2, drop_path=self.drop_path, layer_scale_init_value=0,
                                      act_layer=nn.ReLU,
                                      norm_layer=nn.BatchNorm3d, pconv_fw_type='split_cat', double_input=True)
            self.Pconvblock.append(pconvblock)

        self.shortcut = nn.ModuleList()
        for i in range(5):
            # patch merging: norm-conv
            if i <= 1:
                shortcut = nn.Sequential(nn.Conv3d(dim[i + 1], dim[i + 1], 1),
                                         # nn.MaxPool3d(2),
                                         LayerNorm3D(dim[i + 1]),
                                         )
            else:
                shortcut = nn.Sequential(nn.Conv3d(dim[3], dim[3], 1),
                                         # nn.MaxPool3d(2),
                                         LayerNorm3D(dim[3]),
                                         )
            self.shortcut.append(shortcut)

        self.SA = nn.ModuleList()

        for _ in range(5):
            sa = SpatialAttention()
            self.SA.append(sa)


        self.classificatier = nn.Sequential(nn.Linear(self.dim[-1], 1))
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.pre_head_norm = LayerNorm3D(dim[-1])
        self.conv_last = nn.Conv3d(dim[-1], dim[-1], 1)

        self.sig = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        # ---------------------------stage1-----------------------------
        xc, xxc = self.Pconvblock[0](x)
        xb = self.biformerblock1(x, xxc)
        xc = self.SA[0](xb) * xc
        xc = self.convdownsample_layers[0](xc)
        xb = self.bifordownsample_layers[0](xb)

        # ---------------------------stage2-----------------------------
        xc, xxc = self.Pconvblock[1](xc, xb)
        xb = self.biformerblock2(xb, xxc)
        xc = self.SA[1](xb) * xc
        xc = self.convdownsample_layers[1](xc)
        xb = self.bifordownsample_layers[1](xb)
        # ---------------------------stage3-----------------------------
        xc, xxc = self.Pconvblock[2](xc, xb)
        xb = self.biformerblock3(xb, xxc)
        xc = self.SA[2](xb) * xc

        xc, xxc = self.Pconvblock[3](xc, xb)
        xb = self.biformerblock3(xb, xxc)
        xc = self.SA[3](xb) * xc

        xc, xxc = self.Pconvblock[4](xc, xb)
        xb = self.biformerblock3(xb, xxc)
        xc = self.SA[4](xb) * xc

        out_conv = self.avgpool(xc).flatten(1)
        out_bifor = self.pre_head_norm(xb).mean([2, 3, 4])

        out = out_conv + out_bifor
        out = self.classificatier(out)

        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


if __name__ == "__main__":
    input = torch.rand((1, 1, 112, 112, 112)).cuda()
    net = FusionNet()
    net.cuda()
    flops, params = profile(net, inputs=(input,))
    print('FLOPs = ' + str(flops / (1000 ** 3)) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    param_size = 0
    param_sum = 0
    for param in net.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    # print('模型总大小为：{:.3f}MB'.format(all_size))
    # print("占用GPU：{:.3f} MB".format(torch.cuda.memory_allocated(0) / 1024 / 1024))
    out = net(input)
    print(out.shape)
    # print(net)
