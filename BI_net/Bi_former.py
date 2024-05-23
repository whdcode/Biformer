"""
BiFormer-STL (Swin-Tiny-Layout) model
From
"author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com"

Model dimensions are extended to 3D
The author of the re-creation: Huidong Wu
"""

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

    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # attention & mlp
        x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, C, H, W)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, C, H, W)
        return x


class BasicLayer(nn.Module):
    """
    Stack several BiFormer Blocks
    """

    def __init__(self, dim, depth, num_heads, n_win, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):
        super().__init__()
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
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        for blk in self.blocks:
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class nchwBiFormerSTL(nn.Module):
    """
    Replace WindowAttn-ShiftWindowAttn in Swin-T model with Bi-Level Routing Attention
    """
    def __init__(self, in_chans=1, num_classes=1,
                 depth=None,
                 embed_dim=None,
                 head_dim=32, qk_scale=None,
                 drop_path_rate=0.15, drop_rate=0.,
                 # before_attn_dwconv=3,
                 mlp_ratios=None,
                 norm_layer=LayerNorm3D,
                 pre_head_norm_layer=None,
                 ######## biformer specific ############
                 n_wins: Union[int, Tuple[int]] = (),
                 topks: Union[int, Tuple[int]] = (),
                 side_dwconv: int = 5,
                 #######################################
                 ):
        super().__init__()
        if mlp_ratios is None:
            mlp_ratios = []
        if depth is None:
            depth = []
        if embed_dim is None:
            embed_dim = []
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # patch embedding: conv-norm
        stem = nn.Sequential(nn.Conv3d(in_chans, embed_dim[0], kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=1),
                             norm_layer(embed_dim[0]),
                             )
        self.downsample_layers.append(stem)

        for i in range(2):
            # patch merging: norm-conv
            downsample_layer = nn.Sequential(
                norm_layer(embed_dim[i]),
                nn.Conv3d(embed_dim[i], embed_dim[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        ##########################################################################
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in embed_dim]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        for i in range(3):
            stage = BasicLayer(dim=embed_dim[i],
                               depth=depth[i],
                               num_heads=nheads[i],
                               mlp_ratio=mlp_ratios[i],
                               drop_path=dp_rates[sum(depth[:i]):sum(depth[:i + 1])],
                               ####### biformer specific ########
                               n_win=n_wins[i], topk=topks[i], side_dwconv=side_dwconv
                               ##################################
                               )
            self.stages.append(stage)

        ##########################################################################
        pre_head_norm = pre_head_norm_layer or norm_layer
        self.norm = pre_head_norm(embed_dim[-1])
        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor):
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        # x = x.flatten(2).mean(-1)
        x = x.mean([2, 3, 4])
        x = self.head(x)
        return x


@register_model
def biformer_stl_nchw(**kwargs):
    model = nchwBiFormerSTL(in_chans=1, num_classes=1,
                            depth=[2, 2, 3],  # 1, 1, 3
                            embed_dim=[64, 192, 576],  # 64, 128, 576
                            mlp_ratios=[1, 1, 1],
                            head_dim=16,  # 16
                            drop_path_rate=0.2,
                            norm_layer=nn.BatchNorm3d,
                            ######## biformer specific ############
                            n_wins=(4, 2, 1),  # 7, 7, 7
                            topks=(3, 2, 1),  # 1， 2， 8    Mci2 4. 16
                            side_dwconv=5, # 5
                            #######################################
                            **kwargs)

    return model


if __name__ == "__main__":
    input = torch.rand((8, 1, 112, 112, 112)).to("cuda:0")
    # input = torch.rand((1, 56, 14, 14, 14)).cuda()
    net = biformer_stl_nchw()
    # net = nchwBRA(128, topk=12)
    # net = CDAF_Block(32)
    net.to("cuda:0")
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
