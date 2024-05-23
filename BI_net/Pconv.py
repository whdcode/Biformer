import torch
import torch.nn as nn
from thop import profile
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div       # 计算参数卷积的通道数dim_conv3
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv3d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type,
                 double_input=False,
                 double_out=True
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        self.di = double_input
        self.do = double_out

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv3d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv3d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(  # 部分卷积
            dim,
            self.n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            if self.di:
                self.forward = self.forward1
            else:
                self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        xb = x
        x = shortcut + self.drop_path(self.mlp(x))
        if self.do:
            return x, xb
        else:
            return x

    def forward1(self, x, x1) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x + x1)
        xb = x
        x = shortcut + self.drop_path(self.mlp(x))
        if self.do:
            return x, xb
        else:
            return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


if __name__ == "__main__":
    input = torch.rand((1, 32, 28, 28, 28)).cuda()
    # input = torch.rand((1, 56, 14, 14, 14)).cuda()
    net = MLPBlock(32, 4, mlp_ratio=2, drop_path=0.1, layer_scale_init_value=0, act_layer=nn.ReLU,
                   norm_layer=nn.BatchNorm3d, pconv_fw_type='split_cat')
    # net = nchwBRA(128, topk=12)
    # net = CDAF_Block(32)
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
    out,_ = net(input)
    print(out.shape)
    print(net)
