import torch
import torch.nn as nn
import math
import torch.nn as nn
from thop import profile


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=None, stride=1, conv_groups=None):
        super(PSAModule, self).__init__()
        if conv_groups is None:
            conv_groups = [1, 4, 8]
        if conv_kernels is None:
            conv_kernels = [1, 3]
        self.conv = conv(inplans, planes // 2, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                         stride=stride, groups=conv_groups[0])
        self.conv_1 = nn.Sequential(
            conv(inplans, inplans, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                 stride=stride, groups=conv_groups[1]),
            conv(inplans, planes // 4, kernel_size=conv_kernels[1],
                 padding=conv_kernels[1] // 2,
                 stride=stride, groups=conv_groups[1]),
        )
        self.conv_2 = nn.Sequential(
            conv(inplans, inplans, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                 stride=stride, groups=conv_groups[2]),
            conv(inplans, inplans, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                 stride=stride, groups=conv_groups[2]),
            conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                 stride=stride, groups=conv_groups[2]),
        )

        self.se = SEWeightModule(planes // 2)
        self.se1 = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)
        self.planes = planes

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x = self.conv(x)
        feats = torch.cat((x, x1, x2), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3], feats.shape[4])

        x_se = self.se(x)
        x1_se = self.se1(x1)
        x2_se = self.se1(x2)
        x_se = torch.cat((x_se, x1_se, x2_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class SEWeightModule3D(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv3D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard 3D convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1_3D(in_planes, out_planes, stride=1):
    """1x1 3D convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule3D(nn.Module):

    def __init__(self, inplanes, planes, conv_kernels=None, stride=1, conv_groups=None):
        super(PSAModule3D, self).__init__()
        if conv_groups is None:
            conv_groups = [1, 4, 8, 16]
        if conv_kernels is None:
            conv_kernels = [3, 5, 7, 9]
        self.conv_1 = conv3D(inplanes, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                             stride=stride, groups=conv_groups[0])
        self.conv_2 = conv3D(inplanes, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                             stride=stride, groups=conv_groups[1])
        self.conv_3 = conv3D(inplanes, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                             stride=stride, groups=conv_groups[2])
        self.conv_4 = conv3D(inplanes, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                             stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule3D(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3], feats.shape[4])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

if __name__ == "__main__":
    input = torch.rand((1,64, 28, 28, 28)).cuda()
    net = PSAModule(64, 64)
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
