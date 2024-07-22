import torch
from thop import profile
from torch import nn


class SE_resblock(nn.Module):
    """
    Args:
        normalized_shape (int or tuple): Input shape from an expected input. If it is a single integer,
            it is treated as a singleton tuple. Default: 1
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6
    """

    def __init__(self, in_channels):
        super().__init__()
        self.Conv1 = nn.Conv3d(in_channels, 2 * in_channels, 1, 1)
        self.Resbranch = nn.Sequential(
            nn.Conv3d(in_channels, 2 * in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm3d(2 * in_channels),
            nn.ReLU(),
            nn.Conv3d(2 * in_channels, in_channels, 1, 1),
            nn.BatchNorm3d(in_channels)
        )
        self.SEbranch = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Linear(in_channels // 2, in_channels),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1, 1),
            nn.BatchNorm3d(in_channels),)

    def forward(self, x):
        res = x
        x = self.Conv1(x)
        rb, sb = x.chunk(2, dim=1)
        rb = self.Resbranch(rb)
        sb = self.SEbranch(sb.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = torch.cat([rb, sb], dim=1)
        x = self.Conv2(x) + res
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.rand((1, 64, 28, 28, 28)).to(device)
    net = SE_resblock(64)
    net.to(device)
    flops, params = profile(net, inputs=(input,))
    print('FLOPs = ' + str(flops / (1000 ** 3)) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    out = net(input)
    print(out.shape)
    # print(net)
