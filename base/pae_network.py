## This is after one random stroll on the github repo of the original paper
## And found that the implementation was changed
## Repo link: https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE/PAE.py

import torch
import torch.nn as nn

class LN_v2(nn.Module):
    ## From https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022/PyTorch/Library/Utility.py
    ## Probably just a layer normalization effort from
    ## here: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class fft_layer(nn.Module):
    def __init__(self, dim, time_range, freqs):
        super(fft_layer, self).__init__()
        self.dim = dim
        self.time_range = time_range
        self.freqs = freqs

    def forward(self, x):
        ## Real FFT is used. Ref: https://pytorch.org/docs/stable/generated/torch.fft.rfft.html#torch.fft.rfft
        ## Only contain positive frequencies
        rfft = torch.fft.rfft(x, dim=self.dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]
        power = spectrum**2

        freq = torch.sum(self.freqs * power, dim=self.dim) / torch.sum(
            power, dim=self.dim
        )
        amp = 2 * torch.sqrt(torch.sum(power, dim=self.dim)) / self.time_range
        bias = rfft.real[:, :, 0] / self.time_range

        return freq, amp, bias, magnitudes


class ps_layer(nn.Module):
    """Returns the parameter S through
    fully connected layers, one network per channel
    """

    def __init__(self, in_channels, phase_channels, tpi):
        super(ps_layer, self).__init__()
        self.in_channels = in_channels
        self.phase_channels = phase_channels
        self.tpi = tpi

        self.fc = nn.ModuleList()

        for _ in range(self.phase_channels):
            self.fc.append(nn.Linear(self.in_channels, 2))

    def forward(self, x):
        ## Iterate through every phase and fit one set of layers to each channel
        p = torch.empty(
            (x.shape[0], self.phase_channels), dtype=torch.float32, device=x.device
        )
        for i in range(self.phase_channels):
            v = self.fc[i](x[:, i, :])
            p[:, i] = atan2(self.tpi, v[:, 1], v[:, 0]) / self.tpi
        return p


def atan2(tpi, y, x):
    ans = torch.atan(y / x)
    ans = torch.where((x < 0) * (y >= 0), ans + 0.5 * tpi, ans)
    ans = torch.where((x < 0) * (y < 0), ans - 0.5 * tpi, ans)
    return ans


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, window_size):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.inter_channels = in_channels // 2
        self.conv = torch.nn.Sequential()

        self.conv.add_module(
            "conv1",
            nn.Conv1d(
                self.in_channels,
                self.inter_channels,
                kernel_size=self.window_size,
                stride=1,
                padding=int((self.window_size - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )
        self.conv.add_module("LN1", LN_v2(self.window_size))
        self.conv.add_module("elu1", nn.ELU())

        self.conv.add_module(
            "conv2",
            nn.Conv1d(
                self.inter_channels,
                self.out_channels,
                kernel_size=self.window_size,
                stride=1,
                padding=int((self.window_size - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, joint_channels, window_size):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.joint_channels = joint_channels
        self.out_channels = joint_channels * 2
        self.window_size = window_size
        self.conv = torch.nn.Sequential()

        self.conv.add_module(
            "deconv1",
            nn.Conv1d(
                self.in_channels,
                self.joint_channels,
                kernel_size=self.window_size,
                stride=1,
                padding=int((self.window_size - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )
        self.conv.add_module("LN1", LN_v2(self.window_size))
        self.conv.add_module("elu1", nn.ELU())

        self.conv.add_module(
            "deconv2",
            nn.Conv1d(
                self.joint_channels,
                self.out_channels,
                kernel_size=self.window_size,
                stride=1,
                padding=int((self.window_size - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)
        return x
