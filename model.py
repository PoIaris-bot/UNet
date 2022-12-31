import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layers(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layers(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1)

    def forward(self, x, feature_map):
        out = self.layer(F.interpolate(x, scale_factor=2, mode='nearest'))
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = ConvBlock(3, 64)
        self.down_sample1 = DownSample(64)
        self.conv2 = ConvBlock(64, 128)
        self.down_sample2 = DownSample(128)
        self.conv3 = ConvBlock(128, 256)
        self.down_sample3 = DownSample(256)
        self.conv4 = ConvBlock(256, 512)
        self.down_sample4 = DownSample(512)
        self.conv5 = ConvBlock(512, 1024)
        self.up_sample1 = UpSample(1024)
        self.conv6 = ConvBlock(1024, 512)
        self.up_sample2 = UpSample(512)
        self.conv7 = ConvBlock(512, 256)
        self.up_sample3 = UpSample(256)
        self.conv8 = ConvBlock(256, 128)
        self.up_sample4 = UpSample(128)
        self.conv9 = ConvBlock(128, 64)
        self.conv10 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.down_sample1(conv1))
        conv3 = self.conv3(self.down_sample2(conv2))
        conv4 = self.conv4(self.down_sample3(conv3))
        conv5 = self.conv5(self.down_sample4(conv4))
        conv6 = self.conv6(self.up_sample1(conv5, conv4))
        conv7 = self.conv7(self.up_sample2(conv6, conv3))
        conv8 = self.conv8(self.up_sample3(conv7, conv2))
        conv9 = self.conv9(self.up_sample4(conv8, conv1))
        output = self.sigmoid(self.conv10(conv9))
        return output
