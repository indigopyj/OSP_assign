import torch.nn as nn
import torch

###########################################################################
# Question 1 : Implement the UNet model code.
# Understand architecture of the UNet in practice lecture 15 -> slides 5-6 (30 points)

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3은 kernel size
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        ########## fill in the blanks (Hint : check out the channel size in practice lecture 15 ppt slides 5-6)
        self.convDown1 = conv(in_channels=in_channels, out_channels=64)
        self.convDown2 = conv(in_channels=64, out_channels=128)
        self.convDown3 = conv(in_channels=128, out_channels=256)
        self.convDown4 = conv(in_channels=256, out_channels=512)
        self.convDown5 = conv(in_channels=512, out_channels=1024)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convUp4 = conv(in_channels=1536, out_channels=512)
        self.convUp3 = conv(in_channels=768, out_channels=256)
        self.convUp2 = conv(in_channels=384, out_channels=128)
        self.convUp1 = conv(in_channels=192, out_channels=64)
        self.convUp_fin = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)



    def forward(self, x):
        conv1 = self.convDown1(x) # C = 64
        x = self.maxpool(conv1)
        conv2 = self.convDown2(x) # C = 128
        x = self.maxpool(conv2)
        conv3 = self.convDown3(x) # C = 256
        x = self.maxpool(conv3)
        conv4 = self.convDown4(x) # C = 512
        x = self.maxpool(conv4)
        conv5 = self.convDown5(x)   # C = 1024
        x = self.upsample(conv5)    # C = 1024
        x = torch.cat((x, conv4), dim=1) # C = 512 + 1024 = 1536
        x = self.convUp4(x)     # C = 512
        x = self.upsample(x)    # C = 512
        x = torch.cat((x, conv3), dim=1)    # C= 256 + 512 = 768
        x = self.convUp3(x)     # C = 256
        x = self.upsample(x)    # C = 256
        x = torch.cat((x, conv2), dim=1)    # C = 128 + 256 = 384
        x = self.convUp2(x)     # C = 128
        x = self.upsample(x)    # C = 64
        x = torch.cat((x, conv1), dim=1)    # C = 64 + 128 = 192
        x = self.convUp1(x)     # C = 64
        out = self.convUp_fin(x)    # C = 2

        return out
