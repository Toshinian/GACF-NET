import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUpsampleNeck(nn.Module):
    def __init__(self, in_channels: int = 1024, mid_channels: int = 256, out_channels: int = 64):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16 -> 32
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32 -> 64
            nn.Conv2d(mid_channels // 2, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(mid_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64 -> 128
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.stage4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128 -> 256
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


