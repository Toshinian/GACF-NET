import torch
import torch.nn as nn
import torch.nn.functional as F

 
class RadarProcessor(nn.Module):
    """用于处理雷达掩码的normal conv网络"""
    def __init__(self):
        super().__init__()
        # 输入尺寸假设为 (720, 720)，输出尺寸对齐BEV特征的16x16
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 360x360
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 180x180
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 90x90
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 45x45
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 22x22
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # 22x22
            nn.AdaptiveAvgPool2d((16, 16))  # 最终输出16x16
        )

    def forward(self, x):
        # 输入x形状: [B, H, W, C] -> 转换为 [B, C, H, W]
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        return self.downsample(x)
class MMWaveSparsityGuidedProcessor(nn.Module):

    def __init__(self, input_channels=3, output_channels=1024, output_size=(16, 16)):
        super().__init__()
        self.output_size = output_size
        
        self.sparse_aware_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 1/2
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 1/4
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 1/8
            nn.ReLU()
        )
        
        self.sparsity_attention = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(512, output_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size)
        )
        
    def forward(self, x):
        """
        输入: [B, H, W, C] 毫米波栅格化点云数据
        输出: [B, 1024, 16, 16] 处理后的特征
        """
        # 输入预处理
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        
        features = self.sparse_aware_extractor(x)
        attention = self.sparsity_attention(features)
        
        attended_features = features * attention
        
        enhanced_features = self.feature_enhancer(attended_features)
        
        output = self.output_layer(enhanced_features)
        
        return output

