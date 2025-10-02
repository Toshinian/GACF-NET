import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedCrossViewFusion(nn.Module):
    def __init__(self, in_dim=64, reduction_ratio=4):
        super(EnhancedCrossViewFusion, self).__init__()
        self.in_dim = in_dim
        self.reduction_ratio = reduction_ratio
        
        # 多尺度特征提取
        self.multiscale_extractor = MultiScaleFeatureExtractor(in_dim)
        
        # 自适应特征压缩
        self.adaptive_compression = AdaptiveCompression(in_dim)
        
        # 相似性计算模块
        self.similarity_calculator = SimilarityCalculator(in_dim // reduction_ratio)
        
        # 空间-通道注意力
        self.spatial_channel_attention = SpatialChannelAttention(in_dim)
        
        # 残差融合模块
        self.residual_fusion = ResidualFusionModule(in_dim)
        
        # 通道扩展模块
        self.channel_expand = nn.Sequential(
            nn.Conv2d(in_dim // reduction_ratio, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, in_dim, 1)
        )
        
        # 自适应权重参数
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, front_x, bev_x):
        """
        参数:
            front_x: 前视特征 [B, 64, 256, 256]
            bev_x: BEV特征 [B, 64, 256, 256]
        返回:
            增强后的BEV特征 [B, 64, 256, 256]
        """
        front_x = front_x.contiguous(memory_format=torch.contiguous_format)
        bev_x = bev_x.contiguous(memory_format=torch.contiguous_format)
        B, C, H, W = front_x.size()
        
        # 1. 多尺度特征提取
        front_multiscale = self.multiscale_extractor(front_x)  # [B, 64, 256, 256]
        bev_multiscale = self.multiscale_extractor(bev_x)     # [B, 64, 256, 256]
        
        # 2. 自适应特征压缩
        front_compressed = self.adaptive_compression(front_multiscale)  # [B, 16, 1, 256]
        bev_compressed = self.adaptive_compression(bev_multiscale)      # [B, 16, 256, 256]
        
        # 3. 相似性计算与融合
        similarity_map = self.similarity_calculator(front_compressed, bev_compressed)  # [B, 16, 256, 256]
        
        # 4. 加权特征生成
        weighted_front = similarity_map * front_compressed.expand(-1, -1, H, -1)  # [B, 16, 256, 256]
        
        # 5. 上采样到原始分辨率
        weighted_front_upsampled = F.interpolate(
            weighted_front, size=(H, W), mode='bilinear'
        )  # [B, 16, 256, 256]
        
        # 6. 通道扩展
        weighted_front_expanded = self.channel_expand(weighted_front_upsampled)  # [B, 64, 256, 256]
        
        # 7. 空间-通道注意力
        attention_weight = self.spatial_channel_attention(bev_x)  # [B, 64, 256, 256]
        
        # 8. 残差融合
        enhanced_bev = self.residual_fusion(bev_x, weighted_front_expanded, attention_weight)
        
        # 9. 最终输出投影
        final_output = torch.cat([bev_x, enhanced_bev], dim=1)  # [B, 128, 256, 256]
        output = self.output_projection(final_output)  # [B, 64, 256, 256]
        
        # 10. 自适应加权输出
        final_bev = self.alpha * bev_x + self.beta * output
        
        return final_bev


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        
        # 不同尺度的卷积核
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim//4, 1),  # 1x1
            nn.Conv2d(in_dim, in_dim//4, 3, padding=1),  # 3x3
            nn.Conv2d(in_dim, in_dim//4, 5, padding=2),  # 5x5
            nn.Conv2d(in_dim, in_dim//4, 7, padding=3), # 7x7
        ])
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        scale_features = []
        for conv in self.scale_convs:
            scale_features.append(conv(x))
        
        # 拼接多尺度特征
        multiscale_feat = torch.cat(scale_features, dim=1)  # [B, 64, H, W]
        
        # 融合
        return self.fusion_conv(multiscale_feat)


class AdaptiveCompression(nn.Module):
    """自适应特征压缩模块"""
    def __init__(self, in_dim, reduction_ratio=4):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.compressed_dim = in_dim // reduction_ratio
        
        # 通道压缩
        self.channel_compress = nn.Sequential(
            nn.Conv2d(in_dim, self.compressed_dim, 1),
            nn.BatchNorm2d(self.compressed_dim),
            nn.ReLU()
        )
        
        # 自适应池化策略
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # 保持宽度，压缩高度
        
        # 空间注意力增强
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.compressed_dim, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道压缩
        compressed = self.channel_compress(x)  # [B, 16, H, W]
        
        # 空间压缩（沿高度维度）
        compressed = self.adaptive_pool(compressed)  # [B, 16, 1, W]
        
        # 空间注意力增强
        attention = self.spatial_attention(compressed)  # [B, 1, 1, W]
        enhanced = compressed * attention
        
        return enhanced


class SimilarityCalculator(nn.Module):
    """相似性计算模块 - 兼容版本"""
    def __init__(self, compressed_dim):
        super().__init__()
        self.compressed_dim = compressed_dim
        
        # 温度参数（用于控制相似性分布的锐度）
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        
    def forward(self, front_compressed, bev_compressed):
        """
        参数:
            front_compressed: [B, 16, 1, W]
            bev_compressed: [B, 16, H, W]
        返回:
            similarity_map: [B, 16, H, W]
        """
        B, C, H, W = bev_compressed.size()
        
        # 扩展front特征到与bev相同的空间尺寸
        front_expanded = front_compressed.expand(-1, -1, H, -1)  # [B, 16, H, W]
        
        # 计算特征相似性 (手动计算余弦相似度，兼容旧版本)
        front_norm = F.normalize(front_expanded, p=2, dim=1)
        bev_norm = F.normalize(bev_compressed, p=2, dim=1)
        similarity = torch.sum(front_norm * bev_norm, dim=1)  # [B, H, W]
        similarity = similarity.unsqueeze(1)  # [B, 1, H, W]
        
        # 应用温度参数
        similarity = similarity / self.temperature
        
        # 扩展到所有通道
        similarity_map = similarity.expand(-1, C, -1, -1)  # [B, 16, H, W]
        
        return similarity_map


class SpatialChannelAttention(nn.Module):
    """空间-通道注意力模块"""
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim//16, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim//16, in_dim, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_dim, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 融合权重
        self.fusion_weight = nn.Parameter(torch.ones(2))
        
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        
        # 空间注意力
        spatial_att = self.spatial_attention(x)
        
        # 加权融合
        weights = F.softmax(self.fusion_weight, dim=0)
        attention = weights[0] * channel_att + weights[1] * spatial_att
        
        return attention


class ResidualFusionModule(nn.Module):
    """残差融合模块"""
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        
        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, bev_feat, enhanced_feat, attention_weight):
        """
        参数:
            bev_feat: 原始BEV特征 [B, 64, H, W]
            enhanced_feat: 增强特征 [B, 64, H, W]
            attention_weight: 注意力权重 [B, 64, H, W]
        """
        # 特征拼接
        combined = torch.cat([bev_feat, enhanced_feat], dim=1)  # [B, 128, H, W]
        
        # 特征变换
        transformed = self.feature_transform(combined)  # [B, 64, H, W]
        
        # 门控机制
        gate = self.gate(combined)  # [B, 64, H, W]
        
        # 残差连接
        output = bev_feat + gate * transformed * attention_weight
        
        return output

