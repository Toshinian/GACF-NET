import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalBEVFeatureFusion2(nn.Module):
    """
    融合反向投影迭代和相似性计算的稳定BEV特征融合模块
    输入: fs_features [B, 1024, 16, 16], bev_features [B, 1024, 16, 16]
    输出: [B, 1024, 16, 16] 
    """
    def __init__(self, dim=1024, num_iterations=2):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations
        
        # 特征预处理
        self.fs_preprocess = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
        self.bev_preprocess = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
        # 相似性融合层
        self.similarity_fusion = StableSimilarityFusion(dim)
        
        # 简化的上下文更新
        self.context_update = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )
        
        # 最终输出投影
        self.output_projection = nn.Conv2d(dim, dim, 1)
        
    def forward(self, fs_features, bev_features):
        """
        融合反向投影迭代和相似性计算的稳定融合
        参数:
            fs_features: fs特征 [B, 1024, 16, 16]
            bev_features: BEV特征 [B, 1024, 16, 16]
        返回:
            最终的BEV视角特征 [B, 1024, 16, 16]
        """
        # 如果fs_features是元组，取第一个元素
        if isinstance(fs_features, tuple):
            fs_features = fs_features[0]
        
        # 特征预处理
        fs_processed = self.fs_preprocess(fs_features)
        bev_processed = self.bev_preprocess(bev_features)
        
        # 初始化上下文向量
        context_vector = torch.zeros_like(fs_processed)
        
        # 简化的迭代融合过程
        for iteration in range(self.num_iterations):
            # 应用上下文向量 (简化版)
            fs_contextual = fs_processed + 0.1 * context_vector
            bev_contextual = bev_processed + 0.1 * context_vector
            
            # 相似性融合
            similarity_fused = self.similarity_fusion(fs_contextual, bev_contextual)
            
            # 更新上下文向量 (简化版)
            if iteration < self.num_iterations - 1:
                context_vector = self.context_update(similarity_fused)
        
        # 最终输出投影
        final_bev_features = self.output_projection(similarity_fused)
        
        return final_bev_features


class StableSimilarityFusion(nn.Module):
    """稳定的基于相似性的融合模块 - 优化用于迭代融合"""
    def __init__(self, dim):
        super().__init__()
        
        # 查询、键、值投影 (进一步减少维度)
        self.query_conv = nn.Conv2d(dim, dim // 32, 1)  # 进一步减少维度
        self.key_conv = nn.Conv2d(dim, dim // 32, 1)
        self.value_conv = nn.Conv2d(dim, dim // 8, 3, padding=1)  # 减少值维度
        
        # 空间池化减少计算量
        self.spatial_pool = nn.AdaptiveAvgPool2d(4)  # 16x16 -> 4x4 进一步减少
        self.spatial_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        # 输出投影
        self.output_proj = nn.Conv2d(dim // 8, dim, 1)
        
        # 简化的门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(dim*2, dim//16, 1),
            nn.ReLU(),
            nn.Conv2d(dim//16, 1, 1),
            nn.Sigmoid()
        )
        
        # 残差权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, fs_features, bev_features):
        """
        稳定的基于相似性的融合
        参数:
            fs_features: fs特征 [B, dim, H, W]
            bev_features: BEV特征 [B, dim, H, W]
        返回:
            相似性融合结果 [B, dim, H, W]
        """
        B, C, H, W = fs_features.size()
        
        # 空间池化减少计算量
        fs_pooled = self.spatial_pool(fs_features)  # [B, C, 4, 4]
        bev_pooled = self.spatial_pool(bev_features)  # [B, C, 4, 4]
        
        # 投影到查询、键、值空间
        proj_query = self.query_conv(bev_pooled)  # [B, C//32, 4, 4]
        proj_key = self.key_conv(fs_pooled)      # [B, C//32, 4, 4]
        proj_value = self.value_conv(fs_pooled)  # [B, C//8, 4, 4]
        
        # 重塑为序列格式
        proj_query = proj_query.view(B, C//32, 16).permute(0, 2, 1)  # [B, 16, C//32]
        proj_key = proj_key.view(B, C//32, 16).permute(0, 2, 1)      # [B, 16, C//32]
        proj_value = proj_value.view(B, C//8, 16).permute(0, 2, 1)   # [B, 16, C//8]
        
        # 计算相似性矩阵 (使用缩放避免梯度问题)
        similarity = torch.bmm(proj_query, proj_key.permute(0, 2, 1))  # [B, 16, 16]
        similarity = similarity / (C//32) ** 0.5  # 缩放
        
        # 归一化
        similarity = F.softmax(similarity, dim=-1)
        
        # 应用相似性权重
        weighted_value = torch.bmm(similarity, proj_value)  # [B, 16, C//8]
        
        # 重塑回原始格式
        weighted_value = weighted_value.permute(0, 2, 1).view(B, C//8, 4, 4)
        
        # 上采样回原始尺寸
        weighted_value = self.spatial_upsample(weighted_value)  # [B, C//8, 16, 16]
        
        # 输出投影
        similarity_fused = self.output_proj(weighted_value)  # [B, C, 16, 16]
        
        # 简化的门控机制
        gate_input = torch.cat([fs_features, bev_features], dim=1)
        gate_weight = self.gate(gate_input)
        
        # 门控融合 + 残差连接
        gated_similarity = similarity_fused * gate_weight
        output = gated_similarity + self.residual_weight * (fs_features + bev_features)
        
        return output
class SparseFeatureEnhancer(nn.Module):
    """针对毫米波点云稀疏特性的特征增强模块"""
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        
        # 稀疏响应增强
        self.response_boost = nn.Sequential(
            nn.Conv2d(dim, dim//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim//8, dim, kernel_size=1),
            nn.Sigmoid()  # 生成0-1的响应图
        )
        
        # 多尺度特征融合
        self.pyramid = nn.Sequential(
            nn.AvgPool2d(2),  # 8x8
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 16x16
        )
        
    def forward(self, x):
        # 增强稀疏目标响应
        response_map = self.response_boost(x) + 0.5  # 提高稀疏点响应
        enhanced = x * response_map
        
        # 多尺度特征融合（增强小目标）
        pyramid_feat = self.pyramid(enhanced)
        return torch.cat([enhanced, pyramid_feat], dim=1)[:, :self.dim]  # 保持通道数


class ScaleAwareAttention(nn.Module):
    """尺度感知注意力机制：专门增强小目标特征"""
    def __init__(self, dim=1024):
        super().__init__()
        self.scale_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim//4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        scale_weight = self.scale_extractor(x)
        return x * scale_weight + x


class MaritimeFusionModule(nn.Module):
    """水上目标检测专用融合模块（修复尺寸问题）"""
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        
        # BEV分支处理（针对点云数据）
        self.bev_processor = nn.Sequential(
            SparseFeatureEnhancer(dim),
            # 保持尺寸不变：使用padding=1的3×3卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1),
            ScaleAwareAttention(dim)
        )
        
        # FV分支处理（针对视觉特征）
        self.fv_processor = nn.Sequential(
            # 保持尺寸不变：使用padding=1的3×3卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            ScaleAwareAttention(dim)
        )
        
        # 跨模态融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim*2, dim*4, kernel_size=3, padding=1),  # 保持尺寸
            nn.ReLU(),
            nn.Conv2d(dim*4, dim, kernel_size=1),  # 1×1卷积不影响尺寸
            nn.ReLU()
        )
        
        # 空间注意力（BEV主导）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),  # 保持尺寸
            nn.Sigmoid()
        )

    def forward(self, bev, fv):
        # BEV特征处理（针对毫米波点云）
        bev_proc = self.bev_processor(bev)
        
        # FV特征处理（针对视觉特征）
        fv_proc = self.fv_processor(fv)
        
        # 特征融合
        fused = torch.cat([bev_proc, fv_proc], dim=1)
        fused = self.fusion(fused)
        
        # BEV主导的空间注意力
        spatial_weights = self.spatial_attn(bev_proc) + 0.3
        fused = fused * spatial_weights
        
        return fused