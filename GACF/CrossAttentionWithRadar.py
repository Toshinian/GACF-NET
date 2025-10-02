import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionWithRadar(nn.Module):
    def __init__(self, bev_channels=1024):
        super().__init__()
        # 修改投影维度为相同通道数
        self.bev_proj = nn.Conv2d(bev_channels, 512, 1)  # 修改为512
        self.radar_proj = nn.Conv2d(1024, 512, 1)       # 修改为512
        self.attention = nn.MultiheadAttention(512, 8)   # 匹配维度
        self.out_proj = nn.Conv2d(512, bev_channels, 1)  # 恢复原始通道数

    def forward(self, bev_feats, radar_feats):
        # 保持空间维度
        B, C, H, W = bev_feats.shape

        # 投影处理
        q = self.bev_proj(bev_feats).flatten(2).permute(2, 0, 1)  # [H*W, B, 512]
        k = self.radar_proj(radar_feats).flatten(2).permute(2, 0, 1)
        v = k

        # 计算注意力
        attn_out, _ = self.attention(q, k, v)

        # 恢复形状
        attn_out = attn_out.permute(1, 2, 0).reshape(B, 512, H, W)  # [B, 512, H, W]

        # 恢复原始通道数
        output = bev_feats + self.out_proj(attn_out)
        return output


class CrossModalRadarAttention(nn.Module):

    def __init__(self, bev_channels=1024, radar_channels=1024, hidden_dim=256):
        super().__init__()
        
        # 特征投影到统一维度
        self.bev_proj = nn.Conv2d(bev_channels, hidden_dim, 1)
        self.radar_proj = nn.Conv2d(radar_channels, hidden_dim, 1)
        
        # 简化的注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, bev_channels, 1)
        
        # 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, bev_features, radar_features):

        # 1. 特征投影
        bev_proj = self.bev_proj(bev_features)      # [B, 256, 16, 16]
        radar_proj = self.radar_proj(radar_features)  # [B, 256, 16, 16]
        
        # 2. 计算注意力权重
        attention_input = torch.cat([bev_proj, radar_proj], dim=1)  # [B, 512, 16, 16]
        attention_weight = self.attention(attention_input)  # [B, 1, 16, 16]
        
        # 3. 应用注意力
        attended_features = radar_proj * attention_weight  # [B, 256, 16, 16]
        
        # 4. 输出投影
        enhanced_bev = self.output_proj(attended_features)  # [B, 1024, 16, 16]
        
        # 5. 残差连接
        residual_weight = torch.sigmoid(self.residual_weight)
        final_bev = bev_features + residual_weight * enhanced_bev
        
        return final_bev