import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedFSBEVFusion(nn.Module):
    """
    渐进融合：
    - 以雷达(radar)为关键参考，驱动FV与BEV在BEV空间的门控残差更新
    - 使用ConvGRU式的z/r门控进行多步融合
    """
    def __init__(self, dim: int = 1024, num_iterations: int = 3, bottleneck_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations
        hidden = dim
        bottleneck = max(1, dim // bottleneck_ratio)

        # 预处理，将各模态压缩到瓶颈维度，减少融合计算
        gn_b = min(32, max(1, bottleneck))
        gn_h = 32 if (self.dim % 32 == 0) else 16
        self.pre_radar = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1, bias=False),
            nn.GroupNorm(gn_b, bottleneck),
            nn.ReLU(inplace=True),
        )
        self.pre_fv = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1, bias=False),
            nn.GroupNorm(gn_b, bottleneck),
            nn.ReLU(inplace=True),
        )
        self.pre_bev = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1, bias=False),
            nn.GroupNorm(gn_b, bottleneck),
            nn.ReLU(inplace=True),
        )

        # ConvGRU风格门控单元（雷达为锚，融合fv+bev）
        in_ch = bottleneck * 3  # [bev_bottleneck, radar_bottleneck, fv_bottleneck]
        self.gate_z = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.gate_r = nn.Conv2d(in_ch, hidden, 3, padding=1)
        # 将隐藏态先降到瓶颈再参与候选更新，降低大通道卷积对齐风险
        self.h_proj = nn.Conv2d(hidden, bottleneck, 1)
        self.cand_h = nn.Sequential(
            nn.Conv2d(bottleneck * 3, hidden, 3, padding=1),
            nn.GroupNorm(gn_h, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )

        # 输出投影
        self.out_proj = nn.Conv2d(hidden, dim, 1)

        # 初始化轻微残差缩放（帮助稳定）
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, radar_feats: torch.Tensor, fv_features: torch.Tensor, bev_features: torch.Tensor) -> torch.Tensor:
        radar_b = self.pre_radar(radar_feats)
        fv_b = self.pre_fv(fv_features)
        bev_b = self.pre_bev(bev_features)

        h = bev_features  # 隐状态直接在高维通道上更新，更有利于保留检测表征

        for _ in range(self.num_iterations):
            # 门控：由(b, r, f)共同决定更新强度
            gate_in = torch.cat([bev_b.contiguous(), radar_b.contiguous(), fv_b.contiguous()], dim=1)
            z = torch.sigmoid(self.gate_z(gate_in))
            r = torch.sigmoid(self.gate_r(gate_in))

            # 候选更新：在高维空间，结合重置后的隐状态与(b, r, f)低维线索
            h_r = r * h
            h_r_b = self.h_proj(h_r.contiguous())
            cand_in = torch.cat([h_r_b.contiguous(), radar_b.contiguous(), fv_b.contiguous()], dim=1)
            h_tilde = self.cand_h(cand_in)

            # 渐进更新（ConvGRU样式）
            h = (1 - z) * h + z * h_tilde

        # 输出：保留少量残差以稳定
        out = self.out_proj(h)
        out = out + self.residual_weight * bev_features
        return out.contiguous()


