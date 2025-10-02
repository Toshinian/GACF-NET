from torch import nn
import torch
from einops import rearrange
import torch.nn.functional as F
import math

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=k//2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MSCAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, kernel_sizes=[3,5,7],
                 strides=[1,1,1], paddings=[1,2,3], k1=2, k2=3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 自动计算特征图尺寸，假设输入为256点序列 (16x16)
        self.feat_size = int(math.sqrt(256))  # 16x16
        
        # 创建MSC模块
        self.msc = MSC(
            dim_x=d_model, dim_y=d_model, out_dim=d_model,
            num_heads=num_heads, kernel=kernel_sizes, s=strides, pad=paddings,
            k1=k1, k2=k2
        )
    
    def forward(self, x):
        """输入格式: [batch, seq_len, features]"""
        B, N, C = x.shape
        assert C == self.d_model, f"特征维度不匹配: 输入{C} vs 初始化{self.d_model}"
        
        # 将序列转换为2D特征图
        x_4d = rearrange(x, 'b (h w) c -> b c h w', h=self.feat_size, w=self.feat_size)
        
        # MSC处理（自注意力）
        output_4d = self.msc((x_4d, x_4d))
        
        # 转换回序列格式
        output = rearrange(output_4d, 'b c h w -> b (h w) c')
        return output  # [12, 256, 1024]

class MSC(nn.Module):
    def __init__(self, dim_x, dim_y, out_dim, num_heads=8, topk=True, kernel=[3,5,7],
                 s=[1,1,1], pad=[1,2,3], qkv_bias=False, qk_scale=None, 
                 attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super().__init__()
        # 使用最大维度作为统一维度
        dim = max(dim_x, dim_y, out_dim)
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 维度变换层
        self.x_proj = Conv(dim_x, dim, k=1) if dim_x != dim else nn.Identity()
        self.y_proj = Conv(dim_y, dim, k=1) if dim_y != dim else nn.Identity()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1 = k1
        self.k2 = k2

        # 自适应权重参数
        self.attn1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        # 多尺度池化层
        self.avgpool1 = nn.AvgPool2d(kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel[2], stride=s[2], padding=pad[2])
        
        self.layer_norm = nn.LayerNorm(dim)
        self.topk = topk
        self.final_proj = Conv(dim, out_dim, k=1) if dim != out_dim else nn.Identity()

    def forward(self, inputs):
        x, y = inputs
        
        # 维度统一
        x = self.x_proj(x)
        y = self.y_proj(y)
        
        # 多尺度特征提取 (y分支)
        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        y_pooled = y1 + y2 + y3
        
        # 准备注意力输入
        B, C, H, W = y_pooled.shape
        y_seq = rearrange(y_pooled, 'b c h w -> b (h w) c')
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        
        # 归一化
        y_seq = self.layer_norm(y_seq)

        # Q/K/V计算
        kv = self.kv(y_seq).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, H, N1, D]

        q = self.q(x_seq).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B, H, N, D]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N1]
        
        # Top-k注意力掩码1
        N1 = k.size(2)  # 键值对数量
        mask1 = torch.zeros(B, self.num_heads, q.size(2), N1, device=x.device)
        k_val = max(1, int(N1 / self.k1))
        index = torch.topk(attn, k=k_val, dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, -1e9))
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)  # [B, H, N, D]

        # Top-k注意力掩码2
        mask2 = torch.zeros(B, self.num_heads, q.size(2), N1, device=x.device)
        k_val = max(1, int(N1 / self.k2))
        index = torch.topk(attn, k=k_val, dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, -1e9))
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)

        # 双注意力融合
        out = out1 * self.attn1 + out2 * self.attn2
        out = out.transpose(1, 2).reshape(B, -1, C)
        
        # 输出变换
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # 恢复空间特征图
        out_4d = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return self.final_proj(out_4d)