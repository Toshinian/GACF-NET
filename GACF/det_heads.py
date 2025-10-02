import torch
import numpy as np
import torch
import numpy as np
from torch import nn

from .utils.model_utils import sigmoid_hm, _fill_fc_weights

class Bev_predictor(nn.Module):
    def __init__(self, num_class, in_channels):
        super(Bev_predictor, self).__init__()

        self.num_classes = num_class - 1  # 目标类别数（不包括背景）
        
        # 每个类别的回归参数配置
        self.regression_head_cfg = [['offset'], ['loc_z'], ['dim'], ['ori_cls', 'ori_offset']]
        self.regression_channel_cfg = [[2,], [1,], [3,], [8, 8]]
        
        # 计算每个类别的回归通道数
        self.reg_channels_per_class = sum([sum(channels) for channels in self.regression_channel_cfg])
        # 2 + 1 + 3 + 8 + 8 = 22通道/类别
        
        # 总回归通道数 = 类别数 × 每类别通道数
        self.total_reg_channels = self.num_classes * self.reg_channels_per_class
        
        self.head_conv = 64

        use_norm = "BN"
        if use_norm == 'BN': norm_func = nn.BatchNorm2d
        else: norm_func = nn.Identity

        self.bn_momentum = 0.1
        self.abn_activision = 'leaky_relu'

        # ###########################################
        # ###############  Cls Heads ################
        # ########################################### 

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
            norm_func(self.head_conv, momentum=self.bn_momentum), 
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.num_classes, kernel_size=1, padding=1 // 2, bias=True),
        )
        
        self.class_head[-1].bias.data.fill_(- np.log(1 / 0.01 - 1))

        ###########################################
        ############  Regression Heads ############
        ########################################### 
        
        # 简化的回归头设计：为所有类别输出回归参数
        self.reg_features = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
            norm_func(self.head_conv, momentum=self.bn_momentum), 
            nn.ReLU(inplace=True)
        )
        
        # 输出头：输出所有类别的回归参数
        self.reg_output = nn.Conv2d(self.head_conv, self.total_reg_channels, kernel_size=1, bias=True)
        
        # 初始化回归输出权重
        _fill_fc_weights(self.reg_output, 0)

    def forward(self, features):
        # output classification
        output_cls = self.class_head(features)
        
        # output regression: 所有类别的回归参数
        reg_feature = self.reg_features(features)
        output_regs = self.reg_output(reg_feature)
                
        output_cls = sigmoid_hm(output_cls) # sigmoid & clamp
        
        return output_cls, output_regs