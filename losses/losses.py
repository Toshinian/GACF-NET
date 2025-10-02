import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as PLT
import numpy as np
import cv2


def _gather_feat(feat, ind, mask=None):
    """Gather feature map.
        Given feature map and index, return indexed feature map.
        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.
        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
    """
    dim = feat.size(2)
    # 添加边界检查，防止索引越界
    ind = ind.clamp(0, feat.size(1) - 1)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target, class_weights: torch.Tensor = None):
        positive_index = target.eq(1).float()
        negative_index = (target.lt(1) & target.ge(0)).float()
        ignore_index = target.eq(-1).float() # ignored pixels

        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        pos_logits = torch.clamp(prediction, 1e-6, 1-1e-6)
        neg_logits = torch.clamp(1 - prediction, 1e-6, 1-1e-6)

        positive_loss = torch.log(pos_logits) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(neg_logits) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        if class_weights is not None:
            if class_weights.dim() == 1:
                cw = class_weights.view(1, -1, 1, 1).to(prediction.device)
            else:
                cw = class_weights.to(prediction.device)
            positive_loss = positive_loss * cw
            negative_loss = negative_loss * cw

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        loss = - negative_loss - positive_loss

        return loss, num_positive


class compute_transform_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_transform_losses, self).__init__()
        self.device = device
        self.l1_loss = L1Loss()

    def forward(self, outputs, retransform_output):
        loss = F.l1_loss(outputs, retransform_output, size_average=False)
        return loss

def compute_multibin_losses(vector_ori, gt_ori, num_bin=4, sample_weights: torch.Tensor = None):
    gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst
    vector_ori = vector_ori.view(-1, vector_ori.shape[-1])

    cls_losses = 0.0
    reg_losses = 0.0
    reg_cnt = 0.0
    reg_denominator = 0.0
    if sample_weights is not None:
        sample_weights = sample_weights.view(-1).to(vector_ori.device)
    for i in range(num_bin):
        # bin cls loss
        cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
        # regression loss
        valid_mask_i = (gt_ori[:, i] == 1)
        if sample_weights is not None:
            cls_losses += (cls_ce_loss * sample_weights).mean()
        else:
            cls_losses += cls_ce_loss.mean()
        if valid_mask_i.sum() > 0:
            s = num_bin * 2 + i * 2
            e = s + 2
            pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
            reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
                        F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

            if sample_weights is not None:
                reg_losses += (reg_loss * sample_weights[valid_mask_i]).sum()
                reg_denominator += sample_weights[valid_mask_i].sum()
            else:
                reg_losses += reg_loss.sum()
                reg_denominator += valid_mask_i.sum()

    reg_term = reg_losses / torch.clamp(reg_denominator if sample_weights is not None else reg_cnt, min=1.0)
    return cls_losses / num_bin + reg_term

class compute_det_losses():
    def __init__(self, device='cuda'):

        self.orien_bin_size = 4
        self.dim_modes = [False, True]
        # 维度统计：每个类别的维度均值和标准差
        # 注意：维度顺序应该是 (width, height, length) 对应 (0.45, 1.1, 0.45)
        self.dim_mean = torch.tensor([
                                        (0.4500, 1.1000, 0.4500),
                                        (6.2896, 37.7375, 11.0068),
                                        (4.7718, 28.6307, 9.5436),
                                        (3.0893, 18.5356, 7.7232)
                                ]).to(device)
        self.dim_std = torch.tensor([
                                (0.0000, 0.0000, 0.0000),
                                (0.0782, 0.4693, 0.1369),
                                (0.0160, 0.0958, 0.0319),
                                (0.0811, 0.4868, 0.2028)
                                ]).to(device)
        # self.dim_mean = torch.tensor([(1.5261, 3.8840, 1.6286), (0.8423, 1.7607, 0.6602), (1.7635, 1.7372, 0.5968)]).to(device)
        # self.dim_std = torch.tensor([(0.1367, 0.4259, 0.1022), (0.2349, 0.1133, 0.1427), (0.1766, 0.0948, 0.1242)]).to(device)
	 
        self.cls_loss = FocalLoss()
        self.reg_loss = F.l1_loss
    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
                cls_id: each object id
                dims_offset: dimension offsets, shape = (N, 3)

        Returns:
        '''
        cls_id = cls_id.long()
        cls_dimension_mean = self.dim_mean[cls_id, :]

        if self.dim_modes[0] == 'exp':
            dims_offset = dims_offset.exp()
        if self.dim_modes[1]:
            cls_dimension_std = self.dim_std[cls_id, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
            dimensions = dims_offset * cls_dimension_mean
            
        return dimensions

    def __call__(self, preds, targets):

        t_maps = targets["bev_map"]
        t_boxes = targets["bev_boxes"]
        t_inds = targets["bev_inds"]
        t_masks = targets["bev_masks"]
        t_ori = targets["bev_ori"]
        t_cls = targets["cls_ids"]

        p_maps = preds['det_cls']

        # 确保 t_cls 的形状与 t_inds 兼容
        t_cls_gathered = _gather_feat(t_cls.unsqueeze(-1), t_inds).squeeze(-1)
        
        # 为每个目标选择对应类别的回归参数
        batch_size, max_objs = t_inds.shape
        p_loc = torch.zeros(batch_size, max_objs, 3, device=p_maps.device)
        p_dim = torch.zeros(batch_size, max_objs, 3, device=p_maps.device)
        p_ori = torch.zeros(batch_size, max_objs, 16, device=p_maps.device)
        
        # 遍历每个batch和每个目标
        for b in range(batch_size):
            for i in range(max_objs):
                # 检查该目标是否有效
                if t_masks[b, i] == 0:
                    continue
                    
                # 获取该目标的类别ID
                cls_id = int(t_cls_gathered[b, i].item())
                # 确保类别ID在有效范围内
                cls_id = max(0, min(cls_id, len(self.dim_mean) - 1))

                # 计算该类别的回归参数起始位置
                start_ch = cls_id * 22  # 每类别22个通道
                
                # 提取该类别的回归参数
                p_reg_cls = preds['det_reg'][b:b+1, start_ch:start_ch+22, :, :]
                
                # 解析回归参数
                p_loc_cls = p_reg_cls[:, :3, :, :].permute(0,2,3,1).contiguous()
                p_loc_cls = p_loc_cls.view(p_loc_cls.size(0), -1, p_loc_cls.size(3))
                p_loc_cls = _gather_feat(p_loc_cls, t_inds[b:b+1, i:i+1])
                p_loc[b, i] = p_loc_cls[0, 0]
                
                p_dim_cls = p_reg_cls[:, 3:6, :, :].permute(0,2,3,1).contiguous()
                p_dim_cls = p_dim_cls.view(p_dim_cls.size(0), -1, p_dim_cls.size(3))
                p_dim_cls = _gather_feat(p_dim_cls, t_inds[b:b+1, i:i+1])
                p_dim[b, i] = p_dim_cls[0, 0]
                
                p_ori_cls = p_reg_cls[:, 6:, :, :].permute(0,2,3,1).contiguous()
                p_ori_cls = p_ori_cls.view(p_ori_cls.size(0), -1, p_ori_cls.size(3))
                p_ori_cls = _gather_feat(p_ori_cls, t_inds[b:b+1, i:i+1])
                p_ori[b, i] = p_ori_cls[0, 0]

        p_reg = torch.cat((p_loc, p_dim), 2)

        # === 类别权重（基于batch内每类正样本频次的sqrt反比，均值归一，范围裁剪） ===
        with torch.no_grad():
            pos_mask = (t_maps > 0.5).float()  # [B, C, H, W]
            per_class_pos = pos_mask.sum(dim=(0, 2, 3))  # [C]
            class_weights = 1.0 / torch.sqrt(per_class_pos + 1.0)
            if class_weights.numel() > 0:
                class_weights = class_weights / (class_weights.mean() + 1e-6)
            class_weights = class_weights.clamp(0.5, 2.0)

        # det热图分类损失（按类别权重）
        map_loss, num_map_pos = self.cls_loss(p_maps, t_maps, class_weights=class_weights)
        map_loss = map_loss / torch.clamp(num_map_pos, 1)

        # 回归损失（按对象类别权重）
        obj_class_ids = t_cls_gathered.long()  # [B, max_obj]
        obj_weights = class_weights.index_select(0, obj_class_ids.view(-1))  # [B*max_obj]
        obj_weights = obj_weights.view_as(obj_class_ids).to(p_reg.device)  # [B, max_obj]

        reg_per_elem = self.reg_loss(p_reg, t_boxes, reduction='none')  # [B, max_obj, 6]
        reg_weights = (t_masks.unsqueeze(2).float() * obj_weights.unsqueeze(2)).expand_as(reg_per_elem)
        reg_weighted_sum = (reg_per_elem * reg_weights).sum()
        reg_denominator = (t_masks.float() * obj_weights).sum() * p_reg.size(2)
        reg_denominator = torch.clamp(reg_denominator, min=1.0)
        reg_loss = reg_weighted_sum / reg_denominator

        # 朝向损失（按对象类别权重）
        t_masks_ori = t_masks.unsqueeze(2).expand_as(p_ori).float()
        ori_loss = compute_multibin_losses(p_ori * t_masks_ori, t_ori, num_bin=self.orien_bin_size, sample_weights=obj_weights)

		# stop when the loss has NaN or Inf
        for v in [map_loss, reg_loss, ori_loss]:
            if torch.isnan(v).sum() > 0:
                pdb.set_trace()
            if torch.isinf(v).sum() > 0:
                pdb.set_trace()
        
        return map_loss, reg_loss, ori_loss


class compute_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_losses, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss() #douga 此处修改损失函数
        self.det_loss = compute_det_losses(self.device)

    def forward(self, opt, weight, inputs, outputs):
        losses = {}
        self.opt = opt

        losses["bev_seg_loss"] = self.compute_topview_loss(
            outputs["bev_seg"],
            inputs["bev_seg"],
            weight)
        losses["fv_seg_loss"] = self.compute_topview_loss(
            outputs["fv_seg"],
            inputs["fv_seg"],
            weight)

        losses["det_map_loss"], losses["det_reg_loss"], losses["det_ori_loss"] = self.det_loss(outputs, inputs)
        
        losses["loss"] = 1 * losses["bev_seg_loss"] + losses["fv_seg_loss"] +\
                         losses["det_map_loss"] * 1 + losses["det_reg_loss"] * 1 + losses["det_ori_loss"] * 1

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):
        # outputs: [B, C, H, W]，true_top_view: [B, H, W]，标签值范围 0..C-1
        generated_top_view = outputs
        true_top_view = torch.squeeze(true_top_view.long())
        # 多类别权重：背景权重1，其余类别同一权重，可根据需要调参
        num_classes = generated_top_view.shape[1]
        class_weights = torch.ones(num_classes, device=generated_top_view.device)
        if num_classes > 1:
            class_weights[1:] = weight
        class_weights[0] = 0.05
        focal_loss = FocalLossSeg(
            alpha=2, 
            beta=4, 
            num_classes=self.opt.num_class + 1,
            class_weights=class_weights,
        )
        
        loss, _ = focal_loss(outputs, true_top_view)
        return loss

    def compute_transform_losses(self, outputs, retransform_output):
        loss = self.L1Loss(outputs, retransform_output)
        return loss

class FocalLossSeg(nn.Module):
    def __init__(self, alpha=2, beta=4, num_classes=5, class_weights=None):
        super(FocalLossSeg, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, prediction, target):
        prediction = torch.clamp(prediction, 1e-6, 1-1e-6)
        
        if target.dim() == 4:
            if target.size(1) != self.num_classes:
                raise ValueError(f"target通道数({target.size(1)})与类别数({self.num_classes})不匹配")
            multi_channel_target = target
        else:
            target = torch.clamp(target, 0, self.num_classes-1)
            multi_channel_target = torch.zeros(
                prediction.size(0), self.num_classes, prediction.size(2), prediction.size(3),
                device=prediction.device
            )
            for cls_idx in range(self.num_classes):
                multi_channel_target[:, cls_idx] = (target == cls_idx).float()
        
        total_loss = 0
        num_positives = 0
        
        # 从1开始遍历忽略背景类
        for cls_idx in range(0, self.num_classes):
            cls_pred = prediction[:, cls_idx, :, :]
            cls_target = multi_channel_target[:, cls_idx, :, :]
            
            positive_index = cls_target.eq(1).float()
            negative_index = (cls_target.lt(1) & cls_target.ge(0)).float()
            ignore_index = cls_target.eq(-1).float()

            negative_weights = torch.pow(1 - cls_target, self.beta)
            
            positive_loss = torch.log(cls_pred + 1e-6) * torch.pow(1 - cls_pred, self.alpha) * positive_index
            negative_loss = torch.log(1 - cls_pred + 1e-6) * torch.pow(cls_pred, self.alpha) * negative_weights * negative_index

            num_positive = positive_index.float().sum()
            cls_loss = - (positive_loss.sum() + negative_loss.sum())
            
            # 应用类别权重（如果提供）
            if self.class_weights is not None and num_positive > 0:
                cls_loss *= self.class_weights[cls_idx]
                
            if num_positive > 0:
                cls_loss /= num_positive
                total_loss += cls_loss
                num_positives += num_positive
        
        # 如果没有任何正样本，返回零损失
        if num_positives > 0:
            return total_loss, num_positives
        else:
            return torch.tensor(0.0, device=prediction.device), 0