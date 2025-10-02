import torch
from torch import nn
from torch.nn import functional as F

class Converter_key2channel(object):
     def __init__(self, keys, channels):
         super(Converter_key2channel, self).__init__()
         
         # flatten keys and channels
         self.keys = [key for key_group in keys for key in key_group]
         self.channels = [channel for channel_groups in channels for channel in channel_groups]

     def __call__(self, key):
        # find the corresponding index
        index = self.keys.index(key)

        s = sum(self.channels[:index])
        e = s + self.channels[index]

        return slice(s, e, 1)

def sigmoid_hm(hm_features):
    x = hm_features.sigmoid_()
    x = x.clamp(min=1e-4, max=1 - 1e-4)

    return x

def nms_hm(heat_map, kernel=3, reso=1):
    kernel = int(kernel / reso)
    if kernel % 2 == 0:
        kernel += 1
    
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat_map,
                        kernel_size=(kernel, kernel),
                        stride=1,
                        padding=pad)

    eq_index = (hmax == heat_map).float()

    return heat_map * eq_index


def select_topk(heat_map, K=100):
    '''
    Args:
        heat_map: heat_map in [N, C, H, W]
        K: top k samples to be selected
        score: detection threshold

    Returns:

    '''
    batch, cls, height, width = heat_map.size()

    # First select topk scores in all classes and batchs
    # [N, C, H, W] -----> [N, C, H*W]
    heat_map = heat_map.view(batch, cls, -1)
    # Both in [N, C, K]
    topk_scores_all, topk_inds_all = torch.topk(heat_map, K)

    # topk_inds_all = topk_inds_all % (height * width) # todo: this seems redudant
    topk_ys = (topk_inds_all / width).float()
    topk_xs = (topk_inds_all % width).float()

    assert isinstance(topk_xs, torch.cuda.FloatTensor)
    assert isinstance(topk_ys, torch.cuda.FloatTensor)

    # Select topK examples across channel (classes)
    # [N, C, K] -----> [N, C*K]
    topk_scores_all = topk_scores_all.view(batch, -1)
    # Both in [N, K]
    topk_scores, topk_inds = torch.topk(topk_scores_all, K)
    topk_clses = (topk_inds / K).float()

    assert isinstance(topk_clses, torch.cuda.FloatTensor)

    # First expand it as 3 dimension
    topk_inds_all = _gather_feat(topk_inds_all.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_inds).view(batch, K)

    return topk_scores, topk_inds_all, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind):
    '''
    Select specific indexs on feature map
    Args:
        feat: all results in 3 dimensions
        ind: positive index

    Returns:

    '''
    channel = feat.size(-1)                                            
    ind = ind.unsqueeze(-1).expand(ind.size(0), ind.size(1), channel)
    feat = feat.gather(1, ind)

    return feat


def select_point_of_interest(batch, index, feature_maps):
    '''
    Select POI(point of interest) on feature map
    Args:
        batch: batch size
        index: in point format or index format
        feature_maps: regression feature map in [N, C, H, W]

    Returns:

    '''
    w = feature_maps.shape[3]
    if len(index.shape) == 3:
        index = index[:, :, 1] * w + index[:, :, 0]
    index = index.view(batch, -1)
    # [N, C, H, W] -----> [N, H, W, C]
    feature_maps = feature_maps.permute(0, 2, 3, 1).contiguous()
    channel = feature_maps.shape[-1]
    # [N, H, W, C] -----> [N, H*W, C]
    feature_maps = feature_maps.view(batch, -1, channel)
    # expand index in channels
    index = index.unsqueeze(-1).repeat(1, 1, channel)
    # select specific features bases on POIs
    feature_maps = feature_maps.gather(1, index.long())

    return feature_maps

def _fill_fc_weights(layers, value=0):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, value)


def _reorder_dims(dims, in_order='lhw', out_order='lhw'):
    """Reorder last-dimension=3 according to desired order.
    dims: [..., 3] in in_order
    returns: [..., 3] in out_order
    """
    if in_order == out_order:
        return dims
    order_map = {ch: idx for idx, ch in enumerate(in_order)}
    idxs = [order_map[ch] for ch in out_order]
    return dims[..., idxs]


def decode_dimensions_stable(
    cls_ids: torch.Tensor,
    pred_offsets: torch.Tensor,
    dim_mean: torch.Tensor,
    dim_std: torch.Tensor,
    mode: str = 'softplus',
    use_std: bool = False,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    in_order: str = 'lhw',
    out_order: str = 'lhw'
):
    """
    Unified, stable dimension decoder with clamping and optional std usage.
    - cls_ids: [...], long/int tensor of class indices
    - pred_offsets: [..., 3], raw network outputs per object
    - dim_mean/std: [num_classes, 3] priors (in in_order)
    - mode: 'exp' | 'softplus' | 'tanh' | 'linear'
    - use_std: if True, dims = offset*std + mean; else dims = offset*mean
    - min_scale/max_scale: clamp decoded scale relative to mean (elementwise)
    - in_order/out_order: component order letters among 'l','h','w'
    Returns: decoded dims with shape [..., 3] in out_order
    """
    assert pred_offsets.shape[-1] == 3, "pred_offsets last dim must be 3"
    device = pred_offsets.device

    cls_ids = cls_ids.long().to(device)
    means = dim_mean.to(device)[cls_ids]
    stds = dim_std.to(device)[cls_ids]

    if mode == 'exp':
        # prevent overflow
        pred = torch.clamp(pred_offsets, -8.0, 8.0).exp()
    elif mode == 'softplus':
        pred = F.softplus(pred_offsets)
    elif mode == 'tanh':
        # map to [min_scale, max_scale]
        s = torch.tanh(pred_offsets)  # [-1,1]
        s = (s + 1) * 0.5  # [0,1]
        pred = s * (max_scale - min_scale) + min_scale
    else:  # 'linear'
        pred = pred_offsets

    if use_std:
        dims = pred * stds + means
    else:
        dims = pred * means

    # clamp each component relative to mean to keep stability
    lower = means * min_scale
    upper = means * max_scale
    dims = torch.max(torch.min(dims, upper), lower)

    # ensure strictly positive
    dims = torch.clamp(dims, min=1e-3)

    # reorder if needed
    dims = _reorder_dims(dims, in_order=in_order, out_order=out_order)
    return dims