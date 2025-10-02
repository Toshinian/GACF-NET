"""
自定义3D目标检测mAP评估模块
支持自定义数据集格式和类别定义
"""

import os
import numpy as np
import torch
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import concurrent.futures as futures
from collections import OrderedDict
import numba
import gc
from tqdm import tqdm

@dataclass
class DetectionConfig:
    """检测评估配置类"""
    # 类别定义
    class_names: List[str]
    class_ids: Dict[str, int]
    
    # IoU阈值配置
    iou_thresholds: Dict[str, float] = None  # 每个类别的IoU阈值
    default_iou_threshold: float = 0.5
    
    # 难度等级定义
    difficulty_levels: List[str] = None  # ['easy', 'moderate', 'hard']
    
    # 评估参数
    min_height_thresholds: Dict[str, float] = None  # 最小高度阈值
    max_occlusion_thresholds: Dict[str, int] = None  # 最大遮挡阈值
    max_truncation_thresholds: Dict[str, float] = None  # 最大截断阈值
    
    # 坐标系统
    coordinate_system: str = 'camera'  # 'camera', 'lidar', 'world'
    
    # 评估类型
    eval_types: List[str] = None  # ['bbox', 'bev', '3d']
    
    def __post_init__(self):
        if self.iou_thresholds is None:
            self.iou_thresholds = {name: self.default_iou_threshold for name in self.class_names}
        
        if self.difficulty_levels is None:
            self.difficulty_levels = ['easy', 'moderate', 'hard']
        
        if self.min_height_thresholds is None:
            self.min_height_thresholds = {name: 25.0 for name in self.class_names}
        
        if self.max_occlusion_thresholds is None:
            self.max_occlusion_thresholds = {name: 2 for name in self.class_names}
        
        if self.max_truncation_thresholds is None:
            self.max_truncation_thresholds = {name: 0.5 for name in self.class_names}
        
        if self.eval_types is None:
            # 默认仅评估 BEV 与 3D
            self.eval_types = ['bev', '3d']


class CustomDataParser:
    """自定义数据格式解析器"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def parse_gt_annotation(self, label_path: str, show_progress: bool = False) -> Dict:
        """
        解析真值标注文件
        支持多种格式：KITTI格式、COCO格式、自定义格式
        
        Args:
            label_path: 标注文件路径
            show_progress: 是否显示进度条
            
        Returns:
            dict: 包含标注信息的字典
        """
        if not os.path.exists(label_path):
            return self._empty_annotation()
        
        # 根据文件扩展名选择解析方法
        ext = Path(label_path).suffix.lower()
        
        if ext == '.txt':
            return self._parse_kitti_format(label_path, show_progress)
        elif ext == '.json':
            return self._parse_coco_format(label_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    def parse_prediction(self, pred_path: str, show_progress: bool = False) -> Dict:
        """
        解析预测结果文件
        
        Args:
            pred_path: 预测文件路径
            show_progress: 是否显示进度条
            
        Returns:
            dict: 包含预测信息的字典
        """
        if not os.path.exists(pred_path):
            return self._empty_annotation()
        
        ext = Path(pred_path).suffix.lower()
        
        if ext == '.txt':
            return self._parse_kitti_format(pred_path, show_progress)
        elif ext == '.json':
            return self._parse_coco_format(pred_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    def _parse_kitti_format(self, file_path: str, show_progress: bool = False) -> Dict:
        """解析KITTI格式文件"""
        annotations = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': []
        }
        
        # 先统计文件行数用于进度条
        total_lines = 0
        if show_progress:
            with open(file_path, 'r') as f:
                total_lines = sum(1 for line in f if line.strip())
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return self._empty_annotation()
        
        # 添加进度条
        if show_progress and total_lines > 0:
            lines = tqdm(lines, desc=f"解析文件: {os.path.basename(file_path)}", 
                        unit="行", total=total_lines)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 15:
                continue
            
            # KITTI格式: type truncated occluded alpha bbox[4] dimensions[3] location[3] rotation_y [score]
            name = parts[0]
            if name not in self.config.class_names:
                continue
            
            truncated = float(parts[1])
            occluded = int(parts[2])
            alpha = float(parts[3])
            bbox = [float(x) for x in parts[4:8]]
            dimensions = [float(x) for x in parts[8:11]]  # h, w, l
            location = [float(x) for x in parts[11:14]]
            rotation_y = float(parts[14])
            
            # 转换dimensions格式: h,w,l -> l,h,w (KITTI标准)
            dimensions = [dimensions[2], dimensions[0], dimensions[1]]
            
            annotations['name'].append(name)
            annotations['truncated'].append(truncated)
            annotations['occluded'].append(occluded)
            annotations['alpha'].append(alpha)
            annotations['bbox'].append(bbox)
            annotations['dimensions'].append(dimensions)
            annotations['location'].append(location)
            annotations['rotation_y'].append(rotation_y)
            
            # 处理得分
            if len(parts) >= 16:
                annotations['score'].append(float(parts[15]))
            else:
                annotations['score'].append(1.0)
        
        # 转换为numpy数组
        for key in annotations:
            annotations[key] = np.array(annotations[key])
        
        return annotations
    
    def _parse_coco_format(self, file_path: str) -> Dict:
        """解析COCO格式文件"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        annotations = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': []
        }
        
        # 创建类别ID到名称的映射
        id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        
        for ann in data.get('annotations', []):
            category_id = ann['category_id']
            if category_id not in id_to_name:
                continue
            
            name = id_to_name[category_id]
            if name not in self.config.class_names:
                continue
            
            # COCO格式的3D信息通常在extra字段中
            extra = ann.get('extra', {})
            
            annotations['name'].append(name)
            annotations['truncated'].append(extra.get('truncated', 0.0))
            annotations['occluded'].append(extra.get('occluded', 0))
            annotations['alpha'].append(extra.get('alpha', 0.0))
            annotations['bbox'].append(ann['bbox'])  # [x, y, w, h]
            annotations['dimensions'].append(extra.get('dimensions', [1.0, 1.0, 1.0]))
            annotations['location'].append(extra.get('location', [0.0, 0.0, 0.0]))
            annotations['rotation_y'].append(extra.get('rotation_y', 0.0))
            annotations['score'].append(ann.get('score', 1.0))
        
        # 转换为numpy数组
        for key in annotations:
            annotations[key] = np.array(annotations[key])
        
        return annotations
    
    def _empty_annotation(self) -> Dict:
        """返回空的标注字典"""
        return {
            'name': np.array([]),
            'truncated': np.array([]),
            'occluded': np.array([]),
            'alpha': np.array([]),
            'bbox': np.array([]).reshape(0, 4),
            'dimensions': np.array([]).reshape(0, 3),
            'location': np.array([]).reshape(0, 3),
            'rotation_y': np.array([]),
            'score': np.array([])
        }


class Custom3DIoU:
    """自定义3D IoU计算器"""
    
    @staticmethod
    def compute_3d_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        计算3D边界框的IoU
        
        Args:
            boxes1: [N, 7] (x, y, z, l, h, w, ry)
            boxes2: [M, 7] (x, y, z, l, h, w, ry)
            
        Returns:
            IoU矩阵 [N, M]
        """
        return _nb_3d_iou_matrix(boxes1.astype(np.float64), boxes2.astype(np.float64))


# ----------------------
# 模块级 numba 函数，避免在 nopython 环境中引用类方法
# ----------------------

@numba.jit(nopython=True)
def _nb_bev_iou(x1: float, z1: float, l1: float, w1: float,
                x2: float, z2: float, l2: float, w2: float) -> float:
    x1_min = x1 - l1 / 2.0
    x1_max = x1 + l1 / 2.0
    z1_min = z1 - w1 / 2.0
    z1_max = z1 + w1 / 2.0

    x2_min = x2 - l2 / 2.0
    x2_max = x2 + l2 / 2.0
    z2_min = z2 - w2 / 2.0
    z2_max = z2 + w2 / 2.0

    x_min = x1_min if x1_min > x2_min else x2_min
    x_max = x1_max if x1_max < x2_max else x2_max
    z_min = z1_min if z1_min > z2_min else z2_min
    z_max = z1_max if z1_max < z2_max else z2_max

    if x_max <= x_min or z_max <= z_min:
        return 0.0

    inter = (x_max - x_min) * (z_max - z_min)
    area1 = l1 * w1
    area2 = l2 * w2
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


@numba.jit(nopython=True)
def _nb_single_3d_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    # box: [x, y, z, l, h, w, ry]
    bev_iou = _nb_bev_iou(box1[0], box1[2], box1[3], box1[5],
                          box2[0], box2[2], box2[3], box2[5])
    h1_min = box1[1] - box1[4] / 2.0
    h1_max = box1[1] + box1[4] / 2.0
    h2_min = box2[1] - box2[4] / 2.0
    h2_max = box2[1] + box2[4] / 2.0
    # height IoU
    h_inter = (h1_max if h1_max < h2_max else h2_max) - (h1_min if h1_min > h2_min else h2_min)
    if h_inter < 0.0:
        h_inter = 0.0
    h_union = (h1_max if h1_max > h2_max else h2_max) - (h1_min if h1_min < h2_min else h2_min)
    if h_union <= 0.0:
        return 0.0
    h_iou = h_inter / h_union
    return bev_iou * h_iou


@numba.jit(nopython=True)
def _nb_3d_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    ious = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            ious[i, j] = _nb_single_3d_iou(boxes1[i], boxes2[j])
    return ious.astype(np.float32)


# BEV IoU 矩阵（使用 numba 加速）
@numba.jit(nopython=True)
def _nb_bev_iou_matrix(bev_boxes1: np.ndarray, bev_boxes2: np.ndarray) -> np.ndarray:
    # bev_boxes: [x, z, l, w]
    N = bev_boxes1.shape[0]
    M = bev_boxes2.shape[0]
    ious = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            ious[i, j] = _nb_bev_iou(
                bev_boxes1[i, 0], bev_boxes1[i, 1], bev_boxes1[i, 2], bev_boxes1[i, 3],
                bev_boxes2[j, 0], bev_boxes2[j, 1], bev_boxes2[j, 2], bev_boxes2[j, 3]
            )
    return ious.astype(np.float32)


class CustomMAPEvaluator:
    """自定义mAP评估器"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.parser = CustomDataParser(config)
        self.iou_calculator = Custom3DIoU()

    def _normalize_name_array(self, names) -> np.ndarray:
        """将输入名称序列规范化为一维字符串numpy数组，避免标量或嵌套结构造成比较异常"""
        if names is None:
            return np.empty((0,), dtype=object)
        if isinstance(names, np.ndarray):
            flat = names.ravel()
        elif isinstance(names, (list, tuple)):
            flat_list = []
            for v in names:
                if isinstance(v, (list, tuple, np.ndarray)):
                    flat_list.extend(np.asarray(v).ravel().tolist())
                else:
                    flat_list.append(v)
            flat = np.array(flat_list, dtype=object)
        else:
            flat = np.array([names], dtype=object)
        # 统一转换为字符串
        try:
            flat = np.array([str(x) for x in flat], dtype=object)
        except Exception:
            flat = flat.astype(object)
        return flat
    
    def evaluate(self, 
                 gt_folder: str, 
                 pred_folder: str, 
                 image_ids: List[str],
                 score_threshold: float = 0.1,
                 show_progress: bool = False) -> Tuple[str, Dict]:
        """
        执行mAP评估
        
        Args:
            gt_folder: 真值标注文件夹
            pred_folder: 预测结果文件夹
            image_ids: 图像ID列表
            score_threshold: 得分阈值
            show_progress: 是否显示进度条
            
        Returns:
            tuple: (结果字符串, 结果字典)
        """
        # 记录进度配置
        self._show_progress = show_progress
        
        # 加载所有标注和预测
        gt_annos = []
        dt_annos = []
        
        # 添加文件处理进度条
        if show_progress:
            image_ids = tqdm(image_ids, desc="files", unit="its")
        
        for img_id in image_ids:
            gt_path = os.path.join(gt_folder, f"{img_id}.txt")
            pred_path = os.path.join(pred_folder, f"{img_id}.txt")
            
            gt_anno = self.parser.parse_gt_annotation(gt_path, show_progress=False)
            dt_anno = self.parser.parse_prediction(pred_path, show_progress=False)
            
            # 过滤低分预测
            if score_threshold > 0:
                mask = dt_anno['score'] >= score_threshold
                for key in dt_anno:
                    dt_anno[key] = dt_anno[key][mask]
            
            gt_annos.append(gt_anno)
            dt_annos.append(dt_anno)
        
        # 执行评估
        return self._compute_map(gt_annos, dt_annos)
    
    def _compute_map(self, gt_annos: List[Dict], dt_annos: List[Dict]) -> Tuple[str, Dict]:
        """计算mAP"""
        result_str = ""
        result_dict = {}
        
        # 为每个类别计算AP
        for class_name in self.config.class_names:
            class_id = self.config.class_ids[class_name]
            
            # 计算不同难度等级的AP（仅 BEV 和 3D）
            for difficulty in self.config.difficulty_levels:
                # if self._show_progress:
                #     print(f"[开始] 计算 {class_name}-{difficulty} 3D and BEV AP caculation finished")
                ap_3d = self._compute_ap_for_class(
                    gt_annos, dt_annos, class_name, difficulty, metric='3d'
                )
                # if self._show_progress:
                #     print(f"[开始] 计算 {class_name}-{difficulty} 3D and BEV AP caculation finished")
                ap_bev = self._compute_ap_for_class(
                    gt_annos, dt_annos, class_name, difficulty, metric='bev'
                )
                
                # 格式化结果
                if self._show_progress:
                    print(f"{class_name}: 3D and BEV AP caculation finished")
                result_str += f"\n{class_name} ({difficulty}):\n"
                result_str += f"  3D AP: {ap_3d:.4f}\n"
                result_str += f"  BEV AP: {ap_bev:.4f}\n"
                
                # 保存到字典
                result_dict[f"{class_name}_3D_AP_{difficulty}"] = ap_3d
                result_dict[f"{class_name}_BEV_AP_{difficulty}"] = ap_bev
        
        return result_str, result_dict
    
    def _compute_ap_for_class(self, 
                             gt_annos: List[Dict], 
                             dt_annos: List[Dict], 
                             class_name: str, 
                             difficulty: str,
                             metric: str = '3d') -> float:
        """为特定类别和难度计算AP"""
        # 收集该类别的所有GT和预测
        gt_boxes = []
        dt_boxes = []
        dt_scores = []
        for i, (gt_anno, dt_anno) in enumerate(zip(gt_annos, dt_annos)):
            # 规范化name为一维numpy数组，避免标量比较导致的FutureWarning
            gt_names = self._normalize_name_array(gt_anno.get('name', []))
            dt_names = self._normalize_name_array(dt_anno.get('name', []))
            # 过滤GT
            gt_mask = (gt_names == class_name)
            gt_mask = self._apply_difficulty_filter(gt_anno, gt_mask, difficulty)
            
            # 过滤预测
            dt_mask = (dt_names == class_name)
            
            if isinstance(gt_mask, (bool, np.bool_)):
                gt_mask = np.array([gt_mask], dtype=bool)
            if gt_mask.size > 0 and int(gt_mask.sum()) > 0:
                gt_class_boxes = self._prepare_boxes(gt_anno, gt_mask, metric)
                gt_boxes.extend(gt_class_boxes)
                
            if isinstance(dt_mask, (bool, np.bool_)):
                dt_mask = np.array([dt_mask], dtype=bool)
            if dt_mask.size > 0 and int(dt_mask.sum()) > 0:
                dt_class_boxes = self._prepare_boxes(dt_anno, dt_mask, metric)
                dt_scores_arr = np.asarray(dt_anno.get('score', np.zeros((0,), dtype=np.float32)))
                if dt_scores_arr.ndim == 0:
                    dt_scores_arr = np.array([float(dt_scores_arr)]) if dt_scores_arr.size != 0 else np.zeros((0,), dtype=np.float32)
                dt_class_scores = dt_scores_arr[dt_mask]
                dt_boxes.extend(dt_class_boxes)
                dt_scores.extend(dt_class_scores)
        if len(gt_boxes) == 0 or len(dt_boxes) == 0:
            return 0.0
        
        # 转换为numpy数组
        gt_boxes = np.array(gt_boxes)
        dt_boxes = np.array(dt_boxes)
        dt_scores = np.array(dt_scores)
        
        # 按得分排序
        sorted_indices = np.argsort(dt_scores)[::-1]
        dt_boxes = dt_boxes[sorted_indices]
        dt_scores = dt_scores[sorted_indices]
        
        # 计算IoU矩阵
        if metric == '3d':
            # 预热 numba，避免首次 JIT 卡顿无输出
            try:
                _ = _nb_3d_iou_matrix(dt_boxes[:1].astype(np.float64), gt_boxes[:1].astype(np.float64))
            except Exception:
                pass
            # 计算 IoU，并加上进度条
            ious = np.zeros((dt_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)
            for i in tqdm(range(dt_boxes.shape[0]), desc=f"Caculating {class_name}-{metric} IoU", unit="its"):
                ious[i:i+1, :] = self.iou_calculator.compute_3d_iou(dt_boxes[i:i+1], gt_boxes)
        elif metric == 'bev':
            # 预热并显示进度
            try:
                _ = _nb_bev_iou_matrix(dt_boxes[:1, :4].astype(np.float64), gt_boxes[:1, :4].astype(np.float64))
            except Exception:
                pass
            ious = np.zeros((dt_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)
            for i in tqdm(range(dt_boxes.shape[0]), desc=f"Caculating {class_name}-bev IoU", unit="its"):
                ious[i:i+1, :] = _nb_bev_iou_matrix(dt_boxes[i:i+1, :4].astype(np.float64), gt_boxes[:, :4].astype(np.float64))
        else:  # bbox
            ious = self._compute_2d_iou(dt_boxes, gt_boxes)
        
        # 计算AP
        # print(f"[AP] Start: {class_name}-{difficulty}-{metric} AP (IoU: {ious.shape})")
        ap = self._compute_ap_from_iou(ious, dt_scores, gt_boxes.shape[0])
        # if self._show_progress:
        #     print(f"[AP] {class_name}-{metric} AP = {ap:.4f}")
        return ap
    
    def _apply_difficulty_filter(self, anno: Dict, mask: np.ndarray, difficulty: str) -> np.ndarray:
        """应用难度过滤器"""
        if difficulty == 'easy':
            height_thresh = self.config.min_height_thresholds.get('easy', 200)
            occlusion_thresh = self.config.max_occlusion_thresholds.get('easy', 0)
            truncation_thresh = self.config.max_truncation_thresholds.get('easy', 0.15)
        elif difficulty == 'moderate':
            height_thresh = self.config.min_height_thresholds.get('moderate', 25)
            occlusion_thresh = self.config.max_occlusion_thresholds.get('moderate', 1)
            truncation_thresh = self.config.max_truncation_thresholds.get('moderate', 0.3)
        else:  # hard
            height_thresh = self.config.min_height_thresholds.get('hard', 25)
            occlusion_thresh = self.config.max_occlusion_thresholds.get('hard', 2)
            truncation_thresh = self.config.max_truncation_thresholds.get('hard', 0.5)
        
        # 计算边界框高度
        bbox_heights = anno['bbox'][:, 3] - anno['bbox'][:, 1]
        
        # 应用过滤器
        height_mask = True
        occlusion_mask = anno['occluded'] <= occlusion_thresh
        truncation_mask = anno['truncated'] <= truncation_thresh
        
        return mask & height_mask & occlusion_mask & truncation_mask
    
    def _prepare_boxes(self, anno: Dict, mask: np.ndarray, metric: str) -> List[List[float]]:
        """准备边界框数据"""
        boxes = []
        # 使用掩码对应的真实索引，避免与原数组未筛选的顺序不一致
        idxs = np.where(mask)[0]
        # 规范化各字段为二维数组形状
        def ensure_2d(arr, second_dim):
            a = np.asarray(arr)
            if a.size == 0:
                return a.reshape(0, second_dim)
            if a.ndim == 1:
                # 若是一维且长度正好等于second_dim，视为单个样本
                if a.shape[0] == second_dim:
                    return a.reshape(1, second_dim)
                # 否则认为每元素一个标量，不适用此字段
                return a.reshape(-1, 1)
            return a
        loc = ensure_2d(anno.get('location', np.zeros((0, 3))), 3)
        dim = ensure_2d(anno.get('dimensions', np.zeros((0, 3))), 3)
        rot = np.asarray(anno.get('rotation_y', np.zeros((0,), dtype=np.float32)))
        if rot.ndim == 0:
            rot = np.array([rot])
        bbox = ensure_2d(anno.get('bbox', np.zeros((0, 4))), 4)
        for i in idxs:
            if metric == '3d':
                # 3D: [x, y, z, l, h, w, ry]
                box = [
                    loc[i, 0],  # x
                    loc[i, 1],  # y
                    loc[i, 2],  # z
                    dim[i, 0],  # l
                    dim[i, 1],  # h
                    dim[i, 2],  # w
                    float(rot[i]) if i < rot.shape[0] else 0.0  # ry
                ]
            elif metric == 'bev':
                # BEV: [x, z, l, w, ry]
                box = [
                    loc[i, 0],  # x
                    loc[i, 2],  # z
                    dim[i, 0],  # l
                    dim[i, 2],  # w
                    float(rot[i]) if i < rot.shape[0] else 0.0  # ry
                ]
            else:  # bbox
                # 2D: [x, y, w, h]
                if i < bbox.shape[0]:
                    box = bbox[i].tolist()
                else:
                    box = [0.0, 0.0, 0.0, 0.0]
            
            boxes.append(box)
        
        return boxes
    
    def _compute_bev_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算BEV IoU"""
        # 简化实现，实际应用中可以使用更精确的旋转IoU算法
        N, M = boxes1.shape[0], boxes2.shape[0]
        ious = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                ious[i, j] = self._single_bev_iou(boxes1[i], boxes2[j])
        
        return ious
    
    def _single_bev_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算单个BEV边界框对的IoU"""
        # 简化的轴对齐边界框IoU
        x1_min, x1_max = box1[0] - box1[2]/2, box1[0] + box1[2]/2
        z1_min, z1_max = box1[1] - box1[3]/2, box1[1] + box1[3]/2
        
        x2_min, x2_max = box2[0] - box2[2]/2, box2[0] + box2[2]/2
        z2_min, z2_max = box2[1] - box2[3]/2, box2[1] + box2[3]/2
        
        # 计算交集
        x_min = max(x1_min, x2_min)
        x_max = min(x1_max, x2_max)
        z_min = max(z1_min, z2_min)
        z_max = min(z1_max, z2_max)
        
        if x_max <= x_min or z_max <= z_min:
            return 0.0
        
        intersection = (x_max - x_min) * (z_max - z_min)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_2d_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算2D IoU"""
        N, M = boxes1.shape[0], boxes2.shape[0]
        ious = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                ious[i, j] = self._single_2d_iou(boxes1[i], boxes2[j])
        
        return ious
    
    def _single_2d_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算单个2D边界框对的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 计算交集
        x_min = max(x1_min, x2_min)
        x_max = min(x1_max, x2_max)
        y_min = max(y1_min, y2_min)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_ap_from_iou(self, ious: np.ndarray, scores: np.ndarray, num_gt: int) -> float:
        """从IoU矩阵计算AP"""
        if num_gt == 0:
            return 0.0
        
        num_dt = len(scores)
        if num_dt == 0:
            return 0.0
        
        # 使用向量化操作优化匹配计算
        iou_threshold = self.config.iou_thresholds.get('default', 0.3)
        
        # 为每个预测找到最佳匹配的GT
        tp = np.zeros(num_dt, dtype=bool)
        matched_gt = np.zeros(num_gt, dtype=bool)
        
        # 按得分降序处理预测
        sorted_indices = np.argsort(scores)[::-1]
        
        for i in sorted_indices:
            # 找到未匹配的GT中IoU最大的
            available_gt = ~matched_gt
            if not available_gt.any():
                break
                
            available_ious = ious[i, available_gt]
            if available_ious.size == 0:
                continue
                
            best_iou_idx = np.argmax(available_ious)
            best_iou = available_ious[best_iou_idx]
            
            if best_iou >= iou_threshold:
                # 找到原始GT索引
                gt_indices = np.where(available_gt)[0]
                original_gt_idx = gt_indices[best_iou_idx]
                tp[i] = True
                matched_gt[original_gt_idx] = True
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(~tp)
        
        # 计算精确率和召回率
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / num_gt
        
        # 计算AP (使用简化方法，避免11点插值的复杂性)
        ap = self._compute_ap_simple(precision, recall)
        
        return ap
    
    def _compute_ap_simple(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """计算简化的AP（使用梯形积分）"""
        if len(precision) == 0 or len(recall) == 0:
            return 0.0
        
        # 确保recall是单调递增的
        sorted_indices = np.argsort(recall)
        recall_sorted = recall[sorted_indices]
        precision_sorted = precision[sorted_indices]
        
        # 使用梯形积分计算AP
        ap = 0.0
        for i in range(1, len(recall_sorted)):
            delta_recall = recall_sorted[i] - recall_sorted[i-1]
            avg_precision = (precision_sorted[i] + precision_sorted[i-1]) / 2.0
            ap += delta_recall * avg_precision
        
        return ap
    
    def _compute_ap_11_point(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """计算11点插值AP"""
        # 11个召回率点
        recall_thresholds = np.linspace(0, 1, 11)
        
        # 为每个召回率阈值找到最大精确率
        max_precision = np.zeros_like(recall_thresholds)
        
        for i, r_thresh in enumerate(recall_thresholds):
            # 找到所有召回率 >= r_thresh的点
            valid_indices = recall >= r_thresh
            if valid_indices.any():
                max_precision[i] = np.max(precision[valid_indices])
        
        # AP是11个点的平均精确率
        return np.mean(max_precision)


def create_custom_config(class_names: List[str], 
                        iou_thresholds: Dict[str, float] = None,
                        **kwargs) -> DetectionConfig:
    """
    创建自定义评估配置
    
    Args:
        class_names: 类别名称列表
        iou_thresholds: 每个类别的IoU阈值
        **kwargs: 其他配置参数
        
    Returns:
        DetectionConfig: 配置对象
    """
    class_ids = {name: i for i, name in enumerate(class_names)}
    
    if iou_thresholds is None:
        iou_thresholds = {name: 0.5 for name in class_names}
    
    return DetectionConfig(
        class_names=class_names,
        class_ids=class_ids,
        iou_thresholds=iou_thresholds,
        **kwargs
    )


# 示例用法
if __name__ == "__main__":
    # 创建自定义配置
    print("\n结果字典:")
