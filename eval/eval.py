"""
简化的自定义mAP3D评估脚本
直接输入路径和类别名即可输出评估结果
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import argparse
from pathlib import Path

# 导入自定义评估模块
from .custom_map3d_eval import CustomMAPEvaluator, create_custom_config

def evaluate_custom_dataset(
    label_folder: str,
    pred_folder: str, 
    class_names: List[str],
    iou_thresholds: Dict[str, float] = None,
    score_threshold: float = 0.1,
    difficulty_levels: List[str] = None,
    val_file: str = None
) -> Tuple[str, Dict]:
    """
    评估自定义数据集的mAP3D结果
    
    Args:
        label_folder: 标签文件夹路径
        pred_folder: 预测结果文件夹路径
        class_names: 类别名称列表
        iou_thresholds: 每个类别的IoU阈值，如果为None则使用默认值0.5
        score_threshold: 预测得分阈值
        difficulty_levels: 难度等级列表
        val_file: 验证集文件列表路径（可选），如果提供则只评估该文件中的文件
        
    Returns:
        tuple: (结果字符串, 结果字典)
    """
    
    # 设置默认参数
    if iou_thresholds is None:
        iou_thresholds = {name: 0.5 for name in class_names}
    
    if difficulty_levels is None:
        difficulty_levels = ['easy']
    
    # 创建评估配置
    config = create_custom_config(
        class_names=class_names,
        iou_thresholds=iou_thresholds,
        difficulty_levels=difficulty_levels
    )
    
    # 创建评估器
    evaluator = CustomMAPEvaluator(config)
    
    # 获取文件ID
    if val_file and os.path.exists(val_file):
        # 从验证集文件中读取文件ID
        # print(f"从验证集文件读取文件列表: {val_file}")
        image_ids = read_val_file(val_file)
        # 确保这些文件在标签和预测文件夹中都存在
        available_ids = get_file_ids(label_folder, pred_folder)
        image_ids = [fid for fid in image_ids if fid in available_ids]
        # print(f"验证集文件中找到 {len(image_ids)} 个有效文件")
    else:
        # 获取所有匹配的文件ID
        image_ids = get_file_ids(label_folder, pred_folder)
        # print(f"自动匹配找到 {len(image_ids)} 个文件")
    
    if not image_ids:
        return "错误: 未找到匹配的标签和预测文件", {}
    
    print(f"num of files: {len(image_ids)} ")
    print(f"classes: {class_names}")
    # print(f"IoU阈值: {iou_thresholds}")
    # print(f"得分阈值: {score_threshold}")
    # print("-" * 50)
    
    # 执行评估
    result_str, result_dict = evaluator.evaluate(
        gt_folder=label_folder,
        pred_folder=pred_folder,
        image_ids=image_ids,
        score_threshold=score_threshold,
        show_progress=True
    )
    
    return result_str, result_dict

def read_val_file(val_file_path: str) -> List[str]:
    """
    从验证集文件中读取文件ID列表
    
    Args:
        val_file_path: 验证集文件路径
        
    Returns:
        list: 文件ID列表
    """
    image_ids = []
    if os.path.exists(val_file_path):
        with open(val_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 去掉可能的扩展名
                    if line.endswith('.txt'):
                        line = line[:-4]
                    image_ids.append(line)
    return image_ids

def get_file_ids(label_folder: str, pred_folder: str) -> List[str]:
    """
    获取标签和预测文件夹中匹配的文件ID
    
    Args:
        label_folder: 标签文件夹路径
        pred_folder: 预测文件夹路径
        
    Returns:
        list: 文件ID列表
    """
    # 获取标签文件
    label_files = set()
    if os.path.exists(label_folder):
        for file in os.listdir(label_folder):
            if file.endswith('.txt'):
                file_id = file[:-4]  # 去掉.txt扩展名
                label_files.add(file_id)
    
    # 获取预测文件
    pred_files = set()
    if os.path.exists(pred_folder):
        for file in os.listdir(pred_folder):
            if file.endswith('.txt'):
                file_id = file[:-4]  # 去掉.txt扩展名
                pred_files.add(file_id)
    
    # 返回交集
    common_files = label_files.intersection(pred_files)
    return sorted(list(common_files))

def parse_kitti_label_file(file_path: str) -> Dict:
    """
    解析KITTI格式的标签文件
    
    Args:
        file_path: 标签文件路径
        
    Returns:
        dict: 解析后的标签信息
    """
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
    
    if not os.path.exists(file_path):
        return annotations
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 15:
            continue
        
        # KITTI格式: type truncated occluded alpha bbox[4] dimensions[3] location[3] rotation_y [score]
        name = parts[0]
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

def analyze_dataset(label_folder: str, pred_folder: str, class_names: List[str]):
    """
    分析数据集的基本信息
    
    Args:
        label_folder: 标签文件夹路径
        pred_folder: 预测文件夹路径
        class_names: 类别名称列表
    """
    print("=== 数据集分析 ===")
    
    # 获取文件ID
    image_ids = get_file_ids(label_folder, pred_folder)
    
    if not image_ids:
        print("未找到匹配的文件")
        return
    
    print(f"文件数量: {len(image_ids)}")
    
    # 分析标签文件
    gt_stats = {name: 0 for name in class_names}
    pred_stats = {name: 0 for name in class_names}
    
    for file_id in image_ids[:5]:  # 只分析前5个文件作为示例
        label_path = os.path.join(label_folder, f"{file_id}.txt")
        pred_path = os.path.join(pred_folder, f"{file_id}.txt")
        
        # 分析标签
        gt_anno = parse_kitti_label_file(label_path)
        for name in gt_anno['name']:
            if name in gt_stats:
                gt_stats[name] += 1
        
        # 分析预测
        pred_anno = parse_kitti_label_file(pred_path)
        for name in pred_anno['name']:
            if name in pred_stats:
                pred_stats[name] += 1
    
    print("\n标签统计 (前5个文件):")
    for name, count in gt_stats.items():
        print(f"  {name}: {count}")
    
    print("\n预测统计 (前5个文件):")
    for name, count in pred_stats.items():
        print(f"  {name}: {count}")
    
    print("-" * 50)

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='自定义mAP3D评估工具')
    parser.add_argument('--label_folder', type=str, required=True, help='标签文件夹路径')
    parser.add_argument('--pred_folder', type=str, required=True, help='预测结果文件夹路径')
    parser.add_argument('--class_names', type=str, nargs='+', required=True, help='类别名称列表')
    parser.add_argument('--iou_thresholds', type=float, nargs='+', help='IoU阈值列表，按类别顺序')
    parser.add_argument('--score_threshold', type=float, default=0.1, help='预测得分阈值')
    parser.add_argument('--val_file', type=str, help='验证集文件列表路径（可选）')
    parser.add_argument('--analyze', action='store_true', help='只分析数据集，不执行评估')
    
    args = parser.parse_args()
    
    # 处理IoU阈值
    iou_thresholds = None
    if args.iou_thresholds:
        if len(args.iou_thresholds) != len(args.class_names):
            print("错误: IoU阈值数量与类别数量不匹配")
            return
        iou_thresholds = {name: threshold for name, threshold in zip(args.class_names, args.iou_thresholds)}
    
    # 检查路径
    if not os.path.exists(args.label_folder):
        print(f"错误: 标签文件夹不存在: {args.label_folder}")
        return
    
    if not os.path.exists(args.pred_folder):
        print(f"错误: 预测文件夹不存在: {args.pred_folder}")
        return
    
    # 分析数据集
    analyze_dataset(args.label_folder, args.pred_folder, args.class_names)
    
    if args.analyze:
        return
    
    # 执行评估
    print("\n开始评估...")
    result_str, result_dict = evaluate_custom_dataset(
        label_folder=args.label_folder,
        pred_folder=args.pred_folder,
        class_names=args.class_names,
        iou_thresholds=iou_thresholds,
        score_threshold=args.score_threshold,
        val_file=args.val_file
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(result_str)
    
    print("\n详细结果:")
    for key, value in result_dict.items():
        print(f"{key}: {value:.4f}")
    
    # 保存结果到文件
    output_file = "evaluation_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("自定义mAP3D评估结果\n")
        f.write("="*60 + "\n")
        f.write(f"标签文件夹: {args.label_folder}\n")
        f.write(f"预测文件夹: {args.pred_folder}\n")
        f.write(f"类别: {args.class_names}\n")
        f.write(f"IoU阈值: {iou_thresholds}\n")
        f.write(f"得分阈值: {args.score_threshold}\n")
        f.write("-"*60 + "\n")
        f.write(result_str + "\n")
        f.write("-"*60 + "\n")
        f.write("详细结果:\n")
        for key, value in result_dict.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\n结果已保存到: {output_file}")

# 简化的函数接口
def quick_evaluate(label_folder: str, pred_folder: str, class_names: List[str], **kwargs):
    """
    快速评估接口
    
    Args:
        label_folder: 标签文件夹路径
        pred_folder: 预测结果文件夹路径  
        class_names: 类别名称列表
        **kwargs: 其他参数 (iou_thresholds, score_threshold, val_file等)
        
    Returns:
        tuple: (结果字符串, 结果字典)
    """
    return evaluate_custom_dataset(label_folder, pred_folder, class_names, **kwargs)

if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) == 1:
        print("""
=== 自定义mAP3D评估工具 ===

使用方法:
python simple_eval.py --label_folder /path/to/labels --pred_folder /path/to/predictions --class_names Class1 Class2 Class3

示例:
python simple_eval.py --label_folder ./1604479904639label --pred_folder ./1604479904639 --class_names Garbage LargeShip Yacht SmallBoat

使用验证集文件:
python simple_eval.py --label_folder ./1604479904639label --pred_folder ./1604479904639 --class_names Garbage LargeShip Yacht SmallBoat --val_file ./val.txt

参数说明:
--label_folder: 标签文件夹路径
--pred_folder: 预测结果文件夹路径
--class_names: 类别名称列表
--iou_thresholds: IoU阈值列表 (可选)
--score_threshold: 预测得分阈值 (默认0.1)
--val_file: 验证集文件列表路径 (可选)
--analyze: 只分析数据集，不执行评估

快速评估函数:
from simple_eval import quick_evaluate
result_str, result_dict = quick_evaluate(
    label_folder='./1604479904639label',
    pred_folder='./1604479904639', 
    class_names=['Garbage', 'LargeShip', 'Yacht', 'SmallBoat'],
    val_file='./val.txt'  # 可选
)
        """)
    else:
        main()
