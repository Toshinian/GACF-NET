import os
import numpy as np
from collections import defaultdict

def parse_kitti_annotation(line):
    """解析KITTI格式的单行标注数据"""
    parts = line.strip().split()
    if len(parts) < 15:  # 确保是有效标注行
        return None
    
    class_name = parts[0]
    dimensions = list(map(float, parts[8:11]))  # 高、宽、长
    
    return class_name, dimensions

def calculate_category_stats(label_dir, target_classes):
    """
    计算目标类别的尺寸统计信息
    
    参数:
    label_dir: KITTI标注文件目录
    target_classes: 目标类别列表
    
    返回:
    mean_stats: 每个类别的平均尺寸 (高, 宽, 长)
    std_stats: 每个类别的尺寸标准差 (高, 宽, 长)
    """
    # 初始化存储结构
    category_dimensions = defaultdict(list)
    
    # 遍历所有标注文件
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    ann = parse_kitti_annotation(line)
                    if ann and ann[0] in target_classes:
                        class_name, dims = ann
                        category_dimensions[class_name].append(dims)
    
    # 计算统计信息
    mean_stats = {}
    std_stats = {}
    
    for class_name, dims in category_dimensions.items():
        dims_array = np.array(dims)
        
        # 计算平均值和标准差
        mean_stats[class_name] = (
            np.mean(dims_array[:, 0]),
            np.mean(dims_array[:, 1]),
            np.mean(dims_array[:, 2])
        )
        
        std_stats[class_name] = (
            np.std(dims_array[:, 0]),
            np.std(dims_array[:, 1]),
            np.std(dims_array[:, 2])
        )
    
    return mean_stats, std_stats

def print_tensor_format(mean_stats, std_stats, class_order):
    """以PyTorch张量格式打印统计结果"""
    print("        # self.dim_mean = torch.tensor([")
    
    # 打印平均值
    for i, class_name in enumerate(class_order):
        if class_name in mean_stats:
            h_std, w_std, l_std = std_stats[class_name]
            h, w, l = mean_stats[class_name]
            line = f"\t\t\t\t\t({h-h_std:.4f}, {w-w_std:.4f}, {l-l_std:.4f})"
            if i < len(class_order) - 1:
                line += ","
            print(line)
        else:
            print(f"\t\t\t\t\t# 警告: 类别 '{class_name}' 在数据集中未找到")
    
    print("\t\t\t\t])")
    print("        # self.dim_std = torch.tensor([")
    
    # 打印标准差
    for i, class_name in enumerate(class_order):
        if class_name in std_stats:
            h_std, w_std, l_std = std_stats[class_name]
            line = f"\t\t\t\t({h_std/2:.4f}, {w_std/2:.4f}, {l_std/2:.4f})"
            if i < len(class_order) - 1:
                line += ","
            print(line)
        else:
            print(f"\t\t\t\t# 警告: 类别 '{class_name}' 在数据集中未找到")
    
    print("\t\t\t\t])")

if __name__ == "__main__":
    # 设置KITTI标注目录路径
    kitti_label_dir = "/datasets/FlowSense_BEV/training/label_2"  # 替换为您的KITTI标注目录
    
    # 定义目标类别及其顺序
    target_classes = ["Garbage","LargeShip","Yacht", "SmallBoat"]
    
    # 计算统计信息
    mean_stats, std_stats = calculate_category_stats(kitti_label_dir, target_classes)
    
    # 打印结果
    print("类别尺寸统计结果:")
    for class_name in target_classes:
        if class_name in mean_stats:
            h, w, l = mean_stats[class_name]
            h_std, w_std, l_std = std_stats[class_name]
            print(f"{class_name}:")
            print(f"  平均值: 高={h:.4f}, 宽={w:.4f}, 长={l:.4f}")
            print(f"  标准差: 高={h_std:.4f}, 宽={w_std:.4f}, 长={l_std:.4f}")
        else:
            print(f"警告: 类别 '{class_name}' 在数据集中未找到")
    
    # 以PyTorch张量格式打印结果
    print("\nPyTorch张量格式输出:")
    print_tensor_format(mean_stats, std_stats, target_classes)