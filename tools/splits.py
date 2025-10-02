import os
import random
from collections import defaultdict, Counter

def split_dataset_by_class(data_dir, train_ratio=0.9, main_class_only=True):
    """
    按类别均衡分割数据集为训练集和验证集
    
    参数:
        data_dir: 标签文件目录路径
        train_ratio: 训练集比例(默认0.8)
        main_class_only: 仅使用文件中的主要类别(True: 取最频繁的类别, False: 考虑文件中所有类别)
    """
    # 类别列表
    target_classes = ['Garbage', 'SmallBoat', 'Yacht', 'LargeShip']
    
    # 按类别存储文件
    class_files = defaultdict(list)
    ignored_files = 0
    
    # 1. 扫描所有标签文件
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not file_path.endswith('.txt') or not os.path.isfile(file_path):
            continue
            
        with open(file_path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines() if line.strip()]
        
        # 提取文件中的所有有效类别
        file_classes = []
        for line in lines:
            if len(line) > 0 and line[0] in target_classes:
                file_classes.append(line[0])
        
        # 忽略不含目标类别的文件
        if not file_classes:
            ignored_files += 1
            continue
        
        file_id = os.path.splitext(file_name)[0]
        
        if main_class_only:
            # 取出现频率最高的类别作为文件类别
            class_count = Counter(file_classes)
            main_class = class_count.most_common(1)[0][0]
            class_files[main_class].append(file_id)
        else:
            # 考虑文件中出现的所有类别
            for cls in set(file_classes):
                class_files[cls].append(file_id)
    
    print(f"扫描完成，发现 {sum(len(v) for v in class_files.values())} 个有效文件")
    print(f"忽略 {ignored_files} 个不含目标类别的文件")
    
    # 2. 统计每个类别的文件数并打印
    print("\n类别分布统计:")
    for cls in target_classes:
        count = len(class_files.get(cls, []))
        print(f"- {cls}: {count} 个文件")
    
    # 3. 按类别分割数据集
    train_files = []
    val_files = []
    
    for cls, files in class_files.items():
        # 随机打乱当前类别的文件顺序
        random.shuffle(files)
        
        # 计算当前类别的分割点
        n_train = max(1, int(len(files) * train_ratio))
        n_val = len(files) - n_train
        
        # 确保每个类别在验证集中至少有一个文件
        if n_val == 0 and len(files) > 1:
            n_train -= 1
            n_val = 1
        
        # 添加到结果集
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:])
        
        print(f"\n类别 '{cls}':")
        print(f"  训练集: {n_train} 个文件 ({n_train/len(files)*100:.1f}%)")
        print(f"  验证集: {n_val} 个文件 ({n_val/len(files)*100:.1f}%)")
    
    # 再次打乱整个数据集(混合不同类别)
    random.shuffle(train_files)
    random.shuffle(val_files)
    
    # 4. 保存结果
    with open('train_files.txt', 'w') as f:
        f.write('\n'.join(train_files))
    
    with open('val_files.txt', 'w') as f:
        f.write('\n'.join(val_files))
    
    print("\n数据集分割完成:")
    print(f"- 训练集总数: {len(train_files)} 个文件 (保存到 train_files.txt)")
    print(f"- 验证集总数: {len(val_files)} 个文件 (保存到 val_files.txt)")

# 使用示例
if __name__ == "__main__":
    dataset_directory = "/datasets/FlowSense_BEV/training/label_2"
    split_dataset_by_class(dataset_directory, train_ratio=0.9, main_class_only=True)