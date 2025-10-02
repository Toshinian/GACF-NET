"""
针对你的数据的超简单评估脚本
直接运行即可评估你的1604479904639label和1604479904639文件夹
"""

import os
import numpy as np
from eval import quick_evaluate

def main():
    """主函数 - 评估你的数据集"""
    
    print("🎯 自定义mAP3D评估工具")
    print("="*50)
    
    # 你的数据路径
    label_folder = ""  # 标签文件夹
    pred_folder = ""        # 预测结果文件夹
    val_file = ""                 # 验证集文件列表（可选）
    
    # 请根据你的数据集修改类别名称
    class_names = ['Garbage','Ship', 'SmallBoat']
    
    # IoU阈值设置 (可以根据需要调整)
    iou_thresholds = {
        'Garbage': 0.5,  
        'Ship': 0.5,   
        'SmallBoat': 0.5    
    }
    
    print(f"📁 标签文件夹: {label_folder}")
    print(f"📁 预测文件夹: {pred_folder}")
    print(f"📄 验证集文件: {val_file}")
    print(f"🏷️  类别: {class_names}")
    print(f"📏 IoU阈值: {iou_thresholds}")
    print("-"*50)
    
    # 检查路径
    if not os.path.exists(label_folder):
        print(f"❌ 错误: 找不到标签文件夹 {label_folder}")
        print("请确保文件夹路径正确")
        return
    
    if not os.path.exists(pred_folder):
        print(f"❌ 错误: 找不到预测文件夹 {pred_folder}")
        print("请确保文件夹路径正确")
        return
    
    # 检查文件数量
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.txt')]
    
    print(f"📊 找到 {len(label_files)} 个标签文件")
    print(f"📊 找到 {len(pred_files)} 个预测文件")
    
    if len(label_files) == 0:
        print("❌ 标签文件夹中没有.txt文件")
        return
        
    if len(pred_files) == 0:
        print("❌ 预测文件夹中没有.txt文件")
        return
    
    # 显示前几个文件名作为示例
    print(f"📄 标签文件示例: {label_files[:3]}")
    print(f"📄 预测文件示例: {pred_files[:3]}")
    print("-"*50)
    
    try:
        print("🚀 开始评估...")
        
        # 执行评估
        result_str, result_dict = quick_evaluate(
            label_folder=label_folder,
            pred_folder=pred_folder,
            class_names=class_names,
            iou_thresholds=iou_thresholds,
            score_threshold=0.1,
            val_file=val_file
        )
        
        # 输出结果
        print("\n" + "="*60)
        print("📊 评估结果")
        print("="*60)
        print(result_str)
        
        print("\n📈 详细指标:")
        for key, value in result_dict.items():
            print(f"  {key}: {value:.4f}")
        
        # 计算并显示平均mAP
        ap_3d_easy = [v for k, v in result_dict.items() if '3D_AP' in k and 'easy' in k]
        ap_bev_easy = [v for k, v in result_dict.items() if 'BEV_AP' in k and 'easy' in k]
        ap_2d_easy = [v for k, v in result_dict.items() if '2D_AP' in k and 'easy' in k]
        
        if ap_3d_easy:
            print(f"\n🎯 平均mAP (Easy难度):")
            print(f"  🎯 3D mAP: {np.mean(ap_3d_easy):.4f}")
            print(f"  🎯 BEV mAP: {np.mean(ap_bev_easy):.4f}")
            print(f"  🎯 2D mAP: {np.mean(ap_2d_easy):.4f}")
        
        # 保存结果到文件
        output_file = "your_evaluation_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("你的数据集mAP3D评估结果\n")
            f.write("="*60 + "\n")
            f.write(f"标签文件夹: {label_folder}\n")
            f.write(f"预测文件夹: {pred_folder}\n")
            f.write(f"类别: {class_names}\n")
            f.write(f"IoU阈值: {iou_thresholds}\n")
            f.write(f"文件数量: {len(label_files)} 个标签文件, {len(pred_files)} 个预测文件\n")
            f.write("-"*60 + "\n")
            f.write(result_str + "\n")
            f.write("-"*60 + "\n")
            f.write("详细结果:\n")
            for key, value in result_dict.items():
                f.write(f"{key}: {value:.4f}\n")
            
            if ap_3d_easy:
                f.write(f"\n平均mAP (Easy难度):\n")
                f.write(f"3D mAP: {np.mean(ap_3d_easy):.4f}\n")
                f.write(f"BEV mAP: {np.mean(ap_bev_easy):.4f}\n")
                f.write(f"2D mAP: {np.mean(ap_2d_easy):.4f}\n")
        
        print(f"\n💾 结果已保存到: {output_file}")
        print("✅ 评估完成!")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        print("\n🔧 故障排除:")
        print("1. 检查标签文件格式是否为KITTI格式")
        print("2. 检查类别名称是否与标签文件中的一致")
        print("3. 确保标签和预测文件数量匹配")
        print("4. 检查文件内容是否完整")

def evaluate_with_your_params(label_folder, pred_folder, class_names, iou_thresholds=None):
    """
    使用你提供的参数进行评估
    
    Args:
        label_folder: 标签文件夹路径
        pred_folder: 预测结果文件夹路径
        class_names: 类别名称列表
        iou_thresholds: IoU阈值字典
        
    Returns:
        tuple: (结果字符串, 结果字典)
    """
    
    if iou_thresholds is None:
        iou_thresholds = {name: 0.5 for name in class_names}
    
    print(f"🎯 评估参数:")
    print(f"  标签文件夹: {label_folder}")
    print(f"  预测文件夹: {pred_folder}")
    print(f"  类别: {class_names}")
    print(f"  IoU阈值: {iou_thresholds}")
    
    # 执行评估
    result_str, result_dict = quick_evaluate(
        label_folder=label_folder,
        pred_folder=pred_folder,
        class_names=class_names,
        iou_thresholds=iou_thresholds,
        score_threshold=0.1
    )
    
    return result_str, result_dict

if __name__ == "__main__":
    # 直接运行评估
    main()
    
    print("\n" + "="*60)
    print("📝 使用说明:")
    print("="*60)
    print("1. 确保你的标签文件夹和预测文件夹在同一目录下")
    print("2. 修改上面的 class_names 列表为你的实际类别")
    print("3. 根据需要调整 iou_thresholds 字典")
    print("4. 运行脚本: python your_evaluation.py")
    print("5. 查看结果文件和输出")
    print("\n💡 提示: 如果类别名称不同，请修改 class_names 列表")
