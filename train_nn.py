
import os
import argparse
import random
import time
import numpy as np
import torch
from torch import nn
import GACF
import math

from torch.optim.lr_scheduler import ExponentialLR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter

import eval
from opt import get_args

import tqdm

from losses import compute_losses
from utils import mean_IU, mean_precision, BatchCollator
from GACF.utils.evaluation import generate_kitti_3d_detection, evaluate_python
from GACF.utils.vis_utils import show_image_with_boxes


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines



class Trainer:
    def __init__(self):
        self.opt = get_args()
        if self.opt.gpu_ids != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_ids
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
        self.num_gpus = torch.cuda.device_count()  
        print(f"Using {self.num_gpus} GPUs: {self.opt.gpu_ids}")      
         
        if not os.path.isdir(self.opt.out_dir):
            os.makedirs(self.opt.out_dir)
            os.makedirs(os.path.join(self.opt.out_dir, 'det'))
        self.device = "cuda"
        self.seed = self.opt.global_seed
        self.find_best = 0
        if self.seed != 0:
            self.set_seed()  # set seed
        self.min_loss= 1.0
        self.models = {}
        self.inputs = {}
        self.parameters_to_train = []
        self.transform_parameters_to_train = []
        self.detection_parameters_to_train = []
        self.base_parameters_to_train = []
        self.parameters_to_train_D = []
        self.weight = self.opt.weight
        
        loss_keys = ["bev_seg_loss", "fv_seg_loss", "det_map_loss", "det_reg_loss", "det_ori_loss"]
        self.epoch_adaptive_factors = {key: [] for key in loss_keys} 
        
        self.balancer = AdaptiveLossBalancer(
            loss_keys,
            config=self.opt,
            use_lossbalancer= self.opt.use_lossbalancer
        )

       
        self.criterion = compute_losses(self.device)

        self.create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.epoch = 0
        self.start_epoch = 0
        self.scheduler = 0
        
        # Save log and models path
        self.opt.log_root = self.opt.save_path
        self.opt.save_path = self.opt.save_path
        self.writer = SummaryWriter(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time))
        self.log = open(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time, '%s.csv' % self.opt.model_name), 'w')
        self.det_output = os.path.join(self.opt.out_dir, 'det')

        # 初始化模型
        self.models["encoder"] = GACF.Encoder(
            18,
            self.opt.height,
            self.opt.width,
            True
        )

        self.models['DecoupleViewProjection'] = GACF.DecoupleViewProjection(in_dim=16)
        self.models["bev_decoder"] = GACF.Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, 
            (self.opt.num_class + 1),  # 包括背景
            "bev_decoder"
        )
        self.models["fv_decoder"] = GACF.Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, 
            (self.opt.num_class + 1),  # 包括背景
            "fv_decoder"
        )

        # 初始化检测头 
        self.models["det_heads"] = GACF.Bev_predictor(
            self.opt.num_class + 1,  
            64
        )
        # 后处理，从神经网络输出的热图和回归特征中解码出输出信息
        self.det_infer = GACF.DetInfer(self.device) 

        # 添加雷达处理模块
        self.models["radar_processor"] = GACF.RadarProcessor()
        # self.models["radar_processor"] = GACF.MMWaveSparsityGuidedProcessor()
        # self.models["radar_attention"] = GACF.CrossModalRadarAttention()
        self.models["radar_attention"] = GACF.CrossAttentionWithRadar()

        self.models["MSC_attention"] = GACF.MSCAttention(1024)   

        self.models["UnifiedFSBEVFusion"] = GACF.UnifiedFSBEVFusion(1024)

        # self.models["CrossViewEnhancement"] = GACF.CrossViewEnhancement(64)
        self.models["EnhancedCrossViewFusion"] = GACF.EnhancedCrossViewFusion(64)
        self.models["det_neck_bev"] = GACF.SimpleUpsampleNeck(1024, 256, 64)
        self.models["det_neck_fv"] = GACF.SimpleUpsampleNeck(1024, 256, 64)
        
        # 将所有模型移到设备并添加到参数列表
        for key in self.models.keys():
            self.models[key].to(self.device)
            self.base_parameters_to_train += list(self.models[key].parameters())

        # 优化器
        self.parameters_to_train = [
            # {"params": self.transform_parameters_to_train, "lr": self.opt.lr_transform},
            {"params": self.base_parameters_to_train, "lr": self.opt.lr}, 
        ]
        
        self.model_optimizer = optim.AdamW(self.parameters_to_train)
        # 设置初始学习率
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] = self.opt.lr
        
        # 创建自定义调度器
        lr_steps=self.opt.lr_steps
        gamma=self.opt.lrgamma
        gamma1 = (gamma[0] / self.opt.lr) ** (1/lr_steps[0])  
        gamma2 = (gamma[1] / gamma[0]) ** (1/lr_steps[1])  
        gamma3 = (gamma[2] / gamma[1]) ** (1/lr_steps[2])  
        gamma4 = (gamma[3] / gamma[2]) ** (1/(self.opt.num_epochs - lr_steps[0]-lr_steps[1]-lr_steps[2]))  
        
        
        lrs=[self.opt.lr]
        lrs= lrs + gamma
        # 创建多阶段调度器
        self.scheduler = MultiStageExponentialLR(
            optimizer=self.model_optimizer,
            milestones=[20,50,100],  # 阶段切换点
            gammas=[gamma1, gamma2, gamma3, gamma4], 
            lrs=lrs
        )
    

        # 数据加载器
        self.dataset = GACF.KITTIObject
        self.fpath = os.path.join(self.opt.data_path, "splits", "{}_files.txt")
        train_filenames = readlines(self.fpath.format("train"))
        val_filenames = readlines(self.fpath.format("val"))
        self.val_filenames = val_filenames
        self.train_filenames = train_filenames

        train_dataset = self.dataset(self.opt, train_filenames)
        val_dataset = self.dataset(self.opt, val_filenames, is_train=False)
        
        collator = BatchCollator()
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True)
        self.val_loader = DataLoader(
            val_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True)

        # 加载权重
        if self.opt.load_weights_folder != "":
            self.load_model()
            
        # 多GPU支持
        for key in self.models.keys():
            if self.num_gpus > 1:
                self.models[key] = nn.DataParallel(self.models[key]) 
                   
        print("Models:")
        for index, model_name in enumerate(self.models.keys(), start=1):
            print(f"{index}. {model_name}")
            
        # 日志设置
        self.log_dir = os.path.join(self.opt.log_root, self.create_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self.train_log_path = os.path.join(self.log_dir, 'train_log.txt')
        self.val_log_path = os.path.join(self.log_dir, 'val_log.txt')
        
        with open(self.train_log_path, 'w') as f:
            f.write("Epoch | lr | Total Loss | BEV Seg Loss | FV Seg Loss | Det Map Loss | Det Reg Loss | Det Ori Loss\n")
        
        with open(self.val_log_path, 'w') as f:
            f.write("Epoch | BEV mIOU | BEV mAP | FV mIOU | FV mAP | Det AP (Garbage) | Det AP (LargeShip) | Det AP (Yacht) | Det AP (SmallBoat)\n")
        
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"There are {len(train_dataset)} training items and {len(val_dataset)} validation items\n")

    def train(self):
        
        if not os.path.isdir(self.opt.log_root):
            os.mkdir(self.opt.log_root)

        for self.epoch in range(self.start_epoch, self.opt.num_epochs + 1):
            
            
            loss = self.run_epoch()

            output = ("Epoch: %d | lr:%.7f | Loss: %.4f | bev seg Loss: %.4f | fv seg Loss: %.4f || det map Loss: %.4f | det reg Loss: %.4f | det ori Loss: %.4f \nloss_factors: bev seg Loss: %.4f | fv seg Loss: %.4f || det map Loss: %.4f | det reg Loss: %.4f | det ori Loss: %.4f"
                      % (self.epoch, self.model_optimizer.param_groups[-1]['lr'], self.total_loss, self.weighted_losses["bev_seg_loss"], self.weighted_losses["fv_seg_loss"], self.weighted_losses["det_map_loss"], self.weighted_losses["det_reg_loss"], self.weighted_losses["det_ori_loss"],
                      self.loss_factor["bev_seg_loss"], self.loss_factor["fv_seg_loss"], self.loss_factor["det_map_loss"], self.loss_factor["det_reg_loss"], self.loss_factor["det_ori_loss"]))

            print(output)
            
            if self.total_loss < self.min_loss:
                self.min_loss = self.total_loss
                if self.epoch >= 600 and self.epoch % self.opt.log_frequency != 0 and self.find_best:
                    self.validation(self.log)
                    self.find_best = 0
                    if self.opt.model_split_save:
                        self.save_model()
                
            self.log.write(output + '\n')
            self.log.flush()
                
            with open(self.train_log_path, 'a') as f:
                f.write(output + '\n')
                

            for loss_name in loss:
                self.writer.add_scalar(loss_name, loss[loss_name], global_step=self.epoch)
                
            if self.epoch % self.opt.log_frequency == 0:
                self.validation(self.log)
                self.find_best = 1 
                if self.opt.model_split_save:
                    self.save_model()
                                                   
                    
        self.save_model()

    def process_batch(self, inputs, validation=False):
        outputs = {}
        
        self.inputs = {key: torch.stack([t[key] for t in inputs]).to(self.device) for key in inputs[0].keys() if key != 'filename'}
        self.inputs['filename'] = [t["filename"] for t in inputs]
    
        # 处理雷达掩码
        radar_mask = self.inputs['radar_mask']  # [B, 720, 720, 3]
        radar_feats = self.models["radar_processor"](radar_mask)  # [B, 1024, 16, 16]
        # 编码图像特征
        features = self.models["encoder"](self.inputs["color"])
        bev_features, fv_features = self.models["DecoupleViewProjection"](features)
        
        # 应用雷达注意力 
        bev_features = self.models["radar_attention"](bev_features, radar_feats)
        
        # 应用注意力到前视图特征                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        #fv_features = self.models["MSCFA"](fv_features)
        original_shape = fv_features.shape
        fv_features = fv_features.permute(0, 2, 3, 1).reshape(original_shape[0], -1, original_shape[1])
        
        fv_features = self.models["MSC_attention"](fv_features)
        fv_features = fv_features.reshape(original_shape[0], original_shape[2], original_shape[3], original_shape[1]).permute(0, 3, 1, 2)
        
        bev_features = self.models["UnifiedFSBEVFusion"](radar_feats, fv_features, bev_features)
        det_bev_feat_64 = self.models["det_neck_bev"](bev_features)
        det_fv_feat_64 = self.models["det_neck_fv"](fv_features)
        
        det_feat = self.models["EnhancedCrossViewFusion"](det_fv_feat_64, det_bev_feat_64)
        
        # 解码BEV和FV分割 
        if validation:
            outputs["bev_seg"], _ = self.models["bev_decoder"](bev_features, False)
            outputs["fv_seg"], _ = self.models["fv_decoder"](fv_features, False)
        else:
            outputs["bev_seg"], _ = self.models["bev_decoder"](bev_features)
            outputs["fv_seg"], _ = self.models["fv_decoder"](fv_features)

        outputs["det_cls"], outputs["det_reg"] = self.models["det_heads"](det_feat)
        
        if validation:
            return outputs 
        
        # 计算多类别损失
        losses = self.criterion(self.opt, self.weight, self.inputs, outputs)
                                                           
        return outputs, losses

    def run_epoch(self):

        loss = {
            "loss": 0.0,
            "bev_seg_loss": 0.0,
            "fv_seg_loss": 0.0,
            "det_map_loss": 0.0,
            "det_reg_loss": 0.0,
            "det_ori_loss": 0.0,
        }
        
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()

            self.adaptive_factors = self.balancer.calculate_factor({
                k: v for k, v in losses.items() if k != "loss"
            })
            self.weighted_losses = {}
            self.total_loss = 0
            self.loss_factor = {}
            
            
            for i, loss_name in enumerate(losses):
                if loss_name != "loss":
                    self.loss_factor[loss_name] = self.adaptive_factors.get(loss_name, 1.0)
                    weighted_loss = self.opt.base_loss_weights[i] * losses[loss_name] * self.loss_factor[loss_name]
                    self.weighted_losses[loss_name] = weighted_loss
                    self.total_loss += weighted_loss
            for key, factor in self.adaptive_factors.items():
                self.epoch_adaptive_factors[key].append(factor)
            # 计算并存储本epoch的平均因子
            self.mean_factors = {}
            for key in self.epoch_adaptive_factors:
                self.mean_factors[key] = np.mean(self.epoch_adaptive_factors[key]) if self.epoch_adaptive_factors[key] else 1.0
        
            # 清空为下个epoch准备
            for key in self.epoch_adaptive_factors:
                self.epoch_adaptive_factors[key] = []
                
            self.total_loss.backward()
            
            # 梯度监控
            total_norm = 0
            for group in self.parameters_to_train:
                for p in group['params']:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.writer.add_scalar('grad_norm', total_norm, global_step=self.epoch * len(self.train_loader) + batch_idx)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.parameters_to_train for p in group['params']], 
                max_norm=1.0
            )
            
            self.model_optimizer.step()

            for loss_name in losses:
                loss[loss_name] += losses[loss_name].item()
        
        
        for loss_name in loss:
            loss[loss_name] /= len(self.train_loader)
            
        self.scheduler.step(self.epoch)
        
        return loss

    def validation(self, log):
        # 初始化每个类别的累计统计量
        num_classes = self.opt.num_class + 1
        # 只为目标类别分配空间
        bev_true_positives = [0] * num_classes
        bev_false_positives = [0] * num_classes
        bev_false_negatives = [0] * num_classes
        
        fv_true_positives = [0] * num_classes
        fv_false_positives = [0] * num_classes
        fv_false_negatives = [0] * num_classes
        
        # 初始化mAP3D评估
        det_predictions = {}
        det_gt_annos = []

        for batch_idx, inputs in tqdm.tqdm(enumerate(self.val_loader)):
            with torch.no_grad():
                outputs = self.process_batch(inputs, True)
            
            # 处理BEV分割
            bev_pred = np.squeeze(torch.argmax(outputs["bev_seg"].detach(), 1).cpu().numpy())
            bev_gt = np.squeeze(self.inputs["bev_seg"].detach().cpu().numpy())
            
            # 处理FV分割
            fv_pred = np.squeeze(torch.argmax(outputs["fv_seg"].detach(), 1).cpu().numpy())
            fv_gt = np.squeeze(self.inputs["fv_seg"].detach().cpu().numpy())
            
            # 检测推理和mAP3D评估
            det_results = self.det_infer(outputs, inputs)
            filename = self.inputs['filename'][0]  # 假设batch_size=1
            # 将CUDA张量转换为CPU numpy数组
            if isinstance(det_results, torch.Tensor):
                det_results = det_results.cpu().numpy()
            det_predictions[filename] = det_results
            
            
            for i in range(bev_pred.shape[0]):
                for j in range(bev_pred.shape[1]):
                    
                    true_cls = int(bev_gt[i, j])
                    pred_cls = int(bev_pred[i, j])

                    # 只处理目标类别
                    if   true_cls < num_classes  and pred_cls < num_classes:
                        # 计算目标类别的索引（0-based）
                        
                        if true_cls == pred_cls:
                            bev_true_positives[true_cls] += 1
                        else:
                            bev_false_positives[pred_cls] += 1
                            bev_false_negatives[true_cls] += 1

            # 更新FV统计量 - 只处理目标类别（1-4）
            for i in range(fv_pred.shape[0]):
                for j in range(fv_pred.shape[1]):
                    true_cls = int(fv_gt[i, j])
                    pred_cls = int(fv_pred[i, j])
                    
                    # 处理目标类别
                    if  true_cls < num_classes and pred_cls < num_classes:
                        # 计算目标类别的索引（0-based）

                        if true_cls == pred_cls:
                            fv_true_positives[true_cls] += 1
                        else:
                            fv_false_positives[pred_cls] += 1
                            fv_false_negatives[true_cls] += 1
        # 计算BEV指标
        bev_class_ious = []
        bev_class_aps = []

        for cls_idx in range(num_classes):
            # 计算IOU
            if bev_true_positives[cls_idx] + bev_false_positives[cls_idx] + bev_false_negatives[cls_idx] == 0:
                iou = 0.0
            else:
                iou = bev_true_positives[cls_idx] / (bev_true_positives[cls_idx] + bev_false_positives[cls_idx] + bev_false_negatives[cls_idx])
            
            # 计算准确率
            if bev_true_positives[cls_idx] + bev_false_positives[cls_idx] == 0:
                ap = 0.0
            else:
                ap = bev_true_positives[cls_idx] / (bev_true_positives[cls_idx] + bev_false_positives[cls_idx])
            
            bev_class_ious.append(iou)
            bev_class_aps.append(ap)
        
        # 计算FV指标
        fv_class_ious = []
        fv_class_aps = []

        for cls_idx in range(num_classes):
            # 计算IOU
            if fv_true_positives[cls_idx] + fv_false_positives[cls_idx] + fv_false_negatives[cls_idx] == 0:
                iou = 0.0
            else:
                iou = fv_true_positives[cls_idx] / (fv_true_positives[cls_idx] + fv_false_positives[cls_idx] + fv_false_negatives[cls_idx])
            
            # 计算准确率
            if fv_true_positives[cls_idx] + fv_false_positives[cls_idx] == 0:
                ap = 0.0
            else:
                ap = fv_true_positives[cls_idx] / (fv_true_positives[cls_idx] + fv_false_positives[cls_idx])
            
            fv_class_ious.append(iou)
            fv_class_aps.append(ap)
            
            
        class_names = self.opt.classes 
        num_classes_show = len(class_names) + 1
        bev_class_aps=bev_class_aps[1:num_classes_show]
        bev_class_ious=bev_class_ious[1:num_classes_show]
        fv_class_aps=fv_class_aps[1:num_classes_show]
        fv_class_ious=fv_class_ious[1:num_classes_show]
        bev_mean_iou = np.mean(bev_class_ious) if bev_class_ious else 0.0
        bev_mean_ap = np.mean(bev_class_aps) if bev_class_aps else 0.0
        fv_mean_iou = np.mean(fv_class_ious) if fv_class_ious else 0.0
        fv_mean_ap = np.mean(fv_class_aps) if fv_class_aps else 0.0
        
        # 输出结果 - 只显示目标类别
        output = ("Epoch %d  Class    BEV mIOU   BEV mAP    FV mIOU   FV mAP\n" % (self.epoch))
        output += "---------------------------------------------------------\n"

        
        for i, cls_name in enumerate(class_names):

            output += (
                f"{cls_name:<15}"
                f"{bev_class_ious[i]:>10.4f}  "
                f"{bev_class_aps[i]:>9.4f}   "
                f"{fv_class_ious[i]:>9.4f}  "
                f"{fv_class_aps[i]:>9.4f}\n"
            )
        
        output += "---------------------------------------------------------\n"
        output += (
            "Mean"
            f"{bev_mean_iou:>21.4f}  "
            f"{bev_mean_ap:>9.4f}   "
            f"{fv_mean_iou:>9.4f}  "
            f"{fv_mean_ap:>9.4f}\n"
        )
        
        # mAP3D评估
        
        # # 生成KITTI格式的检测结果文件
        det_output_dir = os.path.join(self.opt.out_dir,self.create_time, f'epoch_{self.epoch}')
        os.makedirs(det_output_dir, exist_ok=True)
        
            
        label_path = os.path.join(self.opt.data_path, "label_2")
        label_split_file = os.path.join(self.opt.data_path, "splits", "val_files.txt")
        Class_and_ID={ idx: cls_name for idx, cls_name in enumerate(class_names)}
        if 'dair' in self.opt.dataset:
            
            for filename, det_results in det_predictions.items():
                predict_txt = os.path.join(det_output_dir, f"{filename}.txt")
                generate_kitti_3d_detection(Class_and_ID, det_results, predict_txt)
            # 调用mAP3D评估
            for ids in range(0,num_classes_show-1):
                result, ret_dict = evaluate_python(
                    label_path=label_path,
                    result_path=det_output_dir,
                    label_split_file=label_split_file,
                    current_class=ids, 
                    metric='R40'
                )
                result = '\n' + result
                output += result
        else:

            for filename, det_results in det_predictions.items():
                predict_txt = os.path.join(det_output_dir, f"{filename}.txt")
                generate_kitti_3d_detection(Class_and_ID, det_results, predict_txt)
            # 执行评估
            result_str, result_dict = eval.quick_evaluate(
                label_folder=label_path,
                pred_folder=det_output_dir,
                class_names=class_names,
                score_threshold=0.1,
                val_file=label_split_file
            )
            # 输出结果
            
            output += ("\n" + "========================="+"det results"+"=========================")
            # 计算并显示平均mAP
            ap_3d_easy = [v for k, v in result_dict.items() if '3D_AP' in k and 'easy' in k]
            ap_bev_easy = [v for k, v in result_dict.items() if 'BEV_AP' in k and 'easy' in k]
            
            
            if ap_3d_easy:
                output+= (f"\n mAP:"+f"3D mAP: {np.mean(ap_3d_easy):.4f}"+f"  BEV mAP: {np.mean(ap_bev_easy):.4f}")
        with open(self.val_log_path, 'a') as f:
                f.write(output + '\n')       
        print(output)
        log.write(output + '\n')
        log.flush()
                
    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            self.opt.model_name,
            "weights_{}".format(self.epoch)
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            # 处理DataParallel包装
            if isinstance(model, nn.DataParallel):
                real_model = model.module
            else:
                real_model = model
                
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = real_model.state_dict()
            
            state_dict['epoch'] = self.epoch
            if model_name == "encoder":
                state_dict["height"] = self.opt.height
                state_dict["width"] = self.opt.width

            torch.save(state_dict, model_path)

    def load_model(self):
        """
        Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for key in self.models.keys():
            print("Loading {} weights...".format(key))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(key))
            
            # 获取实际模型（处理DataParallel）
            if isinstance(self.models[key], nn.DataParallel):
                real_model = self.models[key].module
            else:
                real_model = self.models[key]
                
            model_dict = real_model.state_dict()
            
            pretrained_dict = torch.load(path)
            if 'epoch' in pretrained_dict:
                self.start_epoch = pretrained_dict['epoch']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            real_model.load_state_dict(model_dict)

        # 加载Adam状态
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def set_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AdaptiveLossBalancer:
    def __init__(self, loss_keys, config, use_lossbalancer):
        self.history = {key: [] for key in loss_keys}
        self.config = config
        self.min_factor = config.adaptive_min_factor
        self.max_factor = config.adaptive_max_factor
        self.history_size = config.adaptive_history_size
        self.use_lossbalancer = use_lossbalancer
        # 添加稳定性控制参数
        self.smoothing_factor = config.adaptive_smoothing  # 新参数
        self.last_factors = {key: 1.0 for key in loss_keys}  # 存储上次的因子
        
    def calculate_factor(self, loss_dict):
        """计算各损失项的自适应权重因子"""
        if self.use_lossbalancer:
            factors = {}
            
            for key, loss_value in loss_dict.items():
                # 获取或初始化该损失的历史记录
                history = self.history[key]
                current_value = loss_value.item()
                
                # 更新历史记录
                history.append(current_value)
                if len(history) > self.history_size:
                    history.pop(0)
                    
                # 初始阶段返回默认值
                if len(history) < 5:
                    factors[key] = 1.0
                    continue
                    
                # 处理异常值
                hist_arr = np.array(history)
                if np.isnan(hist_arr).any() or np.isinf(hist_arr).any():
                    mean_val = np.nanmean(hist_arr)
                    hist_arr = np.nan_to_num(hist_arr, nan=mean_val, posinf=mean_val, neginf=mean_val)
                
                # 计算统计指标
                mean = np.mean(hist_arr)
                std = np.std(hist_arr)
                if std < 1e-7:  # 防止除零错误
                    factors[key] = 1.0
                    continue
                    
                # 复合稳定性指标计算
                cv = std / mean  # 变异系数
                trend = self._calculate_trend(hist_arr)  # 趋势变化率
                divergence = abs(current_value - mean) / std  # 当前偏差
                
                # 稳定性评分 (0-1, 1表示最稳定)
                stability_score = math.exp(-0.5 * cv) * (1 - min(abs(trend), 1)) * math.exp(-0.2 * divergence)
                
                # 根据损失类型调整参数
                if 'det' in key:  # 检测任务更敏感
                    min_fac = self.min_factor * 1.3
                    max_fac = self.max_factor * 1.3
                    steepness = 3.0
                elif 'seg' in key:  # 分割任务较稳定
                    min_fac = self.min_factor * 1.0
                    max_fac = self.max_factor * 1.0
                    steepness = 2.0
                else:  # 默认设置
                    min_fac = self.min_factor
                    max_fac = self.max_factor
                    steepness = 2.0
                    
                # 应用S型映射 (原有代码)
                factor = min_fac + (max_fac - min_fac) / (1 + math.exp(-steepness * (1 - stability_score - 0.5)))
                raw_factor = max(min_fac * 0.95, min(factor, max_fac * 1.05))
                
                # 新增: 应用指数平滑
                last_factor = self.last_factors[key]
                smoothed_factor = self.smoothing_factor * last_factor + (1 - self.smoothing_factor) * raw_factor
                
                # 新增: 限制相邻周期的最大变化幅度
                max_change = self.config.adaptive_max_change  # 新参数 (例如0.2)
                if smoothed_factor > last_factor * (1 + max_change):
                    smoothed_factor = last_factor * (1 + max_change)
                elif smoothed_factor < last_factor * (1 - max_change):
                    smoothed_factor = last_factor * (1 - max_change)
                
                factors[key] = smoothed_factor
                self.last_factors[key] = smoothed_factor  # 更新最后因子
        else:
            factors = {}
            
            for key, loss_value in loss_dict.items():
                factors[key] = 1
        return factors
    
    def _calculate_trend(self, arr):
        """计算数据的线性趋势斜率（-1到1标准化）"""
        if len(arr) < 2:
            return 0.0
            
        arr = np.array(arr)
        if np.all(arr == arr[0]):
            return 0.0
            
        # 标准化数据
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-7:
            return 0.0
            
        normalized_arr = (arr - mean) / std
        
        # 使用协方差计算斜率
        x = np.arange(len(arr))
        cov = np.cov(x, normalized_arr)
        if cov[0, 0] < 1e-7:
            return 0.0
            
        slope = cov[0, 1] / cov[0, 0]
        return max(-1.0, min(slope, 1.0))


from torch.optim.lr_scheduler import ExponentialLR

class MultiStageExponentialLR:
    """多阶段指数衰减学习率调度器"""
    def __init__(self, optimizer, milestones, gammas, lrs=5e-7):
        """
        参数:
            optimizer: 优化器对象
            milestones: 阶段切换点 [30, 100, 300]
            gammas: 每个阶段的衰减率 [gamma1, gamma2, gamma3, gamma4]
            min_lr: 最低学习率
        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gammas = gammas
        self.lrs = lrs
        self.min_lr = lrs[4]
        self.current_stage = 0
        self.current_scheduler = None
        self.stage_start_epoch = 0
        self.set_stage(0)  # 初始化第一阶段
    
    def set_stage(self, stage):
        """设置当前阶段"""
        self.current_stage = stage
        self.stage_start_epoch = self.last_epoch if hasattr(self, 'last_epoch') else 0
        
        # 计算当前阶段的起始学习率
        start_lr = self.lrs[stage]
 
        
        # 设置优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
        
        # 创建当前阶段的指数衰减调度器
        self.current_scheduler = ExponentialLR(
            self.optimizer, 
            gamma=self.gammas[stage]
        )
    
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1 if hasattr(self, 'last_epoch') else 1
        self.last_epoch = epoch
        
        # 检查是否需要切换到下一阶段
        if self.current_stage < len(self.milestones) and epoch >= self.milestones[self.current_stage]:
            self.set_stage(self.current_stage + 1)
        
        # 更新当前阶段的学习率
        self.current_scheduler.step()
        
        # 确保不低于最小学习率
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr
        
        return self.optimizer.param_groups[0]['lr']
    
if __name__ == "__main__":
    start_time = time.ctime()
    print(start_time)
    trainer = Trainer()
    trainer.train()
    end_time = time.ctime()
    print(end_time)