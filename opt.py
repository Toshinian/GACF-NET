import argparse
import os

from easydict import EasyDict as edict


def get_args():
    parser = argparse.ArgumentParser(description="Training options")
    parser.add_argument("--dataset", type=str, default="FlowSense_BEV", help="dair_c_inf or dair_i")
    # parser.add_argument("--dataset", type=str, default="orca_bev", help="dair_c_inf or dair_i")
    # parser.add_argument("--dataset", type=str, default="dair", help="dair_c_inf or dair_i")
    parser.add_argument("--save_path", type=str, default="./models/", help="Path to save models")
    parser.add_argument("--load_weights_folder", type=str, default="", help="Path to a pretrained model used for initialization")
    parser.add_argument("--model_name", type=str, default="GACF", help="Model Name with specifications")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--global_seed", type=int, default=113422, help="seed")
    parser.add_argument("--batch_size", type=int, default=12, help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument('--lrgamma', nargs='+', type=float, default=[0.0002, 0.00002, 0.000002, 0.0000001],
                        help='各阶段的学习率')
    parser.add_argument('--lr_steps', default=[20, 30, 50], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    
    parser.add_argument('--weight_decay', '--wd', default=2e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument("--scheduler_step_size", type=int, default=5, help="step size for the both schedulers")
    parser.add_argument("--weight", type=float, default=2., help="weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=256, help="size of topview occupancy map")
    
    # parser.add_argument("--classes", type=str, default=['Car', 'Pedestrian', 'Cyclist'], help="classes")
    parser.add_argument("--classes", type=str, default=['Garbage','LargeShip','Yacht','SmallBoat'], help="classes")
    # parser.add_argument("--classes", type=str, default=['Garbage'], help="classes")
    # parser.add_argument("--classes", type=str, default=['Car'], help="classes")
    parser.add_argument("--num_class", type=int, default=4, help="Number of classes")
    
    parser.add_argument("--num_epochs", type=int, default=200, help="Max number of training epoch s")
    parser.add_argument("--log_frequency", type=int, default=5,help="Log files every x epochs")
    
    parser.add_argument("--num_workers", type=int, default=16, help="Number of cpu workers for dataloaders")
    parser.add_argument('--model_split_save', type=bool, default=True)
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument('--gpu_ids', type=str, default="6", 
                   help='comma separated gpu ids to use for training')

    # 自适应权重参数
    parser.add_argument('--use_lossbalancer', default=True, type=bool, help='if use loss balancer')
    parser.add_argument('--base_loss_weights', type=float, default=[0.9,0.9,1.3,1.4,1.1],
                     help='Comma-separated weights: bev_seg,fv_seg,det_map,det_reg,det_ori')
 
    parser.add_argument('--adaptive_min_factor', type=float, default=0.5, 
                       help='自适应权重的最小值')
    parser.add_argument('--adaptive_max_factor', type=float, default=2, 
                       help='自适应权重的最大值')
    parser.add_argument('--adaptive_history_size', type=int, default=50, 
                       help='用于计算自适应权重的历史记录数量')
    parser.add_argument('--adaptive_smoothing', type=float, default=0.7,
                       help='自适应因子的平滑系数(0.0-1.0), 越大变化越慢')
    parser.add_argument('--adaptive_max_change', type=float, default=0.15,
                       help='相邻周期间最大变化幅度(0.1-0.5)')
    
    configs = edict(vars(parser.parse_args()))
    configs.data_path = "./datasets/{}/training".format(configs.dataset)

    return configs


def get_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation options")
    parser.add_argument("--dataset", type=str, default="orca/orca_bev", help="dair_c_inf or dair_i")
    parser.add_argument("--pretrained_path", type=str, default="./GACF/encoder_100", help="Path to the pretrained model")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--occ_map_size", type=int, default=256, help="size of topview occupancy map")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of cpu workers for dataloaders")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--num_class", type=int, default=2, help="Number of classes")
    parser.add_argument('--vis', action='store_true', help="visualization")
    configs = edict(vars(parser.parse_args()))
    configs.data_path = "./datasets/{}/training".format(configs.dataset)

    return configs

