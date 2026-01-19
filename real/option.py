import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='mst', help='set template')

# Hardware
parser.add_argument("--gpu_id", type=str, default='0')

# Data 
# add my own mask path (mask.bmp)
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
parser.add_argument('--data_path_CAVE', default='../../datasets/CAVE_512_28/', type=str, help='path of cave')
parser.add_argument('--data_path_KAIST', default='../../datasets/KAIST_CVPR2021/', type=str, help='path of kaist')
parser.add_argument('--mask_path', default='../../datasets/my_mask_2/', type=str, help='path of mask')

# Saving
parser.add_argument('--outf', type=str, default='./exp/mst_s/', help='saving_path')

# Model
# add nC as number of channels
parser.add_argument("--nC", default=10, type=int, help='number of channels')
parser.add_argument('--method', type=str, default='mst_s', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='model path')
parser.add_argument("--input_setting", type=str, default='H', help='input setting')
parser.add_argument("--input_mask", type=str, default='Phi', help='input mask')

# Training
# training: write size as 512
parser.add_argument("--size", default=384, type=int, help='cropped patch size')
parser.add_argument("--seed", default=1, type=int, help='random seed')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='scheduler')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones')
parser.add_argument("--gamma", type=float, default=0.5, help='gamma')
parser.add_argument("--learning_rate", type=float, default=0.0004)

opt = parser.parse_args()

# 动态计算训练数量
#opt.trainset_num = 20000 // ((opt.size // 96) ** 2)
opt.trainset_num = 400 
# 固定每个epoch 迭代次数为400


# 处理 True/False 字符串转换
for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False