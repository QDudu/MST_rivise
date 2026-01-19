import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# 环境配置
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from dataset import dataset
from option import opt
from utils import *

def debug_visualize_v5(dataset_obj, num_samples=3, save_dir='debug_results_new'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"--- 已创建目录: {save_dir} ---")

    # 针对 nC=10，选择起始、中间、结束波段
    channel_indices = [0, 5, 9]

    for i in range(num_samples):
        # 获取数据: input(H,W), label(C,H,W), mask(C,H,W), label_shift(H,W,C)
        input_t, label_t, mask_t, label_s_t, _ = dataset_obj[i]
        
        meas = input_t.numpy() if isinstance(input_t, torch.Tensor) else input_t
        gt = label_t.numpy() if isinstance(label_t, torch.Tensor) else label_t
        mask = mask_t.numpy() if isinstance(mask_t, torch.Tensor) else mask_t
        gt_s = label_s_t # numpy (H, W, C)

        # 创建超大画布：3行 x 4列 (前三列是Band对比，第四列放综合图)
        fig = plt.figure(figsize=(28, 20)) 
        # 进一步压缩子图间距，让图片显得更大
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.15, wspace=0.05)

        # --- 第一行：原始 GT 波段 (放大版) ---
        for idx, band in enumerate(channel_indices):
            ax = fig.add_subplot(3, 4, idx + 1)
            ax.imshow(gt[band, :, :], cmap='gray', interpolation='nearest')
            ax.set_title(f"1. Original GT Band {band}", fontsize=22, fontweight='bold', pad=10)
            ax.axis('off')

        # --- 第二行：平移后的波段 (Shifted，放大版) ---
        for idx, band in enumerate(channel_indices):
            ax = fig.add_subplot(3, 4, idx + 5)
            shift_img = gt_s[:, :, band]
            ax.imshow(shift_img, cmap='gray', interpolation='nearest')
            offset = dataset_obj.mask_shifts[band]
            ax.set_title(f"2. Shifted Band {band} (Offset: {offset}px)", fontsize=22, color='blue', fontweight='bold', pad=10)
            ax.axis('off')

        # --- 第三行：对应的 Mask 波段 (放大版) ---
        for idx, band in enumerate(channel_indices):
            ax = fig.add_subplot(3, 4, idx + 9)
            mask_img = mask[band, :, :]
            ax.imshow(mask_img, cmap='gray', interpolation='nearest')
            ax.set_title(f"3. Mask Band {band}", fontsize=22, fontweight='bold', pad=10)
            ax.axis('off')

        # --- 右侧第四列：放置合成测量图 (跨行或大图) ---
        # 我们把测量图放在第一行的第四个位置，并让它稍微突出
        ax_meas = fig.add_subplot(3, 4, 4) 
        im_m = ax_meas.imshow(meas, cmap='gray', vmin=0, vmax=np.percentile(meas, 99.5))
        ax_meas.set_title("4. Comp Meas (y)", fontsize=24, fontweight='bold', color='red')
        plt.colorbar(im_m, ax=ax_meas, fraction=0.046, pad=0.04)
        ax_meas.axis('off')
        
        # 为了平衡视觉，可以把测量图的差异或总体能量分布放在 8 和 12 的位置，或者留空
        # 这里我们保持 3x4 结构的整齐度

        save_path = os.path.join(save_dir, f'debug_v5_sample_{i}.png')
        # 使用较高的 DPI 确保放大后的清晰度
        plt.savefig(save_path, bbox_inches='tight', dpi=140)
        plt.close()
        print(f"成功保存高清放大分析图：{save_path}")

if __name__ == "__main__":
    opt.nC = 10
    CAVE = prepare_data_cave(opt.data_path_CAVE, 30)
    KAIST = CAVE 
    test_dataset = dataset(opt, CAVE, KAIST)
    debug_visualize_v5(test_dataset, num_samples=3)