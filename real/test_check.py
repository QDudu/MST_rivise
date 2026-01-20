import sys
import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2
import random
from natsort import natsorted

# 用这个函数计算平移量
from skimage.registration import phase_cross_correlation

from architecture import model_generator

CODE_DIR = '/content/drive/MyDrive/MST/real/train_code'
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)
os.chdir(CODE_DIR)

MASK_PATH = '/content/datasets/mask_local/'
GT_PATH = '/content/datasets/CAVE_local/scene01.mat'
MODEL_PATH = '/content/drive/MyDrive/MST/real/exp/mst_c10/2026-01-16_18-59-19/model_100.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_shifts(bmp_dir):
    all_files = natsorted([f for f in os.listdir(bmp_dir) if f.endswith('.bmp')])
    files = all_files[-10:]
    ref_img = cv2.imread(os.path.join(bmp_dir, files[0]), 0)
    shifts = []
    for f in files:
        curr_img = cv2.imread(os.path.join(bmp_dir, f), 0)
        shift, _, _ = phase_cross_correlation(ref_img, curr_img, upsample_factor=10)
        integer_shift = -int(round(shift[1]))
        shifts.append(integer_shift)
    return np.array(shifts)

def main():

    model = model_generator('mst', nC=10).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 与训练时一致，都是裁剪出512*512
    nC = 10
    patch_size = 512
    gt_dict = sio.loadmat(GT_PATH)
    
    # 兼容不同 key 名
    gt_key = 'data_slice' if 'data_slice' in gt_dict else 'data'
    gt_full = gt_dict[gt_key][:,:,-10:].astype(np.float32)

    if gt_full.max() > 1.1: gt_full /= 65535.0

    # 随机裁剪
    h, w, _ = gt_full.shape
    px, py = random.randint(0, h-patch_size), random.randint(0, w-patch_size)
    gt_data = gt_full[px:px+patch_size, py:py+patch_size, :]

    mask_shifts = get_shifts(MASK_PATH)
    mask_files = natsorted([f for f in os.listdir(MASK_PATH) if f.endswith('.bmp')])[-10:]

    pxm, pym = random.randint(0, 2016-patch_size), random.randint(0, 2016-patch_size)
    masks = []
    
    # 与dataset中操作完全一致
    for f in mask_files:
        mask_img = cv2.imread(os.path.join(MASK_PATH, f), 0).astype(np.float32) / 255.0
        masks.append(mask_img[pxm:pxm+patch_size, pym:pym+patch_size])
    mask_3d = np.stack(masks, axis=-1)

    label_shift = np.zeros_like(gt_data)
    for t in range(10):
        s = mask_shifts[t]
        if s == 0: 
            label_shift[:,:,t] = gt_data[:,:,t]
        elif s > 0: 
            label_shift[:, s:, t] = gt_data[:, :-s, t]
        else: 
            label_shift[:, :-abs(s), t] = gt_data[:, abs(s):, t]

    meas = np.sum(mask_3d * label_shift, axis=2)


    simu_meas = (meas / nC) * 2 * 1.2   # 参数1.2与dataset中保持一致
    simu_meas = np.clip(simu_meas, 0, None) # 防止负值导致报错

    # 加入模拟噪声
    QE, bit = 0.4, 2048
    n_counts = (simu_meas * bit / QE).astype(int)
    simu_meas_noisy = np.random.binomial(n_counts, QE).astype(np.float32) / bit

    # 强制让输入的量级基准等于真值，解决残差模型 (+x) 的偏移
    # in_scale = gt_data.mean() / (y_noisy.mean() + 1e-8)
    # y_aligned = y_noisy * in_scale

    y_tensor = torch.FloatTensor(simu_meas_noisy).unsqueeze(0).to(device)
    mask_tensor = torch.FloatTensor(mask_3d).permute(2,0,1).unsqueeze(0).to(device)
    shifts_tensor = torch.FloatTensor(mask_shifts).to(device)

    with torch.no_grad():
        out = model(y_tensor, mask_tensor, shifts=shifts_tensor)
    recon_raw = out.squeeze().cpu().numpy()


    #由于残差分支可能会带来额外的亮度增益,进行校准
    #out_scale = gt_data.mean() / (recon_raw.mean() + 1e-8)
    #recon_final = recon_raw * out_scale

    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1.2, 0.6])

    plot_idx = [0, 5, 9]
    for i, b in enumerate(plot_idx):
        # 第一行：Recon
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(recon_raw[b], cmap='gray', vmin=0, vmax=gt_data.max()*1.1)
        ax1.set_title(f"Recon Band {b}", fontsize=20, fontweight='bold')
        ax1.axis('off')

        # 第二行：GT
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(gt_data[:,:,b], cmap='gray', vmin=0, vmax=gt_data.max()*1.1)
        ax2.set_title(f"GT Band {b}", fontsize=20, fontweight='bold')
        ax2.axis('off')

    '''
    # 第三行：绝对数值光谱曲线对比
    ax_curve = fig.add_subplot(gs[2, :])
    mid = patch_size // 2
    

    # 取中心 10x10 区域的均值，减少单点噪声（?）
    r_curve = recon_raw[:, mid-5:mid+5, mid-5:mid+5].mean(axis=(1,2))
    g_curve = gt_data[mid-5:mid+5, mid-5:mid+5, :].mean(axis=(0,1))

    ax_curve.plot(r_curve, 'r-o', markersize=10, linewidth=3, label='Recon (Physical Aligned)')
    ax_curve.plot(g_curve, 'g--s', markersize=10, linewidth=3, label='GT (Ground Truth)')
    ax_curve.set_title(f"Absolute Spectral Intensity Comparison (Center 10x10 Mean)", fontsize=18)
    ax_curve.set_ylabel("Reflectance Value", fontsize=15)
    ax_curve.set_xlabel("Spectral Bands", fontsize=15)
    ax_curve.legend(fontsize=16)
    ax_curve.grid(True, linestyle='--', alpha=0.7)


    plt.tight_layout(pad=3.0)
    plt.show()
    '''

    print("\n" + "="*40)
    print(f"DEBUG SUMMARY:")
    print(f"Target GT Mean:      {gt_data.mean():.6f}")
    #print(f"1. Input Correction:  x {in_scale:.4f}")
    print(f"Raw Recon Mean:    {recon_raw.mean():.6f}")
    #print(f"3. Output Correction: x {out_scale:.4f}")
    print(f"Final Recon Mean:     {recon_raw.mean():.6f}")
    print("="*40)

if __name__ == "__main__":

    main()
