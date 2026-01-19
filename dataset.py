import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio
import os
import cv2
from skimage.registration import phase_cross_correlation
from natsort import natsorted


class dataset(tud.Dataset):
    def __init__(self, opt, CAVE, KAIST):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size
        # self.path = opt.data_path
        if self.isTrain == True:
            self.num = opt.trainset_num
        else:
            self.num = opt.testset_num
        self.CAVE = CAVE
        self.KAIST = KAIST

        self.nC = 10 #number of channels
        
        ## load shift
        self.mask_shifts = self.get_shifts(opt.mask_path)
        
        ## load mask
        #files = natsorted([f for f in os.listdir(opt.mask_path) if f.endswith('.bmp')])
        all_files = natsorted([f for f in os.listdir(opt.mask_path) if f.endswith('.bmp')])
        files = all_files[-10:]
        
        mask_list=[]
        for f in files:
            img = cv2.imread(os.path.join(opt.mask_path, f), 0)
            img = img.astype(np.float32) / 255.0 # 归一化到 0-1
            mask_list.append(img)
            
        self.mask_3d = np.stack(mask_list, axis=-1)
        #data = sio.loadmat(opt.mask_path)
        #self.mask = data['mask']
        #self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 28))
        
    def get_shifts(self,bmp_dir): # 用phase_cross_correlation计算平移量
        files = natsorted([f for f in os.listdir(bmp_dir) if f.endswith('.bmp')])

        ref_img = cv2.imread(os.path.join(bmp_dir, files[0]), 0)
        shifts = []
        for i, f in enumerate(files):
            curr_img = cv2.imread(os.path.join(bmp_dir, f), 0)
            shift,_,_ = phase_cross_correlation(ref_img, curr_img, upsample_factor=10)

            # shift[1] 是水平位移

            integer_shift = -int(round(shift[1]))
            shifts.append(integer_shift)

        shift_array = np.array(shifts)
        return shift_array


    def __getitem__(self, index):
        if self.isTrain == True:
            # index1 = 0; d=0
            index1   = random.randint(0, 29)
            if self.KAIST is not None:
                d = random.randint(0, 1)
                if d == 0:
                    hsi  =  self.CAVE[:,:,:,index1]
                else:
                    hsi = self.KAIST[:, :, :, index1]
            else: hsi = self.CAVE[:,:,:,index1]
        else:
            index1 = index
            hsi = self.HSI[:, :, :, index1]
        shape = np.shape(hsi)

        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        label = hsi[px:px + self.size:1, py:py + self.size:1, :self.nC]
        # while np.max(label)==0:
        #     px = random.randint(0, shape[0] - self.size)
        #     py = random.randint(0, shape[1] - self.size)
        #     label = hsi[px:px + self.size:1, py:py + self.size:1, :]
        #     print(np.min(), np.max())

        #pxm = random.randint(0, 660 - self.size)
        #pym = random.randint(0, 660 - self.size)
        pxm = random.randint(0, 2016 - self.size) #实际拍摄得到的分辨率是2016*2016
        pym = random.randint(0, 2016 - self.size)
        
        mask_3d = self.mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :self.nC]

        '''
        mask_3d_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        mask_3d_shift[:, 0:self.size, :] = mask_3d
        for t in range(self.nC)
            # 位移操作，直接每个波段往后平移2个像素
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
        mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
        mask_3d_shift_s[mask_3d_shift_s == 0] = 1
        '''
        
        # 对mask图进行归一化 能量归一化
        mask_3d_s = np.sum(mask_3d ** 2, axis=2, keepdims=False)
        mask_3d_s[mask_3d_s == 0] = 1
        
        # 数据增强
        if self.isTrain == True:

            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label  =  np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()
        
        label_shift = np.zeros_like(label)
        for t in range (self.nC):
            s = self.mask_shifts[t]
            if s == 0:
                label_shift[:, :, t] = label[:, :, t]
            elif s > 0:
                # 向右平移：右边截断，左边补0（label_shift默认是0）
                label_shift[:, s:, t] = label[:, :-s, t]
            else:
                # 向左平移：左边截断，右边补0
                s_abs = abs(s)
                label_shift[:, :-s_abs, t] = label[:, s_abs:, t]

        temp = mask_3d * label_shift # 直接相乘
        meas = np.sum(temp, axis=2)  # 二维对应求和
        input = meas/self.nC * 2 * 1.2  # 赋值input并归一化，0.5的采样率要对应*2来补偿，1.2是对比度调整
        
        '''
        temp_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        temp_shift[:, 0:self.size, :] = temp
        for t in range(28):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)
        input = meas / 28 * 2 * 1.2
        '''
        
        QE, bit = 0.4, 2048
        input = np.random.binomial((input * bit / QE).astype(int), QE)
        input = np.float32(input) / np.float32(bit)

        label = torch.FloatTensor(label.copy()).permute(2,0,1)
        input = torch.FloatTensor(input.copy())
        mask_3d= torch.FloatTensor(mask_3d.copy()).permute(2,0,1)
        mask_3d_s = torch.FloatTensor(mask_3d_s.copy())
        return input, label, mask_3d, label_shift, mask_3d_s

    def __len__(self):
        return self.num
