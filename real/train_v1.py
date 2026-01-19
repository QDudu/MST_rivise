from architecture import *
from utils import *
from dataset import dataset
import torch.utils.data as tud
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import os
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# load training data
print("===> Loading CAVE dataset...")
CAVE = prepare_data_cave(opt.data_path_CAVE, 30)
KAIST = None  # 先不使用 KAIST

# 实例dataset(用dataset的数据处理逻辑进行训练) 和 DataLoader
# dataset.__getitem__ 会因KAIST为 None 只采CAVE
train_set = dataset(opt, CAVE, KAIST)
train_set.num_frame = opt.trainset_num

train_loader = tud.DataLoader(
    dataset=train_set, 
    #num_workers=8, 
    num_workers=2,
    batch_size=opt.batch_size, 
    shuffle=True,
    pin_memory=True
)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = os.path.join(opt.outf, date_time)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# 初始化日志记录器 (utils.py 中的函数)
logger = gen_log(opt.outf)
logger.info(f"Start training with {opt.method}. Dataset: CAVE only.")

# model
if opt.method == 'hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path)
    model = model.cuda()
    FDL_loss = FDL_loss.cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path,nC=opt.nC).cuda()

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)

# Loss是逐像素计算的L1Loss
criterion = nn.L1Loss()

if __name__ == "__main__":
    logger.info(f"Random Seed: {opt.seed}")
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    # 计算位移 (MST中 initial_x )
    # 转为 tensor 并放到 GPU 上
    shifts_tensor = torch.from_numpy(train_set.mask_shifts).float().cuda()

    ## pipline of training
    or epoch in range(1, opt.max_epoch + 1):
        model.train()
        
        scheduler.step()
        epoch_loss = 0
        
        start_time = time.time()

        #for i, (input, label, Mask, Phi, Phi_s) in enumerate(loader_train):
        # 对应dataset 五个返回值return: input, label, mask_3d, mask_3d, mask_3d_s
        for i, (input_meas, label, mask_3d, label_shift, mask_3d_s) in enumerate(train_loader):
            input_meas = input_meas.cuda()   # [b, h, w]
            label = label.cuda()             # [b, 23, h, w]
            mask_3d = mask_3d.cuda()         # [b, 23, h, w]
            label_shift = label_shift.cuda()
            mask_3d_s = mask_3d_s.cuda()
          
            # Phi 传的是 label_shift，Phi_s 传的是 mask_3d_s，对应 utils.init_mask
            input_mask = init_mask(mask_3d, label_shift, mask_3d_s, opt.input_mask)
            
            if opt.method == 'mst':
                # MST 的 forward 接收shifts参数
                out = model(input_meas, input_mask, shifts=shifts_tensor)
                loss = criterion(out, label)
            
            elif opt.method in ['cst_s', 'cst_m', 'cst_l']:
                out, diff_pred = model(input_meas, input_mask)
                loss_main = criterion(out, label)
                diff_gt = torch.mean(torch.abs(out.detach() - label), dim=1, keepdim=True)
                loss_sparsity = F.mse_loss(diff_gt, diff_pred)
                loss = loss_main + 2 * loss_sparsity
            
            else:
                out = model(input_meas, input_mask)
                loss = criterion(out, label)
                
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            '''
            if i % (1000) == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch + 1, i, len(Dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
                    datetime.datetime.now()))


        elapsed_time = time.time() - start_time
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(Dataset), elapsed_time))
        torch.save(model, os.path.join(opt.outf, 'model_%03d.pth' % (epoch + 1)))
        '''
        
            if (i + 1) % 200 == 0:
                latest_save_path = os.path.join(opt.outf, 'model_latest.pth')
                torch.save(model.state_dict(), latest_save_path)
                logger.info(f"Intermediate Checkpoint saved at Iter {i+1} to {latest_save_path}")

            if i % 50 == 0:
                msg = 'Epoch [%d/%d], Iter [%d/%d], Loss: %.8f' % (
                    epoch, opt.max_epoch, i, len(train_loader), loss.item() / opt.batch_size)
                logger.info(msg)

        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_set)
        logger.info('===> Epoch %d End: Avg Loss: %.8f, Time: %.2f s' % (epoch, avg_loss, elapsed_time))

        if epoch % 10 == 0 or epoch == opt.max_epoch:
            save_path = os.path.join(opt.outf, f'model_{epoch:03d}.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"Checkpoint saved to {save_path}")

    logger.info("Training Finished.")
    
    