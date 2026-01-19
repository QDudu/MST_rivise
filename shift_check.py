import os
import torch
from option import opt
from dataset import dataset
from utils import prepare_data_cave

def check_mask_shifts():
    print("\n" + "="*60)
    print("      CASSI 10-Channel Shifts Configuration Check")
    print("="*60)

    # 1. 模拟加载数据 (仅加载 CAVE 路径)
    # 注意：确保 opt.data_path_CAVE 指向正确的路径
    print(f"--> Loading dataset from: {opt.data_path_CAVE}")
    try:
        CAVE = prepare_data_cave(opt.data_path_CAVE, 30)
    except Exception as e:
        print(f"错误：加载数据集失败，请检查路径。详情: {e}")
        return

    # 2. 实例化 Dataset
    # 这里的 opt 会自动读取你 option.py 里的 nC (应为 10)
    train_set = dataset(opt, CAVE, KAIST=None)

    # 3. 提取并打印位移信息
    if hasattr(train_set, 'mask_shifts'):
        shifts = train_set.mask_shifts
        nC = len(shifts)
        
        print(f"\n[结果] 检测到波段总数 (nC): {nC}")
        print("-" * 40)
        print(f"{'通道(Index)':^12} | {'累积平移(Pixels)':^18} | {'步长(Interval)':^10}")
        print("-" * 40)

        for i in range(nC):
            current_shift = shifts[i]
            interval = shifts[i] - shifts[i-1] if i > 0 else 0
            print(f"{i:^12} | {int(current_shift):^18} | {int(interval):^10}")
        
        print("-" * 40)
        
        # 4. 物理参数合理性检查
        total_dispersion = shifts[-1] - shifts[0]
        print(f"\n[分析] 总色散宽度: {int(total_dispersion)} 像素")
        print(f"[分析] 裁剪尺寸 (opt.size): {opt.size}")
        print(f"[分析] 最小需要 Mask 宽度: {int(opt.size + total_dispersion)} 像素")
        
        if total_dispersion == 0:
            print("注意：所有位移均为 0，请检查 dataset.py 是否正确处理了色散逻辑。")
    else:
        print("\n[错误] 未在 dataset 类中找到 'mask_shifts' 属性。")
        print("请检查 dataset.py 的 __init__ 函数是否正确定义了 self.mask_shifts。")

    print("="*60 + "\n")

if __name__ == "__main__":
    check_mask_shifts()