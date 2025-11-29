import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
from tqdm import tqdm
import numpy as np

from data import MyValueDataSet
from model_resnet import Generator
from model_MambaMoe import MambaIRUNet
from model_MambaIR import MambaIRv2
from model_DehazeFormer import dehazeformer_b
from model_D4 import HazeRemovalNet
from loss import SSIM

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default='.checkpoints/instancnormoutdoor.pth', help="path to checkpoint")
parser.add_argument("--input_path", type=str, default="datasets/LMHaze/test/hazy", help="input test images path")
parser.add_argument("--target_path", type=str, default="datasets/LMHaze/test/GT", help="target test images path")
parser.add_argument("--output_dir", type=str, default="val_results", help="output directory")
parser.add_argument("--batch_size", type=int, default=1, help="batch size for validation")
parser.add_argument("--start_idx", type=int, default=975, help="start index for saving images")
parser.add_argument("--end_idx", type=int, default=1071, help="end index for saving images")
opt = parser.parse_args()

# 创建输出目录
os.makedirs(opt.output_dir, exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, "inputs"), exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, "outputs"), exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, "targets"), exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, "comparisons"), exist_ok=True)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
# model = MambaIRUNet().to(device)
# model = MambaIRv2().to(device)
# model = dehazeformer_b().to(device)
# model = HazeRemovalNet(64).to(device)
# 加载最佳模型
if os.path.isfile(opt.checkpoint):
    print(f"Loading checkpoint: {opt.checkpoint}")
    checkpoint = torch.load(opt.checkpoint, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 处理DataParallel包装的state_dict
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully")
    else:
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully (direct state_dict)")
else:
    print(f"Checkpoint not found: {opt.checkpoint}")
    exit(1)

model.eval()

# 数据加载
dataset = MyValueDataSet(opt.input_path, opt.target_path)
dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

# 损失函数
criterion_ssim = SSIM().to(device)
criterion_l1 = nn.L1Loss().to(device)

print(f"Total samples in validation set: {len(dataset)}")
print(f"Saving images from index {opt.start_idx} to {opt.end_idx}")

# 验证循环
psnr_list = []
ssim_list = []
l1_list = []

with torch.no_grad():
    for idx, (hazy, clear) in enumerate(tqdm(dataloader, desc="Validating")):
        hazy = hazy.to(device)
        clear = clear.to(device)

        # 前向传播
        output = model(hazy)
        output = torch.clamp(output, 0, 1)  # 确保输出在[0,1]范围内

        # 计算指标
        mse_loss = F.mse_loss(output, clear)
        l1_loss = criterion_l1(output, clear)
        ssim_loss = criterion_ssim(output, clear)

        # 计算PSNR
        if mse_loss == 0:
            psnr = 100.0
        else:
            psnr = 10 * torch.log10(1.0 / mse_loss)

        psnr_list.append(psnr.item())
        ssim_list.append(ssim_loss.item())
        l1_list.append(l1_loss.item())

        # 保存指定范围内的图片
        if opt.start_idx <= idx <= opt.end_idx:
            # 保存输入图像（有雾）
            save_image(hazy[0],
                       os.path.join(opt.output_dir, "inputs", f"sample_{idx:04d}_input.png"),
                       normalize=True)

            # 保存输出图像（去雾结果）
            save_image(output[0],
                       os.path.join(opt.output_dir, "outputs", f"sample_{idx:04d}_output.png"),
                       normalize=True)

            # 保存目标图像（清晰图像）
            save_image(clear[0],
                       os.path.join(opt.output_dir, "targets", f"sample_{idx:04d}_target.png"),
                       normalize=True)

            # 保存对比图像（三图并列）
            comparison = torch.cat([hazy[0], output[0], clear[0]], dim=2)  # 水平排列
            save_image(comparison,
                       os.path.join(opt.output_dir, "comparisons", f"sample_{idx:04d}_comparison.png"),
                       normalize=True)

            print(f"Saved sample {idx}: PSNR={psnr.item():.4f}, SSIM={ssim_loss.item():.4f}")

# 计算总体统计
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
avg_l1 = np.mean(l1_list)

# 计算指定样本范围的统计
specific_psnr = psnr_list[opt.start_idx:opt.end_idx + 1]
specific_ssim = ssim_list[opt.start_idx:opt.end_idx + 1]
specific_l1 = l1_list[opt.start_idx:opt.end_idx + 1]

avg_specific_psnr = np.mean(specific_psnr)
avg_specific_ssim = np.mean(specific_ssim)
avg_specific_l1 = np.mean(specific_l1)

# 输出结果
print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)
print(f"Overall Performance:")
print(f"  PSNR:  {avg_psnr:.4f} dB")
print(f"  SSIM:  {avg_ssim:.4f}")
print(f"  L1:    {avg_l1:.4f}")
print(f"  Samples: {len(psnr_list)}")

print(f"\nSpecific Range [{opt.start_idx}-{opt.end_idx}]:")
print(f"  PSNR:  {avg_specific_psnr:.4f} dB")
print(f"  SSIM:  {avg_specific_ssim:.4f}")
print(f"  L1:    {avg_specific_l1:.4f}")
print(f"  Samples: {len(specific_psnr)}")

print(f"\nImages saved in: {opt.output_dir}")
print(f"  Inputs:     {os.path.join(opt.output_dir, 'inputs')}")
print(f"  Outputs:    {os.path.join(opt.output_dir, 'outputs')}")
print(f"  Targets:    {os.path.join(opt.output_dir, 'targets')}")
print(f"  Comparisons: {os.path.join(opt.output_dir, 'comparisons')}")

# 保存指标到文件
with open(os.path.join(opt.output_dir, "metrics.txt"), "w") as f:
    f.write("Validation Metrics\n")
    f.write("=================\n\n")
    f.write(f"Overall Performance:\n")
    f.write(f"  PSNR:  {avg_psnr:.4f} dB\n")
    f.write(f"  SSIM:  {avg_ssim:.4f}\n")
    f.write(f"  L1:    {avg_l1:.4f}\n")
    f.write(f"  Samples: {len(psnr_list)}\n\n")

    f.write(f"Specific Range [{opt.start_idx}-{opt.end_idx}]:\n")
    f.write(f"  PSNR:  {avg_specific_psnr:.4f} dB\n")
    f.write(f"  SSIM:  {avg_specific_ssim:.4f}\n")
    f.write(f"  L1:    {avg_specific_l1:.4f}\n")
    f.write(f"  Samples: {len(specific_psnr)}\n\n")

    f.write("Per-sample metrics:\n")
    f.write("Index\tPSNR\tSSIM\tL1\n")
    for i in range(len(psnr_list)):
        f.write(f"{i}\t{psnr_list[i]:.4f}\t{ssim_list[i]:.4f}\t{l1_list[i]:.4f}\n")

print(f"\nDetailed metrics saved to: {os.path.join(opt.output_dir, 'metrics.txt')}")