import sys
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
import os

from loss import SSIM
from data import *
from model_resnet import Generator
from model_MambaMoe import MambaIRUNet
from model_MambaIR import MambaIRv2
from model_DehazeFormer import dehazeformer_b
from model_D4 import HazeRemovalNet
inputPathTrain_j = "datasets/LMHaze/train/hazy"  # 输入的训练文件j
targetPathTrain = "datasets/LMHaze/train/GT"  # 输入训练集的无雾图像
inputPathTest = "datasets/LMHaze/test/hazy"
targePathtest = "datasets/LMHaze/test/GT"
# inputPathTrain_j = "datasets/O_HAZY/train/hazy"  # 输入的训练文件j
# targetPathTrain = "datasets/O_HAZY/train/GT"  # 输入训练集的无雾图像
# inputPathTest = "datasets/O_HAZY/test/hazy"
# targePathtest = "datasets/O_HAZY/test/GT"
# inputPathTrain_j = "datasets/NH_HAZE/train/hazy"  # 输入的训练文件j
# targetPathTrain = "datasets/NH_HAZE/train/GT"  # 输入训练集的无雾图像
# inputPathTest = "datasets/NH_HAZE/test/hazy"
# targePathtest = "datasets/NH_HAZE/test/GT"


parser = argparse.ArgumentParser()
parser.add_argument("--EPOCH", type=int, default=200, help="starting epoch")
parser.add_argument("--BATCH_SIZE", type=int, default=32, help="size of the batches")
parser.add_argument("--PATCH_SIZE", type=int, default=256, help="size of the patch")
parser.add_argument("--LEARNING_RATE", type=float, default=2e-5, help="initial learning rate")
parser.add_argument("--RESUME", type=str, default='', help="path to checkpoint to resume from")
parser.add_argument("--RESUME_EPOCH", type=int, default=0, help="epoch to resume from")
opt = parser.parse_args()

criterion1 = SSIM().cuda()
criterion2 = nn.L1Loss().cuda()

# 实例化数据加载器及模型
swinIR2 = Generator()
# swinIR2 = MambaIRUNet()
# swinIR2 = MambaIRv2()
# swinIR2 = dehazeformer_b()
# swinIR2 = HazeRemovalNet(64)
swinIR2.cuda()
device_ids = [i for i in range(torch.cuda.device_count())]
if len(device_ids) > 1:
    swinIR2 = nn.DataParallel(swinIR2, device_ids=device_ids)

optimizer = torch.optim.AdamW([
    {'params': swinIR2.parameters(), "lr": opt.LEARNING_RATE, "betas": [0.9, 0.99], "weight_decay": 1e-4},
])
cosinese = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=2e-5)

# 恢复训练功能
start_epoch = 0
best_psnr = 0

if opt.RESUME:
    if os.path.isfile(opt.RESUME):
        print(f"=> loading checkpoint '{opt.RESUME}'")
        checkpoint = torch.load(opt.RESUME)

        # 加载模型权重
        if isinstance(swinIR2, nn.DataParallel):
            swinIR2.module.load_state_dict(checkpoint['state_dict'])
        else:
            swinIR2.load_state_dict(checkpoint['state_dict'])

        # 加载优化器状态
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded optimizer state")

        # 加载学习率调度器状态
        if 'scheduler' in checkpoint:
            cosinese.load_state_dict(checkpoint['scheduler'])
            print("=> loaded scheduler state")

        # 加载最佳PSNR和起始epoch
        if 'best_psnr' in checkpoint:
            best_psnr = checkpoint['best_psnr']
            print(f"=> loaded best PSNR: {best_psnr:.4f}")

        start_epoch = checkpoint.get('epoch', opt.RESUME_EPOCH) + 1
        print(f"=> loaded checkpoint '{opt.RESUME}' (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"=> no checkpoint found at '{opt.RESUME}'")

datasetTrain = MyTrainDataSet(inputPathTrain_j, targetPathTrain, patch_size=opt.PATCH_SIZE)
trainLoader = DataLoader(dataset=datasetTrain, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=20, drop_last=True,
                         pin_memory=True)
datasetTest = MyValueDataSet(inputPathTest, targePathtest)
valueLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=True, num_workers=20,
                         pin_memory=True)

output_images = 'results/'

# 创建输出目录
os.makedirs(output_images, exist_ok=True)
os.makedirs('.checkpoints', exist_ok=True)

print('-------------------------------------------------------------------------------------------------------')
print(f'Starting from epoch: {start_epoch}')
print(f'Best PSNR so far: {best_psnr:.4f}')

for epoch in range(start_epoch, opt.EPOCH):
    swinIR2.train(True)
    # 进度条
    iters = tqdm(trainLoader, file=sys.stdout)
    epochLoss1 = 0
    epochLoss2 = 0
    tureloss = 0

    for index, (j, ture_j) in enumerate(iters, 0):
        swinIR2.zero_grad()
        optimizer.zero_grad()
        # 包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息
        j_, ture_j_ = Variable(j).cuda(), Variable(ture_j).cuda()
        j2 = swinIR2(j_)
        loss1 = criterion1(j2, ture_j_)
        loss2 = criterion2(j2, ture_j_)

        loss = 0.7 * loss2 + 0.3 *(1 - loss1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(swinIR2.parameters(), max_norm=1.0)
        optimizer.step()
        # 进度条
        epochLoss1 += loss1.item()
        epochLoss2 += loss2.item()

        iters.set_description('Training !!!  Epoch %d / %d,  SSIM loss %.6f, L1 loss %.6f' % (
            epoch + 1, opt.EPOCH, loss1.item(), loss2.item()))

    cosinese.step()

    # 打印训练损失
    avg_loss1 = epochLoss1 / len(trainLoader)
    avg_loss2 = epochLoss2 / len(trainLoader)
    print(f'Epoch {epoch + 1}/{opt.EPOCH}, Average SSIM loss: {avg_loss1:.6f}, Average L1 loss: {avg_loss2:.6f}')

    # 验证阶段
    psnr_val_rgb = []
    psnr_val_rgb2 = []

    # special_samples = [64, 756, 816]
    # special_saved = {sample_idx: False for sample_idx in special_samples}
    # 在验证循环中添加详细的调试
    if epoch >= 0:
        swinIR2.eval()
        val_iters = tqdm(valueLoader, desc=f'Validating Epoch {epoch + 1}', file=sys.stdout)

        save_sample = True
        abnormal_count = 0
        total_samples = len(valueLoader)
        print_every = max(1, total_samples // 3)  # 只输出约3次

        for index, (test_j, test_ture_j) in enumerate(val_iters, 0):
            test_j_, test_ture_j_ = test_j.cuda(), test_ture_j.cuda()
            with torch.no_grad():
                test_fake_j = swinIR2(test_j_)

            # # 只输出最后5个样本或每隔一定间隔输出
            # if index % print_every == 0:
            #     print(f"\nSample {index}:")
            #     print(f"Input range: [{test_j_.min().item():.3f}, {test_j_.max().item():.3f}]")
            #     print(f"Output range: [{test_fake_j.min().item():.3f}, {test_fake_j.max().item():.3f}]")
            #     print(f"Target range: [{test_ture_j_.min().item():.3f}, {test_ture_j_.max().item():.3f}]")
            # 检查是否有异常值
            if (test_fake_j.min() < -10 or test_fake_j.max() > 10 or
                    torch.isnan(test_fake_j).any() or torch.isinf(test_fake_j).any()):
                print(f"❌ ABNORMAL OUTPUT DETECTED at sample {index}!")
                abnormal_count += 1
                continue

            # 计算损失前先裁剪到合理范围
            test_fake_j = torch.clamp(test_fake_j, 0, 1)

            # 计算MSE和L1
            mse_loss = F.mse_loss(test_fake_j, test_ture_j_)
            l1_loss = F.l1_loss(test_fake_j, test_ture_j_)

            # # 只在输出range时显示MSE/L1
            # if index % print_every == 0:
            #     print(f"MSE: {mse_loss.item():.6f}, L1: {l1_loss.item():.6f}")

            # 安全的PSNR计算
            if mse_loss == 0:
                ps = torch.tensor(100.0).cuda()
            else:
                ps = 10 * torch.log10(1.0 / mse_loss)

            ssim = criterion1(test_fake_j, test_ture_j_)

            # 检查计算结果
            if torch.isnan(ps) or torch.isinf(ps) or ps < 0:
                print(f"❌ Invalid PSNR: {ps.item()}")
                continue
            if torch.isnan(ssim) or ssim < 0 or ssim > 1:
                print(f"❌ Invalid SSIM: {ssim.item()}")
                continue
            # if 65 <= index <= 186:
            #     special_dir = os.path.join(output_images, 'special_samples')
            #     os.makedirs(special_dir, exist_ok=True)
            #     print(f"Sample {index}: PSNR = {ps.item():.4f}, SSIM = {ssim.item():.4f}")
            #     comparison = torch.cat([test_j_, test_fake_j, test_ture_j_], dim=0)
            #     save_image(comparison,
            #                 os.path.join(special_dir, f'epoch_{epoch + 1}_sample_{index}_comparison.png'),
            #                 nrow=3,
            #                 normalize=True)

            # 只保存第一组正常图像
            if save_sample and ps > 0 and 0 <= ssim <= 1:
                comparison = torch.cat([test_j_, test_fake_j, test_ture_j_], dim=0)
                save_image(comparison,
                           os.path.join(output_images, f'epoch_{epoch + 1}_comparison.png'),
                           nrow=3,
                           normalize=True)
                save_sample = False

            psnr_val_rgb2.append(ssim.item())
            psnr_val_rgb.append(ps.item())

            val_iters.set_postfix(PSNR=f'{ps.item():.2f}', SSIM=f'{ssim.item():.4f}')

        print(f"Abnormal samples: {abnormal_count}/{len(valueLoader)}")

        # 在验证循环结束后计算平均值
        if psnr_val_rgb:
            avg_psnr = sum(psnr_val_rgb) / len(psnr_val_rgb)
            avg_ssim = sum(psnr_val_rgb2) / len(psnr_val_rgb2)

            # 保存最佳模型
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr

                # 保存完整的检查点（包含训练状态）
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': swinIR2.module.state_dict() if isinstance(swinIR2,
                                                                            nn.DataParallel) else swinIR2.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': cosinese.state_dict(),
                    'best_psnr': best_psnr,
                }
                torch.save(checkpoint, '.checkpoints/instancnormoutdoor.pth')
                print(f'New best model saved with PSNR: {avg_psnr:.4f}')
            else:
                # 定期保存检查点（每5个epoch）
                if (epoch + 1) % 5 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'state_dict': swinIR2.module.state_dict() if isinstance(swinIR2,
                                                                                nn.DataParallel) else swinIR2.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': cosinese.state_dict(),
                        'best_psnr': best_psnr,
                    }
                    torch.save(checkpoint, f'.checkpoints/checkpoint_epoch_{epoch + 1}.pth')
                    print(f'Checkpoint saved at epoch {epoch + 1}')

            print(f'Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
        print(f'Comparison image saved: {os.path.join(output_images, f"epoch_{epoch + 1}_comparison.png")}')