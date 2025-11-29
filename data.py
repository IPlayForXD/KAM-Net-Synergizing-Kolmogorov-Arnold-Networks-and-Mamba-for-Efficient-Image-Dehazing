import os
import random
import numpy as np
import natsort
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf
from PIL import Image, ImageEnhance, ImageFilter
#
# class MyValueDataSet(Dataset):
#     def __init__(self, inputPathTest_j, targetPathTest):
#         super(MyValueDataSet, self).__init__()
#         self.inputPath_j = inputPathTest_j
#         self.inputImages_j = natsort.natsorted(os.listdir(inputPathTest_j), alg=natsort.ns.PATH)
#         self.targetPath = targetPathTest
#         self.targetImages = natsort.natsorted(os.listdir(targetPathTest), alg=natsort.ns.PATH)
#
#         # 确保文件数量匹配
#         assert len(self.inputImages_j) == len(self.targetImages), \
#             f"验证集数量不匹配: 有雾图像{len(self.inputImages_j)}张, 无雾图像{len(self.targetImages)}张"
#
#
#     def __len__(self):
#         return len(self.inputImages_j)
#
#     def __getitem__(self, index):
#         # 使用相同的index确保配对
#         inputImagePath_j = os.path.join(self.inputPath_j, self.inputImages_j[index])
#         inputImage_j = Image.open(inputImagePath_j).convert('RGB')
#
#         targetImagePath = os.path.join(self.targetPath, self.targetImages[index])  # 使用相同的index
#         targetImage = Image.open(targetImagePath).convert('RGB')
#
#         # 转换成张量
#         inputImage_j = ttf.to_tensor(inputImage_j)
#         targetImage = ttf.to_tensor(targetImage)
#
#         return inputImage_j, targetImage
class MyValueDataSet(Dataset):
    def __init__(self, inputPathTest_j, targetPathTest, patch_size=None):
        super(MyValueDataSet, self).__init__()
        self.inputPath_j = inputPathTest_j
        self.inputImages_j = natsort.natsorted(os.listdir(inputPathTest_j), alg=natsort.ns.PATH)
        self.targetPath = targetPathTest
        self.targetImages = natsort.natsorted(os.listdir(targetPathTest), alg=natsort.ns.PATH)
        self.patch_size = patch_size

        # 确保文件数量匹配
        assert len(self.inputImages_j) == len(self.targetImages), \
            f"验证集数量不匹配: 有雾图像{len(self.inputImages_j)}张, 无雾图像{len(self.targetImages)}张"

    def _make_size_divisible(self, img, divisor=100):
        """调整图像尺寸使其能被divisor整除"""
        w, h = img.size

        # 计算新的尺寸（向下取整到最近的100的倍数）
        new_w = (w // divisor) * divisor
        new_h = (h // divisor) * divisor

        # 如果计算后尺寸为0，使用原始尺寸
        if new_w == 0:
            new_w = divisor
        if new_h == 0:
            new_h = divisor

        # 调整尺寸
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.BILINEAR)


        return img

    def __len__(self):
        return len(self.inputImages_j)

    def __getitem__(self, index):
        # 使用相同的index确保配对
        inputImagePath_j = os.path.join(self.inputPath_j, self.inputImages_j[index])
        inputImage_j = Image.open(inputImagePath_j).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])  # 使用相同的index
        targetImage = Image.open(targetImagePath).convert('RGB')

        # 确保两个图像尺寸相同
        if inputImage_j.size != targetImage.size:
            # 以有雾图像尺寸为准调整无雾图像
            targetImage = targetImage.resize(inputImage_j.size, Image.BILINEAR)

        # 调整尺寸使其能被100整除
        inputImage_j = self._make_size_divisible(inputImage_j, divisor=100)
        targetImage = self._make_size_divisible(targetImage, divisor=100)

        # 如果指定了patch_size，进行中心裁剪
        if self.patch_size is not None:
            w, h = inputImage_j.size
            start_x = (w - self.patch_size) // 2
            start_y = (h - self.patch_size) // 2
            inputImage_j = inputImage_j.crop((start_x, start_y, start_x + self.patch_size, start_y + self.patch_size))
            targetImage = targetImage.crop((start_x, start_y, start_x + self.patch_size, start_y + self.patch_size))

        # 转换成张量
        inputImage_j = ttf.to_tensor(inputImage_j)
        targetImage = ttf.to_tensor(targetImage)

        return inputImage_j, targetImage
class MyTrainDataSet(Dataset):
    def __init__(self, inputPathTrain_j, targetPathTrain, patch_size=256):
        super(MyTrainDataSet, self).__init__()
        self.inputPath_j = inputPathTrain_j
        self.inputImages_j = natsort.natsorted(os.listdir(inputPathTrain_j), alg=natsort.ns.PATH)
        self.targetPath = targetPathTrain
        self.targetImages = natsort.natsorted(os.listdir(targetPathTrain), alg=natsort.ns.PATH)
        self.ps = patch_size

        assert len(self.inputImages_j) == len(self.targetImages), \
            f"训练集数量不匹配: 有雾图像{len(self.inputImages_j)}张, 无雾图像{len(self.targetImages)}张"

    def __len__(self):
        return len(self.inputImages_j)

    # ============= 单边增强（只对输入有雾图像）=============
    def _apply_color_jitter(self, img):
        if random.random() < 0.7:
            if random.random() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-0.2, 0.2))
            if random.random() < 0.5:
                img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-0.2, 0.2))
            if random.random() < 0.5:
                img = ImageEnhance.Color(img).enhance(1 + random.uniform(-0.2, 0.2))
        return img

    def _apply_gaussian_blur(self, img):
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        return img

    def _apply_gamma(self, img):
        if random.random() < 0.4:
            gamma = random.uniform(0.7, 1.5)
            arr = np.array(img) / 255.0
            arr = np.power(arr, gamma)
            img = Image.fromarray(np.uint8(arr * 255))
        return img

    def _add_noise(self, img):
        if random.random() < 0.2:
            arr = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, 0.02, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 1)
            img = Image.fromarray(np.uint8(arr * 255))
        return img

    # ============= 对齐几何增强 =============
    def _random_flip(self, a, b):
        if random.random() < 0.5:
            a = a.transpose(Image.FLIP_LEFT_RIGHT)
            b = b.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            a = a.transpose(Image.FLIP_TOP_BOTTOM)
            b = b.transpose(Image.FLIP_TOP_BOTTOM)
        return a, b

    def _random_rotate(self, a, b):
        if random.random() < 0.3:
            angle = random.uniform(-30, 30)
            a = a.rotate(angle, resample=Image.BILINEAR)
            b = b.rotate(angle, resample=Image.BILINEAR)
        return a, b

    def _random_affine(self, a, b):
        """仿射增强（平移 + 轻微缩放）"""
        if random.random() < 0.3:
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            a = ttf.affine(a, angle=0, translate=translate, scale=1.0, shear=0)
            b = ttf.affine(b, angle=0, translate=translate, scale=1.0, shear=0)
        return a, b

    # ============= 遮挡增强（对齐遮挡）=============
    def _random_erasing(self, a, b):
        if random.random() < 0.2:
            x = random.randint(0, a.width - 1)
            y = random.randint(0, a.height - 1)
            w = random.randint(20, 80)
            h = random.randint(20, 80)
            erase_a = Image.new("RGB", (w, h), (random.randint(0, 255),)*3)
            erase_b = Image.new("RGB", (w, h), (0, 0, 0))

            a.paste(erase_a, (x, y))
            b.paste(erase_b, (x, y))
        return a, b
        # =============（新增）雾气合成增强 =============

    def _apply_fog(self, img):
        """
        基于物理模型的雾气合成:
        I = J * t + A * (1 - t)
        """
        if random.random() > 0.4:  # 40% 概率添加雾
            return img

        img_np = np.array(img).astype(np.float32) / 255.0
        H, W, _ = img_np.shape

        # ------ 随机大气光 A （亮度偏高，接近白色）------
        A = random.uniform(0.7, 1.0)

        # ------ 模拟深度：随机噪声 + 大范围模糊，使雾分布柔和 ------
        depth = np.random.uniform(0.1, 1.0, (H, W)).astype(np.float32)

        # 让雾有区域变化（大核模糊）
        depth = cv2.GaussianBlur(depth, (101, 101), sigmaX=50)

        # ------ 控制雾浓度 beta ------
        beta = random.uniform(0.6, 2.0)

        # 透射率 t(x) = exp(-beta * depth)
        t = np.exp(-beta * depth)[..., None]

        # ------ 应用物理模型 ------
        foggy = img_np * t + A * (1 - t)
        foggy = np.clip(foggy, 0, 1)

        return Image.fromarray((foggy * 255).astype(np.uint8))
    # ================== 主逻辑 ==================
    def __getitem__(self, index):
        ps = self.ps

        inp = Image.open(os.path.join(self.inputPath_j, self.inputImages_j[index])).convert('RGB')
        tar = Image.open(os.path.join(self.targetPath, self.targetImages[index])).convert('RGB')

        # ---- 对齐增强 ----
        inp, tar = self._random_flip(inp, tar)
        inp, tar = self._random_rotate(inp, tar)
        inp, tar = self._random_affine(inp, tar)
        inp, tar = self._random_erasing(inp, tar)

        # ---- 基础 resize（缩到比 patch 稍大）----
        inp = inp.resize((540, 540), Image.BILINEAR)
        tar = tar.resize((540, 540), Image.BILINEAR)

        # ---- 再随机裁剪 patch ----
        rr = random.randint(0, 540 - ps)
        cc = random.randint(0, 540 - ps)
        inp = inp.crop((cc, rr, cc + ps, rr + ps))
        tar = tar.crop((cc, rr, cc + ps, rr + ps))

        # ---- 单边增强（只对 input haze）----
        # （添加）雾气增强
        inp = self._apply_fog(inp)
        inp = self._apply_color_jitter(inp)
        inp = self._apply_gaussian_blur(inp)
        inp = self._apply_gamma(inp)
        inp = self._add_noise(inp)

        # ---- 转 tensor ----
        inp = ttf.to_tensor(inp)
        tar = ttf.to_tensor(tar)

        return inp, tar