import cv2
import math
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def calculate_metrics(original, enhanced):
    """计算PSNR和SSIM指标"""
    # 确保图像在0-255范围内
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

    # 计算PSNR
    psnr_value = psnr(original, enhanced)

    # 计算SSIM（多通道）
    ssim_value = ssim(original, enhanced, channel_axis=2)

    return psnr_value, ssim_value


def dcp_dehaze(hazy_image_path, output_dir="./output"):
    """DCP去雾主函数"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取有雾图像
    src = cv2.imread(hazy_image_path)
    if src is None:
        print(f"错误：无法读取图像 {hazy_image_path}")
        return None

    # 转换为浮点数并归一化
    I = src.astype('float64') / 255

    # DCP去雾流程
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)

    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(hazy_image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_dehazed.png")

    # 保存去雾结果
    cv2.imwrite(output_path, J * 255)

    print(f"去雾完成！结果保存至: {output_path}")

    return J


def process_with_ground_truth(hazy_path, gt_path, output_dir="./output"):
    """处理有雾图像并与真实无雾图像比较"""

    # 执行去雾
    dehazed_result = dcp_dehaze(hazy_path, output_dir)

    if dehazed_result is None:
        return

    # 读取真实无雾图像
    gt_image = cv2.imread(gt_path)
    if gt_image is None:
        print(f"错误：无法读取真实无雾图像 {gt_path}")
        return

    # 转换为浮点数并归一化
    gt_float = gt_image.astype('float64') / 255

    # 确保尺寸一致
    if dehazed_result.shape != gt_float.shape:
        print("调整真实图像尺寸以匹配去雾结果...")
        gt_float = cv2.resize(gt_float, (dehazed_result.shape[1], dehazed_result.shape[0]))

    # 计算指标
    psnr_value, ssim_value = calculate_metrics(gt_float, dehazed_result)

    print("\n=== 去雾效果评估 ===")
    print(f"PSNR: {psnr_value:.4f} dB")
    print(f"SSIM: {ssim_value:.4f}")

    # 保存评估结果到文件
    base_name = os.path.splitext(os.path.basename(hazy_path))[0]
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.txt")

    with open(metrics_path, 'w') as f:
        f.write(f"图像: {base_name}\n")
        f.write(f"PSNR: {psnr_value:.4f} dB\n")
        f.write(f"SSIM: {ssim_value:.4f}\n")

    print(f"指标结果保存至: {metrics_path}")

    return dehazed_result, psnr_value, ssim_value


def batch_process(hazy_dir, gt_dir, output_dir="./output"):
    """批量处理文件夹中的图像"""

    if not os.path.exists(hazy_dir):
        print(f"错误：有雾图像目录 {hazy_dir} 不存在")
        return

    # 获取有雾图像文件
    hazy_images = [f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not hazy_images:
        print("在有雾图像目录中未找到图像文件")
        return

    print(f"找到 {len(hazy_images)} 个有雾图像文件")

    all_psnr = []
    all_ssim = []

    for hazy_file in hazy_images:
        print(f"\n处理图像: {hazy_file}")

        hazy_path = os.path.join(hazy_dir, hazy_file)

        # 构建对应的真实无雾图像路径
        if gt_dir:
            gt_file = hazy_file  # 假设文件名相同
            gt_path = os.path.join(gt_dir, gt_file)

            if os.path.exists(gt_path):
                result = process_with_ground_truth(hazy_path, gt_path, output_dir)
                if result:
                    _, psnr_val, ssim_val = result
                    all_psnr.append(psnr_val)
                    all_ssim.append(ssim_val)
            else:
                print(f"警告：未找到对应的真实无雾图像 {gt_path}，仅执行去雾")
                dcp_dehaze(hazy_path, output_dir)
        else:
            # 如果没有真实无雾图像，只执行去雾
            dcp_dehaze(hazy_path, output_dir)

    # 如果有多个图像且都有真实图像，输出平均指标
    if all_psnr and len(all_psnr) > 1:
        print(f"\n=== 平均指标 ===")
        print(f"平均 PSNR: {np.mean(all_psnr):.4f} dB")
        print(f"平均 SSIM: {np.mean(all_ssim):.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DCP去雾算法')
    parser.add_argument('--hazy', type=str, help='有雾图像路径或目录')
    parser.add_argument('--gt', type=str, default='', help='真实无雾图像目录（可选）')
    parser.add_argument('--output', type=str, default='./output', help='输出目录')

    args = parser.parse_args()

    if not args.hazy:
        print("请提供有雾图像路径")
        print("使用方法:")
        print("单个图像: python script.py --hazy path/to/hazy.jpg --gt path/to/gt.jpg")
        print("批量处理: python script.py --hazy path/to/hazy_dir --gt path/to/gt_dir")
        exit(1)

    # 判断是单个文件还是目录
    if os.path.isfile(args.hazy):
        if args.gt and os.path.isfile(args.gt):
            process_with_ground_truth(args.hazy, args.gt, args.output)
        else:
            dcp_dehaze(args.hazy, args.output)
    elif os.path.isdir(args.hazy):
        batch_process(args.hazy, args.gt, args.output)
    else:
        print(f"错误：路径 {args.hazy} 不存在")