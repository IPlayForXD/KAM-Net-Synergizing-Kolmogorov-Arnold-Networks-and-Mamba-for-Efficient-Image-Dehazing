import math
import random

import  natsort
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from kat import KATVisionTransformer
# kat
class KA_FFN(nn.Module):

    def __init__(self, in_channels: int, depth: int = 2, kat_ctor_kwargs: dict = None, use_proj: bool = False):
        super().__init__()
        kat_ctor_kwargs = kat_ctor_kwargs or {}
        # 强制一些关键参数以减少兼容问题
        kat_defaults = dict(
            img_size=0,            # 会被 dynamic_img_size 接管
            patch_size=1,          # 每像素为一个 token（瓶颈分辨率下可接受）
            in_chans=in_channels,
            embed_dim=in_channels, # 保持 embed_dim == channels，避免投影
            depth=depth,
            class_token=False,
            pos_embed='none',
            dynamic_img_size=True,
            global_pool='',        # 不要 pool
        )
        kat_defaults.update(kat_ctor_kwargs)
        # 构造 KAT 模型实例（你提供的类名必须可见）
        self.kat = KATVisionTransformer(**kat_defaults)

        # 是否使用 1x1 投影以保证输入/输出通道数匹配（如果 KAT 输出维度不等于 in_channels）
        self.use_proj = use_proj
        if use_proj:
            self.proj_in = nn.Conv2d(in_channels, kat_defaults['embed_dim'], 1)
            self.proj_out = nn.Conv2d(kat_defaults['embed_dim'], in_channels, 1)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()

        # 对输入做简单归一化/稳定
        self.pre_norm = nn.BatchNorm2d(in_channels)
        # 若需要还可以在输出后加 BN
        self.post_norm = nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        identity = x

        x = self.pre_norm(x)
        x = self.proj_in(x)  # 可能是 identity
        # 保存原始的 H, W 用于后续处理
        self.kat.patch_grid = (h, w)  # 动态设置分辨率
        # KATVisionTransformer.forward_features 接受 image 张量；它会做 patch_embed + blocks
        # # 这里传入原始 NHWC or NCHW 需要与 KAT 的实现兼容 — 你的实现是以 NCHW 为主
        # kat_feats = self.kat.forward_features(x)  # 返回 (B, N, E) where N = H*W (dynamic)
        # 修改 KAT 的 forward_features 来传递 H, W
        kat_feats = self.kat.forward_features(x, h, w)  # 传递 H, W
        # 把 (B, N, E) -> (B, E, H, W)
        # 注意： KAT 的 forward_features 在 dynamic 模式下会根据输入 H,W 计算 N 顺序
        out = kat_feats.permute(0, 2, 1).contiguous().view(b, c, h, w)

        out = self.proj_out(out)  # 可能是 identity
        out = self.post_norm(out)
        # 残差连接（保留 conv 局部能力）
        return identity + out



#baseline
class Layer(nn.Module):
    def __init__(self, in_features):
        super(Layer, self).__init__()
        self.norm = nn.BatchNorm2d(in_features)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_features, in_features,3,1,1,padding_mode="reflect"),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.Conv2d(in_features, in_features, 3,1,1,padding_mode="reflect"),
            nn.ReLU(True),
            nn.BatchNorm2d(in_features),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_features, in_features,3,1,1,padding_mode="reflect"),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.Conv2d(in_features, in_features, 3,1,1,padding_mode="reflect"),
            nn.ReLU(True),
            nn.BatchNorm2d(in_features),
        )

        self.conv = nn.Conv2d(in_features, in_features,1,groups=in_features)
    def forward(self, x):
        x = self.norm(x)
        mid = self.conv_block1(x)*self.conv_block2(x)
        out = self.conv(mid)
        return x+out
class Down(nn.Module):
    def __init__(self, in_features,out_features):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_features, out_features, 2, stride=2),
        )
    def forward(self, x):
        out = self.down(x)
        return out
class Up(nn.Module):
    def __init__(self, in_features,out_features):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_features, out_features * 4, 1),
            nn.PixelShuffle(2),
        )
    def forward(self, x):
        out = self.up(x)
        return out
class AF_Module(nn.Module):
    def __init__(self, in_features):
        super(AF_Module, self).__init__()

        self.shallow = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
        )
        self.MLP_A = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features,in_features*2,1,bias=False),

            nn.Conv2d(in_features*2, in_features , 1,bias=False),
        )
        self.MLP_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features,2*in_features,1,bias=False),

            nn.Conv2d(2*in_features,in_features, 1,bias=False),
        )
    def forward(self, x):
        mid = self.shallow(x)

        t = self.MLP_T(mid)
        b,c,h,w = t.shape
        a = self.MLP_A(mid)
        one = torch.ones((b,c,h,w)).cuda()
        t = torch.clamp(t, 0.1, 10.0)  # 限制t的范围
        a = torch.clamp(a, -1.0, 1.0)  # 限制a的范围
        return (x + a * (t - one)) * t
class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3):
        super(Generator, self).__init__()
        in_features = 96
        self.inconv = nn.Conv2d(3, in_features,1)

        self.down1 = Down(in_features,in_features*2)
        self.skip1 = nn.Conv2d(in_features*2,in_features*2,1)
        # self.layer1_1 = Layer(in_features*2)
        # self.layer1 = Layer(in_features * 2)


        self.down2 = Down(in_features*2,in_features*4)
        self.skip2 = nn.Conv2d(in_features*4,in_features*4,1)
        # self.layer2 = Layer(in_features*4)
        # self.layer2_2 = Layer(in_features * 4)
        kat_kwargs = dict()
        self.layer_mid = KA_FFN(in_features * 4, depth=2, kat_ctor_kwargs=kat_kwargs)
        self.layer_mid_2 = KA_FFN(in_features * 4, depth=2, kat_ctor_kwargs=kat_kwargs)
        # self.layer_mid   = Layer(in_features * 4)
        # self.layer_mid_2 = Layer(in_features * 4)
        self.layer_mid_3 = Layer(in_features * 4)
        self.layer_mid_4 = Layer(in_features * 4)
        self.layer_mid_5 = Layer(in_features * 4)
        self.layer_mid_6 = Layer(in_features * 4)
        self.layer_mid_7 = Layer(in_features * 4)
        self.layer_mid_8 = Layer(in_features * 4)



        self.AF_Module1 = AF_Module(in_features*4)
        self.up1 = Up(in_features * 4, in_features * 2)


        self.AF_Module2 = AF_Module(in_features*2)
        self.up2 = Up(in_features * 2, in_features)

        self.outconv = nn.Conv2d(in_features,output_nc,1)
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(True)
    def forward(self, x):
        skpi = []
        out = self.inconv(x)

        out = self.down1(out)
        skpi.append((out))


        out = self.down2(out)
        skpi.append((out))


        out = self.layer_mid(out)
        out = self.layer_mid_2(out)
        out = self.layer_mid_3(out)
        out = self.layer_mid_4(out)
        # out = self.layer_mid_5(out)
        # out = self.layer_mid_6(out)
        # out = self.layer_mid_7(out)
        # out = self.layer_mid_8(out)



        out = self.AF_Module1(out)+skpi[-1]
        out = self.up1(out)


        out = self.AF_Module2(out)+skpi[-2]
        out = self.up2(out)

        out = self.outconv(out)
        # 使用Tanh获得更广范围
        out = torch.tanh(out)  # [-1, 1]
        out = (out + 1) / 2  # [0, 1]

        # 保持残差连接
        out = 0.3 * x + 0.7 * out
        out = torch.clamp(out, 0, 1)
        # out = torch.tanh(out)
        #
        # out = x + 0.1 * out
        # out = torch.clamp(out, 0, 1)
        return out


if __name__ == "__main__":
    from thop import profile, clever_format
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    model = Generator().to('cuda')
    x = torch.randn(1, 3, 256, 256).to('cuda')
    # 使用 thop 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(x,))
    y = model(x).to('cuda')
    # 格式化输出
    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

    print(f"Param: {params_formatted} ({params:,})")
    print(f"FLOPs: {flops_formatted} ({flops:,})")
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")