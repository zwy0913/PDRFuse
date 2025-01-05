import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import time
from misc.general import DeviceChecker
import torchvision.transforms.v2.functional as tv2f
from misc.general import fusion_channel_sf
from misc.general import perturb_seg
from nets.FSSM import FSSM
from tqdm import tqdm
from nets.ETB import ETBlock
from nets.repETB import ETBs

DEVICE = DeviceChecker.DEVICE


# DEVICE = 'cpu'


class Extractor2(nn.Module):
    def __init__(self):
        super().__init__()
        self.inconv = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ETBs(dim_in=16,
                 dim_out=16,
                 dim_multiplier=4,
                 depth=1),
        )
        self.down1 = nn.Conv2d(16, 32, 3, 2, 1)
        self.layer2 = nn.Sequential(
            ETBs(dim_in=32,
                 dim_out=32,
                 dim_multiplier=4,
                 depth=1),
        )
        self.down2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.layer3 = nn.Sequential(
            ETBs(dim_in=64,
                 dim_out=64,
                 dim_multiplier=4,
                 depth=1),
        )

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(self.layer1(x1))
        x = self.down2(self.layer2(x2))
        x = self.layer3(x)
        return x, x1, x2


class Upsample2(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels, depth):
        super().__init__()
        self.etb1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            ETBs(dim_in=out_channels,
                 dim_out=out_channels,
                 dim_multiplier=4,
                 depth=1),
        )

    def forward(self, x, up):
        x = F.interpolate(input=x, size=[up.shape[2], up.shape[3]], mode='bilinear', align_corners=False)
        p = self.etb1(torch.cat([x, up], 1))
        return p


class ConvUpsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels, depth):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, up):
        x = F.interpolate(input=x, size=[up.shape[2], up.shape[3]], mode='bilinear', align_corners=False)

        p = self.conv(torch.cat([x, up], 1))

        return p


class PDRF(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference
        self.feats1 = Extractor2()

        self.fm_final_28 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.fm_final_56 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.fm_final_21 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

        self.fm_up_1 = Upsample2(64, 64 + 32, 32, 2)
        self.fm_up_2 = Upsample2(32, 32 + 16, 16, 1)
        self.fm_up_3 = ConvUpsample(16, 16 + 4, 16, 1)

        self.mixer = FSSM(dim_in=64,
                          dim_embed=64,
                          patch_size=3,
                          stride=1,
                          depth=1,
                          # =========================
                          ssm_d_state=16,
                          ssm_ratio=1.0,
                          ssm_dt_rank="auto",
                          ssm_act_layer=nn.SiLU,
                          ssm_conv=13,
                          ssm_conv_bias=True,
                          ssm_drop_rate=0.,
                          ssm_init="v0",
                          forward_type="v2",
                          # =========================
                          mlp_ratio=1.0,
                          mlp_act_layer=nn.GELU,
                          mlp_drop_rate=0.,
                          # =========================
                          drop_path_rate=0.,
                          )

        self.conn1 = nn.Conv2d(32 * 2, 32, 1, 1, 0)
        self.conn2 = nn.Conv2d(16 * 2, 16, 1, 1, 0)
        self.conn3 = nn.Conv2d(4 * 2, 4, 1, 1, 0)

    def forward(self, A, B, sf, ref):

        sfA = sf
        sfB = 1 - sf
        """
        Focus property detection
        """
        fm = {}
        pA0 = torch.cat((A, sfA), 1)
        pB0 = torch.cat((B, sfB), 1)

        pA, f_1_A, f_2_A = self.feats1(pA0)
        pB, f_1_B, f_2_B = self.feats1(pB0)

        # pF = self.mixer(torch.cat([pA, pB], dim=1))
        pF = self.mixer(pA, pB)

        if not self.inference:
            fm_out_28 = self.fm_final_28(pF)
            # fm_out_28 = F.interpolate(fm_final_28, scale_factor=8, mode='bilinear', align_corners=False)

        sc1 = self.conn1(torch.cat([f_2_A, f_2_B], dim=1))
        pF = self.fm_up_1(pF, sc1)
        if not self.inference:
            fm_out_56 = self.fm_final_56(pF)
            # fm_out_56 = F.interpolate(fm_final_56, scale_factor=4, mode='bilinear', align_corners=False)

        sc2 = self.conn2(torch.cat([f_1_A, f_1_B], dim=1))
        pF = self.fm_up_2(pF, sc2)
        sc3 = self.conn3(torch.cat([pA0, pB0], dim=1))
        pF = self.fm_up_3(pF, sc3)
        pF = self.fm_final_21(pF)

        pred_224 = torch.sigmoid(pF)

        fm['pred_224'] = pred_224
        fm['out_224'] = pF
        if not self.inference:
            fm['pred_28'] = torch.sigmoid(fm_out_28)
            fm['pred_56'] = torch.sigmoid(fm_out_56)
            fm['out_28'] = fm_out_28
            fm['out_56'] = fm_out_56

        """
        Data argumentation
        """
        if not self.inference:
            fm_to_refine = perturb_seg(fm['pred_224'], intensity=64)
        else:
            fm_to_refine = fm['pred_224']

        """
        Focus map refinement
        """

        images = {}
        p0 = torch.cat((ref, fm_to_refine), 1)
        ip0 = torch.cat((ref, 1 - fm_to_refine), 1)

        p, f_1_p, f_2_p = self.feats1(p0)
        ip, f_1_ip, f_2_ip = self.feats1(ip0)

        # pF = self.mixer(torch.cat([p, ip], dim=1))
        pF = self.mixer(p, ip)

        if not self.inference:
            out_28 = self.fm_final_28(pF)
            # out_28 = F.interpolate(final_28, scale_factor=8, mode='bilinear', align_corners=False)

        sc1 = self.conn1(torch.cat([f_2_p, f_2_ip], dim=1))
        pF = self.fm_up_1(pF, sc1)
        if not self.inference:
            out_56 = self.fm_final_56(pF)
            # out_56 = F.interpolate(final_56, scale_factor=4, mode='bilinear', align_corners=False)

        sc2 = self.conn2(torch.cat([f_1_p, f_1_ip], dim=1))
        pF = self.fm_up_2(pF, sc2)
        sc3 = self.conn3(torch.cat([p0, ip0], dim=1))
        pF = self.fm_up_3(pF, sc3)
        pF = self.fm_final_21(pF)

        pred_224 = torch.sigmoid(pF)

        images['pred_224'] = pred_224
        images['out_224'] = pF
        if not self.inference:
            images['pred_28'] = torch.sigmoid(out_28)
            images['pred_56'] = torch.sigmoid(out_56)
            images['out_28'] = out_28
            images['out_56'] = out_56

        return fm, images


if __name__ == '__main__':
    A = torch.ones((1, 3, 224, 224)).to(DEVICE)
    B = torch.ones((1, 1, 224, 224)).to(DEVICE)
    # LA = tv2f.resize(A, [A.shape[2] // 4, A.shape[3] // 4], antialias=False)
    # LB = tv2f.resize(B, [B.shape[2] // 4, B.shape[3] // 4], antialias=False)
    model = PDRF().to(DEVICE)
    # model.load_state_dict(torch.load('../debug_model_S.pth'))
    # model.eval()
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
    # allocated_memory = torch.cuda.max_memory_allocated()
    # print(f"Max allocated GPU memory: {allocated_memory / 1024 ** 3} GB")

    # from thop import profile, clever_format
    #
    # flops, params = profile(model, (test_tensor_A.unsqueeze(0)), verbose=False)
    # flops, params = clever_format([flops, params], "%.5f")
    #
    # print('flops: {}, params: {}\n'.format(flops, params))

    # allocated_memory = torch.cuda.max_memory_allocated()

    tic0 = time.time()
    for i in tqdm(range(1000)):
        tex = model(A, A, B, A)
        # str = model_str(test_tensor_A)
        # print(tex[0])
        # print(str[0].shape)
    print(time.time() - tic0)
    allocated_memory = torch.cuda.max_memory_allocated()
    print(f"Max allocated GPU memory: {allocated_memory / 1024 ** 3} GB")
