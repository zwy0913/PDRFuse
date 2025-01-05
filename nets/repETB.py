import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tv2f

class SeqConv3x3(nn.Module):
    # Referenced to ECBSR: https://github.com/xindongzhang/ECBSR
    def __init__(self,
                 seq_type,
                 in_planes,
                 out_planes,
                 dim_multiplier=1):
        super().__init__()

        self.type = seq_type
        self.in_planes = in_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * dim_multiplier)
            conv0 = torch.nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            self.conv1 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1,
                                   groups=self.out_planes)
            self.conv1.weight = self.mask
            self.b1 = self.conv1.bias

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            self.conv1 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1,
                                   groups=self.out_planes)
            self.conv1.weight = self.mask
            self.b1 = self.conv1.bias

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            self.conv1 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1,
                                   groups=self.out_planes)
            self.conv1.weight = self.mask
            self.b1 = self.conv1.bias
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        pass

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + self.b1
        return RK, RB


class ETBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 dim_multiplier,
                 with_idt=True):
        super().__init__()

        self.dim_multiplier = dim_multiplier
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.act = nn.ReLU(inplace=True)

        if with_idt and (self.in_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, stride=1, padding=1)
        self.lkc1 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=9, stride=1, padding=4,
                                    groups=self.out_planes)
        self.lkc2 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=11, stride=1, padding=5,
                                    groups=self.out_planes)
        self.lkc3 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=13, stride=1, padding=6,
                                    groups=self.out_planes)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.in_planes, self.out_planes, self.dim_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.in_planes, self.out_planes)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.in_planes, self.out_planes)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.in_planes, self.out_planes)
        self.bn = nn.BatchNorm2d(out_planes)
        self.RK = None
        self.RB = None

    def forward(self, x):
        if hasattr(self, "conv3x3"):
            self.RK, self.RB = self.rep_params()
            self.__delattr__("conv3x3")
        if hasattr(self, "conv1x1_3x3"):
            self.__delattr__("conv1x1_3x3")
        if hasattr(self, "lkc1"):
            self.__delattr__("lkc1")
        if hasattr(self, "lkc2"):
            self.__delattr__("lkc2")
        if hasattr(self, "lkc3"):
            self.__delattr__("lkc3")
        if hasattr(self, "conv1x1_sbx"):
            self.__delattr__("conv1x1_sbx")
        if hasattr(self, "conv1x1_sby"):
            self.__delattr__("conv1x1_sby")
        if hasattr(self, "conv1x1_lpl"):
            self.__delattr__("conv1x1_lpl")
        y = F.conv2d(input=x, weight=self.RK, bias=self.RB, stride=1, padding=6, dilation=1)
        y = self.act(y)
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        # KK = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=DEVICE)
        # for i in range(self.out_planes):
        #     KK[i, i, 1, 1] = 1.0
        temp = K0 + K1 + K2 + K3 + K4
        temp = F.pad(temp, (5, 5, 5, 5), 'constant', 0)

        K5 = torch.zeros((self.out_planes, self.out_planes, 11, 11), device=K0.device)
        for i in range(self.out_planes):
            K5[i, i, :, :] = self.lkc2.weight[i, 0, :, :]
        K5 = F.pad(K5, (1, 1, 1, 1), 'constant', 0)
        B5 = self.lkc2.bias

        K6 = torch.zeros((self.out_planes, self.out_planes, 13, 13), device=K0.device)
        for i in range(self.out_planes):
            K6[i, i, :, :] = self.lkc3.weight[i, 0, :, :]
        B6 = self.lkc3.bias

        RK = temp + K6 + K5

        RB = B0 + B1 + B2 + B3 + B4 + B6 + B5

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 13, 13, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 6, 6] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


class ETBs(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 dim_multiplier,
                 depth):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_multiplier = dim_multiplier
        self.depth = depth

        ETBs = []
        for i in range(self.depth):
            if self.depth == 1:
                ETBs.append(
                    ETBlock(in_planes=self.dim_in,
                            out_planes=self.dim_out,
                            dim_multiplier=self.dim_multiplier)
                )
            else:
                if i == (self.depth - 1):
                    ETBs.append(
                        ETBlock(in_planes=self.dim_in,
                                out_planes=self.dim_out,
                                dim_multiplier=self.dim_multiplier)
                    )
                else:
                    ETBs.append(
                        ETBlock(in_planes=self.dim_in,
                                out_planes=self.dim_in,
                                dim_multiplier=self.dim_multiplier)
                    )
        self.ETBs = nn.Sequential(*ETBs)
        self.bn = nn.BatchNorm2d(self.dim_out)

    def forward(self, x):
        res = x
        for idx, etb in enumerate(self.ETBs):
            x = etb(x)
        return x