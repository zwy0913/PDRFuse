# Referenced to: https://github.com/walking-shadow/Official_Remote_Sensing_Mamba
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from nets.ETB import ETBlock
from nets.ETB import ETBs

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
try:
    import selective_scan_cuda_core
except Exception as e:
    ...
try:
    import selective_scan_cuda
except Exception as e:
    ...


class SelectiveScanMamba(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        # all in float
        # if u.stride(-1) != 1:
        #     u = u.contiguous()
        # if delta.stride(-1) != 1:
        #     delta = delta.contiguous()
        # if D is not None and D.stride(-1) != 1:
        #     D = D.contiguous()
        # if B.stride(-1) != 1:
        #     B = B.contiguous()
        # if C.stride(-1) != 1:
        #     C = C.contiguous()
        # if B.dim() == 3:
        #     B = B.unsqueeze(dim=1)
        #     ctx.squeeze_B = True
        # if C.dim() == 3:
        #     C = C.unsqueeze(dim=1)
        #     ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        # dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        # dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


# =============
def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
        dt_low_rank=True,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    # ZSJ 这里把矩阵拆分成不同方向的序列，并进行扫描
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    # ZSJ 这里把处理之后的序列融合起来，并还原回原来的矩阵形式
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # ===============================
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )

        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        # ZSJ k_group 指的是扫描的方向
        k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1
        # k_group = 8 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj_x = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_y = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if d_conv > 1:
            self.conv2d_x = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_y = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

            # self.conv2dy_x = ETBlock(in_planes=d_inner,
            #                       out_planes=d_inner,
            #                       dim_multiplier=2)

            # self.conv2d = ETBlock(in_planes=d_inner,
            #                       out_planes=d_inner,
            #                       dim_multiplier=2)

            # self.conv2d = ETBs(dim_in=d_inner,
            #                    dim_out=d_inner,
            #                    dim_multiplier=2,
            #                    depth=1),

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj_x = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.out_proj_y = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        # ZSJ V2版本使用的mamba，要改扫描方向在这里改
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
        )
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs):

        with_dconv = (self.d_conv > 1)
        x = self.in_proj_x(x)
        if not self.disable_z:
            x, xz = x.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                xz = self.act(xz)
        if with_dconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d_x(x)  # (b, d, h, w)
            # x = x.permute(0, 2, 3, 1).contiguous()
        x = self.act(x)

        y = self.in_proj_y(y)
        if not self.disable_z:
            y, yz = y.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                yz = self.act(yz)
        if with_dconv:
            y = y.permute(0, 3, 1, 2).contiguous()
            y = self.conv2d_y(y)  # (b, d, h, w)
            # y = y.permute(0, 2, 3, 1).contiguous()
        y = self.act(y)



        # B, C, H, W = x.shape
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        # h_tensor = torch.empty(B, C, H, 2*W, dtype=torch.float32).to(DEVICE)
        h_tensor = torch.cat([x, x], dim=3)
        # Fill in odd columns with A and even columns with B
        h_tensor[:, :, :, ::2] = x  # Odd columns
        h_tensor[:, :, :, 1::2] = y  # Even columns

        ss_out = self.forward_core(h_tensor, channel_first=with_dconv)
        x = ss_out[:, :, ::2, :]
        y = ss_out[:, :, 1::2, :]


        if not self.disable_z:
            x = x * xz
            y = y * yz

        # out = x + y
        # out = self.dropout(self.out_proj_x(out))

        x = self.dropout(self.out_proj_x(x))
        y = self.dropout(self.out_proj_y(y))
        out = x + y

        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # y = self.fc1(y)
        # y = self.act(y)
        # y = self.drop(y)
        # y = self.fc2(y)
        # y = self.drop(y)
        return x


class VSSMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            depth: int = 1,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                norm_layer(hidden_dim),
                SS2D(
                    d_model=hidden_dim,
                    d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    dt_rank=ssm_dt_rank,
                    act_layer=ssm_act_layer,
                    # ==========================
                    d_conv=ssm_conv,
                    conv_bias=ssm_conv_bias,
                    # ==========================
                    dropout=ssm_drop_rate,
                    # bias=False,
                    # ==========================
                    # dt_min=0.001,
                    # dt_max=0.1,
                    # dt_init="random",
                    # dt_scale="random",
                    # dt_init_floor=1e-4,
                    initialize=ssm_init,
                    # ==========================
                    forward_type=forward_type,
                ),
                norm_layer(hidden_dim),
                Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                    drop=mlp_drop_rate, channels_first=False),
            ]))


        # self.ssm_branch = ssm_ratio > 0
        # self.mlp_branch = mlp_ratio > 0
        # self.use_checkpoint = use_checkpoint
        # self.post_norm = post_norm
        #
        # if self.ssm_branch:
        #     self.norm = norm_layer(hidden_dim)
        #     self.op = SS2D(
        #         d_model=hidden_dim,
        #         d_state=ssm_d_state,
        #         ssm_ratio=ssm_ratio,
        #         dt_rank=ssm_dt_rank,
        #         act_layer=ssm_act_layer,
        #         # ==========================
        #         d_conv=ssm_conv,
        #         conv_bias=ssm_conv_bias,
        #         # ==========================
        #         dropout=ssm_drop_rate,
        #         # bias=False,
        #         # ==========================
        #         # dt_min=0.001,
        #         # dt_max=0.1,
        #         # dt_init="random",
        #         # dt_scale="random",
        #         # dt_init_floor=1e-4,
        #         initialize=ssm_init,
        #         # ==========================
        #         forward_type=forward_type,
        #     )
        #
        # self.drop_path = DropPath(drop_path)
        #
        # if self.mlp_branch:
        #     self.norm2 = norm_layer(hidden_dim)
        #     mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        #     self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
        #                    drop=mlp_drop_rate, channels_first=False)


    def forward(self, x, y):
        for norm, op, norm2, mlp in self.layers:
            # tempx, tempy = op(norm(x), norm(y))
            # x = tempx + x
            # y = tempy + y
            # tempx, tempy = mlp(norm2(x), norm2(y))
            # x = tempx + x
            # y = tempy + y
            out = op(norm(x), norm(y))
            out = out + mlp(norm2(out))
        return out


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvEmbed(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_embed,
                 patch_size,
                 stride):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(dim_in, dim_embed, kernel_size=patch_size, stride=stride, padding=(patch_size - 1) // 2),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(dim_embed)
        )

    def forward(self, x, y):
        x = self.embed(x)
        y = self.embed(y)
        return x, y


class FSSM(nn.Module):
    def __init__(self,
                 dim_in=3,
                 dim_embed=16,
                 patch_size=3,
                 stride=1,
                 depth=1,
                 # =========================
                 ssm_d_state=16,
                 ssm_ratio=1.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer=nn.SiLU,
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_drop_rate=0.0,
                 ssm_init="v0",
                 forward_type="v2",
                 # =========================
                 mlp_ratio=1.0,
                 mlp_act_layer=nn.GELU,
                 mlp_drop_rate=0.0,
                 # =========================
                 drop_path_rate=0.1,
                 patch_norm=True,
                 norm_layer=nn.LayerNorm,
                 **kwargs
                 ):
        super().__init__()

        self.convemb = ConvEmbed(dim_in=dim_in,
                                 dim_embed=dim_embed,
                                 patch_size=patch_size,
                                 stride=stride)
        self.vssm = VSSMBlock(hidden_dim=dim_embed,
                              depth=depth,
                              drop_path=drop_path_rate,
                              norm_layer=norm_layer,
                              ssm_d_state=ssm_d_state,
                              ssm_ratio=ssm_ratio,
                              ssm_dt_rank=ssm_dt_rank,
                              ssm_act_layer=ssm_act_layer,
                              ssm_conv=ssm_conv,
                              ssm_conv_bias=ssm_conv_bias,
                              ssm_drop_rate=ssm_drop_rate,
                              ssm_init=ssm_init,
                              forward_type=forward_type,
                              mlp_ratio=mlp_ratio,
                              mlp_act_layer=mlp_act_layer,
                              mlp_drop_rate=mlp_drop_rate)
        self.ss2d = SS2D(
                    d_model=dim_embed,
                    d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    dt_rank=ssm_dt_rank,
                    act_layer=ssm_act_layer,
                    # ==========================
                    d_conv=ssm_conv,
                    conv_bias=ssm_conv_bias,
                    # ==========================
                    dropout=ssm_drop_rate,
                    # bias=False,
                    # ==========================
                    # dt_min=0.001,
                    # dt_max=0.1,
                    # dt_init="random",
                    # dt_scale="random",
                    # dt_init_floor=1e-4,
                    initialize=ssm_init,
                    # ==========================
                    forward_type=forward_type,
                )
        self.perm = Permute(0, 3, 1, 2)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, stride=1, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def forward(self, x, y):
        x, y = self.convemb(x, y)
        out = self.ss2d(x, y)
        return self.perm(out)


DEVICE = 'cuda:0'

if __name__ == '__main__':
    A = torch.ones((1, 3, 224, 224)).to(DEVICE)
    B = torch.ones((1, 3, 224, 224)).to(DEVICE)
    # LA = tv2f.resize(A, [A.shape[2] // 4, A.shape[3] // 4], antialias=False)
    # LB = tv2f.resize(B, [B.shape[2] // 4, B.shape[3] // 4], antialias=False)
    model = FSSM().to(DEVICE)
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
    for i in tqdm(range(10)):
        tex = model(A, B)
        # str = model_str(test_tensor_A)
        # print(tex[0])
        # print(str[0].shape)
    print(time.time() - tic0)
    allocated_memory = torch.cuda.max_memory_allocated()
    print(f"Max allocated GPU memory: {allocated_memory / 1024 ** 3} GB")
