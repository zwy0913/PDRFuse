import random

import torch
import numpy as np
import os
import datetime
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tv2f


class DeviceChecker:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        # print('\nCUDA is available. (' + str(
        #     torch.cuda.get_device_name(torch.cuda.current_device())) + ')')
    else:
        DEVICE = 'cpu'
        # print('\nOOPS! CUDA is not available! Calculation is performing on CPU.')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=6, verbose=False, delta=0.):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 6
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, model, val_loss, current_epoch, save_every_model=True):

        torch.save(model.state_dict(), 'debug_model.ckpt')

        if save_every_model:
            model_save_path = self.save_path + '/models' + str(current_epoch) + '.ckpt'
            torch.save(model.state_dict(), model_save_path)

        score = -val_loss

        if self.best_score is None:
            print('')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'\033[0;33mEarlyStopping counter: {self.counter} out of {self.patience}\033[0m')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print('')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves models when validation loss decrease.'''
        if self.verbose:
            tqdm.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving models ...')
        # path = os.path.join(self.save_path, 'best_network' + '_ws='  + str(window_size) + '*' + str(window_size) + '_bs=' + str(batch_size) + '_lr=' + str(lr) + '_iav=' +  str(img_argument_val) + '_dim=' + str(dim) + '_depth=' + str(depth) + '_heads=' + str(heads) + '_mlpdim=' + str(mlp_dim) + '.pth')
        path = os.path.join(self.save_path, 'best_network' + '.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


class Logging_SaveWeights_ES:
    def __init__(self, savepath, patience, hyperparas=None):
        self.SAVEPATH = savepath
        self.start_training_time = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        os.mkdir(os.path.join(savepath, self.start_training_time))
        self.ChildDir = os.path.join(savepath, self.start_training_time)
        self.tlf = open(self.ChildDir + '\\' + self.start_training_time + '.txt', 'w')
        if hyperparas != None:
            self.tlf.write('HyperParameter:\n')
            self.tlf.write(hyperparas)
            self.tlf.write('\n\nTraining log:\n')
        # To avoid overfitting.
        self.ES = EarlyStopping(os.path.join(os.getcwd(), self.ChildDir), patience=patience)
        self.ENDTRAIN = False
        # print("Trained weights will be saved in the folder: " + os.path.join(os.getcwd(), self.ChildDir) + "\n")
        print("Weights saved in: " + os.path.join(os.getcwd(), self.ChildDir) + "\n")

    def __call__(self, model, current_epoch, log_contents, val_loss, save_every_model):
        self.Logging(log_contents)  # Logging
        self.SaveWeights(model, val_loss, current_epoch, save_every_model)  # Save models weights

    def Logging(self, contents):
        self.tlf.write(contents)

    def SaveWeights(self, model, val_loss, current_epoch, save_every_model):
        # To avoid overfitting.
        self.ES(model, val_loss, current_epoch, save_every_model)
        if self.ES.early_stop:
            self.tlf.close()
            self.ENDTRAIN = True


class Logging_SaveWeights:
    def __init__(self, savepath, hyperparas=None, save_every_weights=False):
        self.save_every_weights = save_every_weights
        if self.save_every_weights:
            self.SAVEPATH = savepath
            self.start_training_time = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
            os.mkdir(os.path.join(savepath, self.start_training_time))
            self.ChildDir = os.path.join(savepath, self.start_training_time)
            self.tlf = open(self.ChildDir + '\\' + self.start_training_time + '.txt', 'w')
            if hyperparas != None:
                self.tlf.write('HyperParameter:\n')
                self.tlf.write(hyperparas)
                self.tlf.write('\n\nTraining log:\n')
            # print("Trained weights will be saved in the folder: " + os.path.join(os.getcwd(), self.ChildDir) + "\n")
            print("Weights saved in: " + os.path.join(os.getcwd(), self.ChildDir) + "\n")

    def __call__(self, SNet, current_epoch, log_contents):
        if self.save_every_weights:
            # torch.save(TNet.state_dict(), 'debug_model_T.pth')
            # torch.save(PNet.state_dict(), 'debug_model_P.pth')
            torch.save(SNet.state_dict(), 'debug_model.ckpt')
            self.Logging(log_contents)  # Logging
            self.SaveWeights(SNet, current_epoch)  # Save models weights
        else:
            # torch.save(TNet.state_dict(), 'debug_model_T.pth')
            # torch.save(PNet.state_dict(), 'debug_model_P.pth')
            torch.save(SNet.state_dict(), 'debug_model.ckpt')

    def Logging(self, contents):
        self.tlf.write(contents)

    def SaveWeights(self, SNet, current_epoch):
        # model_save_path = os.path.join(os.getcwd(), self.ChildDir) + '/model_T' + str(current_epoch) + '.ckpt'
        # torch.save(TNet.state_dict(), model_save_path)
        # model_save_path = os.path.join(os.getcwd(), self.ChildDir) + '/model_P' + str(current_epoch) + '.ckpt'
        # torch.save(PNet.state_dict(), model_save_path)
        model_save_path = os.path.join(os.getcwd(), self.ChildDir) + '/model_S' + str(current_epoch) + '.ckpt'
        torch.save(SNet.state_dict(), model_save_path)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_net_para(net):
    num_params = 0
    for p in net.parameters():
        num_params += p.numel()
    return num_params


def fusion_channel_sf(imgA, imgB, kernel_radius=5):
    """
    Perform channel sf fusion two features
    """
    device = imgA.device
    b, c, h, w = imgA.shape
    r_shift_kernel = torch.FloatTensor([[0, 0, 0],
                                        [1, 0, 0],
                                        [0, 0, 0]]).to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    b_shift_kernel = torch.FloatTensor([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]]).to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    f1_r_shift = F.conv2d(imgA, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(imgA, b_shift_kernel, padding=1, groups=c)
    f2_r_shift = F.conv2d(imgB, r_shift_kernel, padding=1, groups=c)
    f2_b_shift = F.conv2d(imgB, b_shift_kernel, padding=1, groups=c)

    f1_grad = torch.pow((f1_r_shift - imgA), 2) + torch.pow((f1_b_shift - imgA), 2)
    f2_grad = torch.pow((f2_r_shift - imgB), 2) + torch.pow((f2_b_shift - imgB), 2)

    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1, keepdim=True)
    # save_image(f.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), '../11.png')
    f2_sf = torch.sum(F.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c), dim=1, keepdim=True)
    # weight_zeros = torch.zeros(f1_sf.shape).to(device)
    # weight_ones = torch.ones(f1_sf.shape).to(device)

    # get decision map
    dm_tensor = torch.where(f1_sf > f2_sf, 1., 0.).to(device)
    # dm_np = dm_tensor.squeeze().cpu().numpy().astype(int)

    return dm_tensor

def tensor_erode(bin_img, ksize):  # 已测试
    eroded = 1 - tensor_dilate(1 - bin_img, ksize)
    return eroded

def tensor_dilate(bin_img, ksize): #
    # 首先为原图加入 padding，防止图像尺寸缩小
    pad = ksize // 2
    # bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    dilate = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=pad)
    return dilate

def compute_iou(seg, gt):
    intersection = seg * gt
    union = seg + gt
    torch.count_nonzero(intersection)
    iou = (torch.count_nonzero(intersection) + 1e-6) / (torch.count_nonzero(union) + 1e-6)
    return iou


def perturb_seg(s_fm_to_refine, intensity=35):
    # save_image(s_fm_to_refine, '1.png')
    s_fm_to_refine = torch.where(s_fm_to_refine > 0.5, 1., 0.)
    # s_fm_to_refine = tv2f.resize(s_fm_to_refine, [s_fm_to_refine.shape[2] // 16, s_fm_to_refine.shape[3] // 16], antialias=False)
    # s_fm_to_refine = tv2f.resize(s_fm_to_refine, [A.shape[2], B.shape[3]], antialias=False)
    b, c, h, w = s_fm_to_refine.shape
    for ib in range(b):
        for _ in range(0, intensity):
            lx, ly = random.randint(0, w - 2), random.randint(0, h - 2)
            lw, lh = random.randint(lx + 1, w), random.randint(ly + 1, h)
            cx = int((lx + lw) / 2)
            cy = int((ly + lh) / 2)
            if random.random() < 0.75:
                s_fm_to_refine[ib, :, cy, cx] = random.randint(0, 1)
            if random.random() < 0.75:
                s_fm_to_refine[ib, 0, ly:lh, lx:lw] = tensor_dilate(
                    s_fm_to_refine[ib:ib + 1, 0:, ly:lh, lx:lw], ksize=random.choice([5, 7, 9]))
            else:
                s_fm_to_refine[ib, 0, ly:lh, lx:lw] = tensor_erode(
                    s_fm_to_refine[ib:ib + 1, 0:, ly:lh, lx:lw], ksize=random.choice([5, 7, 9]))
    # save_image(s_fm_to_refine, '2.png')

    return s_fm_to_refine