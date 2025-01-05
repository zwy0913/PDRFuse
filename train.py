import sys
import datetime
import random

import torch
from torch import nn
from tqdm import tqdm
from nets.PDRF import PDRF
from misc.dataloader import get_dataloader
from misc.general import DeviceChecker
from misc.general import sigmoid_rampup
from misc.general import get_net_para
from misc.general import Logging_SaveWeights
from misc.general import fusion_channel_sf

from misc.sobel_op import SobelComputer
from misc.metrics_compute import compute_loss_and_metrics_focus


class Trainer(object):
    def __init__(self,
                 data_path=[
                     'K:/Datasets/Train_Valid/MFFdatasets-duts-te',
                     'L:/Datasets/Train_Valid/MFFdatasets-duts-tr',
                     'L:/Datasets/Train_Valid/MFFdatasets-voc12',
                     # 'L:/Datasets/Train_Valid/MFFdatasets-dut-omron',
                     # 'L:/Datasets/Train_Valid/MFFdatasets-ECSSD',
                     # 'L:/Datasets/Train_Valid/MFFdatasets-msra10k',
                 ],
                 init_size=400,
                 model_input_size=384,
                 save_path='./trained_weights',
                 set_size=99999,
                 batchsize=8,
                 epochs=66,
                 lr=5e-4,
                 gamma=0.9,
                 scheduler_step=1,
                 save_every_weights=True,
                 load=False):
        # Hyper-parameters
        self.DEVICE = DeviceChecker().DEVICE
        self.DATAPATH = data_path
        self.INIT_SIZE = init_size
        self.ModelInputSize = model_input_size
        self.SAVEPATH = save_path
        self.SETSIZE = set_size
        self.BATCHSIZE = batchsize
        self.EPOCHS = epochs
        self.LR = lr
        self.GAMMA = gamma
        self.SCHEDULER_STEP = scheduler_step
        self.SAVE_EVERY_WEIGHTS = save_every_weights
        self.LOAD = load


    def __call__(self, *args, **kwargs):
        # nets
        Net = PDRF().to(self.DEVICE)
        # tqdm.write("Learnable parameters of TNet: {} M".format(round(get_net_para(Net) / 10e5, 6)))
        if self.LOAD:
            Net.load_state_dict(torch.load('debug_model_S.pth'))
        Net.train()

        # data
        train_loader, train_data_size = get_dataloader(self.DATAPATH, self.BATCHSIZE, self.SETSIZE, self.INIT_SIZE,self.ModelInputSize)

        data_source = '\n'
        for i in self.DATAPATH:
            data_source += i
            data_source += '\n'
        # print training settings
        self.log_para = ('Data_source: ' + str(data_source)
                         + '\nLoader_length: ' + str(train_data_size) + ' // ' + str(self.BATCHSIZE) + ' = ' + str(len(train_loader))
                         + '\nIni_size: ' + str(self.INIT_SIZE)
                         + '\nModel_input_size: ' + str(self.ModelInputSize)
                         + '\nSet_size: ' + str(self.SETSIZE)
                         + '\nBatchsize: ' + str(self.BATCHSIZE)
                         + '\nEpochs: ' + str(self.EPOCHS)
                         + '\nLR: ' + str(self.LR)
                         + '\nGamma: ' + str(self.GAMMA)
                         + '\nScheduler_step: ' + str(self.SCHEDULER_STEP)
                         + '\nLoad: ' + str(self.LOAD)
                         + '\nNetwork_params: ' + str(round(get_net_para(Net) / 10e5, 6))
                         )
        self.LS = Logging_SaveWeights(savepath=self.SAVEPATH, hyperparas=self.log_para, save_every_weights=self.SAVE_EVERY_WEIGHTS)
        tqdm.write(self.log_para)


        # opts
        optimizer = torch.optim.AdamW(Net.parameters(), lr=self.LR, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.SCHEDULER_STEP, gamma=self.GAMMA)

        # loss func
        l1_loss = nn.L1Loss()
        l2_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()
        sobel_compute = SobelComputer()

        # init
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler()
        scaler1 = torch.cuda.amp.GradScaler()

        # iter
        iter_num = 0
        alpha = 0
        beta = 0

        tqdm.write('\nStart iteration...\n')
        for epoch in range(self.EPOCHS):

            epoch_loss_focus = 0
            epoch_loss_refine = 0

            # progress bar
            train_loader_tqdm = tqdm(train_loader, colour='green', leave=False, file=sys.stdout)
            for A, B, DM, GT in train_loader_tqdm:
                optimizer.zero_grad()

                # inputs
                beta = sigmoid_rampup(iter_num // 40, 80.)

                # only update student
                with ((torch.autocast(device_type=self.DEVICE, dtype=torch.float16))):

                    SF = fusion_channel_sf(A, B, random.randint(6, 6))
                    fm, dm = Net(A, B, SF, GT)
                    fm['gt'] = DM
                    dm['gt'] = DM
                    sobel_compute.compute_edges(fm)
                    sobel_compute.compute_edges(dm)

                    # ————————————————————————————————————————— loss —————————————————————————————————————————————
                    loss_fm = compute_loss_and_metrics_focus(fm)
                    loss_focus = loss_fm['total_loss']

                    loss_dm = compute_loss_and_metrics_focus(dm)
                    loss_refine = loss_dm['total_loss']

                    S_LOSS = 1. * loss_focus + 1. * loss_refine
                    epoch_loss_focus += loss_focus / len(train_loader)
                    epoch_loss_refine += loss_refine / len(train_loader)
                    # ————————————————————————————————————————————————————————————————————————————————————————————
                # backward
                scaler.scale(S_LOSS).backward()
                scaler.step(optimizer)
                scaler.update()

                # update bar

                train_loader_tqdm.set_postfix(focus=float(loss_focus),
                                              refine=float(loss_refine))
                train_loader_tqdm.set_description("Epoch %s" % (str(epoch + 1)))

                # iter cnt
                iter_num += 1

                if iter_num % (len(train_loader_tqdm) // 4) == 0:
                    self.LS(Net, epoch + 1, '')

            # update LR
            scheduler.step()

            # log to file
            log_contents = (f"[{str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S'))}] Epoch {epoch + 1} "
                            f"- focus : {epoch_loss_focus:.4f} "
                            f"- refine : {epoch_loss_refine:.4f}\n")

            # print results
            tqdm.write(log_contents)

            self.LS(Net, epoch + 1, log_contents)


if __name__ == '__main__':
    t = Trainer()
    t()
