import os
import sys
import glob
import time

import cv2
import torch

from tqdm import tqdm
from nets.PDRF import PDRF
from misc.dataloader import testloader
from torch.utils.data import DataLoader
from misc.general import DeviceChecker
import torchvision.transforms.v2.functional as tv2f
from misc.general import fusion_channel_sf


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class Tester:
    def __init__(self,
                 modelpath='./trained_weights/2024-10-17 14.36.04/model_S66.ckpt',
                 dataroot='./test_datasets',
                 savepath='./results',
                 dataset_name='hrmff',
                 kernel_radius=6,
                 binarization=False,
                 ):
        self.DEVICE = DeviceChecker().DEVICE
        self.MODELPATH = modelpath
        self.DATAPATH = os.path.join(dataroot, dataset_name)
        self.DATASET_NAME = dataset_name
        self.SAVEPATH = savepath + '/' + dataset_name
        self.KERNEL_RADIUS = kernel_radius
        self.BINARIZATION = binarization


    def __call__(self, *args, **kwargs):
        # data
        eval_list_A = sorted(glob.glob(os.path.join(self.DATAPATH, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(self.DATAPATH, 'sourceB', '*.*')))


        Net = PDRF(inference=True).to(self.DEVICE)
        Net.load_state_dict(torch.load(self.MODELPATH))
        Net.eval()
        num_params = 0
        for p in Net.parameters():
            num_params += p.numel()
        tqdm.write("The number of model parameters: {} M".format(round(num_params / 10e5, 6)))
        Net(torch.ones([1, 3, 8, 8]).cuda(),torch.ones([1, 3, 8, 8]).cuda(),torch.ones([1, 1, 8, 8]).cuda(),torch.ones([1, 3, 8, 8]).cuda())


        if not os.path.exists(self.SAVEPATH):
            os.mkdir(self.SAVEPATH)
        eval_data = testloader(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=1,
                                 shuffle=False, )
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)

        cnt = 1
        running_time = []
        flops_total = 0

        with torch.no_grad():

            for A, B in eval_loader_tqdm:

                originalA = cv2.imread(eval_list_A[cnt - 1])
                originalB = cv2.imread(eval_list_B[cnt - 1])


                start_time = time.time()


                dm = None

                if min(originalA.shape[0], originalA.shape[1]) // 16 > 384:
                    IA = tv2f.resize(A, [originalA.shape[0] // 16 // 16 * 16, originalA.shape[1] // 16 // 16 * 16], antialias=False)
                    IB = tv2f.resize(B, [originalA.shape[0] // 16 // 16 * 16, originalA.shape[1] // 16 // 16 * 16], antialias=False)
                    if dm == None:
                        SF = fusion_channel_sf(IA, IB, self.KERNEL_RADIUS)
                    else:
                        SF = tv2f.resize(dm['pred_224'], [originalA.shape[0] // 16 // 16 * 16, originalA.shape[1] // 16 // 16 * 16], antialias=False)

                    # flops, _ = profile(Net, inputs=[torch.ones_like(IA).cuda(),
                    #                                      torch.ones_like(IB).cuda(),
                    #                                      torch.ones_like(SF).cuda(),
                    #                                      torch.ones_like(IA).cuda(),], verbose=False)
                    # flops_total += flops

                    fm, dm = Net(IA, IB, SF, IA)

                if min(originalA.shape[0], originalA.shape[1]) // 8 > 384:
                    IA = tv2f.resize(A, [originalA.shape[0] // 8 // 16 * 16, originalA.shape[1] // 8 // 16 * 16], antialias=False)
                    IB = tv2f.resize(B, [originalA.shape[0] // 8 // 16 * 16, originalA.shape[1] // 8 // 16 * 16], antialias=False)
                    if dm == None:
                        SF = fusion_channel_sf(IA, IB, self.KERNEL_RADIUS)
                    else:
                        SF = tv2f.resize(dm['pred_224'], [originalA.shape[0] // 8 // 16 * 16, originalA.shape[1] // 8 // 16 * 16], antialias=False)

                    # flops, _ = profile(Net, inputs=[torch.ones_like(IA).cuda(),
                    #                                      torch.ones_like(IB).cuda(),
                    #                                      torch.ones_like(SF).cuda(),
                    #                                      torch.ones_like(IA).cuda(),], verbose=False)
                    # flops_total += flops

                    fm, dm = Net(IA, IB, SF, IA)


                if min(originalA.shape[0], originalA.shape[1]) // 4 > 384:
                    IA = tv2f.resize(A, [originalA.shape[0] // 4 // 16 * 16, originalA.shape[1] // 4 // 16 * 16], antialias=False)
                    IB = tv2f.resize(B, [originalA.shape[0] // 4 // 16 * 16, originalA.shape[1] // 4 // 16 * 16], antialias=False)
                    if dm == None:
                        SF = fusion_channel_sf(IA, IB, self.KERNEL_RADIUS)
                    else:
                        SF = tv2f.resize(dm['pred_224'], [originalA.shape[0] // 4 // 16 * 16, originalA.shape[1] // 4 // 16 * 16], antialias=False)

                    # flops, _ = profile(Net, inputs=[torch.ones_like(IA).cuda(),
                    #                                 torch.ones_like(IB).cuda(),
                    #                                 torch.ones_like(SF).cuda(),
                    #                                 torch.ones_like(IA).cuda(), ], verbose=False)
                    # flops_total += flops

                    fm, dm = Net(IA, IB, SF, IA)


                if min(originalA.shape[0], originalA.shape[1]) // 2 > 384:
                    IA = tv2f.resize(A, [originalA.shape[0] // 2 // 16 * 16, originalA.shape[1] // 2 // 16 * 16], antialias=False)
                    IB = tv2f.resize(B, [originalA.shape[0] // 2 // 16 * 16, originalA.shape[1] // 2 // 16 * 16], antialias=False)
                    if dm == None:
                        SF = fusion_channel_sf(IA, IB, self.KERNEL_RADIUS)
                    else:
                        SF = tv2f.resize(dm['pred_224'], [originalA.shape[0] // 2 // 16 * 16, originalA.shape[1] // 2 // 16 * 16], antialias=False)

                    # flops, _ = profile(Net, inputs=[torch.ones_like(IA).cuda(),
                    #                                 torch.ones_like(IB).cuda(),
                    #                                 torch.ones_like(SF).cuda(),
                    #                                 torch.ones_like(IA).cuda(), ], verbose=False)
                    # flops_total += flops

                    fm, dm = Net(IA, IB, SF, IA)

                else:
                    SF = fusion_channel_sf(A, B, self.KERNEL_RADIUS)

                    # flops, _ = profile(Net, inputs=[torch.ones_like(A).cuda(),
                    #                                 torch.ones_like(B).cuda(),
                    #                                 torch.ones_like(SF).cuda(),
                    #                                 torch.ones_like(A).cuda(), ], verbose=False)
                    # flops_total += flops

                    fm, dm = Net(A, B, SF, A)

                dm = tv2f.resize(dm['pred_224'], size=[originalA.shape[0], originalA.shape[1]], antialias=False)

                if self.BINARIZATION:
                    dm = torch.where(dm > 0.5, 1., 0.)

                running_time.append(time.time() - start_time)

                dm = torch.einsum('c w h -> w h c', dm[0]).clone().detach().cpu().numpy()

                fused = originalA * dm + originalB * (1 - dm)

                cv2.imwrite(self.SAVEPATH + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '.jpg', fused)

                cnt += 1

        running_time_total = 0
        # if self.DATASET_NAME == 'hrmff':
        #     print(f'avg flops:{flops_total / len(running_time) / 1000 ** 4:.6f} T')
        # else:
        #     print(f'avg flops:{flops_total / len(running_time) / 1000 ** 3:.6f} G')
        for i in range(len(running_time)):
            # print("process_time: {} s".format(running_time[i]))
            if i != 0:
                running_time_total += running_time[i]
        print(f"avg_process_time: {running_time_total / len(running_time) :.6f} s")
        running_time.append(running_time_total / len(running_time))
        # np.savetxt('time.csv', np.array(running_time), delimiter=',', fmt='%.10f')
        print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.6f} GB")
        print('Testing finished!')



if __name__ == '__main__':
    f = Tester()
    f()
