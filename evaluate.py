from dataset.live import LIVE_dataset
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.misc as m
import imageio
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model.model_eval import DeepQANet

import skimage.measure


# @todo approch evaluating system
def showfigures_LIVE():
    testset = LIVE_dataset('C:/Users/zbb/Desktop/GraduateProject/DeepQA-with-Pytorch-master/DeepQA-with'
                                '-Pytorch-master/data/LIVE_dataset/',
                                transform=None,
                                type='test')
    testloader = DataLoader(testset,
                            shuffle=True,
                            batch_size=1,
                            num_workers=4,
                            pin_memory=True)

    # load the trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepQANet(device).to(device)
    model = torch.nn.DataParallel(model).cuda()
    model_dict = torch.load(r'C:\Users\zbb\Desktop\FR_DEEPIQA-master\FR_DEEPIQA-master\model_st\deepQA_LIVE_seed12_0.9708_0.9665_epoch2430.pth', encoding='latin1')

    model.load_state_dict(model_dict,strict=False)
    model.eval()
    model.to('cuda')

    for batch_index, (r_patch_set, d_patch_set, mos_set) in enumerate(testloader):
        mos_set = mos_set.type('torch.FloatTensor')
        r_patch_set = r_patch_set.squeeze(0)
        d_patch_set = d_patch_set.squeeze(0)
        r_patch_set, d_patch_set, mos_set, = Variable(r_patch_set.cuda()), \
                                Variable(d_patch_set.cuda()), \
                                Variable(mos_set.cuda())

        mos_set = mos_set

        loss, score_pred, senMap, error_np = model(r_patch_set, d_patch_set, mos_set)
        # print(score_pred.data.cpu().numpy(), score_gt.data.cpu().numpy())
        # print(senMap.shape)

        score_pred_np = score_pred.data.cpu().numpy()
        score_gt_np = mos_set.data.cpu().numpy()

        print(batch_index, score_gt_np, score_pred_np)

        img_np = r_patch_set.data.cpu().numpy()
        error_np = error_np.data.cpu().numpy()
        senMap_np = senMap.data.cpu().numpy()
        d_patch_set = d_patch_set.data.cpu().numpy()

        error_np = np.squeeze(error_np)
        img_np = np.squeeze(img_np)

        # error_np_resize = get_downsample_filter(get_downsample_filter(error_np))
        error_np_resize = skimage.measure.block_reduce(error_np, (4), np.mean)

        perceptual = error_np_resize*senMap_np

        plt.figure(figsize=(12, 8))
        plt.suptitle('Ground Truth:%.4f, Predit Score:%.4f' % (score_gt_np, score_pred_np),
                     fontsize=16)

        d_patch_set = d_patch_set[:1,:,:,:].squeeze(0)
        d_patch_set = d_patch_set.squeeze(0)

        plt.subplot(231)
        plt.imshow(d_patch_set)
        plt.xlabel('Distorted Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        img_np = img_np[:1,:,:].squeeze(0)

        plt.subplot(232)
        plt.imshow(img_np)
        plt.xlabel('Reference Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])


        error_np = error_np[:1,:,:].squeeze(0)

        plt.subplot(233)
        plt.imshow(error_np)
        plt.xlabel('Error Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        senMap_np = senMap_np[:1,:,:].squeeze(0)
        senMap_np = senMap_np.squeeze(0)

        plt.subplot(234)
        plt.imshow(senMap_np)
        plt.xlabel('Sensitivity Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        perceptual = perceptual[:1,:,:].squeeze(0)
        perceptual = perceptual[:1,:,:].squeeze(0)

        plt.subplot(235)
        plt.imshow(perceptual)
        plt.xlabel('Perceptual Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        if batch_index == 2:
            plt.savefig('LIVE_exp.png', dpi=500)

        plt.show()

        if batch_index > 3:
            break







if __name__=='__main__':
    showfigures_LIVE()