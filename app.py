from flask import Flask, render_template , request

import os
import imageio
from dataset.live import LIVE_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.model_eval import DeepQANet

app = Flask(__name__)
model = DeepQANet

@app.route('/', methods = ['GET'])
def iqa():
    return render_template('index.html')

# @todo Deploy model
@app.route('/', methods = ['POST'])
def predict():
    r_imagefile = request.files['r_imagefile']
    r_image_path = "./r_images/" + r_imagefile.filename
    r_imagefile.save(r_image_path)

    d_imagefile = request.files['d_imagefile']
    d_image_path = "./d_images/" + d_imagefile.filename
    d_imagefile.save(d_image_path)

    # load the trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepQANet(device).to(device)
    model = torch.nn.DataParallel(model).cuda()
    model_dict = torch.load(r'C:\Users\zbb\Desktop\FR_DEEPIQA-master\FR_DEEPIQA-master\model_st\deepQA_LIVE_seed12_0.9708_0.9665_epoch2430.pth', encoding='latin1')

    model.load_state_dict(model_dict,strict=False)  # copyWeights(model, model_dict, fre eze=False)
    model.eval()
    model.to('cuda')


    #preprocessing images
    def load_reference_img(img_relative_path):
        """
        load single reference_img to patches
        """

        patch_size = [112,112]
        patch_step = [80,80]
        color = 'gray'
        patch_mode = 'shift_center'

        ref_top_left_set = []
        r_pat_set = []
        pass_list = []

        img = imageio.v2.imread(img_relative_path)
        current_h = img.shape[0]
        current_w = img.shape[1]

        # Gray
        img = convert_color2(img, color)

        # Local normalization

        r_img_norm = img.astype('float32') / 255.

        if color == 'gray':
            r_img_norm = r_img_norm[:, :, None]

        # numbers of patches along y and x axes
        ny = (current_h - patch_size[0]) // patch_step[0] + 1
        nx = (current_w - patch_size[1]) // patch_step[1] + 1
        patch_info = (int(ny * nx), ny, nx)

        # get non-covered length along y and x axes
        cov_height = patch_step[0] * (ny - 1) + patch_size[0]
        cov_width = patch_step[1] * (nx - 1) + patch_size[1]
        nc_height = current_h - cov_height
        nc_width = current_w - cov_width

        # Shift center
        if patch_mode == 'shift_center':
            shift = [(nc_height + 1) // 2, (nc_width + 1) // 2]
            if shift[0] % 2 == 1:
                shift[0] -= 1
            if shift[1] % 2 == 1:
                shift[1] -= 1
            shift = tuple(shift)
        else:
            shift = (0, 0)

        # generate top_left_set of patches
        top_left_set = np.zeros((nx * ny, 2), dtype=np.int64)
        for yidx in range(ny):
            for xidx in range(nx):
                top = (yidx * patch_step[0] + shift[0])
                left = (xidx * patch_step[1] + shift[1])
                top_left_set[yidx * nx + xidx] = [top, left]
        ref_top_left_set.append(top_left_set)

        # Crop the images to patches
        for idx in range(ny * nx):
            [top, left] = top_left_set[idx]

            r_crop_norm = r_img_norm[top:top + patch_size[0],
                          left:left + patch_size[1]]

            r_pat_set.append(r_crop_norm)

        return patch_info, ref_top_left_set, r_pat_set

    def load_distored_img(patch_info,ref_top_left_set,img_relative_path):


        patch_size = [112,112]
        color = 'gray'
        fr_met = None
        fr_met_scale = 1.0
        random_crops = 0
        std_filt_r = 1.0

        n_patches = 0
        npat_img_list = []
        d_pat_set = []
        loc_met_set = []
        filt_idx_list = []
        dis2ref_idx = []

        pat_idx = 0


        # Read ref. and dist. images
        d_img_raw = imageio.v2.imread(img_relative_path)

        cur_h = d_img_raw.shape[0]
        cur_w = d_img_raw.shape[1]

        # Gray or RGB
        d_img = convert_color2(d_img_raw, color)

        # Read local metric scores
        # if fr_met:
        #     ext = int(1. / fr_met_scale) - 1
        #     met_size = (int((cur_h + ext) * fr_met_scale),
        #                 int((cur_w + ext) * fr_met_scale))
        #     met_pat_size = (int((patch_size[0] + ext) * fr_met_scale),
        #                     int((patch_size[1] + ext) * fr_met_scale))
        #     if fr_met == 'SSIM_now':
        #         raise NotImplementedError()
        #     else:
        #         met_s_fname = (img_relative_path +
        #                        fr_met_suffix + fr_met_ext)
        #         loc_q_map = np.fromfile(
        #             os.path.join(self.fr_met_path, self.fr_met_subpath,
        #                          met_s_fname),
        #             dtype='float32')
        #         loc_q_map = loc_q_map.reshape(
        #             (met_size[1], met_size[0])).transpose()


        d_img_norm = d_img.astype('float32') / 255.

        if color == 'gray':
            d_img_norm = d_img_norm[:, :, None]

        top_left_set = ref_top_left_set

        if np.array(top_left_set).shape[0]==1:
           top_left_set=ref_top_left_set[0]

        cur_n_patches=np.array(top_left_set).shape[0]

        if random_crops > 0:
            if random_crops < cur_n_patches:
                n_crops = random_crops
                rand_perm = np.random.permutation(cur_n_patches)
                sel_patch_idx = sorted(rand_perm[:n_crops])
                top_left_set = top_left_set[sel_patch_idx].copy()
            else:
                n_crops = cur_n_patches
                sel_patch_idx = np.arange(cur_n_patches)

            npat_filt = n_crops
            npat_img_list.append((npat_filt, 1, npat_filt))
            n_patches += npat_filt

            idx_set = list(range(npat_filt))
            filt_idx_list.append(idx_set)

        else:
            # numbers of patches along y and x axes
            npat, ny, nx = patch_info
            npat_filt = int(npat * std_filt_r)

            npat_img_list.append((npat_filt, ny, nx))
            n_patches += npat_filt

            if std_filt_r < 1.0:
                std_set = np.zeros((nx * ny))
                for idx, top_left in enumerate(top_left_set):
                    top, left = top_left
                    std_set[idx] = np.std(
                        d_img[top:top + patch_size[0],
                              left:left + patch_size[1]])

            # Filter the patches with low std
            if std_filt_r < 1.0:
                idx_set = sorted(list(range(len(std_set))),
                                 key=lambda x: std_set[x], reverse=True)
                idx_set = sorted(idx_set[:npat_filt])
            else:
                idx_set = list(range(npat_filt))
            filt_idx_list.append(idx_set)

        # Crop the images to patches
        for idx in idx_set:
            [top, left] = top_left_set[idx]


            d_crop_norm = d_img_norm[top:top + patch_size[0],
                                     left:left + patch_size[1]]

            d_pat_set.append(d_crop_norm)


            # Crop the local metric scores
            # if fr_met:
            #     ext = int(1. / fr_met_scale) - 1
            #     top_r = int((top + ext) * fr_met_scale)
            #     left_r = int((left + ext) * fr_met_scale)
            #
            #
            #     loc_met_crop = loc_q_map[top_r:top_r + met_pat_size[0],
            #                              left_r:left_r + met_pat_size[1]]
            #
            #
            #     if self.fr_met_avg:
            #         loc_met_set.append(
            #             np.mean(loc_met_crop, keepdims=True))
            #     else:
            #         loc_met_set.append(loc_met_crop)

            pat_idx += 1



        n_patches = n_patches
        npat_img_list = npat_img_list
        d_pat_set = d_pat_set
        if fr_met:
            loc_met_set = loc_met_set
        filt_idx_list = filt_idx_list
        dis2ref_idx = dis2ref_idx


        return d_pat_set

    patch_info, ref_top_left_set, r_pat_set = load_reference_img(img_relative_path=r_imagefile)
    d_pat_set = load_distored_img(patch_info, ref_top_left_set, img_relative_path=d_imagefile)
    mos_set = [1.0]

    r_pat_set = np.array(r_pat_set).transpose(0, 3, 1, 2)
    d_pat_set = np.array(d_pat_set).transpose(0, 3, 1, 2)
    mos_set = np.array(mos_set)

    r_pat_set = torch.from_numpy(r_pat_set)
    d_pat_set = torch.from_numpy(d_pat_set)
    mos_set = torch.from_numpy(mos_set)

    loss, score_pred, senMap, error_np = model(r_pat_set, d_pat_set, mos_set)


    predict_score = '%.4f' % score_pred

    return render_template('index.html', prediction=predict_score, r_imagefile=r_image_path, d_imagefile=d_image_path)

def convert_color2(img, color):
    """ Convert image into gray or RGB or YCbCr.
    (In case of gray, dimension is not increased for
    the faster local normalization.)
    """
    assert len(img.shape) in [2, 3]
    if color == 'gray':
        # if d_img_raw.shape[2] == 1:
        if len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = rgb2gray(img)

    return img_

def rgb2gray(rgb):
    assert rgb.shape[2] == 3
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
    app.run(port=3000, debug=True)