import os.path
import logging
import argparse

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_benchmark


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/FFDNet

@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo             # model_zoo
   |--ffdnet_gray        # model_name, for color images
   |--ffdnet_color
   |--ffdnet_color_clip  # for clipped uint8 color images
   |--ffdnet_gray_clip
|--testset               # testsets
   |--set12              # testset_name
   |--bsd68
   |--cbsd68
|--results               # results
   |--set12_ffdnet_gray  # result_name = testset_name + '_' + model_name
   |--set12_ffdnet_color
   |--cbsd68_ffdnet_color_clip
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ffdnet_gray',
                        help='ffdnet_gray | ffdnet_color | ffdnet_color_clip | ffdnet_gray_clip')
    parser.add_argument('--testset_name', type=str, default='bsd68', help='test set, bsd68 | cbsd68 | set12')
    parser.add_argument('--noise_level_img', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--show_img', type=bool, default=False, help='show the image')
    parser.add_argument('--model_pool', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')
    parser.add_argument('--model_path', type=str, default=None, help='explicit full path to the .pth; overrides --model_pool/--model_name')
    parser.add_argument('--testset_path', type=str, default=None, help='explicit full path to the test set folder; overrides --testsets/--testset_name')
    parser.add_argument('--results', type=str, default='results', help='path of results')
    parser.add_argument('--need_degradation', type=bool, default=True, help='add noise or not')
    args = parser.parse_args()

    noise_level_img = args.noise_level_img     # noise level for noisy image
    noise_level_model = noise_level_img        # noise level for model
    model_name = args.model_name
    need_degradation = args.need_degradation

    task_current = 'dn'       # 'dn' for denoising | 'sr' for super-resolution
    sf = 1                    # unused for denoising
    if 'color' in model_name:
        n_channels = 3        # setting for color image
        nc = 96               # setting for color image
        nb = 12               # setting for color image
    else:
        n_channels = 1        # setting for grayscale image
        nc = 64               # setting for grayscale image
        nb = 15               # setting for grayscale image
    if 'clip' in model_name:
        use_clip = True       # clip the intensities into range of [0, 1]
    else:
        use_clip = False

    if args.model_path:
        model_path = args.model_path
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    else:
        model_path = os.path.join(args.model_pool, args.model_name + '.pth')
        model_basename = args.model_name

    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    if args.testset_path:
        L_path = args.testset_path
        testset_basename = os.path.basename(os.path.normpath(args.testset_path))
    else:
        L_path = os.path.join(args.testsets, args.testset_name) # L_path, for Low-quality images
        testset_basename = args.testset_name
    H_path = L_path                               # H_path, for High-quality images
    result_name = testset_basename + '_' + model_basename     # fixed
    E_path = os.path.join(args.results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        if need_degradation:  # degradation process
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
            if use_clip:
                img_L = util.uint2single(util.single2uint(img_L))

        util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if args.show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        sigma = torch.full((1,1,1,1), noise_level_model/255.).type_as(img_L)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        img_E = model(img_L, sigma)
        img_E = util.tensor2uint(img_E)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if args.show_img else None

        # ------------------------------------
        # save results
        # ------------------------------------

        # util.imsave(img_E, os.path.join(E_path, img_name+ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))
        noise_level = 'real' if 'real' in testset_basename.lower() else noise_level_model
        utils_benchmark.save_benchmark(model_basename, testset_basename, ave_psnr, ave_ssim,
                                       noise_level=noise_level)

if __name__ == '__main__':

    main()
