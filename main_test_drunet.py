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
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN

@misc{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
'''


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='drunet_color',
                        help='drunet_color | drunet_gray | drunet_deblocking_color | drunet_deblocking_grayscale')
    parser.add_argument('--testset_name', type=str, default='bsd68', help='test set')
    parser.add_argument('--noise_level_img', type=int, default=25, help='noise level: 15, 25, 50')
    parser.add_argument('--show_img', type=bool, default=False, help='show the image')
    parser.add_argument('--model_pool', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')
    parser.add_argument('--model_path', type=str, default=None, help='explicit full path to the .pth; overrides --model_pool/--model_name')
    parser.add_argument('--testset_path', type=str, default=None, help='explicit full path to the test set folder; overrides --testsets/--testset_name')
    parser.add_argument('--results', type=str, default='results', help='path of results')
    parser.add_argument('--need_degradation', type=bool, default=True, help='add noise or not')
    args = parser.parse_args()

    # DRUNet feeds (noisy image || noise-level map) along the channel dim, so
    # in_nc = n_channels + 1. The released weights are color (in_nc=4) or gray (in_nc=2).
    if 'color' in args.model_name:
        n_channels = 3        # color image
        nc = [64, 128, 256, 512]
    else:
        n_channels = 1        # grayscale image
        nc = [64, 128, 256, 512]
    nb = 4                   # fixed for DRUNet
    act_mode = 'R'           # ReLU, no BN (matches the released weights)
    downsample_mode = 'strideconv'
    upsample_mode = 'convtranspose'
    bias = False             # fixed for DRUNet

    border = 0               # shave border to calculate PSNR and SSIM (denoising)

    if args.model_path:
        model_path = args.model_path
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    else:
        model_path = os.path.join(args.model_pool, args.model_name + '.pth')
        model_basename = args.model_name

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
        args.need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=nc, nb=nb, act_mode=act_mode,
                downsample_mode=downsample_mode, upsample_mode=upsample_mode, bias=bias)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, image sigma:{}'.format(args.model_name, args.noise_level_img))
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        if args.need_degradation:  # degradation process
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, args.noise_level_img/255., img_L.shape)

        util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(args.noise_level_img)) if args.show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # DRUNet's 4-stage U-Net (stride-2 convs) needs H/W divisible by 16.
        # Pad with reflect and crop back to the original size after inference.
        h, w = img_L.shape[2], img_L.shape[3]
        pad_h = int(np.ceil(h / 16) * 16) - h
        pad_w = int(np.ceil(w / 16) * 16) - w
        img_L = torch.nn.functional.pad(img_L, (0, pad_w, 0, pad_h), mode='reflect')

        # DRUNet takes the noisy image concatenated with a noise-level map.
        sigma = torch.full((1, 1, img_L.size(2), img_L.size(3)), args.noise_level_img/255.).type_as(img_L)
        img_E = model(torch.cat((img_L, sigma), dim=1))
        img_E = util.tensor2uint(img_E[..., :h, :w])

        if need_H:

            # ------------------------------------
            # (3) img_H
            # ------------------------------------
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()

            # ------------------------------------
            # PSNR and SSIM
            # ------------------------------------

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
        noise_level = 'real' if 'real' in testset_basename.lower() else args.noise_level_img
        utils_benchmark.save_benchmark(model_basename, testset_basename, ave_psnr, ave_ssim,
                                       noise_level=noise_level)

if __name__ == '__main__':

    main()
