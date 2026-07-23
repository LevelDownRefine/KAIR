import random
import numpy as np
import torch
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetDPSR(BaseDataset):
    """Noisy-image SR dataset (DPSR): L = bicubic-downsampled H + AWGN, concatenated with a noise-level map M.

    ``_make_sample`` modcrops and bicubic-downsamples H to build L, then (train)
    randomly crops+augments the pair, adds AWGN (PyTorch RNG) and concatenates
    the noise-level map M. ``__getitem__`` only loads, calls ``_make_sample`` and
    converts to float32 tensors.
    """

    def __init__(self, opt):
        super(DatasetDPSR, self).__init__(opt)
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf
        self.sigma = opt['sigma'] if opt['sigma'] else [0, 50]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 0

        assert self.paths_H, 'Error: H path is empty.'

    def _make_sample(self, img_H, index):
        """Build (H, L): modcrop+bicubic L; (train) crop+augment; add AWGN(noise_level) and concat noise-level map M."""
        img_H = util.uint2single(img_H)
        img_H = util.modcrop(img_H, self.sf)
        H, W, _ = img_H.shape
        img_L = util.imresize_np(img_H, 1 / self.sf, True)

        if self.opt['phase'] == 'train':
            H, W, C = img_L.shape
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # select noise level and get Gaussian noise
        # ------------------------------------
        if self.opt['phase'] == 'train':
            if random.random() < 0.1:
                noise_level = 0.0
            else:
                noise_level = np.random.uniform(self.sigma_min, self.sigma_max) / 255.0
        else:
            noise_level = self.sigma_test

        # ------------------------------------
        # add noise
        # ------------------------------------
        noise = torch.randn(img_L.shape).mul_(noise_level).numpy()
        img_L = img_L + noise

        # ------------------------------------
        # get noise level map M and concat to L
        # ------------------------------------
        M = np.full((img_L.shape[0], img_L.shape[1], 1), noise_level, dtype=np.float32)
        img_L = np.concatenate((img_L, M), axis=2)

        return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'L', 'H', 'L_path', 'H_path'}``; L has C=4 (RGB + noise-level map)."""
        H_path = self.paths_H[index] if self.paths_H is not None else ''
        img_H = self._load_img_H(index)
        img_H, img_L = self._make_sample(img_H, index)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        L_path = H_path
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
