import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetDnCNN(BaseDataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__(opt)
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

    def _make_noisy(self, img, sigma, seed=None):
        """Add zero-mean AWGN of std ``sigma/255`` using ``np.random.normal``.

        seed given  -> ``np.random.seed(seed)`` is set first, so the noise is
                       reproducible (test path, exact-assertable in value tests)
        seed None   -> global RNG used as-is -> fresh random noise each call
                       (train path), aligned with the original implementation.
        """
        if seed is not None:
            np.random.seed(seed)
        return img + np.random.normal(0, sigma / 255.0, img.shape)

    def _make_sample(self, img_H, index):
        if self.opt['phase'] == 'train':
            # get L/H patch pairs
            H, W, _ = img_H.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation - flip, rotate
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            img_H = util.uint2single(patch_H)
            img_L = np.copy(img_H)
            img_L = self._make_noisy(img_L, self.sigma)
            return img_H, img_L
        else:
            # get L/H image pairs (deterministic -> directly testable)
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            img_L = self._make_noisy(img_L, self.sigma_test, seed=0)
            return img_H, img_L

    def __getitem__(self, index):
        H_path = self.paths_H[index] if self.paths_H is not None else ''
        img_H = self._load_img_H(index)
        img_H, img_L = self._make_sample(img_H, index)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': H_path}
