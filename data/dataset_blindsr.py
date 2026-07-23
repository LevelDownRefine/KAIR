import os
import random
import numpy as np
import utils.utils_image as util
from utils import utils_blindsr as blindsr
from data.base_dataset import BaseDataset


class DatasetBlindSR(BaseDataset):
    """Blind-SR dataset for BSRGAN / BSRGAN+ degradations (e.g. BSRGAN).

    Builds an L/H pair from one high-quality image using a stochastic
    degradation model (``degradation_bsrgan`` or ``degradation_bsrgan_plus``).
    In train mode a random crop plus flip/rotate augmentation is applied before
    the degradation; in test mode the whole image is degraded deterministically.
    """

    def __init__(self, opt):
        super(DatasetBlindSR, self).__init__(opt)
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize * self.sf
        assert self.paths_H, 'Error: H path is empty.'
        self.current_img_name = ''

    def _make_sample(self, img_H, index):
        """Build (H, L): tile small images, then (train) crop+augment, then apply the BSRGAN degradation."""
        # tile small images so they are at least patch_size on each side
        H, W, C = img_H.shape
        if H < self.patch_size or W < self.patch_size:
            img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8),
                            (self.patch_size, self.patch_size, 1))

        if self.opt['phase'] == 'train':
            H, W, C = img_H.shape
            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            if 'face' in self.current_img_name:
                mode = random.choice([0, 4])
            else:
                mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)

            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob,
                                                              use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)
        else:
            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob,
                                                              use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'L', 'H', 'L_path', 'H_path'}`` as float32 tensors."""
        H_path = self.paths_H[index] if self.paths_H is not None else ''
        self.current_img_name = os.path.splitext(os.path.basename(H_path))[0]
        img_H = self._load_img_H(index)
        img_H, img_L = self._make_sample(img_H, index)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        L_path = H_path
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
