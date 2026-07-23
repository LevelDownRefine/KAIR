import random
import numpy as np
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetFDnCNN(BaseDataset):
    """FDnCNN dataset: L = H + AWGN(sigma) concatenated with a noise-level map M.

    sigma is drawn per sample from [sigma_min, sigma_max] (train) or fixed at
    sigma_test (test, seeded for reproducibility); M is a constant map carrying
    the noise level so the network can condition on it.
    """

    def __init__(self, opt):
        super(DatasetFDnCNN, self).__init__(opt)
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else [0, 75]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 25

    def _make_sample(self, img_H, index):
        """Build (H, L): train crops+augments then adds AWGN(sigma) + M; test adds seeded AWGN(sigma_test) + M.

        L has one extra channel (the noise-level map M) appended to H's channels.
        The map is constant over space and equals sigma/255 (sigma_test/255 in test).
        """
        if self.opt['phase'] == 'train':
            # get L/H/M patch pairs
            H, W, _ = img_H.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation - flip, rotate
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            img_H = util.uint2single(patch_H)
            img_L = np.copy(img_H)
            noise_level = np.random.uniform(self.sigma_min, self.sigma_max) / 255.0
            noise = (np.random.randn(*img_L.shape).astype(np.float32)) * np.float32(noise_level)
            img_L += noise
        else:
            # get L/H/M image pairs (deterministic -> directly testable)
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)
            noise_level = self.sigma_test / 255.0

        # concat L and noise level map M
        H, W, _ = img_L.shape
        noise_level_map = (np.ones((H, W, 1), dtype=np.float32) * np.float32(noise_level))
        img_L = np.concatenate([img_L, noise_level_map], axis=2)
        return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'L', 'H', 'L_path', 'H_path'}`` as float32 tensors."""
        H_path = self.paths_H[index] if self.paths_H is not None else ''
        img_H = self._load_img_H(index)
        img_H, img_L = self._make_sample(img_H, index)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        return {'L': img_L, 'H': img_H, 'L_path': H_path, 'H_path': H_path}
