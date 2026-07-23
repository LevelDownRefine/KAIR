import random
import numpy as np
import torch
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetDnPatch(BaseDataset):
    """Denoising dataset that pre-extracts H patches and corrupts them with fixed-sigma AWGN.

    All H patches are sampled once in ``__init__`` into ``self.H_data``. In train
    mode a patch is randomly flipped/rotated and corrupted with ``sigma/255``
    (PyTorch RNG) AWGN; in test mode the input image is degraded with seeded
    ``sigma_test/255`` numpy AWGN (deterministic, testable).
    """

    def __init__(self, opt):
        super(DatasetDnPatch, self).__init__(opt)
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma
        self.num_patches_per_image = opt['num_patches_per_image'] if opt['num_patches_per_image'] else 40
        self.num_sampled = opt['num_sampled'] if opt['num_sampled'] else 3000

        assert self.paths_H, 'Error: H path is empty.'

        self.num_sampled = min(self.num_sampled, len(self.paths_H))

        self.total_patches = self.num_sampled * self.num_patches_per_image
        self.H_data = np.zeros([self.total_patches, self.patch_size, self.patch_size, self.n_channels], dtype=np.uint8)
        self.update_data()

    def update_data(self):
        """Sample ``num_sampled`` images and extract ``num_patches_per_image`` random patches from each."""
        self.index_sampled = random.sample(range(0, len(self.paths_H), 1), self.num_sampled)
        n_count = 0
        for i in range(len(self.index_sampled)):
            H_patches = self.get_patches(self.index_sampled[i])
            for H_patch in H_patches:
                self.H_data[n_count, :, :, :] = H_patch
                n_count += 1

    def get_patches(self, index):
        """Read image ``index`` and crop ``num_patches_per_image`` random patches from it."""
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        H, W = img_H.shape[:2]
        H_patches = []
        num = self.num_patches_per_image
        for _ in range(num):
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            H_patches.append(img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :])
        return H_patches

    def _load_img_H(self, index):
        """Return the pre-extracted H patch (train) or read the full H image (test) from disk."""
        if self.opt['phase'] == 'train':
            return self.H_data[index]
        return util.imread_uint(self.paths_H[index], self.n_channels)

    def __len__(self):
        return len(self.H_data)

    def _make_sample(self, img_H, index):
        """Build (H, L): train augments a patch then adds torch AWGN(sigma); test adds seeded numpy AWGN(sigma_test)."""
        if self.opt['phase'] == 'train':
            mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            noise = torch.randn(img_L.shape).mul_(self.sigma / 255.0).numpy()
            img_L = img_L + noise
            return img_H, img_L
        else:
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)
            return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'L', 'H', 'L_path', 'H_path'}`` as float32 tensors."""
        if self.opt['phase'] == 'train':
            H_path = 'toy.png'
        else:
            H_path = self.paths_H[index]
        img_H = self._load_img_H(index)
        img_H, img_L = self._make_sample(img_H, index)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        L_path = H_path
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
