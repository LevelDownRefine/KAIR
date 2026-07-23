import random
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetPlain(BaseDataset):
    """Image-to-image mapping dataset: loads both L and H (paths_L and paths_H required)."""

    def __init__(self, opt):
        super(DatasetPlain, self).__init__(opt)
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def _make_sample(self, img_H, index):
        """Build (H, L, L_path): test returns the full paired images; train crops+augments a paired patch."""
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        if self.opt['phase'] == 'train':
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the L/H patch pair
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            img_L, img_H = patch_L, patch_H

        return img_H, img_L, L_path

    def __getitem__(self, index):
        """Return ``{'L', 'H', 'L_path', 'H_path'}`` as float32 tensors (uint8->tensor)."""
        H_path = self.paths_H[index] if self.paths_H is not None else ''
        img_H = self._load_img_H(index)
        img_H, img_L, L_path = self._make_sample(img_H, index)
        img_H, img_L = util.uint2tensor3(img_H), util.uint2tensor3(img_L)
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
