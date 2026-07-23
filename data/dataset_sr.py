import random
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetSR(BaseDataset):
    """SISR dataset (e.g. SRResNet): synthesizes L on the fly via bicubic downsample when paths_L is absent."""

    def __init__(self, opt):
        super(DatasetSR, self).__init__(opt)
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def _make_sample(self, img_H, index):
        """Build (H, L, L_path): test synthesizes L via bicubic; train crops+augments a patch pair."""
        L_path = None
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        return img_H, img_L, L_path

    def __getitem__(self, index):
        """Return ``{'L', 'H', 'L_path', 'H_path'}`` as float32 tensors."""
        H_path = self.paths_H[index] if self.paths_H is not None else ''
        img_H = self._load_img_H(index)
        img_H, img_L, L_path = self._make_sample(img_H, index)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        if L_path is None:
            L_path = H_path
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
