import random
import cv2
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetJPEG(BaseDataset):
    """JPEG compression-artifact reduction (deblocking) with a quality factor; only dataroot_H is needed."""

    def __init__(self, opt):
        super(DatasetJPEG, self).__init__(opt)
        print('Dataset: JPEG compression artifact reduction (deblocking) with quality factor. Only dataroot_H is needed.')
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 128
        self.quality_factor = opt['quality_factor'] if opt['quality_factor'] else 40
        self.quality_factor_test = opt['quality_factor_test'] if opt['quality_factor_test'] else 40
        self.is_color = opt['is_color'] if opt['is_color'] else False

    def _make_sample(self, img_H, index):
        """Build (H, L): test JPEG-compresses H (color or grayscale); train crops+augments then JPEG-compresses a patch."""
        if self.opt['phase'] == 'train':
            H, W = img_H.shape[:2]
            patch_size_plus = self.patch_size + 8

            # ---------------------------------
            # randomly crop a large patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - patch_size_plus))
            rnd_w = random.randint(0, max(0, W - patch_size_plus))
            patch_H = img_H[rnd_h:rnd_h + patch_size_plus, rnd_w:rnd_w + patch_size_plus, ...]

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            img_L = patch_H.copy()

            # ---------------------------------
            # set quality factor
            # ---------------------------------
            quality_factor = self.quality_factor

            if self.is_color:  # color image
                img_H = img_L.copy()
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
                _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 1)
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
            else:
                if random.random() > 0.5:
                    img_L = util.rgb2ycbcr(img_L)
                else:
                    img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2GRAY)
                img_H = img_L.copy()
                _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 0)

            # ---------------------------------
            # randomly crop a patch
            # ---------------------------------
            H, W = img_H.shape[:2]
            if random.random() > 0.5:
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
            else:
                rnd_h = 0
                rnd_w = 0
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            return img_H, img_L
        else:
            # ---------------------------------
            # set quality factor
            # ---------------------------------
            quality_factor = self.quality_factor_test

            if self.is_color:  # color JPEG image deblocking
                img_L = img_H.copy()
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
                _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 1)
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
            else:
                img_H = util.rgb2ycbcr(img_H)
                _, encimg = cv2.imencode('.jpg', img_H, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 0)
            return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'L', 'H', 'L_path', 'H_path'}`` as float32 tensors (uint8->tensor)."""
        H_path = self.paths_H[index] if self.paths_H is not None else ''
        L_path = H_path
        img_H = self._load_img_H(index)
        img_H, img_L = self._make_sample(img_H, index)
        img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
