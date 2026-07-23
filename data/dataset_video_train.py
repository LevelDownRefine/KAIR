import random
import torch
from pathlib import Path
from torchvision import transforms

import utils.utils_video as utils_video
from data.base_dataset import BaseDataset


class VideoRecurrentTrainDataset(BaseDataset):
    """Video dataset for training recurrent networks, built from a meta_info txt.

    Each line of the meta file lists a clip folder, frame count, image shape and
    start frame. Frames are read through ``utils_video.FileClient``; ``__getitem__``
    selects a neighboring window, loads the LQ/GT frames and delegates the paired
    random crop + flip/rotate augmentation to ``_make_sample``.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.scale = opt['scale'] if opt['scale'] else 4
        self.gt_size = opt['gt_size'] if opt['gt_size'] else 256
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.filename_tmpl = opt['filename_tmpl'] if opt['filename_tmpl'] else '08d'
        self.filename_ext = opt['filename_ext'] if opt['filename_ext'] else 'png'
        self.num_frame = opt['num_frame']

        keys = []
        total_num_frames = []  # some clips may not have 100 frames
        start_frames = []  # some clips may not start from 00000
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _, start_frame = line.split(' ')
                keys.extend([f'{folder}/{i:{self.filename_tmpl}}' for i in range(int(start_frame), int(start_frame) + int(frame_num))])
                total_num_frames.extend([int(frame_num) for _ in range(int(frame_num))])
                start_frames.extend([int(start_frame) for _ in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['name'] == 'REDS':
            if opt['val_partition'] == 'REDS4':
                val_partition = ['000', '011', '015', '020']
            elif opt['val_partition'] == 'official':
                val_partition = [f'{v:03d}' for v in range(240, 270)]
            else:
                raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                                 f"Supported ones are ['official', 'REDS4'].")
        else:
            val_partition = []

        self.keys = []
        self.total_num_frames = []  # some clips may not have 100 frames
        self.start_frames = []
        if opt['test_mode']:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
        else:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] not in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list'] if opt['interval_list'] else [1]
        self.random_reverse = opt['random_reverse'] if opt['random_reverse'] else False
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def _make_sample(self, imgs_H, imgs_L, index, gt_path=None):
        """Paired random crop + flip/rotate the GT/LQ frame lists, then convert
        to (t,c,h,w) tensors stacked along the time dimension.

        ``imgs_H`` / ``imgs_L`` are the already-loaded lists of float32 GT/LQ
        frames; ``gt_path`` is only used for crop error messages.
        """
        img_gts, img_lqs = utils_video.paired_random_crop(imgs_H, imgs_L, self.gt_size, self.scale, gt_path)
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_results = utils_video.img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)
        return img_gts, img_lqs

    def __getitem__(self, index):
        """Load the neighboring LQ/GT window and return ``{L, H, key}`` tensors."""
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
                img_gt_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # crop + augment + tensorize, then build the sample dict
        img_gts, img_lqs = self._make_sample(img_gts, img_lqs, index, img_gt_path)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)


class VideoRecurrentTrainNonblindDenoisingDataset(VideoRecurrentTrainDataset):
    """Video dataset for training recurrent architectures in non-blind video denoising.

    GT frames are loaded and cropped/augmented; the LQ sequence is synthesized by
    adding AWGN whose level is sampled uniformly in [sigma_min, sigma_max], and the
    noise level is concatenated as an extra channel.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.sigma_min = self.opt['sigma_min'] / 255.
        self.sigma_max = self.opt['sigma_max'] / 255.

    def _make_sample(self, imgs_H, index, gt_path=None):
        """Crop+augment the GT frames, stack to (t,c,h,w), then synthesize LQ by
        adding AWGN(sigma in [sigma_min, sigma_max]); concat the noise level as an
        extra channel. ``imgs_H`` is the loaded list of float32 GT frames.
        """
        img_gts, _ = utils_video.paired_random_crop(imgs_H, imgs_H, self.gt_size, 1, gt_path)

        # augmentation - flip, rotate
        img_gts = utils_video.augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])

        img_gts = utils_video.img2tensor(img_gts)
        img_gts = torch.stack(img_gts, dim=0)

        # we add noise in the network
        noise_level = torch.empty((1, 1, 1, 1)).uniform_(self.sigma_min, self.sigma_max)
        noise = torch.normal(mean=0, std=noise_level.expand_as(img_gts))
        img_lqs = img_gts + noise

        t, _, h, w = img_lqs.shape
        img_lqs = torch.cat([img_lqs, noise_level.expand(t, 1, h, w)], 1)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return img_gts, img_lqs

    def __getitem__(self, index):
        """Load the GT window and return ``{L, H, key}`` tensors (L synthesized in-network)."""
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring GT frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            else:
                img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        img_gts, img_lqs = self._make_sample(img_gts, index, img_gt_path)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)


class VideoRecurrentTrainVimeoDataset(BaseDataset):
    """Vimeo90K dataset for training recurrent networks, built from a meta_info txt.

    Neighboring frame indices are precomputed from ``num_frame``; ``__getitem__``
    loads the window and delegates the paired crop + flip/rotate augmentation (and
    optional mirror/pad sequence expansion) to ``_make_sample``.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.temporal_scale = opt['temporal_scale'] if opt['temporal_scale'] else 1

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])][::self.temporal_scale]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        print(f'Random reverse is {self.random_reverse}.')

        self.mirror_sequence = opt['mirror_sequence'] if opt['mirror_sequence'] else False
        self.pad_sequence = opt['pad_sequence'] if opt['pad_sequence'] else False

    def _make_sample(self, imgs_H, imgs_L, index, gt_path=None):
        """Paired random crop + flip/rotate the LQ/GT window, convert to (t,c,h,w)
        tensors, then apply mirror/pad sequence expansion if configured.

        ``imgs_H`` / ``imgs_L`` are the loaded lists of float32 GT/LQ frames.
        """
        img_gts, img_lqs = utils_video.paired_random_crop(imgs_H, imgs_L, self.opt['gt_size'], self.opt['scale'], gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        n_lq = len(self.neighbor_list)
        img_lqs = torch.stack(img_results[:n_lq], dim=0)
        img_gts = torch.stack(img_results[n_lq:], dim=0)

        if self.mirror_sequence:  # mirror the sequence: 7 frames to 14 frames
            img_lqs = torch.cat([img_lqs, img_lqs.flip(0)], dim=0)
            img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)
        elif self.pad_sequence:  # pad the sequence: 7 frames to 8 frames
            img_lqs = torch.cat([img_lqs, img_lqs[-1:, ...]], dim=0)
            img_gts = torch.cat([img_gts, img_gts[-1:, ...]], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return img_gts, img_lqs

    def __getitem__(self, index):
        """Load the Vimeo window and return ``{L, H, key}`` tensors."""
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            # LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            # GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        img_gts, img_lqs = self._make_sample(img_gts, img_lqs, index, img_gt_path)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)


class VideoRecurrentTrainVimeoVFIDataset(VideoRecurrentTrainVimeoDataset):
    """Vimeo90K VFI training dataset: a window of LQ frames plus a single center GT.

    The LQ window and the single center GT are loaded and cropped/augmented
    together; ``_make_sample`` stacks them (GT last) and optionally applies the
    shared color-jitter transform.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.color_jitter = self.opt['color_jitter'] if self.opt['color_jitter'] else False

        if self.color_jitter:
            self.transforms_color_jitter = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)

    def _make_sample(self, imgs_H, imgs_L, index, gt_path=None):
        """Crop+augment the concatenated LQ window and center GT, convert to
        tensors, stack (GT last), then apply color jitter if configured.

        ``imgs_H`` is the single center GT frame (ndarray); ``imgs_L`` is the list
        of LQ float32 frames.
        """
        img_gts, img_lqs = utils_video.paired_random_crop([imgs_H], imgs_L, self.opt['gt_size'], self.opt['scale'], gt_path)

        # augmentation - flip, rotate
        img_lqs.extend([img_gts])
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        img_results = torch.stack(img_results, dim=0)

        if self.color_jitter:  # same color_jitter for img_lqs and img_gts
            img_results = self.transforms_color_jitter(img_results)

        img_lqs = img_results[:-1, ...]
        img_gts = img_results[-1:, ...]

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return img_gts, img_lqs

    def __getitem__(self, index):
        """Load the LQ window + center GT and return ``{L, H, key}`` tensors."""
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            # LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # GT
        if self.is_lmdb:
            img_gt_path = f'{clip}/{seq}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'

        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = utils_video.imfrombytes(img_bytes, float32=True)

        img_gts, img_lqs = self._make_sample(img_gt, img_lqs, index, img_gt_path)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}
