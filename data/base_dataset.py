import torch.utils.data as data
import utils.utils_image as util


class BaseDataset(data.Dataset):
    """Shared base for KAIR/BasicSR-style datasets.

    Subclasses store their own configuration in ``__init__`` and implement
    ``_make_sample(self, img_H, index)`` -- the deterministic core transform
    that turns one loaded H (high-quality) image into the (H, L, aux) sample.
    ``__getitem__`` only needs to call ``self._load_img_H`` then
    ``self._make_sample`` and wrap the result into tensors.

    Keeping the core transform in ``_make_sample`` makes it directly
    unit-testable: a test can call ``dataset._make_sample(known_H, 0)`` with a
    known array instead of going through ``__getitem__`` (and disk I/O), which
    is exactly what ``tests/data/test_dataset_*.py`` does.

    If a subclass has no core transform of its own (it only loads a sample
    as-is), it may omit ``_make_sample`` and do the load + tensor wrapping
    entirely inside ``__getitem__``; the base default raises
    ``NotImplementedError``.
    """

    def __init__(self, opt, n_channels_default=3):
        super().__init__()
        self.opt = opt
        self.n_channels = opt.get('n_channels') or n_channels_default
        self.paths_H = opt.get('paths_H') or util.get_image_paths(opt.get('dataroot_H'))
        self.paths_L = opt.get('paths_L') or util.get_image_paths(opt.get('dataroot_L'))

    def _load_img_H(self, index):
        """Read a uint8 HWC (RGB) high-quality image for ``index`` from disk."""
        return util.imread_uint(self.paths_H[index], self.n_channels)

    def _load_img_L(self, index):
        """Read a uint8 HWC (RGB) low-quality image for ``index`` from disk."""
        return util.imread_uint(self.paths_L[index], self.n_channels)

    def __len__(self):
        return len(self.paths_H or [])

    def _make_sample(self, img_H, index):
        raise NotImplementedError
