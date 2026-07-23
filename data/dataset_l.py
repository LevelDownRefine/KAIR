import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetL(BaseDataset):
    """Load L images in testing; only dataroot_L is needed, no H and no degradation."""

    def __init__(self, opt):
        super(DatasetL, self).__init__(opt)
        print('Read L in testing. Only "dataroot_L" is needed.')
        assert self.paths_L, 'Error: L paths are empty.'

    def __getitem__(self, index):
        """Return ``{'L', 'L_path'}`` as a float32 tensor; only L is loaded."""
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)
        img_L = util.uint2tensor3(img_L)
        return {'L': img_L, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_L)
