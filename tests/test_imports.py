"""Import smoke test for the core (non-CUDA-extension) modules.

Every module listed below must import cleanly. This guards two things the
code review flagged:

* The dependency fixes in ``pyproject.toml`` (``scipy`` / ``matplotlib`` /
  ``numpy``) actually resolve — e.g. ``utils.utils_image`` imports
  ``matplotlib`` at module top level, so a missing install crashes here.
* No import-time breakage from the numpy 2.x / Python 3.12 upgrade
  (e.g. ``distutils`` usage, removed aliases).

Modules that require the JIT-compiled CUDA extensions
(``deform_attn`` / ``upfirdn2d`` / ``fused_act``) — ``network_vrt``,
``network_rvrt``, ``network_faceenhancer``, ``model_vrt`` — are
*intentionally excluded*: they need the CUDA 12.8 toolkit (``nvcc``) and
are tracked separately, not as plain import smoke tests.
"""
import importlib

import pytest

SAFE_MODULES = [
    # utils
    "utils.utils_image",
    "utils.utils_option",
    "utils.utils_model",
    # data
    "data.dataset_dncnn",
    "data.dataset_sr",
    "data.dataset_srmd",
    "data.dataset_usrnet",
    "data.dataset_ffdnet",
    "data.dataset_fdncnn",
    "data.dataset_plain",
    "data.dataset_dnpatch",
    "data.dataset_l",
    "data.dataset_plainpatch",
    "data.dataset_jpeg",
    "data.dataset_blindsr",
    "data.dataset_dpsr",
    "data.dataset_video_test",
    "data.dataset_video_train",
    # models - networks (CUDA-extension modules excluded)
    "models.network_dncnn",
    "models.network_dpsr",
    "models.network_ffdnet",
    "models.network_imdn",
    "models.network_msrresnet",
    "models.network_rrdb",
    "models.network_rrdbnet",
    "models.network_srmd",
    "models.network_swinir",
    "models.network_unet",
    "models.network_usrnet",
    "models.network_usrnet_v1",
    "models.network_feature",
    "models.network_discriminator",
    # models - model wrappers
    "models.model_base",
    "models.model_plain",
    "models.model_plain2",
    "models.model_plain4",
    "models.model_gan",
]


@pytest.mark.parametrize("module_name", SAFE_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
