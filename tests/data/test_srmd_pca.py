import os

import hdf5storage
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KERNELS = os.path.join(ROOT, "kernels")
PYTORCH_MAT = os.path.join(KERNELS, "srmd_pca_pytorch.mat")
MATLAB_MAT = os.path.join(KERNELS, "srmd_pca_matlab.mat")


def _load_p():
    return hdf5storage.loadmat(PYTORCH_MAT)["p"]


def _load_P():
    return hdf5storage.loadmat(MATLAB_MAT)["P"]


def test_pytorch_mat_matches_matlab_source():
    """srmd_pca_pytorch.mat 是从 srmd_pca_matlab.mat 的 'P' 转存为 'p'，数值须逐元素一致。"""
    p = _load_p()
    P = _load_P()
    assert p.shape == (15, 225), p.shape
    assert P.shape == (15, 225), P.shape
    max_diff = float(np.abs(p - P).max())
    print(f"[srmd_pca] shape={p.shape} dtype={p.dtype} max|P-p|={max_diff:.3e}")
    assert np.allclose(p, P, atol=1e-9), f"与源文件数值不一致，max diff = {max_diff}"


def test_p_is_orthonormal_pca_basis():
    """PCA 基向量应互相正交且单位范数：p @ p.T ≈ I_15。"""
    p = _load_p()
    gram = p @ p.T
    off = float(np.abs(gram - np.eye(15)).max())
    print(f"[srmd_pca] ||p@p.T - I||_max = {off:.3e}")
    assert np.allclose(gram, np.eye(15), atol=1e-6), f"非正交 PCA 基，max off = {off}"


def test_projection_matches_dataset_code_path():
    """复刻 dataset_srmd 的用法：各向异性核 + order='F' 扁平化，投影应得 15 维且可低误差重建。"""
    from utils import utils_sisr

    p = _load_p()
    kernel = utils_sisr.anisotropic_Gaussian(ksize=15, theta=np.pi / 3, l1=4.0, l2=2.0)
    k = np.reshape(kernel, (-1), order="F")  # 225 维
    k_reduced = np.dot(p, k)  # 15 维 PCA 系数
    assert k_reduced.shape == (15,), k_reduced.shape
    k_recon = p.T @ k_reduced  # 投影回 225 维
    rel_err = float(np.linalg.norm(k - k_recon) / np.linalg.norm(k))
    print(f"[srmd_pca] 投影维度={k_reduced.shape[0]} 重建相对误差={rel_err:.4f}")
    assert rel_err < 0.2, f"PCA 重建相对误差过大: {rel_err:.4f}"
