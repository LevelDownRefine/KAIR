# KAIR `data/` 模块规则

本文件汇总 `data/` 下数据集类（`DatasetSR` / `DatasetDnCNN` / `DatasetUSRNet` /
`DatasetSRMD`）的设计约定、dtype 契约与测试规则。重构与新增数据集类时请照此执行。

## 1. 基类与样本构建

- `BaseDataset(data.Dataset)` 提供共享能力：`_load_img_H(index)`、`_load_img_L(index)`
  （读取 uint8 HWC RGB）、`__len__`、以及抽象方法 `_make_sample(img_H, index)`。
- 每个具体数据集类继承 `BaseDataset`，**只**实现 `_make_sample`；`__getitem__` 仅需
  `_load_img_H → _make_sample → single2tensor3` 三步走，不再重写退化逻辑。
- `_make_sample` 内联退化 / 加噪逻辑；**不要**在其中再抽 inner helper
  （如 `_degrade` / `_prepare_paired` / `_sample_kernel` 等）。train 分支只比 test
  分支多「随机裁剪 + 翻转/旋转增广」一段。
- `_make_sample` 直接接收已加载的 `img_H`（uint8），返回 `(H, L, *aux)`。
  把核心变换留在 `_make_sample` 内，使其可脱离磁盘 I/O 做数值单元测试。

## 2. opt 配置

- 子类 `__init__` 用 `opt['key']` 直接取，**不要用 `.get()`**；键缺失即 `KeyError`，无兜底。
  （`BaseDataset.__init__` 里的 `n_channels`/`paths_H`/`paths_L` 用了 `.get()` 且有默认值，
  那是基类通用项；各子类的专属配置键仍按"必填"处理。）
- 取值常用 `opt['key'] if opt['key'] else default` 形式：**键必须存在**，值为假时才回退默认值。
- `phase`（train/test）决定 `_make_sample` 走哪条分支。
- 各子类必填键：

  | 类 | 必填键（无 `.get()`） |
  | --- | --- |
  | `DatasetSR` | `scale`, `H_size` |
  | `DatasetDnCNN` | `H_size`, `sigma`, `sigma_test` |
  | `DatasetUSRNet` | `H_size`, `sigma_max`, `scales`, `sf_validation` |
  | `DatasetSRMD` | `scale`, `H_size`, `sigma`, `sigma_test` |

## 3. dtype 与退化契约

- 输入 `img_H` 为 uint8；`_make_sample` 内通常以 `util.uint2single` 转 single（float32）。
- `DatasetDnCNN._make_noisy`：`L = img + np.random.normal(0, sigma/255, shape)`。
  float32 图 + 默认 float64 噪声 → 返回 **float64**；下游 `single2tensor3` 再转 float32 tensor。
- `DatasetUSRNet`：退化**必须先作用在 uint8 上**（`ndimage.filters.convolve(patch_H, ...)`），
  再 `util.uint2single`；返回 `L, H, k, noise_level, sf`，其中 `k` 为 float32 tensor、
  `noise_level` 为 `torch.FloatTensor`。
- `DatasetSRMD`：`k = np.reshape(kernel, (-1), order='F')`；`k_reduced = np.dot(self.p, k)`
  （PCA 投影），转 float tensor。最终 `L` 通道数 = `3 + (len(k_reduced) + 1)`，
  多出的通道为退化图 `M`（reduced kernel + noise level 拼接）。
- `DatasetSR`：无 `paths_L` 时 `L` 由 `util.imresize_np(H, 1/sf, True)`（bicubic）合成；
  `H` 先 `util.modcrop(H, sf)`。

## 4. 测试规则

- **每类一个文件**：`test_dataset_sr.py` / `test_dataset_dncnn.py` /
  `test_dataset_usrnet.py` / `test_dataset_srmd.py`，另加 `test_base_dataset.py`。
- **数值验证**：直调 `dataset._make_sample(known_H, 0)`（用合成 `known_H` 绕过磁盘 I/O），
  用相同的 `util` / `utils_sisr` 原语独立复算，断言 `array_equal` / `allclose`，验证接线与顺序正确。
- **覆盖方法**：`__init__` / `__len__` / `_make_sample` / `__getitem__` 端到端；
  `DatasetDnCNN` 额外测 `_make_noisy`（seeded 精确复现、unseeded 加 AWGN）。
- **抽象测试**：用裸 `BaseDataset(opt)._make_sample(...)` 应 `raise NotImplementedError`
  （不要用 stub 子类覆盖后误判）。
- **kernel .mat 加载**：`USRNet` / `SRMD` 用 `os.path.join('kernels', ...)`，测试需
  `monkeypatch.chdir(ROOT)`（ROOT 指项目根）使相对路径可解析。
- **fixture**：`make_image_dir`（生成 n 张合成 PNG 目录）、`write_rgb_png`（写合成 RGB PNG），
  定义在 `tests/data/conftest.py`。
- 运行：`.venv/Scripts/python.exe -m pytest tests/data -k "not cuda" -q -p no:cacheprovider`。

## 5. 代码风格

- **docstring：函数文档风格**。
- **不留未使用的 import**：只 `import` 实际引用的名字。仅加载 kernel `.mat` 的文件需要
  `os` / `from pathlib import Path`（用于 `ROOT` chdir）；只消费 `monkeypatch` fixture 的测试文件
  不需要 `import pytest`。提交前用 AST / pyflakes 扫一遍新文件，删掉死 import。
