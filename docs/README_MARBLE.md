# 在 KAIR 中复现 MARBLE

本仓库现在提供 MARBLE 的独立图任务入口：读取原版导出的图与采样序列，
在 KAIR 的 PyTorch CUDA 13 环境中从随机初始权重训练，保存原版可读取的
检查点，最后拟合位置解码器。当前支持 **Achilles 单只大鼠、一阶梯度、
无扩散、完整邻居、无可训练内积变换**。不是 MARBLE 全部模型或全论文的复现。

## 环境和数据

KAIR 环境按 [CUDA 13 安装说明](README_CUDA13.md)建立：

```powershell
uv sync --locked --all-groups --python 3.12
```

构图与采样仍使用原版 PyG 的 CPU 环境。这避免用新的近邻图或不同采样器
代替原算法。先在 KAIR 的同级目录准备
[MARBLE fork](https://github.com/LevelDownRefine/MARBLE)，固定为
`872e46bd6dff2d092f8554a8c084701450e84904`，按其
[reproduction/README.md](https://github.com/LevelDownRefine/MARBLE/blob/872e46bd6dff2d092f8554a8c084701450e84904/reproduction/README.md)
运行 `setup.ps1` 和 `prepare_data.py`。首次 Windows 安装原版扩展需要 C++ Build Tools。
已配好的 MARBLE 环境可以直接复用，KAIR 训练环境无需安装旧版 PyG 或 CEBRA。

公开文件来自 Harvard Dataverse；原版准备工具按固定大小和 SHA-256 校验：

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| [rat_data.pkl](https://dataverse.harvard.edu/api/access/datafile/7609512) | 公开神经活动和行为标签 | `29dffd951853a98615827e7b8f3507907f9e0f0ef67e875410778ab4aadd42ab` |
| [marble_achilles_32D.pth](https://dataverse.harvard.edu/api/access/datafile/7659512) | 读取超参数；训练不加载作者权重 | `960e62bcbc275da79987331833ae5f1a5c4126b6d9c6b1e9e4e305406438a1ba` |

## 完整运行

以下命令均从 KAIR 根目录执行。输入导出目录和训练目录必须是新目录，脚本
遇到已有目录会报错，避免覆盖实验记录。可自行更换目录名称。

```powershell
# 1. 原版 CPU 构图、随机初始化、导出三个种子的 100 轮采样序列
uv run --locked python -m scripts.marble.prepare `
  --marble-repo ../MARBLE `
  --python ../MARBLE/reproduction/.venv/Scripts/python.exe `
  --output results/marble-input

# 2. KAIR CUDA 13 训练和位置解码
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = "$pwd/results/.matplotlib"
uv run --locked python main_train_marble.py `
  --opt options/marble/train_achilles.json `
  --input results/marble-input `
  --output results/marble-run

# 3. 原版 MARBLE 回读 KAIR 检查点，核对表征与原版 CEBRA kNN 解码
& ../MARBLE/reproduction/.venv/Scripts/python.exe -m scripts.marble.check_original `
  --marble-repo ../MARBLE `
  --input results/marble-input `
  --run results/marble-run
```

Linux 将 CPU Python 路径换成 `../MARBLE/reproduction/.venv/bin/python`，
命令按 shell 语法换行。已导出的目录可重复用于不同的新训练目录；每次训练
都会读取其随机初始权重、重新建立优化器，不会读取之前训练所得的最佳权重。
省略 `--output` 时训练入口创建带时间戳的目录。暂不提供中断续训入口。

## 固定实验协议

- 公开记录为 10,000 个时间 bin、120 个神经元；前 8,000 个 bin 训练，后
  2,000 个测试。PCA 仅在训练段拟合，维数 20；差分后 7,999/1,999 个节点。
- `pos` 是神经状态的 PCA 坐标，`x` 是神经活动的变化向量，均不是动物位置。
  行为标签只在表征训练完成后用于拟合和评价解码器。
- 原版导出固定梯度核、原始采样节点及初始化。重复节点不得去重：
  `K[:, ids] @ x[ids] = K @ (bincount(ids) * x)`，之后选取目标行。
- 特征包含坐标、变化向量和一阶梯度；MLP 隐藏层 64、输出 32，输出做 L2
  归一化。对比损失按原版的 anchor/positive/negative 三块计算。
- seeds 0、1、2，各 100 轮；SGD，lr=1，momentum=0.9；
  ReduceLROnPlateau 监测训练损失，最佳检查点由训练图内验证损失最小值选择。
  `loss_history.json` 中的 `test_loss` 是训练图内部的第三组节点，不能当作
  独立测试记录的解码指标。
- 36 邻居、余弦距离的位置回归。主结果 `eval` 冻结 BN；`notebook` 使用批次
  BN 统计量，单独报告。两种模式从同一个最佳状态开始，测试统计量不会写回检查点。
- 保存每个测试点实际选择的邻居编号。新旧 NumPy 在第 36 个邻居距离并列时
  可能选择不同节点。原版解码核验保持数值容差，并仅在证明所有替换节点都
  与边界距离完全相等时接受例外；例外点及预测偏差会明确记录。
- 训练前必须通过真实图上的三步原版 SGD 数值对照，容差为
  `rtol=2e-4, atol=2e-5`。关闭 TF32，但 CUDA 稀疏累加不保证逐位确定性。

## 代码与结果

| 文件 | 职责 |
| --- | --- |
| `data/dataset_marble.py` | 校验导出的图、标签、初始化与采样哈希 |
| `models/network_marble.py` | 原生稀疏图特征和与原版检查点兼容的编码器 |
| `models/model_marble.py` | 对比学习、数值核验、最佳模型选择、表征导出 |
| `main_train_marble.py` | 专用训练入口与实验记录 |
| `utils/utils_marble.py` | 位置解码、指标、哈希与绘图 |
| `scripts/marble/` | 原版数据准备及训练后独立核验 |

数据和模型已注册到 KAIR 的 `define_Dataset` / `define_Model` 工厂。
运行时使用 `main_train_marble.py`；图任务不能套用图像训练入口中的 L/H、
像素损失、PSNR 或 `tensor2uint` 流程。

每次训练目录保存参数、输入与源码哈希、数值核验、每个种子的损失曲线、
最佳/最后检查点、表征、预测、MAE/R² 和汇总。大文件位于 Git 忽略的 `results/`；
可提交的小型报告见 [MARBLE_REPRODUCTION.md](MARBLE_REPRODUCTION.md)。

```powershell
uv run --locked python -m pytest tests/test_marble.py -q
uv run --locked python -m pytest tests -q
```

## 解释边界

目前只有单只动物、一次时间划分和三个随机种子，不支持跨动物结论。
保留原实现的时间单位和 spike count 处理：25 ms 的 bin 索引被当作毫秒，
非零计数当作一次事件；PCA=20、lr=1 遵循本次 checkpoint 协议，与论文
Methods 或 notebook 的部分设置不同。测试图使用整个测试窗口，属于离线解码。
本次没有重新训练 CEBRA，也没有完成其他动物、扩散或内积不变特征等实验。

来源：[MARBLE 论文](https://www.nature.com/articles/s41592-024-02582-2)、
[已核验的 CUDA 后端](https://github.com/LevelDownRefine/MARBLE/blob/872e46bd6dff2d092f8554a8c084701450e84904/reproduction/src/modern_gpu.py)。
衍生代码的原许可证保留在 [licenses/MARBLE.txt](../licenses/MARBLE.txt)。
