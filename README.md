# CBM 能量沉积模拟与 cGAN 预测代码

整理自现有 CBM 项目的核心 Python 脚本，覆盖随机微结构、电子输运与能量沉积、条件生成网络训练、预测预览和消融实验。
本次整理保留原脚本计算逻辑与默认参数；将原文件 `step3train300 .py` 重命名为 `step3train300.py`。

## 文件与运行顺序

| 文件 | 用途 |
| --- | --- |
| `step1gpu.py` | CUDA 批量生成四通道训练数据 |
| `step1single.py` | CPU 单样本模拟与预览，独立示例 |
| `step2.py` | 四通道数据预览与 MDFS/线性映射比较 |
| `step3train300.py` | 当前 BalancedRerun300 cGAN 训练 |
| `step4blackbox.py` | 加载 BalancedRerun300 权重并预览预测 |
| `step3train.py` | 较早的 200 轮训练版本，供比较 |
| `step3ablationf.py` | NF=4、8、16、32、64、128 容量消融 |
| `step3ablationpurecnn.py` | 纯 CNN 消融 |
| `step3ablationlinearscaling.py` | 线性缩放消融 |
| `step3ablation_arch_schedule.py` | 跳跃连接与训练阶段消融 |

## 环境与安装

建议创建独立 Python 环境，再安装依赖：

```sh
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`step1gpu.py` 需要可用的 NVIDIA CUDA 环境。PyTorch 的 CUDA 构建需要与机器环境匹配；
`requirements.txt` 列出直接依赖，未锁定历史实验的精确版本。

所有命令均在仓库根目录执行：

```sh
# 独立的 CPU 单样本示例（不会创建完整训练集）
python step1single.py

# 完整流程：生成数据 → 预览 → 训练 → 推断
python step1gpu.py
python step2.py
python step3train300.py
python step4blackbox.py
```

批量生成默认运行 10 个能级 × 2 种几何 × 300 样本，共 6000 个 512×512 样本，
每个样本模拟 100000 个电子。先检查 `step1gpu.py` 末尾的能级、样本数、分辨率与进程数，再决定是否运行完整任务。
训练默认 300 轮、batch size 8、种子 42，按能级和几何类型分组进行 70%/10%/20% 划分。
生成器使用与时间有关的随机种子，不能保证重新生成数据时逐位一致。

## 数据与权重

数据目录：`Dataset_MultiScale_512/<能量>keV/<Scatter或DigHole>/*.npy`。
数组为 float32，形状 `(H, W, 4)`：

| 通道 | 内容 |
| --- | --- |
| 0 | 组分加权密度 |
| 1 | 组分加权原子序数 |
| 2 | 累计沉积能量除以像素面积，并乘以固体掩码 |
| 3 | 按像素沉积能量降序累计得到的百分比场，经过平滑并将非固体区设为 100 |

通道 2 的代码分母是 `pixel_size**2`；原脚本部分标签写作 eV/nm³，
本整理版未修改计算，因此使用时应根据模型定义核对单位。
该百分比场不是逐像素概率密度。

网络输入为密度、原子序数、从路径解析的入射能量三通道；输出为 MDFS 能量映射和累计百分比两通道。
MDFS 映射：`clip(20*ln(max(ED, 1e-9))+100, 0, 300)`。

推断默认权重：
`Training_Outputs_Cement_Q1_ThreeCondition_cGAN_BalancedRerun300/weights/netG_BEST.pth`。
可将已有数据和对应权重放在这些相对路径，或先运行模拟与训练产生它们。
数据集、训练权重和论文原稿不包含在本代码包中；忽略规则已排除它们。

## 消融实验

```sh
python step3ablationf.py
python step3ablationpurecnn.py
python step3ablationlinearscaling.py
python step3ablation_arch_schedule.py --dry-run --nf 16
python step3ablation_arch_schedule.py --variant all --nf 16 --epochs 300 --no-resume
```

NF 脚本默认依次训练六种宽度，其余脚本参数主要位于文件顶部。
`step4blackbox.py` 从整个数据目录随机选取样本用于预览，不构成独立测试集的性能报告。
CPU 单样本脚本与 CUDA 批量脚本的几何、剂量和画布配置不同，不能当作逐样本等价实现。

## 整理与验证记录

`SOURCE_MANIFEST.json` 记录原文件名、文件大小和 SHA-256。十个脚本均与来源逐字节一致。
本次进行语法编译检查；额外执行的轻量验证见 `VALIDATION.md`。
完整 6000 样本生成、300 轮训练和论文结果复现没有在本次整理中重新运行。
未添加开源许可；发布公开版本前由作者确定授权方式。
