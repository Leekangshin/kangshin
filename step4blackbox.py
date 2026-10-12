import os
import glob
import re
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ==============================================================================
# 1. 配置区 (🌟 彻底对齐 BalancedRerun300 文件夹)
# ==============================================================================
TRAIN_TASK = "Cement_Q1_ThreeCondition_cGAN_BalancedRerun300" 
DATA_ROOT = "Dataset_MultiScale_512"

WEIGHT_PATH = os.path.join(f"Training_Outputs_{TRAIN_TASK}", "weights", "netG_BEST.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CHANNELS = 2

# ==============================================================================
# 2. 工具函数 & 预处理
# ==============================================================================
def extract_energy_keV(file_path: str) -> float:
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(path_parts):
        m = re.match(r"(\d+)\s*keV", part, re.IGNORECASE)
        if m: return float(m.group(1))
    basename = os.path.basename(file_path)
    m = re.search(r"_E(\d+)keV", basename, re.IGNORECASE)
    if m: return float(m.group(1))
    return 12.0

# ==============================================================================
# 3. 网络结构 (🌟 完美复刻 A版 的 DownBlock, UpBlock 和 InstanceNorm2d)
# ==============================================================================
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat((x, skip), dim=1)

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        self.d1 = DownBlock(in_ch, 64, normalize=False)
        self.d2 = DownBlock(64, 128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512, dropout=0.5)
        self.d5 = DownBlock(512, 512, dropout=0.5)
        self.d6 = DownBlock(512, 512, dropout=0.5)
        self.d7 = DownBlock(512, 512, dropout=0.5)
        self.d8 = DownBlock(512, 512, normalize=False, dropout=0.5)

        self.u1 = UpBlock(512, 512, dropout=0.5)
        self.u2 = UpBlock(1024, 512, dropout=0.5)
        self.u3 = UpBlock(1024, 512, dropout=0.5)
        self.u4 = UpBlock(1024, 512, dropout=0.0)
        self.u5 = UpBlock(1024, 256, dropout=0.0)
        self.u6 = UpBlock(512, 128, dropout=0.0)
        self.u7 = UpBlock(256, 64, dropout=0.0)

        # 🌟 完美还原：带 nn.Sequential 和 nn.Identity 的输出层
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, 4, stride=2, padding=1),
            nn.Identity(),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)

        return self.final(u7)

# ==============================================================================
# 4. 主程序：加载数据、推断并画灰度图
# ==============================================================================
def show_grayscale_pred_comparison():
    print("🔍 启动 [纯灰度 Log 放缩] 预测对比观测器...")
    
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 找不到权重文件！请检查路径：{WEIGHT_PATH}")
        return

    netG = UNetGenerator(in_ch=3, out_ch=OUT_CHANNELS).to(DEVICE)
    netG.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE), strict=True)
    netG.eval()

    all_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*", "*", "*.npy")))
    # 随机挑一个测试样本 (为了方便观察，可以多跑几次)
    file_path = np.random.choice(all_files)
    basename = os.path.basename(file_path)
    
    data = np.load(file_path).astype(np.float32)
    energy_keV = extract_energy_keV(file_path)

    raw_density = data[:, :, 0]
    raw_z = data[:, :, 1]
    raw_ed = data[:, :, 2]
    
    # 屏蔽真空区背景底噪
    solid_mask = (raw_density > 0.05).astype(np.float32)
    
    # --- 准备网络输入 ---
    energy_channel = np.full_like(raw_density, fill_value=energy_keV, dtype=np.float32)
    norm_D = 2.0 * np.sqrt(np.clip(raw_density / 11.0, 0.0, 1.0)) - 1.0
    norm_Z = 2.0 * np.sqrt(np.clip(raw_z / 80.0, 0.0, 1.0)) - 1.0
    norm_E = 2.0 * np.sqrt(np.clip(energy_channel / 30.0, 0.0, 1.0)) - 1.0
    x_input = np.stack([norm_D, norm_Z, norm_E], axis=0)
    
    X_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # --- 网络推断 ---
    with torch.no_grad():
        fake_Y = netG(X_tensor)
        pred_y_norm = fake_Y[0, 0].cpu().numpy()

    # ==========================================
    # 物理还原：将真值和预测都转回 MDFS 空间 (Log 放缩)
    # ==========================================
    # 1. MC 真值 MDFS
    raw_ed_clipped = np.clip(raw_ed, 1e-9, None)
    gt_mdfs = 20.0 * np.log(raw_ed_clipped) + 100.0
    gt_mdfs = np.clip(gt_mdfs, 0.0, 300.0) * solid_mask

    # 2. AI 预测 MDFS
    pred_mdfs = np.clip((pred_y_norm + 1.0) * 150.0, 0.0, 300.0) * solid_mask

    # ==========================================
    # 🎨 绘图：纯粹的黑白对比
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Grayscale Log-Scaled (MDFS) Comparison\nFile: {basename}", fontsize=16, fontweight='bold')

    # 图 1：MC 真值 (黑白 Log)
    im0 = axes[0].imshow(gt_mdfs, cmap='gray', aspect='equal', vmin=0, vmax=300)
    axes[0].set_title("1. Original MC Simulation (Log-Scaled MDFS)", fontsize=14, pad=10)
    axes[0].axis("off")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("MDFS Value", fontsize=12)

    # 图 2：AI 预测 (黑白 Log)
    im1 = axes[1].imshow(pred_mdfs, cmap='gray', aspect='equal', vmin=0, vmax=300)
    axes[1].set_title("2. AI Prediction Result (Log-Scaled MDFS)", fontsize=14, pad=10)
    axes[1].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("MDFS Value", fontsize=12)

    plt.tight_layout()
    
    save_path = f"Grayscale_Pred_{basename.replace('.npy', '.pdf')}"
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"🎉 灰度预测对比图已保存为: {save_path}")

if __name__ == "__main__":
    show_grayscale_pred_comparison()