import os
import re
import glob
import math
import random
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm, clip_grad_norm_


# ==================================================
# Linear scaling ablation version aligned with the current clean/balanced cGAN code
# 目标：只替换缩放逻辑为 y = kx + b 线性映射，其余训练框架与当前主模型对齐。
# 说明：训练阶段仍为 LSGAN + Global L1，不重新加入 physics-aware mask。
# ==================================================
TRAIN_TASK = "Cement_Q1_Ablation_cGAN_LinearScaling_Rerun300"
DATA_ROOT = "Dataset_MultiScale_512"
OUTPUT_DIR = f"Training_Outputs_{TRAIN_TASK}"
TRACKING_DIR = os.path.join(OUTPUT_DIR, "Tracking_Evolution")
WEIGHT_DIR = os.path.join(OUTPUT_DIR, "weights")
CSV_LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")

BATCH_SIZE = 8
EPOCHS = 300

# Stage 1: coarse-to-stable learning, keep the main energy field accurate.
LR_G = 2.0e-4
LR_D = 1.0e-5
LAMBDA_L1 = 130.0

# Stage 2: gentle refinement, not a long weak-D fine-tune.
FINE_TUNE_START = 221
LR_G_FINE = 5.0e-5
LR_D_FINE = 5.0e-6
LAMBDA_L1_FINE = 120.0

SPLIT_TRAIN = 0.7
SPLIT_VAL = 0.1
SPLIT_TEST = 0.2

NUM_WORKERS = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CHANNELS = 2

AUTO_RESUME = False  # 重新跑 300 轮，禁止自动接旧断点
RESUME_PATH = os.path.join(WEIGHT_DIR, "resume_checkpoint.pth")
BEST_CKPT_PATH = os.path.join(WEIGHT_DIR, "best_checkpoint.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRACKING_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)


def extract_energy_keV(file_path: str) -> float:
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(path_parts):
        m = re.match(r"(\d+)\s*keV", part, re.IGNORECASE)
        if m:
            return float(m.group(1))
    basename = os.path.basename(file_path)
    m = re.search(r"_E(\d+)keV", basename, re.IGNORECASE)
    if m:
        return float(m.group(1))
    raise ValueError(f"无法从路径解析能量信息: {file_path}")


def get_logic_name(file_path: str) -> str:
    path_parts = os.path.normpath(file_path).split(os.sep)
    if len(path_parts) >= 2:
        return path_parts[-2]
    return "Unknown"


def grouped_split(file_paths, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    rng = random.Random(seed)

    groups = {}
    for fp in file_paths:
        key = (extract_energy_keV(fp), get_logic_name(fp))
        groups.setdefault(key, []).append(fp)

    train_files, val_files, test_files = [], [], []
    for _, items in sorted(groups.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val >= n:
            n_train = max(1, int(math.floor(n * train_ratio)))
            n_val = max(1, int(math.floor(n * val_ratio)))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        train_files.extend(items[:n_train])
        val_files.extend(items[n_train:n_train + n_val])
        test_files.extend(items[n_train + n_val:])

    return train_files, val_files, test_files


# Linear scaling range for the main energy deposition field.
# This follows the provided linear baseline logic: raw energy [0, LINEAR_MAX_E] -> [-1, 1].
LINEAR_MAX_E = 400.0

def decode_energy(norm_val):
    return (np.clip(norm_val, -1.0, 1.0) + 1.0) * (LINEAR_MAX_E / 2.0)


class CementDataset(Dataset):
    def __init__(self, file_paths, is_train=True):
        self.file_paths = file_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path).astype(np.float32)
        energy_keV = extract_energy_keV(file_path)

        if self.is_train and np.random.rand() > 0.5:
            data = np.flip(data, axis=1).copy()

        raw_density = data[:, :, 0]
        raw_z = data[:, :, 1]
        raw_ed = data[:, :, 2]
        raw_percentile = data[:, :, 3]

        energy_channel = np.full_like(raw_density, fill_value=energy_keV, dtype=np.float32)

        # Linear ablation follows the provided baseline logic:
        # input density/Z keep sqrt compression to preserve light/heavy phase contrast,
        # incident energy uses simple linear mapping; output energy uses y = kx + b linear scaling.
        norm_D = 2.0 * np.sqrt(np.clip(raw_density / 11.0, 0.0, 1.0)) - 1.0
        norm_Z = 2.0 * np.sqrt(np.clip(raw_z / 80.0, 0.0, 1.0)) - 1.0
        norm_E = 2.0 * np.clip(energy_channel / 30.0, 0.0, 1.0) - 1.0
        x_input = np.stack([norm_D, norm_Z, norm_E], axis=0)

        gt_E = np.clip(raw_ed, 0.0, LINEAR_MAX_E) / (LINEAR_MAX_E / 2.0) - 1.0
        gt_P = np.clip(raw_percentile, 0.0, 100.0) / 50.0 - 1.0
        y_target = np.stack([gt_E, gt_P], axis=0)

        return (
            torch.tensor(x_input, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
            file_path,
        )


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


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        def block(ic, oc, stride=2, norm=True):
            layers = [spectral_norm(nn.Conv2d(ic, oc, 4, stride, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, 64, norm=False),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 1)),
        )

    def forward(self, x, y):
        return self.model(torch.cat((x, y), 1))


def recover_solid_mask_from_x(x: torch.Tensor) -> torch.Tensor:
    density = torch.clamp((x[:, 0:1] + 1.0) / 2.0, min=0.0, max=1.0) ** 2 * 11.0
    return (density > 0.05).float()


def mdfs_to_energy(y_norm: torch.Tensor) -> torch.Tensor:
    # Kept function name for compatibility with validation code; this version uses linear inverse scaling.
    return (torch.clamp(y_norm, -1.0, 1.0) + 1.0) * (LINEAR_MAX_E / 2.0)


def compute_d90_error(pred_energy: torch.Tensor, gt_energy: torch.Tensor, valid_threshold: float = 100.0):
    depth_profile_pred = torch.sum(pred_energy, dim=3)
    depth_profile_gt = torch.sum(gt_energy, dim=3)
    cumsum_pred = torch.cumsum(depth_profile_pred, dim=2)
    cumsum_gt = torch.cumsum(depth_profile_gt, dim=2)
    total_pred = cumsum_pred[:, :, -1:]
    total_gt = cumsum_gt[:, :, -1:]
    d90_pred = torch.argmax((cumsum_pred >= 0.90 * total_pred).float(), dim=2).float()
    d90_gt = torch.argmax((cumsum_gt >= 0.90 * total_gt).float(), dim=2).float()
    valid_mask = (total_gt.squeeze(-1) > valid_threshold).float()
    err_sum = torch.sum(torch.abs(d90_pred - d90_gt) * valid_mask)
    count = torch.sum(valid_mask)
    return err_sum.item(), count.item()


def save_preview_image(netG, val_loader, epoch):
    if epoch % 25 != 0 and epoch != 1:
        return

    netG.eval()
    with torch.no_grad():
        for X_val, Y_val, file_path in val_loader:
            X_val = X_val.to(DEVICE)
            Y_val = Y_val.to(DEVICE)
            fake_Y = netG(X_val)
            x_np = X_val[0].cpu().numpy()
            y_true_np = Y_val[0].cpu().numpy()
            y_pred_np = fake_Y[0].cpu().numpy()
            basename = os.path.basename(file_path[0])
            break

    z_map = ((x_np[1] + 1.0) / 2.0) ** 2 * 80.0
    gt_show = decode_energy(y_true_np[0])
    pr_show = decode_energy(y_pred_np[0])
    vmax = max(np.max(gt_show), np.max(pr_show), 100.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.suptitle(
        f"Epoch {epoch:03d} | Linear Scaling Ablation | cGAN | {basename}",
        fontsize=15,
        fontweight="bold",
    )
    axes[0].imshow(z_map, cmap="viridis")
    axes[0].set_title("Input: Z")
    axes[1].imshow(gt_show, cmap="inferno", vmax=vmax)
    axes[1].set_title("GT Energy")
    axes[2].imshow(pr_show, cmap="inferno", vmax=vmax)
    axes[2].set_title("AI Predicted")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(TRACKING_DIR, f"epoch_{epoch:03d}.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_resume_checkpoint(epoch, netG, netD, optimizer_G, optimizer_D, best_val):
    ckpt = {
        "epoch": epoch,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "best_val_solid_mae": best_val,
    }
    torch.save(ckpt, RESUME_PATH)


def try_resume(netG, netD, optimizer_G, optimizer_D):
    if not AUTO_RESUME or not os.path.exists(RESUME_PATH):
        return 1, 1e9

    ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
    netG.load_state_dict(ckpt["netG"])
    netD.load_state_dict(ckpt["netD"])
    optimizer_G.load_state_dict(ckpt["optimizer_G"])
    optimizer_D.load_state_dict(ckpt["optimizer_D"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_val = float(ckpt.get("best_val_solid_mae", 1e9))
    print(f"🔁 从 Epoch {start_epoch} 继续训练 | 当前 best_val_solid_mae = {best_val:.6f}")
    return start_epoch, best_val


def set_train_hyperparams(epoch, optimizer_G, optimizer_D):
    """Two-stage schedule for a fresh 300-epoch run."""
    if epoch < FINE_TUNE_START:
        lr_g, lr_d, lambda_l1 = LR_G, LR_D, LAMBDA_L1
    else:
        lr_g, lr_d, lambda_l1 = LR_G_FINE, LR_D_FINE, LAMBDA_L1_FINE

    for pg in optimizer_G.param_groups:
        pg["lr"] = lr_g
    for pg in optimizer_D.param_groups:
        pg["lr"] = lr_d
    return lr_g, lr_d, lambda_l1


def train():
    seed_everything(SEED)

    all_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*", "*", "*.npy")))
    if not all_files:
        print("❌ 找不到数据！")
        return

    train_files, val_files, _ = grouped_split(
        all_files,
        train_ratio=SPLIT_TRAIN,
        val_ratio=SPLIT_VAL,
        test_ratio=SPLIT_TEST,
        seed=SEED,
    )

    train_loader = DataLoader(
        CementDataset(train_files, is_train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        CementDataset(val_files, is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    netG = UNetGenerator().to(DEVICE)
    netD = Discriminator(3 + OUT_CHANNELS).to(DEVICE)

    optimizer_G = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

    criterion_GAN = nn.MSELoss()

    start_epoch, best_val = try_resume(netG, netD, optimizer_G, optimizer_D)
    if start_epoch > EPOCHS:
        print(f"✅ 当前断点已经到 Epoch {start_epoch - 1}，把 EPOCHS 改大就能继续。")
        return

    with open(CSV_LOG_FILE, mode="a" if start_epoch > 1 else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if start_epoch == 1:
            writer.writerow([
                "Epoch",
                "Train_L1",
                "Val_Global_MAE",
                "Val_Solid_MAE",
                "Pore_MSE",
                "E_error_percent",
                "D90_error",
            ])

    print("==================================================")
    print("🚀 启动 [Linear Scaling Ablation Re-run 300]：linear y=kx+b + GAN + Global L1，两阶段学习率与L1权重")
    print("==================================================")

    for epoch in range(start_epoch, EPOCHS + 1):
        current_lr_g, current_lr_d, current_lambda_l1 = set_train_hyperparams(epoch, optimizer_G, optimizer_D)

        netG.train()
        netD.train()

        train_L1 = 0.0
        train_D = 0.0
        train_GAN = 0.0

        for X, Y, _ in train_loader:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            fake_Y = netG(X)

            optimizer_D.zero_grad()
            pred_real = netD(X, Y)
            real_label = torch.empty_like(pred_real).uniform_(0.9, 1.0).to(DEVICE)
            loss_real = criterion_GAN(pred_real, real_label)

            pred_fake = netD(X, fake_Y.detach())
            fake_label = torch.empty_like(pred_fake).uniform_(0.0, 0.1).to(DEVICE)
            loss_fake = criterion_GAN(pred_fake, fake_label)

            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            clip_grad_norm_(netD.parameters(), max_norm=5.0)
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_Y = netG(X)
            pred_fake_G = netD(X, fake_Y)
            adv_target = torch.ones_like(pred_fake_G)
            loss_GAN = criterion_GAN(pred_fake_G, adv_target)
            loss_L1 = torch.mean(torch.abs(fake_Y - Y))

            loss_G = loss_GAN + current_lambda_l1 * loss_L1
            loss_G.backward()
            clip_grad_norm_(netG.parameters(), max_norm=5.0)
            optimizer_G.step()

            train_D += loss_D.item()
            train_GAN += loss_GAN.item()
            train_L1 += loss_L1.item()

        netG.eval()
        val_global_mae = 0.0
        val_solid_mae = 0.0
        val_pore_mse = 0.0
        val_energy_error = 0.0
        val_d90_sum_err = 0.0
        val_d90_valid_count = 0.0

        with torch.no_grad():
            for Xv, Yv, _ in val_loader:
                Xv = Xv.to(DEVICE)
                Yv = Yv.to(DEVICE)
                pred = netG(Xv)

                abs_diff = torch.abs(pred - Yv)
                val_global_mae += torch.mean(abs_diff).item()

                solid_mask = recover_solid_mask_from_x(Xv)
                pore_mask = 1.0 - solid_mask

                solid_error = torch.sum(solid_mask * abs_diff[:, 0:1]) / (torch.sum(solid_mask) + 1e-8)
                pore_mse = torch.sum(pore_mask * (pred[:, 0:1] - Yv[:, 0:1]) ** 2) / (torch.sum(pore_mask) + 1e-8)

                pred_E = mdfs_to_energy(pred[:, 0:1]) * solid_mask
                gt_E = mdfs_to_energy(Yv[:, 0:1]) * solid_mask

                sum_pred_E = torch.sum(pred_E, dim=(2, 3))
                sum_gt_E = torch.sum(gt_E, dim=(2, 3))
                e_err = torch.mean(torch.abs(sum_pred_E - sum_gt_E) / (sum_gt_E + 1.0))

                d90_err_sum, d90_count = compute_d90_error(pred_E, gt_E)

                val_solid_mae += solid_error.item()
                val_pore_mse += pore_mse.item()
                val_energy_error += e_err.item()
                val_d90_sum_err += d90_err_sum
                val_d90_valid_count += d90_count

        avg_train_D = train_D / max(len(train_loader), 1)
        avg_train_GAN = train_GAN / max(len(train_loader), 1)
        avg_train_l1 = train_L1 / max(len(train_loader), 1)

        avg_val_global = val_global_mae / max(len(val_loader), 1)
        avg_val_solid = val_solid_mae / max(len(val_loader), 1)
        avg_val_pore = val_pore_mse / max(len(val_loader), 1)
        avg_val_energy = (val_energy_error / max(len(val_loader), 1)) * 100.0
        avg_val_d90 = val_d90_sum_err / val_d90_valid_count if val_d90_valid_count > 0 else 0.0

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"L_D: {avg_train_D:.4f} | "
            f"L_GAN: {avg_train_GAN:.4f} | "
            f"L1: {avg_train_l1:.4f} | "
            f"lrG: {current_lr_g:.1e} | lrD: {current_lr_d:.1e} | λL1: {current_lambda_l1:.1f}\n"
            f"   || 🧪 Val_Global_MAE: {avg_val_global:.4f} | "
            f"Val_Solid_MAE: {avg_val_solid:.4f} | "
            f"Pore_MSE: {avg_val_pore:.4f}\n"
            f"   || ⚛️ E_error: {avg_val_energy:.2f}% | "
            f"⚛️ D90_error: {avg_val_d90:.2f} Pixels"
        )

        with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                avg_train_l1,
                avg_val_global,
                avg_val_solid,
                avg_val_pore,
                avg_val_energy,
                avg_val_d90,
            ])

        save_preview_image(netG, val_loader, epoch)
        save_resume_checkpoint(epoch, netG, netD, optimizer_G, optimizer_D, best_val)

        if avg_val_solid < best_val:
            best_val = avg_val_solid
            best_ckpt = {
                "epoch": epoch,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "best_val_solid_mae": best_val,
            }
            torch.save(best_ckpt, BEST_CKPT_PATH)
            torch.save(netG.state_dict(), os.path.join(WEIGHT_DIR, "netG_BEST.pth"))
            torch.save(netD.state_dict(), os.path.join(WEIGHT_DIR, "netD_BEST.pth"))
            print("   🎯 当前框架精简版最佳模型已更新！")

        if epoch % 25 == 0 or epoch == EPOCHS:
            torch.save(netG.state_dict(), os.path.join(WEIGHT_DIR, f"netG_epoch_{epoch:03d}.pth"))

    print("🎉 Linear Scaling Ablation Re-run 300 训练完成：请优先查看 netG_BEST.pth，而不是最后一轮")


if __name__ == "__main__":
    train()
