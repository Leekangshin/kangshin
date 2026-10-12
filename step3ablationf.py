import os
import re
import glob
import math
import random
import csv
import json
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm, clip_grad_norm_


# ==================================================
# Step3 dedicated ablation code: NF sweep for cGAN + MDFS
# ==================================================

F2_MAX = 300.0
DEFAULT_DATA_ROOT = "Dataset_MultiScale_512"
OUT_CHANNELS = 2

# ==================================================
# Dedicated ablation code: NF sweep for cGAN + MDFS
# Runs NF = 4, 8, 16, 32, 64, 128 sequentially.
# ==================================================
NF_LIST = [4, 8, 16, 32, 64, 128]
MODEL_TYPE = "cgan"
SCALING_MODE = "mdfs"
D_BASE = 64
DATA_ROOT = DEFAULT_DATA_ROOT
EPOCHS = 300
BATCH_SIZE = 8
NUM_WORKERS = 4
SEED = 42
AUTO_RESUME = False
LR_G = 2.0e-4
LR_D = 1.0e-5
LAMBDA_L1 = 130.0
FINE_TUNE_START = 221
LR_G_FINE = 5.0e-5
LR_D_FINE = 5.0e-6
LAMBDA_L1_FINE = 120.0
LINEAR_ED_MAX = "auto_p999"
PREVIEW_EVERY = 25

def parse_args(nf_value=64):
    return SimpleNamespace(
        model=MODEL_TYPE,
        scaling=SCALING_MODE,
        nf=nf_value,
        d_base=D_BASE,
        data_root=DATA_ROOT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        resume=AUTO_RESUME,
        lr_g=LR_G,
        lr_d=LR_D,
        lambda_l1=LAMBDA_L1,
        fine_start=FINE_TUNE_START,
        lr_g_fine=LR_G_FINE,
        lr_d_fine=LR_D_FINE,
        lambda_l1_fine=LAMBDA_L1_FINE,
        linear_ed_max=LINEAR_ED_MAX,
        preview_every=PREVIEW_EVERY,
    )

def make_task_name(args):
    return f"Cement_Q1_Ablation_{args.model.upper()}_{args.scaling.upper()}_NF{args.nf}"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    # DataLoader will pass worker_id. Keep deterministic but unique per worker.
    seed = 42 + worker_id
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


def compute_linear_ed_max(train_files, mode="auto_p999", cache_path=None):
    if mode not in ["auto_p999", "auto_max"]:
        return float(mode)

    values = []
    max_val = 0.0
    print(f"🔎 正在扫描训练集以确定线性能量缩放上限: {mode}")
    for i, fp in enumerate(train_files):
        arr = np.load(fp).astype(np.float32)
        raw_ed = np.clip(arr[:, :, 2], 0.0, None)
        max_val = max(max_val, float(np.max(raw_ed)))
        if mode == "auto_p999":
            # Use per-file high percentile to avoid loading all pixels into memory.
            values.append(float(np.percentile(raw_ed, 99.9)))
        if (i + 1) % 200 == 0:
            print(f"   scanned {i+1}/{len(train_files)} files")

    if mode == "auto_max":
        ed_max = max(max_val, 1e-6)
    else:
        ed_max = max(float(np.percentile(np.array(values), 95.0)), 1e-6)

    # Slight safety margin to reduce clipping.
    ed_max *= 1.05
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"linear_ed_max": ed_max, "mode": mode}, f, indent=2)
    print(f"✅ linear ED_MAX = {ed_max:.6g}")
    return ed_max


class ScalingHelper:
    def __init__(self, mode="mdfs", linear_ed_max=None):
        self.mode = mode
        self.linear_ed_max = linear_ed_max
        if self.mode == "linear" and (self.linear_ed_max is None or self.linear_ed_max <= 0):
            raise ValueError("linear scaling requires positive linear_ed_max")

    def encode_inputs(self, raw_density, raw_z, energy_keV):
        energy_channel = np.full_like(raw_density, fill_value=energy_keV, dtype=np.float32)
        if self.mode == "mdfs":
            norm_D = 2.0 * np.sqrt(np.clip(raw_density / 11.0, 0.0, 1.0)) - 1.0
            norm_Z = 2.0 * np.sqrt(np.clip(raw_z / 80.0, 0.0, 1.0)) - 1.0
            norm_E = 2.0 * np.sqrt(np.clip(energy_channel / 30.0, 0.0, 1.0)) - 1.0
        else:
            # Linear y = kx + b min-max scaling.
            norm_D = 2.0 * np.clip(raw_density / 11.0, 0.0, 1.0) - 1.0
            norm_Z = 2.0 * np.clip(raw_z / 80.0, 0.0, 1.0) - 1.0
            norm_E = 2.0 * np.clip(energy_channel / 30.0, 0.0, 1.0) - 1.0
        return np.stack([norm_D, norm_Z, norm_E], axis=0)

    def encode_targets(self, raw_ed, raw_percentile):
        if self.mode == "mdfs":
            gt_E = np.clip(20.0 * np.log(np.clip(raw_ed, 1e-9, None)) + 100.0, 0.0, 300.0) / 150.0 - 1.0
        else:
            gt_E = 2.0 * np.clip(raw_ed / self.linear_ed_max, 0.0, 1.0) - 1.0
        gt_P = np.clip(raw_percentile, 0.0, 100.0) / 50.0 - 1.0
        return np.stack([gt_E, gt_P], axis=0)

    def decode_energy_numpy(self, y_norm):
        if self.mode == "mdfs":
            f2 = (np.clip(y_norm, -1.0, 1.0) + 1.0) * (F2_MAX / 2.0)
            return np.exp((f2 - 100.0) / 20.0)
        return ((np.clip(y_norm, -1.0, 1.0) + 1.0) / 2.0) * self.linear_ed_max

    def decode_energy_torch(self, y_norm: torch.Tensor) -> torch.Tensor:
        if self.mode == "mdfs":
            mdfs = torch.clamp((y_norm + 1.0) * 150.0, min=0.0, max=300.0)
            return torch.exp((mdfs - 100.0) / 20.0)
        return ((torch.clamp(y_norm, -1.0, 1.0) + 1.0) / 2.0) * self.linear_ed_max

    def recover_density_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mdfs":
            return torch.clamp((x[:, 0:1] + 1.0) / 2.0, min=0.0, max=1.0) ** 2 * 11.0
        return torch.clamp((x[:, 0:1] + 1.0) / 2.0, min=0.0, max=1.0) * 11.0


class CementDataset(Dataset):
    def __init__(self, file_paths, scaler: ScalingHelper, is_train=True):
        self.file_paths = file_paths
        self.scaler = scaler
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

        x_input = self.scaler.encode_inputs(raw_density, raw_z, energy_keV)
        y_target = self.scaler.encode_targets(raw_ed, raw_percentile)

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
    def __init__(self, in_ch=3, out_ch=2, base_nf=64, max_ch=512):
        super().__init__()
        c1 = min(base_nf, max_ch)
        c2 = min(base_nf * 2, max_ch)
        c3 = min(base_nf * 4, max_ch)
        c4 = min(base_nf * 8, max_ch)
        c5 = min(base_nf * 16, max_ch)
        c6 = min(base_nf * 32, max_ch)
        c7 = min(base_nf * 64, max_ch)
        c8 = min(base_nf * 128, max_ch)
        self.channels = [c1, c2, c3, c4, c5, c6, c7, c8]

        self.d1 = DownBlock(in_ch, c1, normalize=False)
        self.d2 = DownBlock(c1, c2)
        self.d3 = DownBlock(c2, c3)
        self.d4 = DownBlock(c3, c4, dropout=0.5)
        self.d5 = DownBlock(c4, c5, dropout=0.5)
        self.d6 = DownBlock(c5, c6, dropout=0.5)
        self.d7 = DownBlock(c6, c7, dropout=0.5)
        self.d8 = DownBlock(c7, c8, normalize=False, dropout=0.5)

        self.u1 = UpBlock(c8, c7, dropout=0.5)
        self.u2 = UpBlock(c7 + c7, c6, dropout=0.5)
        self.u3 = UpBlock(c6 + c6, c5, dropout=0.5)
        self.u4 = UpBlock(c5 + c5, c4, dropout=0.0)
        self.u5 = UpBlock(c4 + c4, c3, dropout=0.0)
        self.u6 = UpBlock(c3 + c3, c2, dropout=0.0)
        self.u7 = UpBlock(c2 + c2, c1, dropout=0.0)
        self.final = nn.ConvTranspose2d(c1 + c1, out_ch, 4, stride=2, padding=1)

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
    def __init__(self, in_channels, base_nf=64):
        super().__init__()
        d1 = base_nf
        d2 = min(base_nf * 2, 512)
        d3 = min(base_nf * 4, 512)
        d4 = min(base_nf * 8, 512)

        def block(ic, oc, stride=2, norm=True):
            layers = [spectral_norm(nn.Conv2d(ic, oc, 4, stride, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, d1, norm=False),
            block(d1, d2),
            block(d2, d3),
            block(d3, d4, stride=1),
            spectral_norm(nn.Conv2d(d4, 1, 4, 1, 1)),
        )

    def forward(self, x, y):
        return self.model(torch.cat((x, y), 1))


def recover_solid_mask_from_x(x: torch.Tensor, scaler: ScalingHelper) -> torch.Tensor:
    density = scaler.recover_density_torch(x)
    return (density > 0.05).float()


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_preview_image(netG, val_loader, scaler, tracking_dir, epoch, preview_every, task_name, device):
    if epoch % preview_every != 0 and epoch != 1:
        return
    netG.eval()
    with torch.no_grad():
        for X_val, Y_val, file_path in val_loader:
            X_val = X_val.to(device)
            Y_val = Y_val.to(device)
            fake_Y = netG(X_val)
            x_np = X_val[0].cpu().numpy()
            y_true_np = Y_val[0].cpu().numpy()
            y_pred_np = fake_Y[0].cpu().numpy()
            basename = os.path.basename(file_path[0])
            break

    z_map = ((x_np[1] + 1.0) / 2.0)
    if scaler.mode == "mdfs":
        z_map = z_map ** 2 * 80.0
    else:
        z_map = z_map * 80.0
    gt_show = scaler.decode_energy_numpy(y_true_np[0])
    pr_show = scaler.decode_energy_numpy(y_pred_np[0])
    err_show = np.abs(pr_show - gt_show)
    vmax = max(np.max(gt_show), np.max(pr_show), 100.0)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    plt.suptitle(f"{task_name} | Epoch {epoch:03d} | {basename}", fontsize=14, fontweight="bold")
    axes[0].imshow(z_map, cmap="viridis")
    axes[0].set_title("Input: Z")
    axes[1].imshow(gt_show, cmap="inferno", vmax=vmax)
    axes[1].set_title("GT Energy")
    axes[2].imshow(pr_show, cmap="inferno", vmax=vmax)
    axes[2].set_title("Predicted")
    axes[3].imshow(err_show, cmap="magma")
    axes[3].set_title("Absolute Error")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(tracking_dir, f"epoch_{epoch:03d}.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_resume_checkpoint(path, epoch, netG, netD, optimizer_G, optimizer_D, best_val, args_dict):
    ckpt = {
        "epoch": epoch,
        "netG": netG.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "best_val_solid_mae": best_val,
        "args": args_dict,
    }
    if netD is not None:
        ckpt["netD"] = netD.state_dict()
    if optimizer_D is not None:
        ckpt["optimizer_D"] = optimizer_D.state_dict()
    torch.save(ckpt, path)


def try_resume(path, netG, netD, optimizer_G, optimizer_D, device):
    if not os.path.exists(path):
        return 1, 1e9
    ckpt = torch.load(path, map_location=device)
    netG.load_state_dict(ckpt["netG"])
    optimizer_G.load_state_dict(ckpt["optimizer_G"])
    if netD is not None and "netD" in ckpt:
        netD.load_state_dict(ckpt["netD"])
    if optimizer_D is not None and "optimizer_D" in ckpt:
        optimizer_D.load_state_dict(ckpt["optimizer_D"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_val = float(ckpt.get("best_val_solid_mae", 1e9))
    print(f"🔁 从 Epoch {start_epoch} 继续训练 | 当前 best_val_solid_mae = {best_val:.6f}")
    return start_epoch, best_val


def set_train_hyperparams(epoch, args, optimizer_G, optimizer_D):
    if epoch < args.fine_start:
        lr_g, lr_d, lambda_l1 = args.lr_g, args.lr_d, args.lambda_l1
    else:
        lr_g, lr_d, lambda_l1 = args.lr_g_fine, args.lr_d_fine, args.lambda_l1_fine
    for pg in optimizer_G.param_groups:
        pg["lr"] = lr_g
    if optimizer_D is not None:
        for pg in optimizer_D.param_groups:
            pg["lr"] = lr_d
    return lr_g, lr_d, lambda_l1


def train(args=None):
    if args is None:
        args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task_name = make_task_name(args)
    output_dir = f"Training_Outputs_{task_name}"
    tracking_dir = os.path.join(output_dir, "Tracking_Evolution")
    weight_dir = os.path.join(output_dir, "weights")
    csv_log_file = os.path.join(output_dir, "training_log.csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.data_root, "*", "*", "*.npy")))
    if not all_files:
        print("❌ 找不到数据！")
        return

    train_files, val_files, test_files = grouped_split(all_files, 0.7, 0.1, 0.2, seed=args.seed)
    linear_ed_max = None
    norm_json = os.path.join(output_dir, "normalization_config.json")
    if args.scaling == "linear":
        linear_ed_max = compute_linear_ed_max(train_files, mode=args.linear_ed_max, cache_path=norm_json)
    else:
        with open(norm_json, "w", encoding="utf-8") as f:
            json.dump({"scaling": "mdfs"}, f, indent=2)
    scaler = ScalingHelper(mode=args.scaling, linear_ed_max=linear_ed_max)

    train_loader = DataLoader(
        CementDataset(train_files, scaler=scaler, is_train=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        CementDataset(val_files, scaler=scaler, is_train=False),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=(device == "cuda"),
    )

    netG = UNetGenerator(base_nf=args.nf).to(device)
    netD = None if args.model == "cnn" else Discriminator(3 + OUT_CHANNELS, base_nf=args.d_base).to(device)
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_D = None if netD is None else optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    criterion_GAN = nn.MSELoss()

    resume_path = os.path.join(weight_dir, "resume_checkpoint.pth")
    best_ckpt_path = os.path.join(weight_dir, "best_checkpoint.pth")
    start_epoch, best_val = (1, 1e9)
    if args.resume:
        start_epoch, best_val = try_resume(resume_path, netG, netD, optimizer_G, optimizer_D, device)

    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config.update({
            "task_name": task_name,
            "num_train": len(train_files),
            "num_val": len(val_files),
            "num_test": len(test_files),
            "generator_params": count_parameters(netG),
            "discriminator_params": 0 if netD is None else count_parameters(netD),
            "linear_ed_max": linear_ed_max,
        })
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(csv_log_file, mode="a" if start_epoch > 1 else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if start_epoch == 1:
            writer.writerow([
                "Epoch", "Train_D", "Train_GAN", "Train_L1", "lrG", "lrD", "lambda_l1",
                "Val_Global_MAE", "Val_Solid_MAE", "Pore_MSE", "E_error_percent", "D90_error",
            ])

    print("==================================================")
    print(f"🚀 启动消融实验: {task_name}")
    print(f"   model={args.model}, scaling={args.scaling}, NF={args.nf}, D_base={args.d_base}")
    print(f"   G params={count_parameters(netG):,}; D params={0 if netD is None else count_parameters(netD):,}")
    print("==================================================")

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr_g, current_lr_d, current_lambda_l1 = set_train_hyperparams(epoch, args, optimizer_G, optimizer_D)
        netG.train()
        if netD is not None:
            netD.train()

        train_L1 = 0.0
        train_D = 0.0
        train_GAN = 0.0

        for X, Y, _ in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            if netD is not None:
                # Train discriminator.
                with torch.no_grad():
                    fake_for_d = netG(X)
                optimizer_D.zero_grad()
                pred_real = netD(X, Y)
                real_label = torch.empty_like(pred_real).uniform_(0.9, 1.0).to(device)
                loss_real = criterion_GAN(pred_real, real_label)
                pred_fake = netD(X, fake_for_d.detach())
                fake_label = torch.empty_like(pred_fake).uniform_(0.0, 0.1).to(device)
                loss_fake = criterion_GAN(pred_fake, fake_label)
                loss_D = 0.5 * (loss_real + loss_fake)
                loss_D.backward()
                clip_grad_norm_(netD.parameters(), max_norm=5.0)
                optimizer_D.step()
            else:
                loss_D = torch.tensor(0.0, device=device)

            # Train generator / CNN regressor.
            optimizer_G.zero_grad()
            fake_Y = netG(X)
            loss_L1 = torch.mean(torch.abs(fake_Y - Y))
            if netD is not None:
                pred_fake_G = netD(X, fake_Y)
                adv_target = torch.ones_like(pred_fake_G)
                loss_GAN = criterion_GAN(pred_fake_G, adv_target)
                loss_G = loss_GAN + current_lambda_l1 * loss_L1
            else:
                loss_GAN = torch.tensor(0.0, device=device)
                # Pure CNN / U-Net regression: no adversarial term.
                loss_G = loss_L1
            loss_G.backward()
            clip_grad_norm_(netG.parameters(), max_norm=5.0)
            optimizer_G.step()

            train_D += float(loss_D.item())
            train_GAN += float(loss_GAN.item())
            train_L1 += float(loss_L1.item())

        netG.eval()
        val_global_mae = 0.0
        val_solid_mae = 0.0
        val_pore_mse = 0.0
        val_energy_error = 0.0
        val_d90_sum_err = 0.0
        val_d90_valid_count = 0.0

        with torch.no_grad():
            for Xv, Yv, _ in val_loader:
                Xv = Xv.to(device)
                Yv = Yv.to(device)
                pred = netG(Xv)
                abs_diff = torch.abs(pred - Yv)
                val_global_mae += torch.mean(abs_diff).item()

                solid_mask = recover_solid_mask_from_x(Xv, scaler)
                pore_mask = 1.0 - solid_mask
                solid_error = torch.sum(solid_mask * abs_diff[:, 0:1]) / (torch.sum(solid_mask) + 1e-8)
                pore_mse = torch.sum(pore_mask * (pred[:, 0:1] - Yv[:, 0:1]) ** 2) / (torch.sum(pore_mask) + 1e-8)

                pred_E = scaler.decode_energy_torch(pred[:, 0:1]) * solid_mask
                gt_E = scaler.decode_energy_torch(Yv[:, 0:1]) * solid_mask
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
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"L_D: {avg_train_D:.4f} | L_GAN: {avg_train_GAN:.4f} | L1: {avg_train_l1:.4f} | "
            f"lrG: {current_lr_g:.1e} | lrD: {current_lr_d:.1e} | λL1: {current_lambda_l1:.1f}\n"
            f"   || 🧪 Val_Global_MAE: {avg_val_global:.4f} | "
            f"Val_Solid_MAE: {avg_val_solid:.4f} | Pore_MSE: {avg_val_pore:.4f}\n"
            f"   || ⚛️ E_error: {avg_val_energy:.2f}% | ⚛️ D90_error: {avg_val_d90:.2f} Pixels"
        )

        with open(csv_log_file, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, avg_train_D, avg_train_GAN, avg_train_l1, current_lr_g, current_lr_d,
                current_lambda_l1, avg_val_global, avg_val_solid, avg_val_pore, avg_val_energy, avg_val_d90,
            ])

        save_preview_image(netG, val_loader, scaler, tracking_dir, epoch, args.preview_every, task_name, device)
        save_resume_checkpoint(resume_path, epoch, netG, netD, optimizer_G, optimizer_D, best_val, vars(args))

        if avg_val_solid < best_val:
            best_val = avg_val_solid
            best_ckpt = {
                "epoch": epoch,
                "netG": netG.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "best_val_solid_mae": best_val,
                "args": vars(args),
            }
            if netD is not None:
                best_ckpt["netD"] = netD.state_dict()
                best_ckpt["optimizer_D"] = optimizer_D.state_dict()
            torch.save(best_ckpt, best_ckpt_path)
            torch.save(netG.state_dict(), os.path.join(weight_dir, "netG_BEST.pth"))
            if netD is not None:
                torch.save(netD.state_dict(), os.path.join(weight_dir, "netD_BEST.pth"))
            print("   🎯 当前实验最佳模型已更新！")

        if epoch % 25 == 0 or epoch == args.epochs:
            torch.save(netG.state_dict(), os.path.join(weight_dir, f"netG_epoch_{epoch:03d}.pth"))

    print(f"🎉 实验完成: {task_name}。请优先查看 netG_BEST.pth，而不是最后一轮。")


if __name__ == "__main__":
    for nf in NF_LIST:
        print("\n" + "=" * 90)
        print(f"🚀 NF 消融开始：NF={nf}")
        print("=" * 90)
        train(parse_args(nf))
