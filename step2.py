import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# ==============================================================================
# 🎨 1. 顶刊级色棒定义 (MDFS 专用)
# ==============================================================================
Z_COLORS = ['#202c81', '#009ee5', '#00d559', '#e8f877', '#9e7b64', '#ffffff']
CMAP_Z = mcolors.LinearSegmentedColormap.from_list('custom_Z', Z_COLORS)

PERCENTAGE_COLORS = ['#FFA500', '#00FF00', '#00BFFF', '#A020F0', '#FF0000', '#FFFFFF']
PERCENTAGE_BOUNDS = [0, 10, 30, 50, 70, 90, 101]
CMAP_PERCENTAGE = mcolors.ListedColormap(PERCENTAGE_COLORS)
NORM_PERCENTAGE = mcolors.BoundaryNorm(PERCENTAGE_BOUNDS, CMAP_PERCENTAGE.N)

# 纯粹的水彩热力图 (不做任何人为压暗处理)
def get_watercolor_turbo():
    base_turbo = plt.get_cmap('turbo')
    nodes = [0.00, 0.01, 0.20, 0.40, 0.60, 0.80, 1.00]
    colors = [
        (1.0, 1.0, 1.0, 1.0),  
        (1.0, 1.0, 1.0, 1.0),  
        base_turbo(0.15),      
        base_turbo(0.35),      
        base_turbo(0.55),      
        base_turbo(0.75),      
        base_turbo(1.00)       
    ]
    return LinearSegmentedColormap.from_list("WatercolorTurbo", list(zip(nodes, colors)))

CMAP_ENERGY = get_watercolor_turbo()

# ==============================================================================
# 📂 2. 数据读取与 K-O 物理尺度反推
# ==============================================================================
DATA_ROOT = "Dataset_MultiScale_512" 

def view_pure_sample():
    print("🔍 启动 [绝对生肉版：无高斯模糊] 阅片器...")
    
    all_files = glob.glob(os.path.join(DATA_ROOT, "*", "*", "*.npy"))
    if not all_files:
        return print(f"❌ 在 {DATA_ROOT} 中没有找到数据！请确保生成脚本已经跑出结果。")

    np.random.seed() 
    selected_file = np.random.choice(all_files)
    sample_data = np.load(selected_file)
    filename = os.path.basename(selected_file)
    resolution = sample_data.shape[0]
    
    logic_match = re.search(r'_(Scatter|DigHole)_', filename)
    logic_type = logic_match.group(1) if logic_match else "Unknown"

    match = re.search(r'_E(\d+)keV', filename)
    if match:
        energy_keV = float(match.group(1))
        canvas_nm = 30.0 * (energy_keV ** 1.67) 
        pixel_nm = canvas_nm / resolution
        scale_info = f"Beam: {int(energy_keV)} keV | Canvas: {canvas_nm:.1f} nm | Pixel: {pixel_nm:.2f} nm/px"
    else:
        scale_info = "Scale: Unknown"

    print(f"✅ 已加载: {filename} ({logic_type} 模式)\n📐 物理尺度: {scale_info}")

    # ==============================================================================
    # 🧠 3. 绝对真实的物理数据提取 (🌟 剥除全部模糊滤镜)
    # ==============================================================================
    # 彻底去掉 scipy.ndimage.gaussian_filter，直接使用原始张量
    density_raw = sample_data[:, :, 0]
    z_raw = sample_data[:, :, 1]
    percentage_raw = sample_data[:, :, 3]

    raw_energy = sample_data[:, :, 2]
    
    # 仅做底噪截断，防止 log(0) 报错，但不做任何空间平滑
    raw_energy_clipped = np.clip(raw_energy, 1e-9, None)
    
    # 原汁原味：严格执行论文公式 f2 = 20 * ln(E) + 100
    mdfs_val = 20.0 * np.log(raw_energy_clipped) + 100.0
    mdfs_val = np.clip(mdfs_val, 0.0, 300.0)

    # 对比展示空间 (线性 vs MDFS)
    gray_linear_255 = np.clip((raw_energy_clipped / 3000.0) * 255.0, 0, 255.0)
    gray_log_255 = np.clip((mdfs_val / 300.0) * 255.0, 0, 255.0)

    # ==============================================================================
    # 🎨 4. 开始绘图 (2x3 矢量六宫格)
    # ==============================================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
    
    fig.suptitle(f"Raw Discrete Viewer (No Blur) | Logic: {logic_type} | {scale_info}\nFile: {filename}", fontsize=18, fontweight='bold', y=0.98)

    def setup_ax(ax, title, border_color='none'):
        ax.axis('off'); ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        if border_color != 'none':
            ax.add_patch(patches.Rectangle((0,0), 1, 1, transform=ax.transAxes, linewidth=4, edgecolor=border_color, facecolor='none'))

    # 第一排
    ax = axes[0, 0]; im0 = ax.imshow(density_raw, cmap='terrain', aspect='equal'); setup_ax(ax, "Ch0: Raw Density ($g/cm^3$)"); plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[0, 1]; im1 = ax.imshow(z_raw, cmap=CMAP_Z, aspect='equal', vmin=0, vmax=80); setup_ax(ax, "Ch1: Raw Atomic Number (Z)"); plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[0, 2]; im3 = ax.imshow(np.ma.masked_where(percentage_raw >= 99.9, percentage_raw), cmap=CMAP_PERCENTAGE, norm=NORM_PERCENTAGE, aspect='equal'); setup_ax(ax, "Ch3: Energy Percent (%)"); ax.set_facecolor('#EEEEEE'); plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, ticks=PERCENTAGE_BOUNDS)

    # 第二排
    ax = axes[1, 0]; im2 = ax.imshow(mdfs_val, cmap=CMAP_ENERGY, aspect='equal', vmin=0, vmax=300); setup_ax(ax, "Ch2: Raw MDFS Energy (Unsmoothed)")
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    tick_eV = [0.01, 0.1, 1, 10, 100, 1000, 3000, 10000]
    cbar2.set_ticks([20.0 * np.log(v) + 100.0 for v in tick_eV]); cbar2.set_ticklabels([f"{v} eV" for v in tick_eV])

    ax = axes[1, 1]; im4 = ax.imshow(gray_linear_255, cmap='gray', aspect='equal', vmin=0, vmax=255); setup_ax(ax, "Raw Linear Baseline\n[Tail Invisible]", '#b22222'); plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[1, 2]; im5 = ax.imshow(gray_log_255, cmap='gray', aspect='equal', vmin=0, vmax=255); setup_ax(ax, "Raw MDFS Grayscale\n[Strict f2 = 20*ln(E)+100]", '#006400'); plt.colorbar(im5, ax=ax, fraction=0.046, pad=0.04)

    # 💾 保存
    out_dir = "Dataset_Previews_Vector"
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, f"Viewer_RawDiscrete_{filename.replace('.npy', '.pdf')}")
    plt.savefig(out_name, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"🎉 验收完成！彻底剥离模糊的纯生肉 PDF 图已保存：{out_name}")

if __name__ == "__main__":
    view_pure_sample()