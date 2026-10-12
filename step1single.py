import os
import time
import math
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# ==========================================
# 0. 基础物理常数与材料库
# ==========================================
ELEMENT_DATA = {
    6:  [6,  12.011, 1.0,   0],  # C (模拟充填了环氧树脂的孔隙)
    14: [14, 28.085, 2.3,   0],  # Si(模拟 C-S-H 凝胶基体)
    29: [29, 63.546, 3.1,   0],  # Cu(模拟未水化水泥熟料高密度相)
    78: [78, 195.08, 21.45, 0]   # Pt(预留极重元素相)
}

class GlobalConfig:
    @staticmethod
    def get_element_info(atomic_numbers):
        return np.array([ELEMENT_DATA.get(int(z), [10.0, 20.0, 2.5, 0]) for z in atomic_numbers])

# ==========================================
# 1. 物理辅助函数
# ==========================================
def get_mott_scattering_angle(Z, E_eV):
    E_keV = max(E_eV / 1000.0, 1e-9)
    alpha = 3.4e-3 * (Z**0.67) / E_keV
    R2 = np.random.random() 
    cos_theta = 1 - (2 * alpha * R2) / (1 + alpha - R2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def get_elastic_mean_free_path(Z, A, Rho, E_eV):
    E_keV = max(E_eV / 1000.0, 0.05) 
    sigma_mott = (3.0e-18 * (Z**1.7)) / (E_keV + 0.005 * (Z**2) * np.sqrt(E_keV))
    return 1.0 if sigma_mott <= 0 else (A / (6.022e23 * Rho * sigma_mott)) * 1e7 

def update_direction(dx, dy, dz, theta, phi):
    cx, cy, cz = dx, dz, dy 
    sin_t, cos_t, sin_p, cos_p = math.sin(theta), math.cos(theta), math.sin(phi), math.cos(phi)
    arg2 = cx**2 + cy**2
    if arg2 > 1e-12:
        arg = math.sqrt(arg2)
        nx = cx * cos_t + sin_t / arg * (cx * cz * cos_p - cy * sin_p)
        ny = cy * cos_t + sin_t / arg * (cy * cz * cos_p + cx * sin_p)
        nz = cz * cos_t - arg * sin_t * cos_p
    else:
        nx, ny, nz = sin_t * cos_p, sin_t * sin_p, cos_t if cz > 0 else -cos_t
    norm = math.sqrt(nx**2 + nz**2 + ny**2)
    return (nx/norm, nz/norm, ny/norm) if norm > 0 else (nx, nz, ny)

# ==========================================
# 2. 蒙特卡洛引擎 (兼容混合孔隙)
# ==========================================
class MCEngine:
    def __init__(self, sample):
        self.sample = sample
        self.comp = sample['comp']
        self.dEdsC1 = sample['dEdsC1']
        self.pixel_size = sample['pixel_size']
        self.h, self.w = self.comp.shape[:2]
        self.vac_nm = sample['size'][3] * self.pixel_size 
        self.elem_props = sample['element_properties_table']
        self.J_vals = np.array([9.76*Z + 58.5*(Z**-0.19) if Z >= 13 else 11.5*Z for Z in self.elem_props[:, 0]])

    def run(self, num_electrons):
        max_steps = 15000
        trajectories = np.full((num_electrons, max_steps, 4), np.nan)
        center_x_nm = (self.w * self.pixel_size) / 2.0
        
        for i in range(num_electrons):
            x, y, z = center_x_nm + np.random.normal(0, self.pixel_size * 0.5), 0.0, 0.0
            dx, dy, dz = 0.0, 1.0, 0.0  
            E = self.sample['energy'] * 1000.0 
            step = 0
            while step < max_steps and E > 50:
                trajectories[i, step, :] = [x, y, z, E]
                px, py = int(x / self.pixel_size), int(y / self.pixel_size)
                if px < 0 or px >= self.w or py >= self.h or py < 0: break 

                if y < self.vac_nm:
                    x += dx*5.0; y += dy*5.0; z += dz*5.0; step += 1
                    continue
                
                local_comps = self.comp[py, px, :]
                
                # 🌟 真·真空判断：挖空法生成的孔隙，各相总和为0，直线穿透不损失能量
                if np.sum(local_comps) < 1e-6:
                    x += dx * self.pixel_size
                    y += dy * self.pixel_size
                    z += dz * self.pixel_size
                    step += 1
                    continue
                
                R4 = np.random.random()
                cum_comp = np.cumsum(local_comps)
                elem_idx = np.searchsorted(cum_comp, R4 * cum_comp[-1])
                Z, A_val, Rho_val = self.elem_props[elem_idx, :3]
                
                lambda_nm = get_elastic_mean_free_path(Z, A_val, Rho_val, E)
                R1 = max(np.random.random(), 1e-9)
                step_size = min(-lambda_nm * np.log(R1), self.pixel_size * 0.8) 
                
                const_val = abs(self.dEdsC1[py, px, elem_idx])
                J = self.J_vals[elem_idx]
                log_term = np.log(1.166 * (E + 0.85*J) / J)
                dE = min(abs((const_val / E * log_term if log_term > 0 else const_val / E * 0.01) * step_size), E)
                E -= dE
                
                theta = get_mott_scattering_angle(Z, E) 
                phi = 2.0 * math.pi * np.random.random() 
                
                dx, dy, dz = update_direction(dx, dy, dz, theta, phi)
                x += dx*step_size; y += dy*step_size; z += dz*step_size; step += 1
        return trajectories

# ==========================================
# 3. 能量场专属提取器
# ==========================================
class EnergyExtractor:
    @staticmethod
    def extract_energy_field(trajectories, sample):
        h, w = sample['comp'].shape[:2]
        pixel_size = sample['pixel_size']
        sum_E = np.zeros((h, w))   
        for traj in trajectories:
            valid = traj[~np.isnan(traj[:, 0])]
            if len(valid) < 2: continue
            dE = np.maximum(valid[:-1, 3] - valid[1:, 3], 0)
            pxs, pys = (valid[:-1, 0]/pixel_size).astype(int), (valid[:-1, 1]/pixel_size).astype(int)
            mask = (pxs >= 0) & (pxs < w) & (pys >= 0) & (pys < h)
            np.add.at(sum_E, (pys[mask], pxs[mask]), dE[mask])
        return sum_E

def calculate_casino_percentage(e_grid, mask_solid):
    e_flat = e_grid.flatten()
    sort_idx = np.argsort(e_flat)[::-1]
    cum_energy = np.cumsum(e_flat[sort_idx])
    total_energy = cum_energy[-1]
    if total_energy < 1e-5: return np.ones_like(e_grid) * 100.0
    percent_flat = np.zeros_like(e_flat)
    percent_flat[sort_idx] = (cum_energy / total_energy) * 100.0
    percent_grid = percent_flat.reshape(e_grid.shape)
    percent_grid[e_grid <= 1e-5] = 100.0
    percent_smoothed = scipy.ndimage.gaussian_filter(percent_grid, sigma=1.5)
    percent_smoothed[~mask_solid] = 100.0
    return np.clip(percent_smoothed, 0.0, 100.0)

# ==========================================
# 4. 几何生成器 (绝对体积守恒版)
# ==========================================
class CementClusterGenerator:
    def __init__(self, elem_info): 
        self.elem_info = elem_info

    def generate(self, options):
        sample_size = options['sampleSize']
        pixel_size, w_pixels, h_pixels, vacuum_pixels = sample_size[2], int(sample_size[0]), int(sample_size[1]), int(sample_size[3])
        h_solid = h_pixels - vacuum_pixels
        
        canvas_width_nm = w_pixels * pixel_size
        
        layer_comp = np.zeros((h_solid, w_pixels, len(self.elem_info)), dtype=np.float32)
        layer_comp[:, :, 1] = 1.0  # 基体默认全填充 Si
        
        y_grid, x_grid = np.ogrid[:h_solid, :w_pixels]

        def add_irregular_shape(cx, cy, base_r_pixels, val_idx, is_pore=False, pore_type='vacuum'):
            num_blobs = np.random.randint(3, 6) if not is_pore else np.random.randint(2, 4)
            for _ in range(num_blobs):
                # 严格限幅畸变倍率，保证多边形边缘但不造成无限膨胀
                a = base_r_pixels * np.random.uniform(0.7, 1.2)
                b = base_r_pixels * np.random.uniform(0.7, 1.2)
                theta = np.random.uniform(0, 2 * np.pi) 
                
                offset_r = base_r_pixels * np.random.uniform(0, 0.4)
                offset_theta = np.random.uniform(0, 2 * np.pi)
                ox = cx + offset_r * np.cos(offset_theta)
                oy = cy + offset_r * np.sin(offset_theta)
                
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                dx = x_grid - ox
                dy = y_grid - oy
                rx = dx * cos_t + dy * sin_t
                ry = -dx * sin_t + dy * cos_t
                
                mask = (rx**2 / max(a**2, 1)) + (ry**2 / max(b**2, 1)) <= 1.0
                
                # 第一步：物理挖坑，保证空间互斥
                for i in range(len(self.elem_info)):
                    layer_comp[mask, i] = 0.0
                    
                # 第二步：依材质填埋
                if is_pore:
                    if pore_type == 'resin':
                        layer_comp[mask, 0] = 1.0 # 树脂孔 (碳)
                    elif pore_type == 'vacuum':
                        pass # 真空孔 (绝对空洞)
                else:
                    layer_comp[mask, val_idx] = 1.0 # 熟料粗骨料

        # --- 1. 生成未水化核心 ---
        target_clinker_ratio = np.random.uniform(0.12, 0.25) 
        current_clinker = 0.0
        
        # 尺寸受控：最大不超过 600nm，且不超过视野宽度的 20%
        max_r_nm = min(600.0, canvas_width_nm * 0.20) 
        attempts = 0
        
        while current_clinker < target_clinker_ratio and attempts < 3000:
            attempts += 1
            r_nm = np.random.exponential(scale=max_r_nm / 3.0)
            if r_nm < 20.0 or r_nm > max_r_nm: 
                continue 
            
            r_pixels = max(1, int(r_nm / pixel_size))
            cx, cy = np.random.randint(0, w_pixels), np.random.randint(0, h_solid)
            
            add_irregular_shape(cx, cy, r_pixels, val_idx=2, is_pore=False)
            current_clinker = np.sum(layer_comp[:, :, 2]) / (w_pixels * h_solid)

        # --- 2. 生成全域随机混合孔隙网络 ---
        target_porosity = options['porosity']
        attempts = 0
        
        while True:
            # 实时精准计分板：真空与树脂总和
            current_vac = np.sum(np.sum(layer_comp, axis=2) < 0.01) / (w_pixels * h_solid)
            current_resin = np.sum(layer_comp[:, :, 0]) / (w_pixels * h_solid)
            current_porosity = current_vac + current_resin
            
            if current_porosity >= target_porosity or attempts > 6000:
                break
            attempts += 1
            
            # 全域轰炸坐标
            cx, cy = np.random.randint(0, w_pixels), np.random.randint(0, h_solid)
            
            pore_r_nm = np.random.exponential(scale=35.0)
            pore_r_nm = np.clip(pore_r_nm, 10.0, 150.0)
            r_pixels = max(1, int(pore_r_nm / pixel_size))
            
            p_type = 'vacuum' if np.random.random() < 0.5 else 'resin'
            add_irregular_shape(cx, cy, r_pixels, val_idx=0, is_pore=True, pore_type=p_type)

        # --- 3. 绝对体积组装 ---
        comp = np.zeros((h_pixels, w_pixels, len(self.elem_info)))
        
        # 🚀 极其关键的修改：抛弃高斯模糊，原汁原味输出，捍卫所有微孔隙的物理面积！
        comp[vacuum_pixels:, :, :] = layer_comp 
        
        # 归一化 (防沉淀机制，仅作物理托底，此时 layer_comp 本身已天然归一化)
        comp_sum = np.sum(comp, axis=2, keepdims=True)
        comp /= np.maximum(comp_sum, 1e-9)
        comp[comp_sum[:, :, 0] < 1e-5] = 0.0 # 坚守真空阵地
        
        dEdsC1 = np.zeros_like(comp)
        for el in range(len(self.elem_info)): 
            dEdsC1[:, :, el] = 7850.0 * comp[:, :, el] * self.elem_info[el, 2] * self.elem_info[el, 0] / self.elem_info[el, 1]
            
        return {'comp': comp, 'size': sample_size, 'energy': options['energy'], 'pixel_size': pixel_size, 'element_properties_table': self.elem_info[:, :4], 'dEdsC1': dEdsC1}

# ==========================================
# 5. 单样本预览主程序
# ==========================================
if __name__ == "__main__":
    
    TARGET_ENERGY = 15.0     
    RESOLUTION = 256         
    POROSITY = 0.2           # 👉 20% 孔隙率，这次肉眼绝对可见！
    OUTPUT_DIR = "Preview_Data"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename_prefix = f"SingleTest_UltimatePhysics_{int(TARGET_ENERGY)}keV"
    
    print(f"🚀 启动终极物理场基准测试 | 电压: {TARGET_ENERGY} keV")
    
    np.random.seed(int(time.time() % 1000)) 
    elem_info = GlobalConfig.get_element_info([6, 14, 29, 78])
    
    adaptive_canvas_nm = 1000.0 * ((TARGET_ENERGY / 5.0) ** 1.2)
    pixel_size = adaptive_canvas_nm / RESOLUTION
    
    print(f"📏 画布自适应调整: 物理尺寸 {adaptive_canvas_nm:.1f} nm, 像素精度 {pixel_size:.2f} nm/px")
    
    opt = {
        'energy': TARGET_ENERGY, 
        'sampleSize': [RESOLUTION, RESOLUTION, pixel_size, int(RESOLUTION*0.1)],
        'porosity': POROSITY
    }
    
    print("🧱 正在生成具备绝对体积守恒的多相微观结构...")
    sample = CementClusterGenerator(elem_info).generate(opt)
    
    print(f"🔫 发射 10000 颗电子推演纯能量沉积场...")
    start_time = time.time()
    trajectories = MCEngine(sample).run(num_electrons=10000) 
    print(f"⏱️ 蒙特卡洛计算耗时: {time.time() - start_time:.2f} 秒")
    
    print("📊 正在提取 4 通道核心数据...")
    ch_density = np.sum(sample['comp'] * elem_info[:, 2], axis=2)
    
    # 掩码剔除外部和内部的绝对真空，仅对固体（包含基体、熟料和树脂）进行能量统计评估
    mask_solid = ch_density > 0.01 
    ch_zeff = np.sum(sample['comp'] * elem_info[:, 0], axis=2)
    
    sum_E = EnergyExtractor.extract_energy_field(trajectories, sample)
    pixel_vol_nm3 = pixel_size ** 2
    ch_E_dens = (sum_E / pixel_vol_nm3) * mask_solid
    ch_E_percent = calculate_casino_percentage(sum_E, mask_solid)
    
    multi_physics = np.stack([
        ch_density, 
        ch_zeff, 
        ch_E_dens, 
        ch_E_percent
    ], axis=-1).astype(np.float32)
    
    npy_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.npy")
    np.save(npy_path, multi_physics)
    
    print("🎨 正在生成完美对称的 2x2 高清预览矩阵...")
    channel_names = [
        'Density (g/cm3)', 
        'Z_eff (Atomic Number)', 
        'Energy Density (eV/nm3)', 
        'Energy Percent (%)'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Energy Deposition Field (Absolute Volume) - {TARGET_ENERGY}keV", fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(multi_physics[:, :, i], cmap='viridis')
        ax.set_title(channel_names[i], fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ 测试完毕！")
    print(f"👉 终极基准图已保存至: {png_path} (准备迎接满屏极其真实的 20% 多尺度孔隙吧！)")