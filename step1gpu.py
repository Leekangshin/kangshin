import os
import time
import math
import numpy as np
import scipy.ndimage
import multiprocessing as mp
import sys
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32

# ==========================================
# 0. 基础物理常数与材料库
# ==========================================
ELEMENT_DATA = {
    6:  [6,  12.011, 2.26,  0],  # C (轻元素/树脂孔隙)
    14: [14, 28.085, 2.65,  0],  # Si(常规基体)
    29: [29, 63.546, 8.96,  0],  # Cu(中重熟料)
    78: [78, 195.08, 21.45, 0]   # Pt(极重元素)
}

class GlobalConfig:
    @staticmethod
    def get_element_info(atomic_numbers):
        return np.array([ELEMENT_DATA.get(int(z), [10.0, 20.0, 2.5, 0]) for z in atomic_numbers]).astype(np.float32)

# ==========================================
# 1. GPU CUDA 物理计算核函数 (设备端)
# ==========================================
@cuda.jit(device=True)
def get_mott_scattering_angle_gpu(rng_states, thread_id, Z, E_eV):
    E_keV = max(E_eV / 1000.0, 1e-9)
    alpha = 3.4e-3 * (Z**0.67) / E_keV
    R2 = xoroshiro128p_uniform_float32(rng_states, thread_id)
    cos_theta = 1.0 - (2.0 * alpha * R2) / (1.0 + alpha - R2)
    cos_theta = min(max(cos_theta, -1.0), 1.0)
    return math.acos(cos_theta)

@cuda.jit(device=True)
def get_elastic_mean_free_path_gpu(Z, A, Rho, E_eV):
    E_keV = max(E_eV / 1000.0, 0.05)
    sigma_mott = (3.0e-18 * (Z**1.7)) / (E_keV + 0.005 * (Z**2) * math.sqrt(E_keV))
    if sigma_mott <= 0: return 1.0
    return (A / (6.022e23 * Rho * sigma_mott)) * 1e7

@cuda.jit(device=True)
def update_direction_gpu(dx, dy, dz, theta, phi):
    cx, cy, cz = dx, dz, dy
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    sin_p, cos_p = math.sin(phi), math.cos(phi)
    arg2 = cx**2 + cy**2
    if arg2 > 1e-12:
        arg = math.sqrt(arg2)
        nx = cx * cos_t + sin_t / arg * (cx * cz * cos_p - cy * sin_p)
        ny = cy * cos_t + sin_t / arg * (cy * cz * cos_p + cx * sin_p)
        nz = cz * cos_t - arg * sin_t * cos_p
    else:
        nx, ny, nz = sin_t * cos_p, sin_t * sin_p, cos_t if cz > 0 else -cos_t
    norm = math.sqrt(nx**2 + nz**2 + ny**2)
    if norm > 0: return nx/norm, nz/norm, ny/norm
    return nx, nz, ny

@cuda.jit
def mc_energy_deposition_kernel(rng_states, comp, dEdsC1, elem_props, J_vals, pixel_size, vac_nm, energy, w, h, sum_E):
    idx = cuda.grid(1)
    if idx >= rng_states.shape[0]: return

    center_x_nm = (w * pixel_size) / 2.0
    x = center_x_nm + xoroshiro128p_normal_float32(rng_states, idx) * 1.0
    y = 0.0; z = 0.0
    dx, dy, dz = 0.0, 1.0, 0.0
    E = energy * 1000.0
    step = 0; max_steps = 15000

    while step < max_steps and E > 50.0:
        px, py = int(x / pixel_size), int(y / pixel_size)
        if px < 0 or px >= w or py >= h or py < 0: break
        if y < vac_nm:
            x += dx * 5.0; y += dy * 5.0; z += dz * 5.0; step += 1
            continue

        comp_sum = 0.0
        for e_i in range(comp.shape[2]): comp_sum += comp[py, px, e_i]
        if comp_sum < 1e-6:
            x += dx * pixel_size; y += dy * pixel_size; z += dz * pixel_size; step += 1
            continue

        R4 = xoroshiro128p_uniform_float32(rng_states, idx)
        target_val = R4 * comp_sum
        cum_comp = 0.0; elem_idx = 0
        for e_i in range(comp.shape[2]):
            cum_comp += comp[py, px, e_i]
            if cum_comp >= target_val:
                elem_idx = e_i; break

        Z, A_val, Rho_val = elem_props[elem_idx, 0], elem_props[elem_idx, 1], elem_props[elem_idx, 2]
        lambda_nm = get_elastic_mean_free_path_gpu(Z, A_val, Rho_val, E)
        R1 = max(xoroshiro128p_uniform_float32(rng_states, idx), 1e-9)
        step_size = min(-lambda_nm * math.log(R1), pixel_size * 0.5)

        const_val = abs(dEdsC1[py, px, elem_idx])
        J = J_vals[elem_idx]
        log_term = math.log(1.166 * (E + 0.85*J) / J)
        dE = (const_val / E * log_term if log_term > 0 else const_val / E * 0.01) * step_size
        dE = min(dE, E)

        cuda.atomic.add(sum_E, (py, px), dE)

        E -= dE
        theta = get_mott_scattering_angle_gpu(rng_states, idx, Z, E)
        phi = 2.0 * math.pi * xoroshiro128p_uniform_float32(rng_states, idx)
        dx, dy, dz = update_direction_gpu(dx, dy, dz, theta, phi)
        x += dx * step_size; y += dy * step_size; z += dz * step_size; step += 1

class GPUMCEngine:
    def __init__(self, sample):
        self.comp = np.ascontiguousarray(sample['comp'].astype(np.float32))
        self.dEdsC1 = np.ascontiguousarray(sample['dEdsC1'].astype(np.float32))
        self.elem_props = np.ascontiguousarray(sample['element_properties_table'].astype(np.float32))
        self.J_vals = np.array([9.76*Z + 58.5*(Z**-0.19) if Z >= 13 else 11.5*Z for Z in self.elem_props[:, 0]], dtype=np.float32)
        self.pixel_size = float(sample['pixel_size'])
        self.vac_nm = float(sample['size'][3] * self.pixel_size)
        self.energy = float(sample['energy'])
        self.h, self.w = self.comp.shape[:2]

    # 🚀 坚守 100,000 颗电子高画质物理底线
    def run(self, num_electrons=100000):
        d_comp = cuda.to_device(self.comp)
        d_dEdsC1 = cuda.to_device(self.dEdsC1)
        d_elem_props = cuda.to_device(self.elem_props)
        d_J_vals = cuda.to_device(self.J_vals)
        d_sum_E = cuda.to_device(np.zeros((self.h, self.w), dtype=np.float32))
        
        rng_states = create_xoroshiro128p_states(num_electrons, seed=int(time.time()*1000)%10000)
        threads_per_block = 256
        blocks_per_grid = (num_electrons + (threads_per_block - 1)) // threads_per_block
        
        mc_energy_deposition_kernel[blocks_per_grid, threads_per_block](
            rng_states, d_comp, d_dEdsC1, d_elem_props, d_J_vals, 
            self.pixel_size, self.vac_nm, self.energy, self.w, self.h, d_sum_E
        )
        return d_sum_E.copy_to_host()

# ==========================================
# 2. 几何引擎 
# ==========================================
class ScatterCircleGenerator:
    def __init__(self, elem_info): 
        self.elem_info = elem_info
        self.num_elements = len(elem_info) 

    def generate(self, options):
        sample_size = options['sampleSize']
        pixel_size, w_pixels, h_pixels, vacuum_pixels = sample_size[2], int(sample_size[0]), int(sample_size[1]), int(sample_size[3])
        h_solid, solid_area = h_pixels - vacuum_pixels, (h_pixels - vacuum_pixels) * w_pixels
        
        layer_comp = np.zeros((h_solid, w_pixels, self.num_elements), dtype=np.float32)
        solid_mask = np.zeros((h_solid, w_pixels), dtype=bool)
        target_solid_ratio = 1.0 - options['porosity']
        
        weights = options['weights'] 
        attempts = 0
        
        while np.count_nonzero(solid_mask) / solid_area < target_solid_ratio and attempts < 15000:
            el = np.random.choice(self.num_elements, p=weights)
            d_nm = np.random.normal(options['meanClusterSize'][el], options['stdClusterSize'][el])
            d_nm = max(d_nm, 2.0)
            
            rx, ry = np.random.randint(0, w_pixels), np.random.randint(0, h_solid)
            y_grid, x_grid = np.ogrid[:h_solid, :w_pixels]
            circle = (x_grid - rx)**2 + (y_grid - ry)**2 <= ((d_nm/pixel_size)/2.0)**2
            
            layer_comp[circle, el] += 0.5
            solid_mask[circle] = True
            attempts += 1
            
        comp = np.zeros((h_pixels, w_pixels, self.num_elements))
        for el in range(self.num_elements): 
            comp[vacuum_pixels:, :, el] = scipy.ndimage.gaussian_filter(layer_comp[:, :, el], sigma=2.0)
            
        comp_sum = np.sum(comp, axis=2, keepdims=True)
        comp /= np.maximum(comp_sum, 1e-9)
        comp[comp_sum[:, :, 0] < 1e-5] = 0.0 
        
        dEdsC1 = np.zeros_like(comp)
        for el in range(self.num_elements): 
            dEdsC1[:, :, el] = 7850.0 * comp[:, :, el] * self.elem_info[el, 2] * self.elem_info[el, 0] / self.elem_info[el, 1]
        return {'comp': comp, 'size': sample_size, 'energy': options['energy'], 'pixel_size': pixel_size, 'element_properties_table': self.elem_info[:, :4], 'dEdsC1': dEdsC1}

class DigHoleGenerator:
    def __init__(self, elem_info): 
        self.elem_info = elem_info

    def generate(self, options):
        sample_size = options['sampleSize']
        pixel_size, w_pixels, h_pixels, vacuum_pixels = sample_size[2], int(sample_size[0]), int(sample_size[1]), int(sample_size[3])
        h_solid = h_pixels - vacuum_pixels
        canvas_width_nm = w_pixels * pixel_size
        
        layer_comp = np.zeros((h_solid, w_pixels, len(self.elem_info)), dtype=np.float32)
        
        matrix_idx = options['matrix_idx']
        layer_comp[:, :, matrix_idx] = 1.0  
        
        clinker_candidates = [i for i in range(4) if i != matrix_idx and i != 0]
        if not clinker_candidates: clinker_candidates = [1] 
        
        y_grid, x_grid = np.ogrid[:h_solid, :w_pixels]

        def add_irregular_shape(cx, cy, base_r_pixels, val_idx, is_vacuum=False):
            num_blobs = np.random.randint(2, 4) if is_vacuum else np.random.randint(3, 5)
            for _ in range(num_blobs):
                a, b = base_r_pixels * np.random.uniform(0.85, 1.1), base_r_pixels * np.random.uniform(0.85, 1.1)
                offset_r = base_r_pixels * np.random.uniform(0, 0.1) 
                theta_off, theta_rot = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)
                ox, oy = cx + offset_r * math.cos(theta_off), cy + offset_r * math.sin(theta_off)
                cos_t, sin_t = math.cos(theta_rot), math.sin(theta_rot)
                rx_p = (x_grid - ox) * cos_t + (y_grid - oy) * sin_t
                ry_p = -(x_grid - ox) * sin_t + (y_grid - oy) * cos_t
                mask = (rx_p**2 / max(a**2, 1)) + (ry_p**2 / max(b**2, 1)) <= 1.0
                
                for i in range(len(self.elem_info)): layer_comp[mask, i] = 0.0
                if not is_vacuum: layer_comp[mask, val_idx] = 1.0 

        target_clinker = np.random.uniform(0.12, 0.25) 
        current_clinker, max_r_nm = 0.0, canvas_width_nm * 0.08 
        while current_clinker < target_clinker:
            r_nm = np.random.exponential(scale=max_r_nm / 3.0)
            if r_nm < (canvas_width_nm * 0.01) or r_nm > max_r_nm: continue 
            add_irregular_shape(np.random.randint(0, w_pixels), np.random.randint(0, h_solid), int(r_nm / pixel_size), np.random.choice(clinker_candidates), False)
            current_clinker = np.sum([np.sum(layer_comp[:, :, c]) for c in clinker_candidates]) / (w_pixels * h_solid)

        target_porosity = options['porosity']
        attempts, dynamic_pore_scale = 0, canvas_width_nm * 0.012 
        
        while attempts < 25000:
            current_vac = np.sum(np.sum(layer_comp, axis=2) < 0.01) / (w_pixels * h_solid)
            if current_vac >= target_porosity: break
            attempts += 1
            cx, cy = np.random.randint(0, w_pixels), np.random.randint(0, h_solid)
            if cy < int(h_solid * 0.15): continue
            if np.sum(layer_comp[cy, cx, :]) < 0.1: continue

            pore_r_nm = np.clip(np.random.exponential(scale=dynamic_pore_scale), canvas_width_nm * 0.005, canvas_width_nm * 0.04)
            add_irregular_shape(cx, cy, max(1, int(pore_r_nm / pixel_size)), 0, True)

        comp = np.zeros((h_pixels, w_pixels, len(self.elem_info)))
        for el in range(len(self.elem_info)): 
            comp[vacuum_pixels:, :, el] = scipy.ndimage.gaussian_filter(layer_comp[:, :, el], sigma=2.0)
        comp_sum = np.sum(comp, axis=2, keepdims=True)
        comp /= np.maximum(comp_sum, 1e-9)
        comp[comp_sum[:, :, 0] < 1e-5] = 0.0 
        
        dEdsC1 = np.zeros_like(comp)
        for el in range(len(self.elem_info)): 
            dEdsC1[:, :, el] = 7850.0 * comp[:, :, el] * self.elem_info[el, 2] * self.elem_info[el, 0] / self.elem_info[el, 1]
        return {'comp': comp, 'size': sample_size, 'energy': options['energy'], 'pixel_size': pixel_size, 'element_properties_table': self.elem_info[:, :4], 'dEdsC1': dEdsC1}

# ==========================================
# 3. 核心任务 Worker
# ==========================================
def worker_generate_sample(task_info):
    sample_id, target_energy, logic_type, output_root_dir, resolution = task_info
    
    # 🌟 核心改进 1：双轨文件夹架构 (将 Scatter 和 DigHole 在物理上隔开)
    logic_name = 'Scatter' if logic_type == 0 else 'DigHole'
    sub_dir = os.path.join(output_root_dir, f"{int(target_energy)}keV", logic_name)
    save_path = os.path.join(sub_dir, f"Geo_S{sample_id:04d}_{logic_name}_E{int(target_energy)}keV.npy")
    
    if os.path.exists(save_path): return f"SKIP: {os.path.basename(save_path)}"

    try:
        np.random.seed(sample_id * 1000 + int(time.time() % 1000))
        elem_info = GlobalConfig.get_element_info([6, 14, 29, 78])
        
        # 🌟 核心改进 2：科学成分定标 (打破平均，引入地质学/水泥学真实概率)
        # 索引对应：0->C(10%), 1->Si(60%), 2->Cu(25%), 3->Pt(5%)
        dominant_idx = np.random.choice([0, 1, 2, 3], p=[0.10, 0.60, 0.25, 0.05])  
        
        alpha = [0.5, 0.5, 0.5, 0.5]
        alpha[dominant_idx] = 6.0 
        weights = np.random.dirichlet(alpha)
        
        ko_ranges_nm = []
        for i in range(4):
            Z, A, Rho = elem_info[i, 0], elem_info[i, 1], elem_info[i, 2]
            r_nm = 1000.0 * 0.0276 * A * (target_energy ** 1.67) / (Rho * (Z ** 0.89))
            ko_ranges_nm.append(r_nm)
            
        base_canvas_nm = np.average(ko_ranges_nm, weights=weights)
        adaptive_canvas_nm = base_canvas_nm * 1.5 
        
        if weights[0] > 0.1:
            adaptive_canvas_nm = max(adaptive_canvas_nm, ko_ranges_nm[0] * 1.1)
            
        pixel_size = adaptive_canvas_nm / resolution
        
        base_size, max_size = adaptive_canvas_nm * 0.015, adaptive_canvas_nm * 0.06
        
        opt = {
            'energy': target_energy, 
            'sampleSize': [resolution, resolution, pixel_size, int(resolution*0.1)],
            # 🌟 核心改进 3：拓扑大跨度历练，孔隙率扩展到 0.05 ~ 0.40
            'porosity': np.random.uniform(0.05, 0.40),
            'meanClusterSize': [np.random.uniform(base_size, max_size) for _ in range(4)],
            'stdClusterSize': [np.random.uniform(base_size*0.1, base_size*0.3) for _ in range(4)],
            'weights': weights,          
            'matrix_idx': dominant_idx   
        }
        
        generator = ScatterCircleGenerator(elem_info) if logic_type == 0 else DigHoleGenerator(elem_info)
        sample = generator.generate(opt)
        
        # 🚀 发射 100,000 颗电子
        sum_E = GPUMCEngine(sample).run(num_electrons=100000)
        
        density = np.sum(sample['comp'] * elem_info[:, 2], axis=2)
        mask = density > 0.01 
        
        # 🚨 绝对原始能量保存 (不做任何归一化)
        multi_physics = np.stack([
            density, 
            np.sum(sample['comp'] * elem_info[:, 0], axis=2), 
            (sum_E / (pixel_size**2)) * mask, 
            np.zeros_like(sum_E) 
        ], axis=-1).astype(np.float32)
        
        e_flat = sum_E.flatten(); sort_idx = np.argsort(e_flat)[::-1]
        cum_E = np.cumsum(e_flat[sort_idx]); total_E = cum_E[-1]
        p_flat = np.zeros_like(e_flat)
        if total_E > 1e-5: p_flat[sort_idx] = (cum_E / total_E) * 100.0
        
        p_grid = scipy.ndimage.gaussian_filter(p_flat.reshape(sum_E.shape), sigma=3.0)
        p_grid[~mask] = 100.0
        multi_physics[:,:,3] = p_grid

        np.save(save_path, multi_physics)
        
        material_names = ["C", "Si", "Cu", "Pt"]
        return f"DONE: {os.path.basename(save_path)} | Matrix: {material_names[dominant_idx]}"
    except Exception as e: return f"ERR: {os.path.basename(save_path)} -> {str(e)}"

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    OUTPUT_FOLDER = "Dataset_MultiScale_512"
    ENERGY_LEVELS = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
    
    # 🌟 核心修改：每个能量下，每种逻辑各生成 300 个样本
    SAMPLES_PER_LOGIC = 300 
    RESOLUTION = 512
    
    # 预先创建双轨文件夹
    for eng in ENERGY_LEVELS: 
        os.makedirs(os.path.join(OUTPUT_FOLDER, f"{int(eng)}keV", "Scatter"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, f"{int(eng)}keV", "DigHole"), exist_ok=True)

    print(f"🚀 CUDA 引擎启动: [各300双轨分离 + Si(60%)主导概率 + 10万电子真实剂量] 正在生成...")
    
    all_tasks = [(int(eng*100)+lid*10000+idx, eng, lid, OUTPUT_FOLDER, RESOLUTION) 
                 for eng in ENERGY_LEVELS for lid in [0, 1] for idx in range(SAMPLES_PER_LOGIC)]

    num_cores = 4 
    with mp.Pool(processes=num_cores, maxtasksperchild=5) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_generate_sample, all_tasks)):
            sys.stdout.write(f"\r进度: [{i+1}/{len(all_tasks)}] {res}    "); sys.stdout.flush()
            
    print("\n✅ 10个能级共计 6000 样本终极架构数据集生成圆满结束！")