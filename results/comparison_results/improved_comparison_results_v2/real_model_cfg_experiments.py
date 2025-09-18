#!/usr/bin/env python3
"""
Real Model CFG Experiments - 使用真實模型進行Static CFG和Heuristic schedule實驗
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from pathlib import Path
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import random
import json
from typing import Dict, List, Tuple, Any

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qrl.envs import create_diffusion_env
from qrl.actors import QuantumActor, ClassicalActor
from qrl.critics import MLPCritic
from qrl.training import QuantumPPOTrainer, PPOTrainer
from qrl.utils.seed import set_seed

# 設置隨機種子
set_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class RealModelCFGGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"使用設備: {self.device}")
        self.quantum_actor = None
        self.classical_actor = None
        self.quantum_critic = None
        self.classical_critic = None
        
    def load_models(self, models_dir: Path):
        """載入已訓練的模型"""
        print("載入已訓練的模型...")
        
        try:
            # 載入量子模型
            quantum_model_path = models_dir / "quantum_actor.pt"
            if quantum_model_path.exists():
                # 創建環境來獲取正確的維度
                temp_env = create_diffusion_env(
                    model_id="runwayml/stable-diffusion-v1-5",
                    target_class=0,
                    max_steps=50,
                    stage="A",
                    device=self.device
                )
                
                self.quantum_actor = QuantumActor(
                    state_dim=temp_env.observation_space.shape[0],
                    action_dim=temp_env.action_space.shape[0],
                    n_qubits=4,
                    n_layers=2,
                    device=self.device
                )
                
                self.quantum_critic = MLPCritic(
                    state_dim=temp_env.observation_space.shape[0],
                    hidden_dims=[64, 32],
                    device=self.device
                )
                
                # 載入權重
                quantum_state = torch.load(quantum_model_path, map_location=self.device)
                self.quantum_actor.load_state_dict(quantum_state)
                self.quantum_actor.to(self.device)
                self.quantum_actor.eval()
                
                print("✅ 量子模型載入成功")
            else:
                print("❌ 量子模型文件不存在")
                
            # 載入古典模型
            classical_model_path = models_dir / "classical_actor.pt"
            if classical_model_path.exists():
                # 創建環境來獲取正確的維度
                temp_env = create_diffusion_env(
                    model_id="runwayml/stable-diffusion-v1-5",
                    target_class=0,
                    max_steps=50,
                    stage="A",
                    device=self.device
                )
                
                self.classical_actor = ClassicalActor(
                    state_dim=temp_env.observation_space.shape[0],
                    action_dim=temp_env.action_space.shape[0],
                    hidden_dims=[64, 32],
                    device=self.device
                )
                
                self.classical_critic = MLPCritic(
                    state_dim=temp_env.observation_space.shape[0],
                    hidden_dims=[64, 32],
                    device=self.device
                )
                
                # 載入權重
                classical_state = torch.load(classical_model_path, map_location=self.device)
                self.classical_actor.load_state_dict(classical_state)
                self.classical_actor.to(self.device)
                self.classical_actor.eval()
                
                print("✅ 古典模型載入成功")
            else:
                print("❌ 古典模型文件不存在")
                
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def create_environment_with_cfg(self, target_class: int, cfg_scale: float = 7.5, 
                                  use_heuristic: bool = False) -> Any:
        """創建帶有CFG配置的環境"""
        control_config = {
            'stage': 'A',
            'delta_cfg': {'min': -2.0, 'max': 2.0, 'base_cfg': cfg_scale},
            'environment': {'max_steps': 50}
        }
        
        if use_heuristic:
            # 啟發式調度：動態調整CFG
            control_config['delta_cfg']['base_cfg'] = 5.0  # 基礎CFG
            control_config['heuristic_schedule'] = True
        
        return create_diffusion_env(
            model_id="runwayml/stable-diffusion-v1-5",
            target_class=target_class,
            max_steps=50,
            stage="A",
            device=self.device,
            reward_config={
                'alpha_cls': 1.0,
                'beta_step': 0.2,
                'action_penalty': 0.005,
                'tv_penalty': 0.0
            },
            control_config=control_config
        )
    
    def generate_with_static_cfg(self, target_class: int, num_images: int = 16, 
                                cfg_scale: float = 7.5, actor_type: str = "classical") -> List[torch.Tensor]:
        """使用Static CFG生成圖像"""
        print(f"使用Static CFG (g={cfg_scale})生成{num_images}張圖像，類別: {target_class}")
        
        if actor_type == "quantum" and self.quantum_actor is None:
            raise ValueError("量子模型未載入")
        elif actor_type == "classical" and self.classical_actor is None:
            raise ValueError("古典模型未載入")
        
        actor = self.quantum_actor if actor_type == "quantum" else self.classical_actor
        generated_images = []
        
        # 確保模型在GPU上
        actor = actor.to(self.device)
        
        for i in range(num_images):
            try:
                print(f"  生成第{i+1}/{num_images}張圖像...")
                
                # 創建環境
                env = self.create_environment_with_cfg(target_class, cfg_scale, use_heuristic=False)
                
                # 重置環境
                obs, _ = env.reset()
                
                # 使用actor進行rollout
                with torch.no_grad():
                    for step in range(50):
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        action = actor.sample(obs_tensor)
                        action_np = action.cpu().numpy().flatten()
                        obs, reward, done, truncated, info = env.step(action_np)
                        if done or truncated:
                            break
                
                # 獲取最終圖像
                final_image = env._decode_latent()
                if final_image is not None:
                    if isinstance(final_image, tuple):
                        final_image = final_image[0]
                    generated_images.append(final_image)
                
                env.close()
                
            except Exception as e:
                print(f"生成第{i}張圖像失敗: {e}")
                continue
        
        print(f"成功生成 {len(generated_images)} 張圖像")
        return generated_images
    
    def generate_with_heuristic_schedule(self, target_class: int, num_images: int = 16, 
                                       actor_type: str = "classical") -> List[torch.Tensor]:
        """使用Heuristic schedule生成圖像"""
        print(f"使用Heuristic schedule生成{num_images}張圖像，類別: {target_class}")
        
        if actor_type == "quantum" and self.quantum_actor is None:
            raise ValueError("量子模型未載入")
        elif actor_type == "classical" and self.classical_actor is None:
            raise ValueError("古典模型未載入")
        
        actor = self.quantum_actor if actor_type == "quantum" else self.classical_actor
        generated_images = []
        
        # 確保模型在GPU上
        actor = actor.to(self.device)
        
        for i in range(num_images):
            try:
                print(f"  生成第{i+1}/{num_images}張圖像...")
                
                # 創建環境（啟發式調度會動態調整CFG）
                env = self.create_environment_with_cfg(target_class, cfg_scale=5.0, use_heuristic=True)
                
                # 重置環境
                obs, _ = env.reset()
                
                # 使用actor進行rollout
                with torch.no_grad():
                    for step in range(50):
                        # 在啟發式調度中，CFG會根據步數動態調整
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        action = actor.sample(obs_tensor)
                        action_np = action.cpu().numpy().flatten()
                        obs, reward, done, truncated, info = env.step(action_np)
                        if done or truncated:
                            break
                
                # 獲取最終圖像
                final_image = env._decode_latent()
                if final_image is not None:
                    if isinstance(final_image, tuple):
                        final_image = final_image[0]
                    generated_images.append(final_image)
                
                env.close()
                
            except Exception as e:
                print(f"生成第{i}張圖像失敗: {e}")
                continue
        
        print(f"成功生成 {len(generated_images)} 張圖像")
        return generated_images

class ImageMetricsCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
    def load_image(self, image_path):
        """載入並預處理圖像"""
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return image
    
    def calculate_psnr(self, img1, img2):
        return psnr(img1, img2, data_range=1.0)
    
    def calculate_ssim(self, img1, img2):
        return ssim(img1, img2, channel_axis=2, data_range=1.0)
    
    def calculate_lpips(self, img1, img2):
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        img1_tensor = img1_tensor * 2.0 - 1.0
        img2_tensor = img2_tensor * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_value = self.lpips_model(img1_tensor, img2_tensor)
        
        return lpips_value.item()
    
    def calculate_metrics(self, generated_path, real_path):
        """計算所有指標"""
        try:
            gen_img = self.load_image(generated_path)
            real_img = self.load_image(real_path)
            
            if gen_img.shape != real_img.shape:
                real_h, real_w = real_img.shape[:2]
                gen_img = cv2.resize(gen_img, (real_w, real_h))
            
            psnr_value = self.calculate_psnr(real_img, gen_img)
            ssim_value = self.calculate_ssim(real_img, gen_img)
            lpips_value = self.calculate_lpips(real_img, gen_img)
            
            return {
                'psnr': psnr_value,
                'ssim': ssim_value,
                'lpips': lpips_value
            }
        except Exception as e:
            print(f"計算指標失敗 {generated_path} vs {real_path}: {e}")
            return {'psnr': np.nan, 'ssim': np.nan, 'lpips': np.nan}

def save_images(images: List[torch.Tensor], output_dir: Path, prefix: str):
    """保存圖像"""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    for i, image in enumerate(images):
        try:
            # 轉換tensor為numpy
            if isinstance(image, torch.Tensor):
                image = image.squeeze(0).cpu().numpy()
                if len(image.shape) == 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
            
            # 轉換為[0, 255]範圍
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
            
            # 保存圖像
            filename = f"{prefix}_{i}.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            saved_paths.append(filepath)
            
        except Exception as e:
            print(f"保存圖像{i}失敗: {e}")
            continue
    
    return saved_paths

def run_real_model_cfg_experiments():
    """運行真實模型CFG實驗"""
    print("🚀 開始真實模型CFG實驗...")
    
    # 定義路徑
    base_dir = Path('.')
    models_dir = base_dir / 'models'
    real_images_dir = base_dir / 'real_images_png'
    
    # 類別 - 只測試前3個類別以加快速度
    categories = ['airplane', 'automobile', 'bird']  # 只測試前3個類別
    
    # 初始化生成器和指標計算器
    generator = RealModelCFGGenerator()
    metrics_calc = ImageMetricsCalculator()
    
    # 載入模型
    generator.load_models(models_dir)
    
    if generator.classical_actor is None:
        print("❌ 無法載入古典模型，退出")
        return []
    
    results = []
    
    for class_id, category in enumerate(categories):
        print(f"\n處理類別: {category} (ID: {class_id})")
        
        # 獲取真實圖像
        real_category_dir = real_images_dir / category
        real_images = list(real_category_dir.glob('*.png'))
        real_images.sort()
        
        if not real_images:
            print(f"未找到{category}的真實圖像")
            continue
        
        # 限制圖像數量以加快測試
        num_images = min(len(real_images), 4)  # 每個類別只生成4張圖像進行測試
        
        # 生成Static CFG圖像
        try:
            static_cfg_images = generator.generate_with_static_cfg(
                class_id, num_images, cfg_scale=7.5, actor_type="classical"
            )
            static_cfg_dir = base_dir / 'generated_images_png' / 'real_classical_static_cfg' / category
            static_cfg_paths = save_images(static_cfg_images, static_cfg_dir, 'static_cfg')
            
            # 計算Static CFG指標
            for i, (real_img, gen_img) in enumerate(zip(real_images[:num_images], static_cfg_paths)):
                try:
                    metrics = metrics_calc.calculate_metrics(gen_img, real_img)
                    results.append({
                        'method': 'real_classical_static_cfg',
                        'category': category,
                        'image_id': i,
                        'real_image': real_img.name,
                        'generated_image': gen_img.name,
                        'psnr': metrics['psnr'],
                        'ssim': metrics['ssim'],
                        'lpips': metrics['lpips']
                    })
                except Exception as e:
                    print(f"計算Static CFG指標失敗: {e}")
        
        except Exception as e:
            print(f"生成Static CFG圖像失敗: {e}")
        
        # 生成Heuristic schedule圖像
        try:
            heuristic_images = generator.generate_with_heuristic_schedule(
                class_id, num_images, actor_type="classical"
            )
            heuristic_dir = base_dir / 'generated_images_png' / 'real_classical_heuristic' / category
            heuristic_paths = save_images(heuristic_images, heuristic_dir, 'heuristic')
            
            # 計算Heuristic指標
            for i, (real_img, gen_img) in enumerate(zip(real_images[:num_images], heuristic_paths)):
                try:
                    metrics = metrics_calc.calculate_metrics(gen_img, real_img)
                    results.append({
                        'method': 'real_classical_heuristic',
                        'category': category,
                        'image_id': i,
                        'real_image': real_img.name,
                        'generated_image': gen_img.name,
                        'psnr': metrics['psnr'],
                        'ssim': metrics['ssim'],
                        'lpips': metrics['lpips']
                    })
                except Exception as e:
                    print(f"計算Heuristic指標失敗: {e}")
        
        except Exception as e:
            print(f"生成Heuristic圖像失敗: {e}")
    
    return results

def save_results_and_analysis(results):
    """保存結果並分析"""
    if not results:
        print("沒有結果可保存")
        return
    
    # 保存到CSV
    df = pd.DataFrame(results)
    df.to_csv('real_model_cfg_experiments_results.csv', index=False)
    print("結果已保存到 real_model_cfg_experiments_results.csv")
    
    # 打印統計
    print("\n=== 真實模型CFG實驗結果 ===")
    print(f"總共處理圖片: {len(results)}")
    
    for method in ['real_classical_static_cfg', 'real_classical_heuristic']:
        method_data = df[df['method'] == method]
        if not method_data.empty:
            print(f"\n{method.upper()}:")
            print(f"  PSNR:  {method_data['psnr'].mean():.4f} ± {method_data['psnr'].std():.4f}")
            print(f"  SSIM:  {method_data['ssim'].mean():.4f} ± {method_data['ssim'].std():.4f}")
            print(f"  LPIPS: {method_data['lpips'].mean():.4f} ± {method_data['lpips'].std():.4f}")
    
    # 與原始結果比較
    try:
        original_df = pd.read_csv('image_metrics_results.csv')
        original_classical = original_df[original_df['method'] == 'classical']
        
        print(f"\n=== 與原始古典方法比較 ===")
        print(f"原始古典:")
        print(f"  PSNR:  {original_classical['psnr'].mean():.4f} ± {original_classical['psnr'].std():.4f}")
        print(f"  SSIM:  {original_classical['ssim'].mean():.4f} ± {original_classical['ssim'].std():.4f}")
        print(f"  LPIPS: {original_classical['lpips'].mean():.4f} ± {original_classical['lpips'].std():.4f}")
        
        # 比較改進
        static_cfg_data = df[df['method'] == 'real_classical_static_cfg']
        heuristic_data = df[df['method'] == 'real_classical_heuristic']
        
        if not static_cfg_data.empty:
            print(f"\n真實Static CFG vs 原始古典:")
            print(f"  PSNR:  {static_cfg_data['psnr'].mean() - original_classical['psnr'].mean():+.4f}")
            print(f"  SSIM:  {static_cfg_data['ssim'].mean() - original_classical['ssim'].mean():+.4f}")
            print(f"  LPIPS: {static_cfg_data['lpips'].mean() - original_classical['lpips'].mean():+.4f}")
        
        if not heuristic_data.empty:
            print(f"\n真實Heuristic vs 原始古典:")
            print(f"  PSNR:  {heuristic_data['psnr'].mean() - original_classical['psnr'].mean():+.4f}")
            print(f"  SSIM:  {heuristic_data['ssim'].mean() - original_classical['ssim'].mean():+.4f}")
            print(f"  LPIPS: {heuristic_data['lpips'].mean() - original_classical['lpips'].mean():+.4f}")
    
    except FileNotFoundError:
        print("原始結果文件未找到，跳過比較")

def main():
    """主函數"""
    print("=== 真實模型CFG實驗 ===")
    print("使用真實訓練的模型進行:")
    print("1. Static CFG (g=7.5)")
    print("2. Heuristic schedule")
    print("並計算PSNR、SSIM、LPIPS指標")
    
    # 運行實驗
    results = run_real_model_cfg_experiments()
    
    # 保存結果和分析
    save_results_and_analysis(results)
    
    print("\n真實模型CFG實驗完成！")

if __name__ == "__main__":
    main()
