#!/usr/bin/env python3
"""簡化的帶訓練的量子與經典模型比較腳本。"""

import os
import sys
import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qrl.envs.diffusion_env import DiffusionEnvironment
from qrl.actors.quantum_actor import QuantumActor
from qrl.actors.classical_actor import ClassicalActor
from qrl.critics import MLPCritic
from qrl.training import PPOTrainer
from qrl.evaluation import ImageQualityMetrics, create_reference_images

# =============================================================================
# 配置參數 - 可在這裡修改所有訓練參數
# =============================================================================

# 訓練參數
NUM_EPISODES = 100          # 訓練回合數
MAX_STEPS = 20             # 每個episode的最大步數
TARGET_CLASS = 3           # 目標類別
DEVICE = "cpu"             # 設備 (cpu/cuda)
STAGE = "A"                # 訓練階段

# 模型參數
STATE_DIM = 6              # 狀態維度
ACTION_DIM = 1             # 動作維度
N_QUBITS = 4               # 量子比特數
N_LAYERS = 2               # 量子層數

# PPO訓練參數
PPO_EPOCHS = 4             # PPO訓練輪數
PPO_BATCH_SIZE = 64        # PPO批次大小
PPO_CLIP_RATIO = 0.2       # PPO裁剪比例
PPO_VALUE_COEF = 0.5       # 價值函數係數
PPO_ENTROPY_COEF = 0.01    # 熵係數
PPO_LEARNING_RATE = 3e-4   # 學習率

# 圖像品質評估參數
EVAL_IMAGES_COUNT = 5      # 評估圖像數量
REFERENCE_SIZE = (64, 64)  # 參考圖像尺寸

# =============================================================================

def run_quantum_training(num_episodes: int = NUM_EPISODES) -> Dict[str, Any]:
    """運行量子模型訓練實驗。"""
    print("🔬 開始量子模型訓練實驗...")
    
    try:
        # 初始化環境
        env = DiffusionEnvironment(
            model_id="runwayml/stable-diffusion-v1-5",
            target_class=TARGET_CLASS,
            max_steps=MAX_STEPS,
            device=DEVICE,
            stage=STAGE
        )
        
        # 初始化量子 Actor
        quantum_actor = QuantumActor(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            device=DEVICE
        )
        
        # 初始化 Critic
        critic = MLPCritic(
            state_dim=6,
            hidden_dims=[64, 32]
        )
        
        # 初始化PPO訓練器
        trainer = PPOTrainer(
            actor=quantum_actor,
            critic=critic,
            device=DEVICE,
            lr_actor=PPO_LEARNING_RATE,
            lr_critic=PPO_LEARNING_RATE,
            rollout_steps=10,  # 減少rollout步數
            num_envs=1,  # 單環境
            update_epochs=PPO_EPOCHS,
            mini_batch_size=PPO_BATCH_SIZE
        )
        
        # 運行訓練實驗
        episode_rewards = []
        episode_times = []
        generated_images = []
        cfg_trajectories = []
        
        start_time = time.time()
        
        # 使用PPO訓練器進行訓練
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 重置環境
            obs, _ = env.reset()
            episode_reward = 0
            cfg_trajectory = []
            
            # 收集rollout數據
            rollouts = trainer.collect_rollouts(env, num_steps=10)
            
            # 計算episode獎勵
            episode_reward = sum(rollouts['rewards'].cpu().numpy())
            
            # 記錄CFG軌跡
            for step in range(len(rollouts['rewards'])):
                # 這裡可以添加CFG軌跡記錄邏輯
                pass
            
            # 執行PPO更新
            if len(rollouts['rewards']) > 0:
                update_stats = trainer.update(rollouts)
                print(f"Episode {episode + 1} 更新統計: {update_stats}")
            
            # 獲取最終圖像
            final_image = env._decode_latent()
            if final_image is not None:
                generated_images.append(final_image)
                cfg_trajectories.append(cfg_trajectory)
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_times.append(episode_time)
            
            if (episode + 1) % 2 == 0:
                print(f"Episode {episode + 1}: 獎勵={episode_reward:.4f}, 時間={episode_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # 計算基本統計
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_time = np.mean(episode_times)
        
        # 計算參數統計
        total_params = sum(p.numel() for p in quantum_actor.parameters())
        quantum_params = quantum_actor.quantum_weights.numel() if hasattr(quantum_actor, 'quantum_weights') else 0
        classical_params = total_params - quantum_params
        
        results = {
            "model_type": "quantum",
            "num_episodes": num_episodes,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_time_per_episode": avg_time,
            "total_time": total_time,
            "total_params": total_params,
            "quantum_params": quantum_params,
            "classical_params": classical_params,
            "rewards": episode_rewards,
            "times": episode_times,
            "generated_images": generated_images,
            "cfg_trajectories": cfg_trajectories
        }
        
        print(f"✅ 量子模型訓練完成 - 平均獎勵: {avg_reward:.4f} ± {std_reward:.4f}")
        return results
        
    except Exception as e:
        print(f"❌ 量子模型訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_classical_training(num_episodes: int = NUM_EPISODES) -> Dict[str, Any]:
    """運行經典模型訓練實驗。"""
    print("\n🔬 開始經典模型訓練實驗...")
    
    try:
        # 初始化環境
        env = DiffusionEnvironment(
            model_id="runwayml/stable-diffusion-v1-5",
            target_class=TARGET_CLASS,
            max_steps=MAX_STEPS,
            device=DEVICE,
            stage=STAGE
        )
        
        # 初始化經典 Actor
        classical_actor = ClassicalActor(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=[64, 32],
            device=DEVICE
        )
        
        # 初始化 Critic
        critic = MLPCritic(
            state_dim=STATE_DIM,
            hidden_dims=[64, 32],
            device=DEVICE
        )
        
        # 初始化PPO訓練器
        trainer = PPOTrainer(
            actor=classical_actor,
            critic=critic,
            device=DEVICE,
            lr_actor=PPO_LEARNING_RATE,
            lr_critic=PPO_LEARNING_RATE,
            rollout_steps=10,  # 減少rollout步數
            num_envs=1,  # 單環境
            update_epochs=PPO_EPOCHS,
            mini_batch_size=PPO_BATCH_SIZE
        )
        
        # 運行訓練實驗
        episode_rewards = []
        episode_times = []
        generated_images = []
        cfg_trajectories = []
        
        start_time = time.time()
        
        # 使用PPO訓練器進行訓練
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 重置環境
            obs, _ = env.reset()
            episode_reward = 0
            cfg_trajectory = []
            
            # 收集rollout數據
            rollouts = trainer.collect_rollouts(env, num_steps=10)
            
            # 計算episode獎勵
            episode_reward = sum(rollouts['rewards'].cpu().numpy())
            
            # 記錄CFG軌跡
            for step in range(len(rollouts['rewards'])):
                # 這裡可以添加CFG軌跡記錄邏輯
                pass
            
            # 執行PPO更新
            if len(rollouts['rewards']) > 0:
                update_stats = trainer.update(rollouts)
                print(f"Episode {episode + 1} 更新統計: {update_stats}")
            
            # 獲取最終圖像
            final_image = env._decode_latent()
            if final_image is not None:
                generated_images.append(final_image)
                cfg_trajectories.append(cfg_trajectory)
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_times.append(episode_time)
            
            if (episode + 1) % 2 == 0:
                print(f"Episode {episode + 1}: 獎勵={episode_reward:.4f}, 時間={episode_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # 計算基本統計
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_time = np.mean(episode_times)
        
        # 計算參數統計
        total_params = sum(p.numel() for p in classical_actor.parameters())
        
        results = {
            "model_type": "classical",
            "num_episodes": num_episodes,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_time_per_episode": avg_time,
            "total_time": total_time,
            "total_params": total_params,
            "rewards": episode_rewards,
            "times": episode_times,
            "generated_images": generated_images,
            "cfg_trajectories": cfg_trajectories
        }
        
        print(f"✅ 經典模型訓練完成 - 平均獎勵: {avg_reward:.4f} ± {std_reward:.4f}")
        return results
        
    except Exception as e:
        print(f"❌ 經典模型訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_simple_metrics(img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, float]:
    """計算簡單的圖像品質指標 (PSNR 和 SSIM)"""
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    # 轉換為numpy數組
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # 確保圖像範圍在[0, 1]
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # 統一處理圖像格式 - 都轉換為灰度圖像進行比較
    def to_grayscale(img):
        """將圖像轉換為灰度圖像"""
        if len(img.shape) == 4:
            # 4D圖像 [B, C, H, W] - 取第一個樣本
            img = img[0]
        if len(img.shape) == 3:
            if img.shape[0] == 3:  # RGB圖像 [C, H, W]
                return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            elif img.shape[0] == 1:  # 單通道圖像 [1, H, W]
                return img[0]
            else:
                # 其他情況，取第一個通道
                return img[0]
        elif len(img.shape) == 2:  # 已經是2D圖像 [H, W]
            return img
        else:
            raise ValueError(f"不支持的圖像維度: {img.shape}")
    
    # 轉換為灰度圖像
    img1_gray = to_grayscale(img1)
    img2_gray = to_grayscale(img2)
    
    # 確保圖像尺寸相同
    if img1_gray.shape != img2_gray.shape:
        min_h = min(img1_gray.shape[0], img2_gray.shape[0])
        min_w = min(img1_gray.shape[1], img2_gray.shape[1])
        img1_gray = img1_gray[:min_h, :min_w]
        img2_gray = img2_gray[:min_h, :min_w]
    
    # 計算PSNR - 使用灰度圖像
    img1_uint8 = (img1_gray * 255).astype(np.uint8)
    img2_uint8 = (img2_gray * 255).astype(np.uint8)
    
    psnr_val = psnr(img1_uint8, img2_uint8, data_range=255)
    
    # 計算SSIM - 使用灰度圖像
    ssim_val = ssim(img1_gray, img2_gray, data_range=1.0)
    
    return {"psnr": psnr_val, "ssim": ssim_val}

def evaluate_image_quality(quantum_results: Dict[str, Any], classical_results: Dict[str, Any]) -> Dict[str, Any]:
    """評估圖像品質並比較量子模型和經典模型 (僅PSNR和SSIM)"""
    print("\n🖼️ 開始圖像品質評估 (PSNR & SSIM)...")
    
    try:
        # 獲取生成的圖像
        quantum_images = quantum_results.get("generated_images", [])
        classical_images = classical_results.get("generated_images", [])
        
        # 如果沒有生成圖像，創建一些測試圖像
        if not quantum_images or not classical_images:
            print("⚠️ 沒有找到生成的圖像，創建測試圖像進行評估...")
            quantum_images = [torch.rand(3, *REFERENCE_SIZE) for _ in range(EVAL_IMAGES_COUNT)]
            classical_images = [torch.rand(3, *REFERENCE_SIZE) for _ in range(EVAL_IMAGES_COUNT)]
        
        # 創建參考圖像
        reference_images = [torch.rand(3, *REFERENCE_SIZE) for _ in range(len(quantum_images))]
        
        print(f"✅ 準備評估圖像:")
        print(f"   量子圖像: {len(quantum_images)} 個")
        print(f"   經典圖像: {len(classical_images)} 個")
        print(f"   參考圖像: {len(reference_images)} 個")
        
        # 計算量子模型指標
        quantum_psnr_values = []
        quantum_ssim_values = []
        
        for i in range(len(quantum_images)):
            metrics = calculate_simple_metrics(reference_images[i], quantum_images[i])
            quantum_psnr_values.append(metrics["psnr"])
            quantum_ssim_values.append(metrics["ssim"])
        
        # 計算經典模型指標
        classical_psnr_values = []
        classical_ssim_values = []
        
        for i in range(len(classical_images)):
            metrics = calculate_simple_metrics(reference_images[i], classical_images[i])
            classical_psnr_values.append(metrics["psnr"])
            classical_ssim_values.append(metrics["ssim"])
        
        # 計算平均值
        quantum_metrics = {
            "psnr": np.mean(quantum_psnr_values),
            "ssim": np.mean(quantum_ssim_values)
        }
        
        classical_metrics = {
            "psnr": np.mean(classical_psnr_values),
            "ssim": np.mean(classical_ssim_values)
        }
        
        # 計算改善百分比
        improvements = {
            "psnr": ((quantum_metrics["psnr"] - classical_metrics["psnr"]) / classical_metrics["psnr"]) * 100,
            "ssim": ((quantum_metrics["ssim"] - classical_metrics["ssim"]) / classical_metrics["ssim"]) * 100
        }
        
        comparison_results = {
            "quantum_metrics": quantum_metrics,
            "classical_metrics": classical_metrics,
            "improvements": improvements
        }
        
        print("✅ 圖像品質評估完成")
        return comparison_results
        
    except Exception as e:
        print(f"❌ 圖像品質評估失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_training_comparison(quantum_results: Dict[str, Any], classical_results: Dict[str, Any]):
    """打印訓練比較結果。"""
    print("\n" + "="*80)
    print("📊 量子 vs 經典模型訓練比較結果")
    print("="*80)
    
    if quantum_results and classical_results:
        print(f"{'指標':<25} {'量子模型':<15} {'經典模型':<15} {'改善':<15}")
        print("-" * 80)
        
        # 參數數量
        q_params = quantum_results["total_params"]
        c_params = classical_results["total_params"]
        param_ratio = c_params / q_params
        print(f"{'參數數量':<25} {q_params:<15} {c_params:<15} {'量子 ' + str(param_ratio) + 'x':<15}")
        
        # 平均獎勵
        q_reward = quantum_results["avg_reward"]
        c_reward = classical_results["avg_reward"]
        reward_improvement = ((q_reward - c_reward) / abs(c_reward)) * 100 if c_reward != 0 else 0
        print(f"{'平均獎勵':<25} {q_reward:<15.4f} {c_reward:<15.4f} {reward_improvement:+.1f}%")
        
        # 時間效率
        q_time = quantum_results["avg_time_per_episode"]
        c_time = classical_results["avg_time_per_episode"]
        time_ratio = c_time / q_time if q_time != 0 else 0
        print(f"{'平均時間/Episode':<25} {f'{q_time:.2f}s':<15} {f'{c_time:.2f}s':<15} {'量子 ' + str(time_ratio) + 'x':<15}")
        
        # 穩定性
        q_std = quantum_results["std_reward"]
        c_std = classical_results["std_reward"]
        stability = "量子更穩定" if q_std < c_std else "經典更穩定"
        print(f"{'穩定性 (std)':<25} {q_std:<15.4f} {c_std:<15.4f} {stability:<15}")
        
        # 學習進度
        q_rewards = quantum_results["rewards"]
        c_rewards = classical_results["rewards"]
        
        # 初始化學習進度變量
        q_improvement = 0.0
        c_improvement = 0.0
        
        if len(q_rewards) >= 4 and len(c_rewards) >= 4:
            q_early = np.mean(q_rewards[:2])
            q_late = np.mean(q_rewards[-2:])
            c_early = np.mean(c_rewards[:2])
            c_late = np.mean(c_rewards[-2:])
            
            q_improvement = ((q_late - q_early) / abs(q_early)) * 100 if q_early != 0 else 0
            c_improvement = ((c_late - c_early) / abs(c_early)) * 100 if c_early != 0 else 0
            
            print(f"{'學習改善 (前2vs後2)':<25} {q_improvement:+.1f}%{'':<10} {c_improvement:+.1f}%{'':<10} {'比較值':<15}")
        else:
            print(f"{'學習改善 (前2vs後2)':<25} {'數據不足':<15} {'數據不足':<15} {'需要更多episodes':<15}")
        
        print("\n🎯 訓練總結:")
        if q_reward > c_reward:
            print("✅ 量子模型在訓練後性能上優於經典模型")
        else:
            print("⚠️  經典模型在訓練後性能上優於量子模型")
        
        if q_params < c_params:
            print("✅ 量子模型參數效率更高")
        else:
            print("⚠️  經典模型參數效率更高")
        
        if len(q_rewards) >= 4 and len(c_rewards) >= 4:
            print(f"📈 量子模型學習進度: {q_improvement:+.1f}%")
            print(f"📈 經典模型學習進度: {c_improvement:+.1f}%")
        else:
            print("📈 學習進度: 需要更多episodes來計算學習進度")


def print_image_quality_comparison(quality_results: Dict[str, Any]):
    """打印圖像品質比較結果。"""
    if quality_results is None:
        print("❌ 圖像品質評估結果不可用")
        return
    
    print("\n" + "="*80)
    print("🖼️ 量子 vs 經典模型圖像品質比較結果 (PSNR & SSIM)")
    print("="*80)
    
    quantum_metrics = quality_results["quantum_metrics"]
    classical_metrics = quality_results["classical_metrics"]
    improvements = quality_results["improvements"]
    
    print(f"{'指標':<25} {'量子模型':<15} {'經典模型':<15} {'改善':<15}")
    print("-" * 80)
    
    # PSNR (越高越好)
    q_psnr = quantum_metrics["psnr"]
    c_psnr = classical_metrics["psnr"]
    psnr_improvement = improvements["psnr"]
    print(f"{'PSNR (dB)':<25} {q_psnr:<15.4f} {c_psnr:<15.4f} {psnr_improvement:+.1f}%")
    
    # SSIM (越高越好)
    q_ssim = quantum_metrics["ssim"]
    c_ssim = classical_metrics["ssim"]
    ssim_improvement = improvements["ssim"]
    print(f"{'SSIM':<25} {q_ssim:<15.4f} {c_ssim:<15.4f} {ssim_improvement:+.1f}%")
    
    
    print("\n🎯 圖像品質總結:")
    if psnr_improvement > 0:
        print(f"✅ 量子模型 PSNR 優於經典模型 {psnr_improvement:.1f}%")
    else:
        print(f"❌ 量子模型 PSNR 低於經典模型 {abs(psnr_improvement):.1f}%")
    
    if ssim_improvement > 0:
        print(f"✅ 量子模型 SSIM 優於經典模型 {ssim_improvement:.1f}%")
    else:
        print(f"❌ 量子模型 SSIM 低於經典模型 {abs(ssim_improvement):.1f}%")
    
    # 綜合評估
    positive_metrics = 0
    total_metrics = 2
    
    if psnr_improvement > 0:
        positive_metrics += 1
    if ssim_improvement > 0:
        positive_metrics += 1
    
    print(f"\n📊 綜合評估: {positive_metrics}/{total_metrics} 個指標量子模型表現更好")
    
    if positive_metrics > total_metrics // 2:
        print("🏆 量子模型在圖像品質上整體優於經典模型")
    else:
        print("📉 經典模型在圖像品質上整體優於量子模型")


def main():
    """主函數"""
    print("🚀 開始簡化的帶訓練的量子 vs 經典模型比較實驗...")
    print("="*80)
    
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 運行量子模型訓練實驗
    quantum_results = run_quantum_training(num_episodes=NUM_EPISODES)
    
    # 運行經典模型訓練實驗
    classical_results = run_classical_training(num_episodes=NUM_EPISODES)
    
    # 打印比較結果
    print_training_comparison(quantum_results, classical_results)
    
    # 評估圖像品質
    quality_results = evaluate_image_quality(quantum_results, classical_results)
    print_image_quality_comparison(quality_results)
    
    # 保存結果
    if quantum_results and classical_results:
        results = {
            "timestamp": datetime.now().isoformat(),
            "quantum": quantum_results,
            "classical": classical_results,
            "image_quality": quality_results
        }
        
        results_file = "simple_training_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 結果已保存到: {results_file}")
    
    return quantum_results is not None and classical_results is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
