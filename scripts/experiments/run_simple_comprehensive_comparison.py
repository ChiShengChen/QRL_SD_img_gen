#!/usr/bin/env python3
"""簡化版全面比較量子與經典模型的腳本。"""

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
from qrl.training import PPOTrainer, QuantumPPOTrainer


def run_quantum_experiment(num_episodes: int = 50) -> Dict[str, Any]:
    """運行量子模型實驗。"""
    print("🔬 開始量子模型實驗...")
    
    try:
        # 初始化環境
        env = DiffusionEnvironment(
            model_id="runwayml/stable-diffusion-v1-5",
            target_class=3,
            max_steps=50,
            device="cpu",
            stage="A"
        )
        
        # 初始化量子 Actor
        quantum_actor = QuantumActor(
            state_dim=6,
            action_dim=1,
            n_qubits=4,
            n_layers=2,
            device="cpu"
        )
        
        # 初始化 Critic
        critic = MLPCritic(
            state_dim=6,
            hidden_dims=[64, 32]
        )
        
        # 運行實驗
        episode_rewards = []
        episode_times = []
        generated_images = []
        cfg_trajectories = []
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 重置環境
            obs, _ = env.reset()
            episode_reward = 0
            cfg_trajectory = []
            
            # 運行一個回合
            for step in range(50):
                # 獲取動作
                action = quantum_actor.sample(torch.FloatTensor(obs).unsqueeze(0).to("cpu"))
                action = action.cpu().numpy()[0]
                
                # 執行動作
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                # 記錄CFG軌跡
                if 'control_result' in info and 'cfg' in info['control_result']:
                    cfg_trajectory.append(info['control_result']['cfg'])
                
                if done or truncated:
                    break
            
            # 獲取最終圖像
            final_image = env._decode_latent()
            if final_image is not None:
                generated_images.append(final_image)
                cfg_trajectories.append(cfg_trajectory)
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_times.append(episode_time)
            
            if (episode + 1) % 10 == 0:
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
        
        print(f"✅ 量子模型完成 - 平均獎勵: {avg_reward:.4f} ± {std_reward:.4f}")
        return results
        
    except Exception as e:
        print(f"❌ 量子模型實驗失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_classical_experiment(num_episodes: int = 50) -> Dict[str, Any]:
    """運行經典模型實驗。"""
    print("\n🔬 開始經典模型實驗...")
    
    try:
        # 初始化環境
        env = DiffusionEnvironment(
            model_id="runwayml/stable-diffusion-v1-5",
            target_class=3,
            max_steps=50,
            device="cpu",
            stage="A"
        )
        
        # 初始化經典 Actor
        classical_actor = ClassicalActor(
            state_dim=6,
            action_dim=1,
            hidden_dims=[64, 32],
            device="cpu"
        )
        
        # 初始化 Critic
        critic = MLPCritic(
            state_dim=6,
            hidden_dims=[64, 32]
        )
        
        # 運行實驗
        episode_rewards = []
        episode_times = []
        generated_images = []
        cfg_trajectories = []
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 重置環境
            obs, _ = env.reset()
            episode_reward = 0
            cfg_trajectory = []
            
            # 運行一個回合
            for step in range(50):
                # 獲取動作
                action = classical_actor.sample(torch.FloatTensor(obs).unsqueeze(0).to("cpu"))
                action = action.cpu().numpy()[0]
                
                # 執行動作
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                # 記錄CFG軌跡
                if 'control_result' in info and 'cfg' in info['control_result']:
                    cfg_trajectory.append(info['control_result']['cfg'])
                
                if done or truncated:
                    break
            
            # 獲取最終圖像
            final_image = env._decode_latent()
            if final_image is not None:
                generated_images.append(final_image)
                cfg_trajectories.append(cfg_trajectory)
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_times.append(episode_time)
            
            if (episode + 1) % 10 == 0:
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
        
        print(f"✅ 經典模型完成 - 平均獎勵: {avg_reward:.4f} ± {std_reward:.4f}")
        return results
        
    except Exception as e:
        print(f"❌ 經典模型實驗失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_basic_metrics(quantum_results: Dict[str, Any], 
                         classical_results: Dict[str, Any]) -> Dict[str, Any]:
    """計算基本評估指標。"""
    print("\n📊 計算基本評估指標...")
    
    metrics = {}
    
    # 基本性能比較
    if quantum_results and classical_results:
        q_reward = quantum_results["avg_reward"]
        c_reward = classical_results["avg_reward"]
        q_std = quantum_results["std_reward"]
        c_std = classical_results["std_reward"]
        q_time = quantum_results["avg_time_per_episode"]
        c_time = classical_results["avg_time_per_episode"]
        q_params = quantum_results["total_params"]
        c_params = classical_results["total_params"]
        
        # 性能改善
        performance_improvement = ((q_reward - c_reward) / abs(c_reward)) * 100 if c_reward != 0 else 0
        
        # 參數效率比
        param_efficiency_ratio = c_params / q_params if q_params > 0 else 0
        
        # 時間效率比
        time_efficiency_ratio = c_time / q_time if q_time > 0 else 0
        
        # 穩定性比較
        stability_improvement = (c_std - q_std) / c_std * 100 if c_std > 0 else 0
        
        metrics["basic_comparison"] = {
            "performance_improvement": performance_improvement,
            "param_efficiency_ratio": param_efficiency_ratio,
            "time_efficiency_ratio": time_efficiency_ratio,
            "stability_improvement": stability_improvement,
            "quantum_avg_reward": q_reward,
            "classical_avg_reward": c_reward,
            "quantum_std": q_std,
            "classical_std": c_std,
            "quantum_time": q_time,
            "classical_time": c_time,
            "quantum_params": q_params,
            "classical_params": c_params
        }
    
    # 控制性能指標
    if quantum_results and classical_results:
        q_cfg_trajectories = quantum_results.get("cfg_trajectories", [])
        c_cfg_trajectories = classical_results.get("cfg_trajectories", [])
        
        # CFG平滑度計算
        def compute_cfg_smoothness(cfg_trajectories):
            if not cfg_trajectories:
                return 0.0
            smoothness_scores = []
            for trajectory in cfg_trajectories:
                if len(trajectory) < 2:
                    continue
                differences = np.abs(np.diff(trajectory))
                smoothness = 1.0 / (1.0 + np.mean(differences))
                smoothness_scores.append(smoothness)
            return np.mean(smoothness_scores) if smoothness_scores else 0.0
        
        q_cfg_smoothness = compute_cfg_smoothness(q_cfg_trajectories)
        c_cfg_smoothness = compute_cfg_smoothness(c_cfg_trajectories)
        
        # CFG範圍計算
        def compute_cfg_range(cfg_trajectories):
            if not cfg_trajectories:
                return 0.0
            all_cfg_values = []
            for trajectory in cfg_trajectories:
                all_cfg_values.extend(trajectory)
            if not all_cfg_values:
                return 0.0
            return np.max(all_cfg_values) - np.min(all_cfg_values)
        
        q_cfg_range = compute_cfg_range(q_cfg_trajectories)
        c_cfg_range = compute_cfg_range(c_cfg_trajectories)
        
        metrics["control_performance"] = {
            "quantum_cfg_smoothness": q_cfg_smoothness,
            "classical_cfg_smoothness": c_cfg_smoothness,
            "quantum_cfg_range": q_cfg_range,
            "classical_cfg_range": c_cfg_range,
            "cfg_smoothness_improvement": (q_cfg_smoothness - c_cfg_smoothness) / c_cfg_smoothness * 100 if c_cfg_smoothness > 0 else 0
        }
    
    # 效率指標
    if quantum_results and classical_results:
        q_reward = quantum_results["avg_reward"]
        c_reward = classical_results["avg_reward"]
        q_params = quantum_results["total_params"]
        c_params = classical_results["total_params"]
        q_time = quantum_results["avg_time_per_episode"]
        c_time = classical_results["avg_time_per_episode"]
        
        # 參數效率
        q_param_efficiency = q_reward / q_params if q_params > 0 else 0
        c_param_efficiency = c_reward / c_params if c_params > 0 else 0
        
        # 時間效率
        q_time_efficiency = 1.0 / q_time if q_time > 0 else 0
        c_time_efficiency = 1.0 / c_time if c_time > 0 else 0
        
        metrics["efficiency"] = {
            "quantum_param_efficiency": q_param_efficiency,
            "classical_param_efficiency": c_param_efficiency,
            "quantum_time_efficiency": q_time_efficiency,
            "classical_time_efficiency": c_time_efficiency,
            "param_efficiency_improvement": (q_param_efficiency - c_param_efficiency) / abs(c_param_efficiency) * 100 if c_param_efficiency != 0 else 0,
            "time_efficiency_improvement": (q_time_efficiency - c_time_efficiency) / c_time_efficiency * 100 if c_time_efficiency > 0 else 0
        }
    
    return metrics


def print_comprehensive_comparison(quantum_results: Dict[str, Any], 
                                 classical_results: Dict[str, Any],
                                 comprehensive_metrics: Dict[str, Any]):
    """打印全面的比較結果。"""
    print("\n" + "="*80)
    print("📊 量子 vs 經典模型全面比較結果")
    print("="*80)
    
    # 基本性能比較
    print("\n🎯 基本性能指標")
    print("-" * 60)
    print(f"{'指標':<25} {'量子模型':<15} {'經典模型':<15} {'改善':<15}")
    print("-" * 60)
    
    if quantum_results and classical_results:
        # 參數數量
        q_params = quantum_results["total_params"]
        c_params = classical_results["total_params"]
        param_ratio = c_params / q_params
        print(f"{'參數數量':<25} {q_params:<15} {c_params:<15} {f'量子 {param_ratio:.1f}x':<15}")
        
        # 平均獎勵
        q_reward = quantum_results["avg_reward"]
        c_reward = classical_results["avg_reward"]
        reward_improvement = ((q_reward - c_reward) / abs(c_reward)) * 100 if c_reward != 0 else 0
        print(f"{'平均獎勵':<25} {q_reward:.4f:<15} {c_reward:.4f:<15} {f'{reward_improvement:+.1f}%':<15}")
        
        # 時間效率
        q_time = quantum_results["avg_time_per_episode"]
        c_time = classical_results["avg_time_per_episode"]
        time_ratio = c_time / q_time if q_time != 0 else 0
        print(f"{'平均時間/Episode':<25} {f'{q_time:.2f}s':<15} {f'{c_time:.2f}s':<15} {f'量子 {time_ratio:.1f}x':<15}")
        
        # 穩定性
        q_std = quantum_results["std_reward"]
        c_std = classical_results["std_reward"]
        stability = "量子更穩定" if q_std < c_std else "經典更穩定"
        print(f"{'穩定性 (std)':<25} {q_std:.4f:<15} {c_std:.4f:<15} {stability:<15}")
    
    # 控制性能指標
    if "control_performance" in comprehensive_metrics:
        print("\n🎮 控制性能指標")
        print("-" * 60)
        ctrl = comprehensive_metrics["control_performance"]
        
        q_cfg_smooth = ctrl["quantum_cfg_smoothness"]
        c_cfg_smooth = ctrl["classical_cfg_smoothness"]
        smooth_improvement = ctrl["cfg_smoothness_improvement"]
        better_smooth = "量子" if q_cfg_smooth > c_cfg_smooth else "經典"
        print(f"{'CFG平滑度':<25} {q_cfg_smooth:.4f:<15} {c_cfg_smooth:.4f:<15} {f'{smooth_improvement:+.1f}% ({better_smooth})':<15}")
        
        q_cfg_range = ctrl["quantum_cfg_range"]
        c_cfg_range = ctrl["classical_cfg_range"]
        print(f"{'CFG範圍':<25} {q_cfg_range:.4f:<15} {c_cfg_range:.4f:<15} {'比較值':<15}")
    
    # 效率指標
    if "efficiency" in comprehensive_metrics:
        print("\n⚡ 效率指標")
        print("-" * 60)
        eff = comprehensive_metrics["efficiency"]
        
        q_param_eff = eff["quantum_param_efficiency"]
        c_param_eff = eff["classical_param_efficiency"]
        param_eff_improvement = eff["param_efficiency_improvement"]
        better_param = "量子" if q_param_eff > c_param_eff else "經典"
        print(f"{'參數效率':<25} {q_param_eff:.6f:<15} {c_param_eff:.6f:<15} {f'{param_eff_improvement:+.1f}% ({better_param})':<15}")
        
        q_time_eff = eff["quantum_time_efficiency"]
        c_time_eff = eff["classical_time_efficiency"]
        time_eff_improvement = eff["time_efficiency_improvement"]
        better_time = "量子" if q_time_eff > c_time_eff else "經典"
        print(f"{'時間效率':<25} {q_time_eff:.4f:<15} {c_time_eff:.4f:<15} {f'{time_eff_improvement:+.1f}% ({better_time})':<15}")
    
    print("\n🎯 總結:")
    if quantum_results and classical_results:
        q_reward = quantum_results["avg_reward"]
        c_reward = classical_results["avg_reward"]
        q_params = quantum_results["total_params"]
        c_params = classical_results["total_params"]
        
        if q_reward > c_reward:
            print("✅ 量子模型在基本性能上優於經典模型")
        else:
            print("⚠️  經典模型在基本性能上優於量子模型")
        
        if q_params < c_params:
            print("✅ 量子模型參數效率更高")
        else:
            print("⚠️  經典模型參數效率更高")


def main():
    """主函數"""
    print("🚀 開始簡化版全面比較實驗...")
    print("="*80)
    
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 運行量子模型實驗
    quantum_results = run_quantum_experiment(num_episodes=30)  # 減少episode數以節省時間
    
    # 運行經典模型實驗
    classical_results = run_classical_experiment(num_episodes=30)
    
    # 計算基本指標
    comprehensive_metrics = compute_basic_metrics(quantum_results, classical_results)
    
    # 打印比較結果
    print_comprehensive_comparison(quantum_results, classical_results, comprehensive_metrics)
    
    # 保存結果
    if quantum_results and classical_results:
        results = {
            "timestamp": datetime.now().isoformat(),
            "quantum": quantum_results,
            "classical": classical_results,
            "comprehensive_metrics": comprehensive_metrics
        }
        
        results_file = "simple_comprehensive_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str處理numpy類型
        
        print(f"\n💾 結果已保存到: {results_file}")
    
    return quantum_results is not None and classical_results is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
