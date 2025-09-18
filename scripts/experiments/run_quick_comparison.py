#!/usr/bin/env python3
"""
快速比較腳本
運行量子 vs 經典模型的快速比較
"""

import torch
import numpy as np
import time
from pathlib import Path
import sys
import json
from datetime import datetime

# 添加項目路徑
sys.path.append(str(Path(__file__).parent))

def run_quantum_experiment():
    """運行量子模型實驗"""
    print("🔬 開始量子模型實驗...")
    
    try:
        from qrl.actors.quantum_actor import QuantumActor
        from qrl.actors.critic import Critic
        from qrl.training.ppo_trainer import QuantumPPOTrainer
        from qrl.envs.diffusion_env import DiffusionEnvironment
        
        # 設置設備
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {device}")
        
        # 創建環境
        env = DiffusionEnvironment(
            model_id="runwayml/stable-diffusion-v1-5",
            max_steps=5,  # 減少步數用於快速測試
            device=device
        )
        
        # 創建量子 Actor
        actor = QuantumActor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            n_qubits=8,
            n_layers=4,
            backend="default.qubit",
            device=device
        )
        
        # 創建 Critic
        critic = Critic(
            state_dim=env.observation_space.shape[0],
            hidden_dims=[64, 32]
        )
        
        # 創建訓練器
        trainer = QuantumPPOTrainer(
            actor=actor,
            critic=critic,
            device=device,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.995,
            gae_lambda=0.95,
            clip_ratio=0.1,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=1,
            mini_batch_size=4,
            rollout_steps=16,
            num_envs=1,
            mixed_precision=False,
            use_quantum_gradients=False
        )
        
        # 運行實驗
        num_episodes = 100
        rewards = []
        times = []
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # 收集經驗
            obs, _ = env.reset()
            episode_reward = 0
            
            for step in range(env.max_steps):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = actor.sample(obs_tensor)
                    action = action.squeeze().cpu().numpy()
                
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                obs = next_obs
                
                if done or truncated:
                    break
            
            episode_time = time.time() - start_time
            rewards.append(episode_reward)
            times.append(episode_time)
            
            print(f"Episode {episode + 1}: 獎勵={episode_reward:.4f}, 時間={episode_time:.2f}s")
        
        # 計算統計
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_time = np.mean(times)
        
        results = {
            "model_type": "quantum",
            "num_episodes": num_episodes,
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "avg_time_per_episode": float(avg_time),
            "total_params": sum(p.numel() for p in actor.parameters()),
            "rewards": rewards,
            "times": times
        }
        
        print(f"✅ 量子模型完成 - 平均獎勵: {avg_reward:.4f} ± {std_reward:.4f}")
        return results
        
    except Exception as e:
        print(f"❌ 量子模型實驗失敗: {e}")
        return None

def run_classical_experiment():
    """運行經典模型實驗"""
    print("\n🔬 開始經典模型實驗...")
    
    try:
        from qrl.actors.classical_actor import ClassicalActor
        from qrl.actors.critic import Critic
        from qrl.training.ppo_trainer import PPOTrainer
        from qrl.envs.diffusion_env import DiffusionEnvironment
        
        # 設置設備
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {device}")
        
        # 創建環境
        env = DiffusionEnvironment(
            model_id="runwayml/stable-diffusion-v1-5",
            max_steps=5,  # 減少步數用於快速測試
            device=device
        )
        
        # 創建經典 Actor
        actor = ClassicalActor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dims=[64, 32]  # 減少參數量
        )
        
        # 創建 Critic
        critic = Critic(
            state_dim=env.observation_space.shape[0],
            hidden_dims=[64, 32]
        )
        
        # 創建訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            device=device,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.995,
            gae_lambda=0.95,
            clip_ratio=0.1,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=1,
            mini_batch_size=4,
            rollout_steps=16,
            num_envs=1,
            mixed_precision=False,
            use_quantum_gradients=False
        )
        
        # 運行實驗
        num_episodes = 100
        rewards = []
        times = []
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # 收集經驗
            obs, _ = env.reset()
            episode_reward = 0
            
            for step in range(env.max_steps):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = actor.sample(obs_tensor)
                    action = action.squeeze().cpu().numpy()
                
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                obs = next_obs
                
                if done or truncated:
                    break
            
            episode_time = time.time() - start_time
            rewards.append(episode_reward)
            times.append(episode_time)
            
            print(f"Episode {episode + 1}: 獎勵={episode_reward:.4f}, 時間={episode_time:.2f}s")
        
        # 計算統計
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_time = np.mean(times)
        
        results = {
            "model_type": "classical",
            "num_episodes": num_episodes,
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "avg_time_per_episode": float(avg_time),
            "total_params": sum(p.numel() for p in actor.parameters()),
            "rewards": rewards,
            "times": times
        }
        
        print(f"✅ 經典模型完成 - 平均獎勵: {avg_reward:.4f} ± {std_reward:.4f}")
        return results
        
    except Exception as e:
        print(f"❌ 經典模型實驗失敗: {e}")
        return None

def print_comparison(quantum_results, classical_results):
    """打印比較結果"""
    print("\n" + "="*60)
    print("📊 量子 vs 經典模型比較結果")
    print("="*60)
    
    if quantum_results and classical_results:
        print(f"{'指標':<20} {'量子模型':<15} {'經典模型':<15} {'優勢':<10}")
        print("-"*60)
        
        # 參數數量比較
        q_params = quantum_results["total_params"]
        c_params = classical_results["total_params"]
        param_ratio = c_params / q_params
        print(f"{'參數數量':<20} {q_params:<15} {c_params:<15} {f'量子 {param_ratio:.1f}x':<10}")
        
        # 平均獎勵比較
        q_reward = quantum_results["avg_reward"]
        c_reward = classical_results["avg_reward"]
        reward_improvement = ((q_reward - c_reward) / abs(c_reward)) * 100 if c_reward != 0 else 0
        q_reward_str = f"{q_reward:.4f}"
        c_reward_str = f"{c_reward:.4f}"
        improvement_str = f"{reward_improvement:+.1f}%"
        print(f"{'平均獎勵':<20} {q_reward_str:<15} {c_reward_str:<15} {improvement_str:<10}")
        
        # 時間效率比較
        q_time = quantum_results["avg_time_per_episode"]
        c_time = classical_results["avg_time_per_episode"]
        time_ratio = c_time / q_time if q_time != 0 else 0
        q_time_str = f"{q_time:.2f}s"
        c_time_str = f"{c_time:.2f}s"
        time_ratio_str = f"量子 {time_ratio:.1f}x"
        print(f"{'平均時間/Episode':<20} {q_time_str:<15} {c_time_str:<15} {time_ratio_str:<10}")
        
        # 穩定性比較
        q_std = quantum_results["std_reward"]
        c_std = classical_results["std_reward"]
        stability = "量子更穩定" if q_std < c_std else "經典更穩定"
        q_std_str = f"{q_std:.4f}"
        c_std_str = f"{c_std:.4f}"
        print(f"{'穩定性 (std)':<20} {q_std_str:<15} {c_std_str:<15} {stability:<10}")
        
        print("\n🎯 結論:")
        if q_reward > c_reward:
            print("✅ 量子模型在性能上優於經典模型")
        else:
            print("⚠️  經典模型在性能上優於量子模型")
        
        if q_params < c_params:
            print("✅ 量子模型參數效率更高")
        else:
            print("⚠️  經典模型參數效率更高")
            
    else:
        print("❌ 無法完成比較，部分實驗失敗")

def main():
    """主函數"""
    print("🚀 開始快速比較實驗...")
    print("="*60)
    
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 運行量子模型實驗
    quantum_results = run_quantum_experiment()
    
    # 運行經典模型實驗
    classical_results = run_classical_experiment()
    
    # 打印比較結果
    print_comparison(quantum_results, classical_results)
    
    # 保存結果
    if quantum_results and classical_results:
        results = {
            "timestamp": datetime.now().isoformat(),
            "quantum": quantum_results,
            "classical": classical_results
        }
        
        results_file = "quick_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 結果已保存到: {results_file}")
    
    return quantum_results is not None and classical_results is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
