#!/usr/bin/env python3
"""帶訓練的量子與經典模型比較腳本。"""

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
from qrl.training.buffers import PPOBuffer


def run_quantum_training(num_episodes: int = 20) -> Dict[str, Any]:
    """運行量子模型訓練實驗。"""
    print("🔬 開始量子模型訓練實驗...")
    
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
        
        # 初始化優化器
        actor_optimizer = torch.optim.Adam(quantum_actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        
        # 運行訓練實驗
        episode_rewards = []
        episode_times = []
        generated_images = []
        cfg_trajectories = []
        
        start_time = time.time()
        
        # 初始化經驗緩衝區
        buffer = PPOBuffer(
            state_dim=6,
            action_dim=1,
            rollout_steps=256,
            num_envs=1,
            device="cpu"
        )
        
        # 訓練參數
        rollout_steps = 8
        update_epochs = 2
        mini_batch_size = 4
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 重置環境
            obs, _ = env.reset()
            episode_reward = 0
            cfg_trajectory = []
            
            # 收集經驗數據
            for step in range(rollout_steps):
                # 獲取動作和價值
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to("cpu")
                
            with torch.no_grad():
                action = quantum_actor.sample(obs_tensor)
                log_prob = quantum_actor.log_prob(obs_tensor, action)
                value = critic(obs_tensor)
            
            action_np = action.detach().cpu().numpy()[0]
            log_prob_np = log_prob.detach().cpu().numpy()[0]
            value_np = value.detach().cpu().numpy()[0]
            
            # 執行動作
            next_obs, reward, done, truncated, info = env.step(action_np)
            episode_reward += reward
            
            # 記錄CFG軌跡
            if 'control_result' in info and 'cfg' in info['control_result']:
                cfg_trajectory.append(info['control_result']['cfg'])
            
            # 存儲經驗
            buffer.add(
                state=torch.FloatTensor(obs).unsqueeze(0),
                action=torch.FloatTensor([action_np]),
                reward=torch.FloatTensor([reward]),
                value=torch.FloatTensor([value_np]),
                log_prob=torch.FloatTensor([log_prob_np]),
                done=torch.BoolTensor([done or truncated])
            )
            
            obs = next_obs
            
            if done or truncated:
                break
            
            # 計算最後的價值
            if not (done or truncated):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to("cpu")
                    last_value = critic(obs_tensor)
                    buffer.store(
                        state=obs,
                        action=0,
                        reward=0,
                        value=last_value.detach().cpu().numpy()[0],
                        log_prob=0,
                        done=True
                    )
            
            # 完成緩衝區
            buffer.finish_path(last_value=0 if (done or truncated) else last_value.detach().cpu().numpy()[0])
            
            # 每收集足夠的經驗就進行更新
            if buffer.size >= rollout_steps:
                # 獲取經驗數據
                rollouts = buffer.get()
                
                # 進行PPO更新
                for _ in range(update_epochs):
                    # 隨機打亂數據
                    indices = torch.randperm(len(rollouts['states']))
                    
                    # 小批次更新
                    for start_idx in range(0, len(rollouts['states']), mini_batch_size):
                        end_idx = min(start_idx + mini_batch_size, len(rollouts['states']))
                        batch_indices = indices[start_idx:end_idx]
                        
                        # 獲取批次數據
                        batch_states = rollouts['states'][batch_indices]
                        batch_actions = rollouts['actions'][batch_indices]
                        batch_old_log_probs = rollouts['log_probs'][batch_indices]
                        batch_advantages = rollouts['advantages'][batch_indices]
                        batch_returns = rollouts['returns'][batch_indices]
                        
                        # 計算新的動作概率和價值
                        new_actions = quantum_actor.sample(batch_states)
                        new_log_probs = quantum_actor.log_prob(batch_states, new_actions)
                        new_values = critic(batch_states)
                        
                        # 計算比率
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        
                        # PPO目標
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - 0.1, 1 + 0.1) * batch_advantages
                        
                        # 策略損失
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # 價值損失
                        value_loss = torch.nn.functional.mse_loss(new_values.squeeze(), batch_returns)
                        
                        # 總損失
                        total_loss = policy_loss + 0.5 * value_loss
                        
                        # 反向傳播
                        total_loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(quantum_actor.parameters(), 0.5)
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                        
                        # 更新參數
                        actor_optimizer.step()
                        critic_optimizer.step()
                        
                        # 清零梯度
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                
                # 清空緩衝區
                buffer.clear()
            
            # 獲取最終圖像
            final_image = env._decode_latent()
            if final_image is not None:
                generated_images.append(final_image)
                cfg_trajectories.append(cfg_trajectory)
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_times.append(episode_time)
            
            if (episode + 1) % 5 == 0:
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


def run_classical_training(num_episodes: int = 20) -> Dict[str, Any]:
    """運行經典模型訓練實驗。"""
    print("\n🔬 開始經典模型訓練實驗...")
    
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
        
        # 初始化優化器
        actor_optimizer = torch.optim.Adam(classical_actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        
        # 運行訓練實驗
        episode_rewards = []
        episode_times = []
        generated_images = []
        cfg_trajectories = []
        
        start_time = time.time()
        
        # 初始化經驗緩衝區
        buffer = PPOBuffer(
            state_dim=6,
            action_dim=1,
            rollout_steps=256,
            num_envs=1,
            device="cpu"
        )
        
        # 訓練參數
        rollout_steps = 8
        update_epochs = 2
        mini_batch_size = 4
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 重置環境
            obs, _ = env.reset()
            episode_reward = 0
            cfg_trajectory = []
            
            # 收集經驗數據
            for step in range(rollout_steps):
                # 獲取動作和價值
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to("cpu")
                
            with torch.no_grad():
                action = classical_actor.sample(obs_tensor)
                log_prob = classical_actor.log_prob(obs_tensor, action)
                value = critic(obs_tensor)
                
                action_np = action.detach().cpu().numpy()[0]
                log_prob_np = log_prob.detach().cpu().numpy()[0]
                value_np = value.detach().cpu().numpy()[0]
                
                # 執行動作
                next_obs, reward, done, truncated, info = env.step(action_np)
                episode_reward += reward
                
                # 記錄CFG軌跡
                if 'control_result' in info and 'cfg' in info['control_result']:
                    cfg_trajectory.append(info['control_result']['cfg'])
                
                # 存儲經驗
                buffer.store(
                    state=obs,
                    action=action_np,
                    reward=reward,
                    value=value_np,
                    log_prob=log_prob_np,
                    done=done or truncated
                )
                
                obs = next_obs
                
                if done or truncated:
                    break
            
            # 計算最後的價值
            if not (done or truncated):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to("cpu")
                    last_value = critic(obs_tensor)
                    buffer.store(
                        state=obs,
                        action=0,
                        reward=0,
                        value=last_value.detach().cpu().numpy()[0],
                        log_prob=0,
                        done=True
                    )
            
            # 完成緩衝區
            buffer.finish_path(last_value=0 if (done or truncated) else last_value.detach().cpu().numpy()[0])
            
            # 每收集足夠的經驗就進行更新
            if buffer.size >= rollout_steps:
                # 獲取經驗數據
                rollouts = buffer.get()
                
                # 進行PPO更新
                for _ in range(update_epochs):
                    # 隨機打亂數據
                    indices = torch.randperm(len(rollouts['states']))
                    
                    # 小批次更新
                    for start_idx in range(0, len(rollouts['states']), mini_batch_size):
                        end_idx = min(start_idx + mini_batch_size, len(rollouts['states']))
                        batch_indices = indices[start_idx:end_idx]
                        
                        # 獲取批次數據
                        batch_states = rollouts['states'][batch_indices]
                        batch_actions = rollouts['actions'][batch_indices]
                        batch_old_log_probs = rollouts['log_probs'][batch_indices]
                        batch_advantages = rollouts['advantages'][batch_indices]
                        batch_returns = rollouts['returns'][batch_indices]
                        
                        # 計算新的動作概率和價值
                        new_actions = classical_actor.sample(batch_states)
                        new_log_probs = classical_actor.log_prob(batch_states, new_actions)
                        new_values = critic(batch_states)
                        
                        # 計算比率
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        
                        # PPO目標
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - 0.1, 1 + 0.1) * batch_advantages
                        
                        # 策略損失
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # 價值損失
                        value_loss = torch.nn.functional.mse_loss(new_values.squeeze(), batch_returns)
                        
                        # 總損失
                        total_loss = policy_loss + 0.5 * value_loss
                        
                        # 反向傳播
                        total_loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(classical_actor.parameters(), 0.5)
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                        
                        # 更新參數
                        actor_optimizer.step()
                        critic_optimizer.step()
                        
                        # 清零梯度
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                
                # 清空緩衝區
                buffer.clear()
            
            # 獲取最終圖像
            final_image = env._decode_latent()
            if final_image is not None:
                generated_images.append(final_image)
                cfg_trajectories.append(cfg_trajectory)
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_times.append(episode_time)
            
            if (episode + 1) % 5 == 0:
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
        
        # 學習進度
        q_rewards = quantum_results["rewards"]
        c_rewards = classical_results["rewards"]
        
        if len(q_rewards) >= 10 and len(c_rewards) >= 10:
            q_early = np.mean(q_rewards[:5])
            q_late = np.mean(q_rewards[-5:])
            c_early = np.mean(c_rewards[:5])
            c_late = np.mean(c_rewards[-5:])
            
            q_improvement = ((q_late - q_early) / abs(q_early)) * 100 if q_early != 0 else 0
            c_improvement = ((c_late - c_early) / abs(c_early)) * 100 if c_early != 0 else 0
            
            print(f"{'學習改善 (前5vs後5)':<25} {f'{q_improvement:+.1f}%':<15} {f'{c_improvement:+.1f}%':<15} {'比較值':<15}")
        
        print("\n🎯 訓練總結:")
        if q_reward > c_reward:
            print("✅ 量子模型在訓練後性能上優於經典模型")
        else:
            print("⚠️  經典模型在訓練後性能上優於量子模型")
        
        if q_params < c_params:
            print("✅ 量子模型參數效率更高")
        else:
            print("⚠️  經典模型參數效率更高")
        
        print(f"📈 量子模型學習進度: {q_improvement:+.1f}%")
        print(f"📈 經典模型學習進度: {c_improvement:+.1f}%")


def main():
    """主函數"""
    print("🚀 開始帶訓練的量子 vs 經典模型比較實驗...")
    print("="*80)
    
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 運行量子模型訓練實驗
    quantum_results = run_quantum_training(num_episodes=20)
    
    # 運行經典模型訓練實驗
    classical_results = run_classical_training(num_episodes=20)
    
    # 打印比較結果
    print_training_comparison(quantum_results, classical_results)
    
    # 保存結果
    if quantum_results and classical_results:
        results = {
            "timestamp": datetime.now().isoformat(),
            "quantum": quantum_results,
            "classical": classical_results
        }
        
        results_file = "training_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 結果已保存到: {results_file}")
    
    return quantum_results is not None and classical_results is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
