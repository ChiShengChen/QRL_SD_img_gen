#!/usr/bin/env python3
"""全面比較量子與經典模型的腳本。"""

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
from qrl.metrics.image_quality import ImageQualityMetrics
from qrl.metrics.control_performance import ControlMetrics
from qrl.metrics.efficiency import EfficiencyMetrics
from qrl.metrics.quantum_metrics import QuantumMetrics


def run_quantum_experiment(num_episodes: int = 100) -> Dict[str, Any]:
    """運行量子模型實驗。"""
    print("🔬 開始量子模型實驗...")
    
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
    
    # 初始化訓練器
    trainer = QuantumPPOTrainer(
        actor=quantum_actor,
        critic=critic,
            device="cpu",
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
    episode_rewards = []
    episode_times = []
    generated_images = []
    cfg_trajectories = []
    
    start_time = time.time()
    
    # 初始化經驗緩衝區
    from qrl.training.buffers import PPOBuffer
    buffer = PPOBuffer(
        state_dim=6,
        action_dim=1,
        rollout_steps=512,
        num_envs=1,
        device="cpu"
    )
    
    # 訓練參數
    rollout_steps = 16
    update_epochs = 4
    mini_batch_size = 8
    
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
                buffer.add(
                    state=torch.FloatTensor(obs).unsqueeze(0),
                    action=torch.FloatTensor([0]),  # 虛擬動作
                    reward=torch.FloatTensor([0]),  # 虛擬獎勵
                    value=torch.FloatTensor([last_value.detach().cpu().numpy()[0]]),
                    log_prob=torch.FloatTensor([0]),  # 虛擬log_prob
                    done=torch.BoolTensor([True])
                )
        
        # 完成緩衝區 - PPOBuffer 不需要 finish_path
        
        # 每收集足夠的經驗就進行更新
        if buffer.is_full():
            # 計算優勢
            last_value = torch.FloatTensor([0])  # 簡化處理
            advantages, returns = buffer.compute_advantages(last_value)
            
            # 獲取經驗數據
            rollouts = buffer.get_batch(mini_batch_size)
            
            # 進行PPO更新
            for _ in range(update_epochs):
                # 獲取批次數據
                batch_states = rollouts['states']
                batch_actions = rollouts['actions']
                batch_old_log_probs = rollouts['log_probs']
                batch_advantages = advantages[:len(batch_states)]
                batch_returns = returns[:len(batch_states)]
                    
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
                trainer.actor_optimizer.step()
                trainer.critic_optimizer.step()
                
                # 清零梯度
                trainer.actor_optimizer.zero_grad()
                trainer.critic_optimizer.zero_grad()
            
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


def run_classical_experiment(num_episodes: int = 100) -> Dict[str, Any]:
    """運行經典模型實驗。"""
    print("\n🔬 開始經典模型實驗...")
    
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
    
    # 初始化訓練器
    trainer = PPOTrainer(
        actor=classical_actor,
        critic=critic,
            device="cpu",
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
    episode_rewards = []
    episode_times = []
    generated_images = []
    cfg_trajectories = []
    
    start_time = time.time()
    
    # 初始化經驗緩衝區
    from qrl.training.buffers import PPOBuffer
    buffer = PPOBuffer(
        state_dim=6,
        action_dim=1,
        rollout_steps=512,
        num_envs=1,
        device="cpu"
    )
    
    # 訓練參數
    rollout_steps = 16
    update_epochs = 4
    mini_batch_size = 8
    
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
                buffer.add(
                    state=torch.FloatTensor(obs).unsqueeze(0),
                    action=torch.FloatTensor([0]),  # 虛擬動作
                    reward=torch.FloatTensor([0]),  # 虛擬獎勵
                    value=torch.FloatTensor([last_value.detach().cpu().numpy()[0]]),
                    log_prob=torch.FloatTensor([0]),  # 虛擬log_prob
                    done=torch.BoolTensor([True])
                )
        
        # 完成緩衝區 - PPOBuffer 不需要 finish_path
        
        # 每收集足夠的經驗就進行更新
        if buffer.is_full():
            # 計算優勢
            last_value = torch.FloatTensor([0])  # 簡化處理
            advantages, returns = buffer.compute_advantages(last_value)
            
            # 獲取經驗數據
            rollouts = buffer.get_batch(mini_batch_size)
            
            # 進行PPO更新
            for _ in range(update_epochs):
                # 獲取批次數據
                batch_states = rollouts['states']
                batch_actions = rollouts['actions']
                batch_old_log_probs = rollouts['log_probs']
                batch_advantages = advantages[:len(batch_states)]
                batch_returns = returns[:len(batch_states)]
                
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
                trainer.actor_optimizer.step()
                trainer.critic_optimizer.step()
                
                # 清零梯度
                trainer.actor_optimizer.zero_grad()
                trainer.critic_optimizer.zero_grad()
            
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


def compute_comprehensive_metrics(quantum_results: Dict[str, Any], 
                                 classical_results: Dict[str, Any]) -> Dict[str, Any]:
    """計算全面的評估指標。"""
    print("\n📊 計算全面評估指標...")
    
    # 初始化指標計算器
    image_metrics = ImageQualityMetrics(device="cuda")
    control_metrics = ControlMetrics()
    efficiency_metrics = EfficiencyMetrics()
    quantum_metrics = QuantumMetrics(device="cuda")
    
    # 創建臨時模型對象用於指標計算
    quantum_actor = QuantumActor(state_dim=6, action_dim=1, hidden_dim=64, 
                                num_qubits=4, num_layers=2, device="cuda")
    classical_actor = ClassicalActor(state_dim=6, action_dim=1, hidden_dim=64, device="cuda")
    
    metrics = {}
    
    # 1. 圖像質量指標
    print("  📸 計算圖像質量指標...")
    if quantum_results.get("generated_images"):
        q_image_metrics = image_metrics.compute_all(quantum_results["generated_images"])
        metrics["quantum_image_quality"] = q_image_metrics
    
    if classical_results.get("generated_images"):
        c_image_metrics = image_metrics.compute_all(classical_results["generated_images"])
        metrics["classical_image_quality"] = c_image_metrics
    
    # 2. 控制性能指標
    print("  🎮 計算控制性能指標...")
    if quantum_results.get("rewards") and quantum_results.get("cfg_trajectories"):
        q_control_metrics = control_metrics.compute_all(
            quantum_results["rewards"], 
            quantum_results["cfg_trajectories"]
        )
        metrics["quantum_control_performance"] = q_control_metrics
    
    if classical_results.get("rewards") and classical_results.get("cfg_trajectories"):
        c_control_metrics = control_metrics.compute_all(
            classical_results["rewards"], 
            classical_results["cfg_trajectories"]
        )
        metrics["classical_control_performance"] = c_control_metrics
    
    # 3. 效率指標
    print("  ⚡ 計算效率指標...")
    if quantum_results.get("generated_images") and quantum_results.get("rewards"):
        q_efficiency_metrics = efficiency_metrics.compute_all(
            quantum_actor, 
            quantum_results["generated_images"], 
            quantum_results["rewards"]
        )
        metrics["quantum_efficiency"] = q_efficiency_metrics
    
    if classical_results.get("generated_images") and classical_results.get("rewards"):
        c_efficiency_metrics = efficiency_metrics.compute_all(
            classical_actor, 
            classical_results["generated_images"], 
            classical_results["rewards"]
        )
        metrics["classical_efficiency"] = c_efficiency_metrics
    
    # 4. 量子特定指標
    print("  🔬 計算量子特定指標...")
    if quantum_results.get("generated_images") and quantum_results.get("rewards"):
        q_quantum_metrics = quantum_metrics.compute_all(
            quantum_actor, 
            classical_actor, 
            quantum_results["generated_images"], 
            quantum_results["rewards"]
        )
        metrics["quantum_specific"] = q_quantum_metrics
    
    # 5. 量子 vs 經典比較
    print("  ⚖️ 計算量子 vs 經典比較...")
    if quantum_results.get("rewards") and classical_results.get("rewards"):
        comparison_metrics = quantum_metrics.compute_quantum_vs_classical_comparison(
            quantum_actor, 
            classical_actor,
            quantum_results["rewards"],
            classical_results["rewards"]
        )
        metrics["quantum_vs_classical"] = comparison_metrics
    
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
    print(f"{'指標':<25} {'量子模型':<15} {'經典模型':<15} {'優勢':<15}")
    print("-" * 60)
    
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
    
    # 圖像質量指標
    if "quantum_image_quality" in comprehensive_metrics and "classical_image_quality" in comprehensive_metrics:
        print("\n📸 圖像質量指標")
        print("-" * 60)
        q_img = comprehensive_metrics["quantum_image_quality"]
        c_img = comprehensive_metrics["classical_image_quality"]
        
        for metric in ["classification_accuracy", "fid_score", "inception_score", "lpips"]:
            if metric in q_img and metric in c_img:
                q_val = q_img[metric]
                c_val = c_img[metric]
                if metric == "fid_score":  # FID越低越好
                    better = "量子" if q_val < c_val else "經典"
                else:  # 其他指標越高越好
                    better = "量子" if q_val > c_val else "經典"
                print(f"{metric:<25} {q_val:.4f:<15} {c_val:.4f:<15} {better:<15}")
    
    # 控制性能指標
    if "quantum_control_performance" in comprehensive_metrics and "classical_control_performance" in comprehensive_metrics:
        print("\n🎮 控制性能指標")
        print("-" * 60)
        q_ctrl = comprehensive_metrics["quantum_control_performance"]
        c_ctrl = comprehensive_metrics["classical_control_performance"]
        
        for metric in ["convergence_speed", "cfg_smoothness", "policy_stability", "cfg_efficiency"]:
            if metric in q_ctrl and metric in c_ctrl:
                q_val = q_ctrl[metric]
                c_val = c_ctrl[metric]
                better = "量子" if q_val > c_val else "經典"
                print(f"{metric:<25} {q_val:.4f:<15} {c_val:.4f:<15} {better:<15}")
    
    # 效率指標
    if "quantum_efficiency" in comprehensive_metrics and "classical_efficiency" in comprehensive_metrics:
        print("\n⚡ 效率指標")
        print("-" * 60)
        q_eff = comprehensive_metrics["quantum_efficiency"]
        c_eff = comprehensive_metrics["classical_efficiency"]
        
        for metric in ["parameter_efficiency", "inference_efficiency", "memory_efficiency", "training_efficiency"]:
            if metric in q_eff and metric in c_eff:
                q_val = q_eff[metric]
                c_val = c_eff[metric]
                better = "量子" if q_val > c_val else "經典"
                print(f"{metric:<25} {q_val:.4f:<15} {c_val:.4f:<15} {better:<15}")
    
    # 量子特定指標
    if "quantum_specific" in comprehensive_metrics:
        print("\n🔬 量子特定指標")
        print("-" * 60)
        q_quantum = comprehensive_metrics["quantum_specific"]
        
        for metric in ["quantum_coherence", "quantum_advantage", "quantum_entanglement", 
                      "quantum_gate_efficiency", "quantum_noise_robustness", "quantum_gradient_quality"]:
            if metric in q_quantum:
                val = q_quantum[metric]
                print(f"{metric:<25} {val:.4f:<15} {'N/A':<15} {'量子專有':<15}")
    
    # 量子 vs 經典比較
    if "quantum_vs_classical" in comprehensive_metrics:
        print("\n⚖️ 量子 vs 經典比較")
        print("-" * 60)
        comparison = comprehensive_metrics["quantum_vs_classical"]
        
        for metric in ["performance_improvement", "parameter_efficiency_ratio", 
                      "speed_improvement", "memory_efficiency_ratio"]:
            if metric in comparison:
                val = comparison[metric]
                print(f"{metric:<25} {val:.4f:<15} {'1.0':<15} {'比較值':<15}")
    
    print("\n🎯 總結:")
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
    print("🚀 開始全面比較實驗...")
    print("="*80)
    
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 運行量子模型實驗
    quantum_results = run_quantum_experiment(num_episodes=50)  # 減少episode數以節省時間
    
    # 運行經典模型實驗
    classical_results = run_classical_experiment(num_episodes=50)
    
    # 計算全面指標
    comprehensive_metrics = compute_comprehensive_metrics(quantum_results, classical_results)
    
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
        
        results_file = "comprehensive_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str處理numpy類型
        
        print(f"\n💾 結果已保存到: {results_file}")
    
    return quantum_results is not None and classical_results is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
