"""PPO 訓練器模組，實現 PPO 算法和量子梯度計算。"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
from tqdm import tqdm
import time

from .buffers import PPOBuffer
from ..actors import QuantumActor, ClassicalActor
from ..critics import MLPCritic
from ..utils.logging import get_logger


class PPOTrainer:
    """PPO 訓練器。"""
    
    def __init__(
        self,
        actor: Union[QuantumActor, ClassicalActor],
        critic: MLPCritic,
        device: str = "cuda",
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.1,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        mini_batch_size: int = 8,
        rollout_steps: int = 512,
        num_envs: int = 8,
        mixed_precision: bool = True,
        use_quantum_gradients: bool = False
    ):
        """
        初始化 PPO 訓練器。
        
        Args:
            actor: Actor 網路
            critic: Critic 網路
            device: 設備
            lr_actor: Actor 學習率
            lr_critic: Critic 學習率
            gamma: 折扣因子
            gae_lambda: GAE lambda 參數
            clip_ratio: PPO clip 比例
            entropy_coef: 熵正則化係數
            value_coef: 價值函數係數
            max_grad_norm: 梯度裁剪範數
            update_epochs: 每次更新的 epoch 數
            mini_batch_size: 小批次大小
            rollout_steps: 每次 rollouts 的步數
            num_envs: 環境數量
            mixed_precision: 是否使用混合精度
            use_quantum_gradients: 是否使用量子梯度
        """
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        
        # 超參數
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.mixed_precision = mixed_precision
        self.use_quantum_gradients = use_quantum_gradients
        
        # 優化器
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr_critic)
        
        # 學習率調度器
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=1000
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=1000
        )
        
        # 緩衝區
        state_dim = actor.state_dim if hasattr(actor, 'state_dim') else 16
        action_dim = actor.action_dim if hasattr(actor, 'action_dim') else 1
        
        self.buffer = PPOBuffer(
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        
        # 混合精度
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 日誌記錄器
        self.logger = get_logger("ppo_trainer")
        
        # 訓練統計
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': [],
            'value_loss': [],
            'policy_loss': []
        }
        
        # 檢查是否為量子 Actor
        self.is_quantum_actor = isinstance(actor, QuantumActor)
        
        self.logger.info(f"PPO 訓練器初始化完成")
        self.logger.info(f"Actor 類型: {'量子' if self.is_quantum_actor else '經典'}")
        self.logger.info(f"使用量子梯度: {self.use_quantum_gradients}")
    
    def collect_rollouts(self, env, num_steps: int) -> Dict[str, Any]:
        """
        收集 rollouts 數據。
        
        Args:
            env: 環境
            num_steps: 收集步數
            
        Returns:
            Dict: rollouts 數據
        """
        self.actor.eval()
        self.critic.eval()
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        state, _ = env.reset()
        episode_rewards = []
        current_episode_reward = 0
        
        for step in range(num_steps):
            # 轉換狀態為 tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 獲取動作
            with torch.no_grad():
                action, log_prob = self._get_action(state_tensor)
                value = self.critic(state_tensor)
            
            # 轉換動作為 numpy array
            action_np = action.cpu().numpy().flatten()
            
            # 執行動作
            next_state, reward, done, truncated, info = env.step(action_np)
            
            # 記錄數據
            states.append(state_tensor)
            actions.append(action)
            rewards.append(torch.tensor(reward, device=self.device))
            values.append(value)
            log_probs.append(log_prob)
            dones.append(torch.tensor(done, device=self.device))
            
            # 更新狀態
            state = next_state
            current_episode_reward += reward
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state, _ = env.reset()
        
        # 組織數據
        rollouts = {
            'states': torch.stack(states),
            'actions': torch.stack(actions),
            'rewards': torch.stack(rewards),
            'values': torch.stack(values),
            'log_probs': torch.stack(log_probs),
            'dones': torch.stack(dones),
            'episode_rewards': episode_rewards
        }
        
        return rollouts
    
    def _get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取動作和對數概率。"""
        mu, log_std = self.actor(state)
        std = torch.exp(log_std)
        
        # 重參數化採樣
        epsilon = torch.randn_like(mu)
        action = mu + std * epsilon
        
        # 計算對數概率
        log_prob = self.actor.log_prob(state, action)
        
        return action, log_prob
    
    def update(self, rollouts: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        執行 PPO 更新。
        
        Args:
            rollouts: rollouts 數據
            
        Returns:
            Dict: 更新統計信息
        """
        self.actor.train()
        self.critic.train()
        
        # 計算優勢
        next_value = self.critic(rollouts['states'][-1])
        advantages, returns = self.buffer.compute_advantages(
            next_value, self.gamma, self.gae_lambda
        )
        
        # 展平優勢和回報以匹配批次數據
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        
        # 標準化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 獲取批次數據
        batch_data = self.buffer.get_batch(self.mini_batch_size)
        
        # 執行多次更新
        update_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': [],
            'value_loss': [],
            'policy_loss': []
        }
        
        for epoch in range(self.update_epochs):
            # 重新獲取批次數據以避免計算圖重複使用
            batch_data = self.buffer.get_batch(self.mini_batch_size)
            
            # 重新計算優勢和回報以避免計算圖重複使用
            next_value = self.critic(rollouts['states'][-1])
            epoch_advantages, epoch_returns = self.buffer.compute_advantages(
                next_value, self.gamma, self.gae_lambda
            )
            epoch_advantages = epoch_advantages.view(-1)
            epoch_returns = epoch_returns.view(-1)
            epoch_advantages = (epoch_advantages - epoch_advantages.mean()) / (epoch_advantages.std() + 1e-8)
            
            # 打印epoch開始信息
            self.logger.info(f"    Epoch {epoch+1}/{self.update_epochs} 開始更新...")
            
            epoch_stats = self._update_epoch(batch_data, epoch_advantages, epoch_returns, epoch)
            
            # 打印epoch完成信息
            self.logger.info(
                f"    Epoch {epoch+1}/{self.update_epochs} 完成 | "
                f"Policy Loss: {epoch_stats['policy_loss']:.6f} | "
                f"Value Loss: {epoch_stats['value_loss']:.6f} | "
                f"Entropy: {epoch_stats['entropy_loss']:.6f} | "
                f"Clip Fraction: {epoch_stats['clip_fraction']:.4f}"
            )
            
            # 累積統計
            for key in update_stats:
                update_stats[key].append(epoch_stats[key])
        
        # 計算平均值
        final_stats = {}
        for key in update_stats:
            final_stats[key] = np.mean(update_stats[key])
        
        # 更新學習率
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 記錄統計
        for key in final_stats:
            self.training_stats[key].append(final_stats[key])
        
        return final_stats
    
    def _update_epoch(
        self,
        batch_data: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, float]:
        """執行單個 epoch 的更新。"""
        # 獲取數據
        states = batch_data['states']
        actions = batch_data['actions']
        old_log_probs = batch_data['log_probs']
        old_values = batch_data['values']
        
        # 獲取對應的優勢和回報
        batch_advantages = advantages[:len(states)]
        batch_returns = returns[:len(states)]
        
        # 計算新的動作概率和價值
        new_mu, new_log_std = self.actor(states)
        new_log_probs = self.actor.log_prob(states, actions)
        new_values = self.critic(states)
        
        # 計算比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO 目標
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
        
        # 策略損失
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 價值損失 - 確保維度匹配
        if new_values.dim() > 1:
            new_values = new_values.squeeze()
        if batch_returns.dim() > 1:
            batch_returns = batch_returns.squeeze()
        value_loss = F.mse_loss(new_values, batch_returns)
        
        # 熵損失
        entropy_loss = -self.actor.entropy(states).mean()
        
        # 總損失
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        # 計算 clip 分數
        clip_fraction = (abs(ratio - 1) > self.clip_ratio).float().mean()
        
        # 反向傳播
        if self.mixed_precision:
            self.scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.actor_optimizer)
            self.scaler.unscale_(self.critic_optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            # 優化器步驟
            self.scaler.step(self.actor_optimizer)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            # 優化器步驟
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # 清零梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        return {
            'actor_loss': policy_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'clip_fraction': clip_fraction.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()
        }
    
    def save_checkpoint(self, path: str):
        """保存檢查點。"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'training_stats': self.training_stats,
            'hyperparameters': {
                'lr_actor': self.lr_actor,
                'lr_critic': self.lr_critic,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
            }
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"檢查點已保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """載入檢查點。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        self.logger.info(f"檢查點已從 {path} 載入")
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """獲取訓練統計信息。"""
        return self.training_stats.copy()
    
    def reset_stats(self):
        """重置訓練統計。"""
        for key in self.training_stats:
            self.training_stats[key].clear()


class QuantumPPOTrainer(PPOTrainer):
    """支持量子梯度的 PPO 訓練器。"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not self.is_quantum_actor:
            raise ValueError("QuantumPPOTrainer 需要量子 Actor")
        
        self.logger.info("量子 PPO 訓練器初始化完成")
    
    def _compute_quantum_gradients(self, loss: torch.Tensor) -> None:
        """使用量子梯度計算。"""
        if not self.use_quantum_gradients:
            return
        
        # 使用 PennyLane 的參數偏移方法
        # 這裡需要根據具體的量子電路實現
        # 暫時使用標準的反向傳播
        pass
    
    def _update_epoch(
        self,
        batch_data: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, float]:
        """執行單個 epoch 的更新（量子版本）。"""
        # 直接使用父類的實現，不重複調用
        return super()._update_epoch(batch_data, advantages, returns, epoch)
    
    def evaluate(self, env, num_episodes: int = 10) -> float:
        """評估當前策略。"""
        self.actor.eval()
        self.critic.eval()
        
        total_rewards = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action, _ = self._get_action(state_tensor)
                    action_np = action.cpu().numpy().flatten()
                    
                    next_state, reward, done, truncated, _ = env.step(action_np)
                    episode_reward += reward
                    state = next_state
                    
                    if truncated:
                        done = True
                
                total_rewards.append(episode_reward)
        
        self.actor.train()
        self.critic.train()
        
        return np.mean(total_rewards)


# 便捷函數
def create_ppo_trainer(
    actor: Union[QuantumActor, ClassicalActor],
    critic: MLPCritic,
    **kwargs
) -> PPOTrainer:
    """創建 PPO 訓練器的便捷函數。"""
    if isinstance(actor, QuantumActor):
        return QuantumPPOTrainer(actor, critic, **kwargs)
    else:
        return PPOTrainer(actor, critic, **kwargs)
