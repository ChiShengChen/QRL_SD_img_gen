"""訓練緩衝區模組，實現經驗回放和優勢計算。"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from collections import deque
import random


class ExperienceBuffer:
    """經驗回放緩衝區。"""
    
    def __init__(
        self,
        capacity: int = 10000,
        device: str = "cpu"
    ):
        """
        初始化經驗緩衝區。
        
        Args:
            capacity: 緩衝區容量
            device: 設備
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor
    ):
        """
        添加經驗到緩衝區。
        
        Args:
            state: 狀態
            action: 動作
            reward: 獎勵
            next_state: 下一個狀態
            done: 是否結束
            log_prob: 動作對數概率
            value: 狀態價值
        """
        experience = {
            'state': state.cpu(),
            'action': action.cpu(),
            'reward': reward.cpu(),
            'next_state': next_state.cpu(),
            'done': done.cpu(),
            'log_prob': log_prob.cpu(),
            'value': value.cpu()
        }
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        採樣經驗批次。
        
        Args:
            batch_size: 批次大小
            
        Returns:
            Dict: 經驗批次
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        # 組織批次數據
        states = torch.stack([exp['state'] for exp in batch]).to(self.device)
        actions = torch.stack([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.stack([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.stack([exp['done'] for exp in batch]).to(self.device)
        log_probs = torch.stack([exp['log_prob'] for exp in batch]).to(self.device)
        values = torch.stack([exp['value'] for exp in batch]).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'log_probs': log_probs,
            'values': values
        }
    
    def clear(self):
        """清空緩衝區。"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class PPOBuffer:
    """PPO 專用的經驗緩衝區。"""
    
    def __init__(
        self,
        rollout_steps: int = 512,
        num_envs: int = 8,
        state_dim: int = 16,
        action_dim: int = 1,
        device: str = "cpu"
    ):
        """
        初始化 PPO 緩衝區。
        
        Args:
            rollout_steps: 每次 rollouts 的步數
            num_envs: 環境數量
            state_dim: 狀態維度
            action_dim: 動作維度
            device: 設備
        """
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 初始化緩衝區
        self.reset()
    
    def reset(self):
        """重置緩衝區。"""
        self.states = torch.zeros(
            (self.rollout_steps, self.num_envs, self.state_dim),
            device=self.device
        )
        self.actions = torch.zeros(
            (self.rollout_steps, self.num_envs, self.action_dim),
            device=self.device
        )
        self.rewards = torch.zeros(
            (self.rollout_steps, self.num_envs),
            device=self.device
        )
        self.values = torch.zeros(
            (self.rollout_steps, self.num_envs),
            device=self.device
        )
        self.log_probs = torch.zeros(
            (self.rollout_steps, self.num_envs),
            device=self.device
        )
        self.dones = torch.zeros(
            (self.rollout_steps, self.num_envs),
            device=self.device
        )
        
        self.step = 0
        self.episode_rewards = []
    
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor
    ):
        """
        添加經驗到緩衝區。
        
        Args:
            state: 狀態 [num_envs, state_dim]
            action: 動作 [num_envs, action_dim]
            reward: 獎勵 [num_envs]
            value: 價值 [num_envs]
            log_prob: 對數概率 [num_envs]
            done: 是否結束 [num_envs]
        """
        if self.step >= self.rollout_steps:
            raise ValueError("緩衝區已滿，請先計算優勢或重置")
        
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.dones[self.step] = done
        
        self.step += 1
    
    def compute_advantages(
        self,
        next_value: torch.Tensor,
        gamma: float = 0.995,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算廣義優勢估計 (GAE)。
        
        Args:
            next_value: 最後一步的價值 [num_envs]
            gamma: 折扣因子
            gae_lambda: GAE lambda 參數
            
        Returns:
            Tuple: (advantages, returns)
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        
        # 從後往前計算 GAE
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value_step = self.values[step + 1]
            
            delta = (
                self.rewards[step] +
                gamma * next_value_step * next_non_terminal -
                self.values[step]
            )
            
            advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )
        
        # 計算 returns
        returns = advantages + self.values
        
        return advantages, returns
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        獲取隨機批次數據。
        
        Args:
            batch_size: 批次大小
            
        Returns:
            Dict: 批次數據
        """
        # 展平所有數據
        states = self.states.view(-1, self.state_dim)
        actions = self.actions.view(-1, self.action_dim)
        rewards = self.rewards.view(-1)
        values = self.values.view(-1)
        log_probs = self.log_probs.view(-1)
        dones = self.dones.view(-1)
        
        # 隨機採樣
        indices = torch.randperm(len(states))[:batch_size]
        
        return {
            'states': states[indices],
            'actions': actions[indices],
            'rewards': rewards[indices],
            'values': values[indices],
            'log_probs': log_probs[indices],
            'dones': dones[indices]
        }
    
    def is_full(self) -> bool:
        """檢查緩衝區是否已滿。"""
        return self.step >= self.rollout_steps
    
    def get_stats(self) -> Dict[str, float]:
        """獲取緩衝區統計信息。"""
        if self.step == 0:
            return {}
        
        return {
            'total_steps': self.step * self.num_envs,
            'mean_reward': self.rewards[:self.step].mean().item(),
            'std_reward': self.rewards[:self.step].std().item(),
            'min_reward': self.rewards[:self.step].min().item(),
            'max_reward': self.rewards[:self.step].max().item(),
        }


class MultiEnvBuffer:
    """多環境緩衝區，支持並行環境。"""
    
    def __init__(
        self,
        num_envs: int = 8,
        buffer_size: int = 1000,
        device: str = "cpu"
    ):
        """
        初始化多環境緩衝區。
        
        Args:
            num_envs: 環境數量
            buffer_size: 每個環境的緩衝區大小
            device: 設備
        """
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.device = device
        
        # 為每個環境創建緩衝區
        self.buffers = [deque(maxlen=buffer_size) for _ in range(num_envs)]
        self.episode_rewards = [[] for _ in range(num_envs)]
    
    def add(
        self,
        env_id: int,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor
    ):
        """
        為指定環境添加經驗。
        
        Args:
            env_id: 環境 ID
            state: 狀態
            action: 動作
            reward: 獎勵
            next_state: 下一個狀態
            done: 是否結束
            log_prob: 動作對數概率
            value: 狀態價值
        """
        if env_id >= self.num_envs:
            raise ValueError(f"環境 ID 超出範圍: {env_id}")
        
        experience = {
            'state': state.cpu(),
            'action': action.cpu(),
            'reward': reward.cpu(),
            'next_state': next_state.cpu(),
            'done': done.cpu(),
            'log_prob': log_prob.cpu(),
            'value': value.cpu()
        }
        
        self.buffers[env_id].append(experience)
        
        # 記錄 episode 獎勵
        if done.item():
            episode_reward = sum(exp['reward'].item() for exp in self.buffers[env_id])
            self.episode_rewards[env_id].append(episode_reward)
    
    def sample_all(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        從所有環境採樣經驗。
        
        Args:
            batch_size: 每個環境的批次大小
            
        Returns:
            Dict: 所有環境的經驗批次
        """
        all_experiences = []
        
        for env_id in range(self.num_envs):
            if len(self.buffers[env_id]) >= batch_size:
                batch = random.sample(self.buffers[env_id], batch_size)
                all_experiences.extend(batch)
        
        if not all_experiences:
            return {}
        
        # 組織批次數據
        states = torch.stack([exp['state'] for exp in all_experiences]).to(self.device)
        actions = torch.stack([exp['action'] for exp in all_experiences]).to(self.device)
        rewards = torch.stack([exp['reward'] for exp in all_experiences]).to(self.device)
        next_states = torch.stack([exp['next_state'] for exp in all_experiences]).to(self.device)
        dones = torch.stack([exp['done'] for exp in all_experiences]).to(self.device)
        log_probs = torch.stack([exp['log_prob'] for exp in all_experiences]).to(self.device)
        values = torch.stack([exp['value'] for exp in all_experiences]).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'log_probs': log_probs,
            'values': values
        }
    
    def get_episode_stats(self) -> Dict[str, float]:
        """獲取 episode 統計信息。"""
        all_rewards = []
        for rewards in self.episode_rewards:
            all_rewards.extend(rewards)
        
        if not all_rewards:
            return {}
        
        return {
            'total_episodes': len(all_rewards),
            'mean_episode_reward': np.mean(all_rewards),
            'std_episode_reward': np.std(all_rewards),
            'min_episode_reward': np.min(all_rewards),
            'max_episode_reward': np.max(all_rewards),
        }
    
    def clear(self):
        """清空所有緩衝區。"""
        for buffer in self.buffers:
            buffer.clear()
        for rewards in self.episode_rewards:
            rewards.clear()
