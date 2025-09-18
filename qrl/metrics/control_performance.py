"""控制性能評估指標。"""

import numpy as np
from typing import List, Dict, Any
import torch


class ControlMetrics:
    """控制性能評估指標。"""
    
    def __init__(self):
        pass
    
    def compute_all(self, episode_rewards: List[float], cfg_trajectories: List[List[float]]) -> Dict[str, float]:
        """計算所有控制性能指標。"""
        metrics = {}
        
        # 回合獎勵
        metrics['episode_reward'] = self.compute_episode_reward(episode_rewards)
        
        # 收斂速度
        metrics['convergence_speed'] = self.compute_convergence_speed(episode_rewards)
        
        # CFG平滑度
        metrics['cfg_smoothness'] = self.compute_cfg_smoothness(cfg_trajectories)
        
        # 策略穩定性
        metrics['policy_stability'] = self.compute_policy_stability(episode_rewards)
        
        # CFG變化範圍
        metrics['cfg_range'] = self.compute_cfg_range(cfg_trajectories)
        
        return metrics
    
    def compute_episode_reward(self, episode_rewards: List[float]) -> float:
        """計算平均回合獎勵。"""
        if not episode_rewards:
            return 0.0
        
        return np.mean(episode_rewards)
    
    def compute_convergence_speed(self, episode_rewards: List[float], threshold: float = 0.95) -> float:
        """計算收斂速度。"""
        if not episode_rewards or len(episode_rewards) < 2:
            return 0.0
        
        # 計算移動平均
        window_size = min(10, len(episode_rewards) // 4)
        if window_size < 2:
            return 0.0
        
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        
        # 找到收斂點
        final_reward = moving_avg[-1]
        target_reward = final_reward * threshold
        
        for i, reward in enumerate(moving_avg):
            if reward >= target_reward:
                # 返回收斂速度 (episode數的倒數)
                return 1.0 / (i + 1)
        
        return 0.0
    
    def compute_cfg_smoothness(self, cfg_trajectories: List[List[float]]) -> float:
        """計算CFG軌跡的平滑度。"""
        if not cfg_trajectories:
            return 0.0
        
        smoothness_scores = []
        
        for trajectory in cfg_trajectories:
            if len(trajectory) < 2:
                continue
            
            # 計算相鄰步驟的差異
            differences = np.abs(np.diff(trajectory))
            
            # 平滑度 = 1 / (1 + 平均差異)
            smoothness = 1.0 / (1.0 + np.mean(differences))
            smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.0
    
    def compute_policy_stability(self, episode_rewards: List[float]) -> float:
        """計算策略穩定性。"""
        if not episode_rewards or len(episode_rewards) < 2:
            return 0.0
        
        # 穩定性 = 1 / (1 + 標準差)
        stability = 1.0 / (1.0 + np.std(episode_rewards))
        
        return stability
    
    def compute_cfg_range(self, cfg_trajectories: List[List[float]]) -> float:
        """計算CFG變化範圍。"""
        if not cfg_trajectories:
            return 0.0
        
        all_cfg_values = []
        for trajectory in cfg_trajectories:
            all_cfg_values.extend(trajectory)
        
        if not all_cfg_values:
            return 0.0
        
        # 計算範圍
        cfg_range = np.max(all_cfg_values) - np.min(all_cfg_values)
        
        return cfg_range
    
    def compute_cfg_efficiency(self, cfg_trajectories: List[List[float]], target_cfg: float = 7.5) -> float:
        """計算CFG效率 (接近目標CFG的程度)。"""
        if not cfg_trajectories:
            return 0.0
        
        efficiency_scores = []
        
        for trajectory in cfg_trajectories:
            if not trajectory:
                continue
            
            # 計算與目標CFG的距離
            distances = np.abs(np.array(trajectory) - target_cfg)
            
            # 效率 = 1 / (1 + 平均距離)
            efficiency = 1.0 / (1.0 + np.mean(distances))
            efficiency_scores.append(efficiency)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0
    
    def compute_action_diversity(self, actions: List[List[float]]) -> float:
        """計算動作多樣性。"""
        if not actions:
            return 0.0
        
        all_actions = np.array(actions)
        
        # 計算動作的標準差
        diversity = np.std(all_actions)
        
        return diversity
    
    def compute_control_consistency(self, cfg_trajectories: List[List[float]]) -> float:
        """計算控制一致性。"""
        if not cfg_trajectories or len(cfg_trajectories) < 2:
            return 0.0
        
        # 計算不同軌跡之間的相似性
        similarities = []
        
        for i in range(len(cfg_trajectories)):
            for j in range(i + 1, len(cfg_trajectories)):
                traj1 = cfg_trajectories[i]
                traj2 = cfg_trajectories[j]
                
                # 對齊長度
                min_len = min(len(traj1), len(traj2))
                if min_len < 2:
                    continue
                
                traj1_aligned = traj1[:min_len]
                traj2_aligned = traj2[:min_len]
                
                # 計算相關係數
                correlation = np.corrcoef(traj1_aligned, traj2_aligned)[0, 1]
                if not np.isnan(correlation):
                    similarities.append(abs(correlation))
        
        return np.mean(similarities) if similarities else 0.0
