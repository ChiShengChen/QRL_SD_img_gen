"""經典 Actor 模組，實現兩層 MLP 網路。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import math


class ClassicalActor(nn.Module):
    """經典 MLP Actor 網路。"""
    
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 1,
        hidden_dims: Optional[list] = None,
        device: str = "cuda"
    ):
        """
        初始化經典 Actor。
        
        Args:
            state_dim: 狀態維度
            action_dim: 動作維度
            hidden_dims: 隱藏層維度列表
            device: 設備
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 設置隱藏層維度
        if hidden_dims is None:
            # 預設隱藏層維度，設計為與量子 Actor 參數量級匹配
            hidden_dims = [128, 64]
        
        # 構建網路
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # 輕微正則化
            ])
            input_dim = hidden_dim
        
        # 輸出層：mu 和 log_std
        layers.append(nn.Linear(input_dim, action_dim * 2))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化權重
        self._init_weights()
        
        # 計算參數數量
        self._count_parameters()
        
        # 驗證參數量對齊
        self._validate_parameter_alignment()
    
    def _init_weights(self):
        """初始化權重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier 初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _count_parameters(self):
        """計算參數數量。"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"經典 Actor 參數統計:")
        print(f"  總參數: {total_params}")
        print(f"  可訓練參數: {trainable_params}")
        
        self.total_params = total_params
        self.trainable_params = trainable_params
    
    def _validate_parameter_alignment(self):
        """
        驗證參數量與量子 Actor 的對齊。
        
        注意：這是一個註解，說明參數量對齊的重要性。
        """
        # 量子 Actor 的參數量級：
        # - 量子電路：n_layers * n_qubits * 4 = 4 * 8 * 4 = 128
        # - 經典後處理：8*64 + 64*32 + 32*2 = 512 + 2048 + 64 = 2624
        # - 總計：約 2752 參數
        
        # 經典 Actor 的參數量級：
        # - 第一層：16*128 + 128 = 2176
        # - 第二層：128*64 + 64 = 8256
        # - 輸出層：64*2 + 2 = 130
        # - 總計：約 10562 參數
        
        # 為了公平比較，我們可以調整隱藏層維度
        # 或者使用更小的網路結構
        print("注意：經典 Actor 參數量較大，建議調整隱藏層維度以匹配量子 Actor")
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播。
        
        Args:
            state: 狀態張量 [B, state_dim]
            
        Returns:
            Tuple: (mu, log_std) 高斯策略參數
        """
        # 確保狀態維度正確
        if state.size(1) != self.state_dim:
            raise ValueError(f"狀態維度錯誤: {state.size(1)}，期望: {self.state_dim}")
        
        # 前向傳播
        output = self.network(state)  # [B, action_dim * 2]
        
        # 分離 mu 和 log_std
        mu = output[:, :self.action_dim]
        log_std = output[:, self.action_dim:]
        
        # 確保 log_std 在合理範圍內
        log_std = torch.clamp(log_std, -20, 2)
        
        return mu, log_std
    
    def sample(self, state: torch.Tensor) -> torch.Tensor:
        """
        從策略中採樣動作。
        
        Args:
            state: 狀態張量
            
        Returns:
            torch.Tensor: 採樣的動作
        """
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 重參數化採樣
        epsilon = torch.randn_like(mu)
        action = mu + std * epsilon
        
        return action
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        計算動作的對數概率。
        
        Args:
            state: 狀態張量
            action: 動作張量
            
        Returns:
            torch.Tensor: 對數概率
        """
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 計算高斯分布的對數概率
        log_prob = -0.5 * ((action - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        
        # 對動作維度求和
        log_prob = log_prob.sum(dim=-1)
        
        return log_prob
    
    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        計算策略的熵。
        
        Args:
            state: 狀態張量
            
        Returns:
            torch.Tensor: 熵值
        """
        _, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 高斯分布的熵
        entropy = 0.5 * (1 + math.log(2 * math.pi)) + log_std
        
        # 對動作維度求和
        entropy = entropy.sum(dim=-1)
        
        return entropy
    
    def get_network_info(self) -> Dict[str, Any]:
        """獲取網路信息。"""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": [m.out_features for m in self.modules() if isinstance(m, nn.Linear)][:-1],
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
        }


class AlignedClassicalActor(ClassicalActor):
    """參數量對齊的經典 Actor。"""
    
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 1,
        target_params: int = 2752,  # 量子 Actor 的參數量
        device: str = "cuda"
    ):
        """
        初始化參數量對齊的經典 Actor。
        
        Args:
            state_dim: 狀態維度
            action_dim: 動作維度
            target_params: 目標參數量
            device: 設備
        """
        # 計算合適的隱藏層維度
        hidden_dims = self._calculate_hidden_dims(state_dim, action_dim, target_params)
        
        super().__init__(state_dim, action_dim, hidden_dims, device)
        
        print(f"參數量對齊完成，目標: {target_params}，實際: {self.total_params}")
    
    def _calculate_hidden_dims(self, state_dim: int, action_dim: int, target_params: int) -> list:
        """計算隱藏層維度以達到目標參數量。"""
        # 簡化計算：假設使用單個隱藏層
        # 參數量 = state_dim * hidden_dim + hidden_dim + hidden_dim * (action_dim * 2) + (action_dim * 2)
        # 簡化為：state_dim * hidden_dim + hidden_dim * (action_dim * 2) + hidden_dim + action_dim * 2
        
        # 求解隱藏層維度
        # target_params ≈ hidden_dim * (state_dim + action_dim * 2 + 1) + action_dim * 2
        # hidden_dim ≈ (target_params - action_dim * 2) / (state_dim + action_dim * 2 + 1)
        
        denominator = state_dim + action_dim * 2 + 1
        numerator = target_params - action_dim * 2
        
        if numerator <= 0:
            # 如果目標參數量太小，使用最小網路
            hidden_dim = max(8, state_dim // 2)
        else:
            hidden_dim = max(8, numerator // denominator)
        
        return [hidden_dim]


# 便捷函數
def create_classical_actor(
    state_dim: int = 16,
    action_dim: int = 1,
    hidden_dims: Optional[list] = None,
    **kwargs
) -> ClassicalActor:
    """創建經典 Actor 的便捷函數。"""
    return ClassicalActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        **kwargs
    )


def create_aligned_classical_actor(
    state_dim: int = 16,
    action_dim: int = 1,
    target_params: int = 2752,
    **kwargs
) -> AlignedClassicalActor:
    """創建參數量對齊的經典 Actor 的便捷函數。"""
    return AlignedClassicalActor(
        state_dim=state_dim,
        action_dim=action_dim,
        target_params=target_params,
        **kwargs
    )
