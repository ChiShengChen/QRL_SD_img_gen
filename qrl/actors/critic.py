"""
Critic 網路實現
用於 PPO 算法中的價值函數估計
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Critic(nn.Module):
    """經典 Critic 網路，用於估計狀態價值函數 V(s)"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # 選擇激活函數
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # 構建網路層
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # 輸出層
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化網路權重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            state: 狀態張量，形狀為 (batch_size, state_dim)
            
        Returns:
            value: 狀態價值估計，形狀為 (batch_size, 1)
        """
        return self.network(state)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        獲取狀態價值估計（與 forward 相同，但語義更清晰）
        
        Args:
            state: 狀態張量
            
        Returns:
            value: 狀態價值估計
        """
        return self.forward(state)


class QuantumCritic(nn.Module):
    """量子 Critic 網路（未來擴展用）"""
    
    def __init__(
        self,
        state_dim: int,
        n_qubits: int = 8,
        n_layers: int = 4,
        backend: str = "default.qubit",
        device: str = "cpu"
    ):
        super().__init__()
        
        # 目前使用經典實現，未來可以擴展為真正的量子 Critic
        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=[64, 32],
            dropout=0.1
        )
        
        print(f"注意：QuantumCritic 目前使用經典實現")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        return self.critic(state)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """獲取狀態價值估計"""
        return self.forward(state)
