"""MLP Critic 模組，實現簡單的 MLP 價值函數網路。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple


class MLPCritic(nn.Module):
    """簡單的 MLP 價值函數網路。"""
    
    def __init__(
        self,
        state_dim: int = 16,
        hidden_dims: Optional[list] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        """
        初始化 MLP Critic。
        
        Args:
            state_dim: 狀態維度
            hidden_dims: 隱藏層維度列表
            activation: 激活函數
            dropout: Dropout 比率
            device: 設備
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.device = device
        
        # 設置隱藏層維度
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        # 設置激活函數
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支援的激活函數: {activation}")
        
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
        
        # 輸出層：單一價值
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化權重
        self._init_weights()
        
        # 計算參數數量
        self._count_parameters()
    
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
        
        print(f"MLP Critic 參數統計:")
        print(f"  總參數: {total_params}")
        print(f"  可訓練參數: {trainable_params}")
        
        self.total_params = total_params
        self.trainable_params = trainable_params
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        
        Args:
            state: 狀態張量 [B, state_dim]
            
        Returns:
            torch.Tensor: 狀態價值 [B, 1]
        """
        # 確保狀態維度正確
        if state.size(1) != self.state_dim:
            raise ValueError(f"狀態維度錯誤: {state.size(1)}，期望: {self.state_dim}")
        
        # 前向傳播
        value = self.network(state)
        
        return value
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        獲取狀態價值。
        
        Args:
            state: 狀態張量
            
        Returns:
            torch.Tensor: 狀態價值
        """
        return self.forward(state)
    
    def get_network_info(self) -> Dict[str, Any]:
        """獲取網路信息。"""
        return {
            "state_dim": self.state_dim,
            "hidden_dims": [m.out_features for m in self.modules() if isinstance(m, nn.Linear)][:-1],
            "activation": str(self.activation),
            "dropout": self.network[2].p if len(self.network) > 2 else 0.0,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
        }


class LightweightCritic(MLPCritic):
    """輕量級 Critic，參數量較少。"""
    
    def __init__(
        self,
        state_dim: int = 16,
        device: str = "cuda"
    ):
        """
        初始化輕量級 Critic。
        
        Args:
            state_dim: 狀態維度
            device: 設備
        """
        # 使用較小的隱藏層
        hidden_dims = [64, 32]
        
        super().__init__(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            dropout=0.05,  # 較小的 dropout
            device=device
        )
        
        print(f"輕量級 Critic 創建完成，參數量: {self.total_params}")


class SharedCritic(MLPCritic):
    """共享 Critic，可以與其他網路共享特徵提取器。"""
    
    def __init__(
        self,
        state_dim: int = 16,
        feature_extractor: Optional[nn.Module] = None,
        hidden_dims: Optional[list] = None,
        device: str = "cuda"
    ):
        """
        初始化共享 Critic。
        
        Args:
            state_dim: 狀態維度
            feature_extractor: 特徵提取器（可選）
            hidden_dims: 隱藏層維度
            device: 設備
        """
        super().__init__(state_dim, hidden_dims, "relu", 0.1, device)
        
        self.feature_extractor = feature_extractor
        
        if feature_extractor is not None:
            print("使用共享特徵提取器")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向傳播（支持特徵提取器）。
        
        Args:
            state: 狀態張量
            
        Returns:
            torch.Tensor: 狀態價值
        """
        if self.feature_extractor is not None:
            # 使用共享特徵提取器
            features = self.feature_extractor(state)
            value = self.network(features)
        else:
            # 直接使用原始網路
            value = super().forward(state)
        
        return value


# 便捷函數
def create_mlp_critic(
    state_dim: int = 16,
    hidden_dims: Optional[list] = None,
    **kwargs
) -> MLPCritic:
    """創建 MLP Critic 的便捷函數。"""
    return MLPCritic(
        state_dim=state_dim,
        hidden_dims=hidden_dims,
        **kwargs
    )


def create_lightweight_critic(
    state_dim: int = 16,
    **kwargs
) -> LightweightCritic:
    """創建輕量級 Critic 的便捷函數。"""
    return LightweightCritic(
        state_dim=state_dim,
        **kwargs
    )


def create_shared_critic(
    state_dim: int = 16,
    feature_extractor: Optional[nn.Module] = None,
    **kwargs
) -> SharedCritic:
    """創建共享 Critic 的便捷函數。"""
    return SharedCritic(
        state_dim=state_dim,
        feature_extractor=feature_extractor,
        **kwargs
    )
