"""量子 Actor 模組，使用 PennyLane 實現變分量子電路。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import math


class QuantumActor(nn.Module):
    """基於變分量子電路的 Actor 網路。"""
    
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 1,
        n_qubits: int = 8,
        n_layers: int = 4,
        shots: Optional[int] = None,
        backend: str = "default.qubit",
        device: str = "cuda"
    ):
        """
        初始化量子 Actor。
        
        Args:
            state_dim: 狀態維度
            action_dim: 動作維度
            n_qubits: 量子比特數量
            n_layers: 電路層數
            shots: 測量次數（None 表示解析期望）
            backend: 量子後端
            device: 設備
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.backend = backend
        self.device = device
        
        # 創建量子設備
        try:
            # 暫時使用 default.qubit 避免版本兼容性問題
            if backend == "lightning.qubit":
                print("警告: 使用 default.qubit 替代 lightning.qubit 以避免版本兼容性問題")
                self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
            else:
                self.dev = qml.device(backend, wires=n_qubits, shots=shots)
        except Exception as e:
            print(f"警告: 無法使用 {backend}，使用 default.qubit: {e}")
            self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        
        # 創建量子電路
        self.qnode = self._create_quantum_circuit()
        
        # 經典後處理網路
        self.post_process = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim * 2),  # mu 和 log_std
        )
        
        # 初始化權重
        self._init_weights()
        
        # 計算參數數量
        self._count_parameters()
    
    def _create_quantum_circuit(self):
        """創建量子電路。"""
        def circuit(inputs, weights):
            # 角度編碼
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            
            # 變分層
            for layer in range(self.n_layers):
                # 旋轉門
                for i in range(self.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
                
                # 糾纏層（環形拓撲）
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                
                # 額外的旋轉門
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 3], wires=i)
            
            # 測量所有量子比特
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return qml.QNode(circuit, self.dev, interface="torch")
    
    def _init_weights(self):
        """初始化權重。"""
        # 量子電路權重
        self.quantum_weights = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 4) * 0.1
        )
        
        # 經典網路權重已在 Sequential 中自動初始化
    
    def _count_parameters(self):
        """計算參數數量。"""
        quantum_params = self.quantum_weights.numel()
        classical_params = sum(p.numel() for p in self.post_process.parameters())
        total_params = quantum_params + classical_params
        
        print(f"量子 Actor 參數統計:")
        print(f"  量子電路參數: {quantum_params}")
        print(f"  經典後處理參數: {classical_params}")
        print(f"  總參數: {total_params}")
        
        self.total_params = total_params
        self.num_params = total_params  # 添加 num_params 屬性以保持兼容性
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播。
        
        Args:
            state: 狀態張量 [B, state_dim]
            
        Returns:
            Tuple: (mu, log_std) 高斯策略參數
        """
        batch_size = state.size(0)
        
        # 確保狀態維度正確
        if state.size(1) != self.state_dim:
            raise ValueError(f"狀態維度錯誤: {state.size(1)}，期望: {self.state_dim}")
        
        # 準備量子電路輸入
        quantum_inputs = []
        for i in range(batch_size):
            # 將狀態映射到量子比特數量
            if self.state_dim > self.n_qubits:
                # 降維：使用 PCA 或簡單的線性投影
                state_i = state[i, :self.n_qubits]
            else:
                # 升維：重複填充
                repeat_times = (self.n_qubits // self.state_dim) + 1
                state_i = state[i].repeat(repeat_times)[:self.n_qubits]
            
            quantum_inputs.append(state_i)
        
        quantum_inputs = torch.stack(quantum_inputs)  # [B, n_qubits]
        
        # 運行量子電路
        quantum_outputs = []
        for i in range(batch_size):
            # 對每個樣本運行量子電路
            output = self.qnode(quantum_inputs[i], self.quantum_weights)
            if isinstance(output, (list, tuple)):
                output = torch.tensor(output, dtype=torch.float32)
            quantum_outputs.append(output)
        
        quantum_outputs = torch.stack(quantum_outputs)  # [B, n_qubits]
        quantum_outputs = quantum_outputs.to(self.device)  # 移動到正確設備
        
        # 經典後處理
        post_output = self.post_process(quantum_outputs)  # [B, action_dim * 2]
        
        # 分離 mu 和 log_std
        mu = post_output[:, :self.action_dim]
        log_std = post_output[:, self.action_dim:]
        
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
    
    def get_quantum_info(self) -> Dict[str, Any]:
        """獲取量子電路信息。"""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "backend": self.backend,
            "shots": self.shots,
            "total_params": self.total_params,
        }


# 便捷函數
def create_quantum_actor(
    state_dim: int = 16,
    action_dim: int = 1,
    n_qubits: int = 8,
    n_layers: int = 4,
    **kwargs
) -> QuantumActor:
    """創建量子 Actor 的便捷函數。"""
    return QuantumActor(
        state_dim=state_dim,
        action_dim=action_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        **kwargs
    )
