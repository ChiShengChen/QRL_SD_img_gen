"""Actor 形狀測試。"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qrl.actors import QuantumActor, ClassicalActor, AlignedClassicalActor


class TestActorShapes:
    """測試 Actor 的形狀。"""
    
    @pytest.fixture
    def state_dim(self):
        """狀態維度。"""
        return 64
    
    @pytest.fixture
    def action_dim(self):
        """動作維度。"""
        return 1
    
    @pytest.fixture
    def batch_size(self):
        """批次大小。"""
        return 8
    
    def test_quantum_actor_shapes(self, state_dim, action_dim, batch_size):
        """測試量子 Actor 的形狀。"""
        # 創建量子 Actor
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        # 創建測試輸入
        state = torch.randn(batch_size, state_dim)
        
        # 測試前向傳播
        mu, log_std = actor(state)
        
        # 檢查輸出形狀
        assert mu.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)
        
        # 檢查輸出範圍
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(log_std))
        
        # 測試採樣
        action, log_prob = actor.sample(state)
        
        # 檢查採樣輸出形狀
        assert action.shape == (batch_size, action_dim)
        assert log_prob.shape == (batch_size,)
        
        # 檢查採樣輸出範圍
        assert torch.all(torch.isfinite(action))
        assert torch.all(torch.isfinite(log_prob))
        
        # 測試 log_prob 計算
        log_prob_calc = actor.log_prob(state, action)
        assert log_prob_calc.shape == (batch_size,)
        assert torch.all(torch.isfinite(log_prob_calc))
        
        # 測試熵計算
        entropy = actor.entropy(state)
        assert entropy.shape == (batch_size,)
        assert torch.all(torch.isfinite(entropy))
    
    def test_classical_actor_shapes(self, state_dim, action_dim, batch_size):
        """測試古典 Actor 的形狀。"""
        # 創建古典 Actor
        actor = ClassicalActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        # 創建測試輸入
        state = torch.randn(batch_size, state_dim)
        
        # 測試前向傳播
        mu, log_std = actor(state)
        
        # 檢查輸出形狀
        assert mu.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)
        
        # 檢查輸出範圍
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(log_std))
        
        # 測試採樣
        action, log_prob = actor.sample(state)
        
        # 檢查採樣輸出形狀
        assert action.shape == (batch_size, action_dim)
        assert log_prob.shape == (batch_size,)
        
        # 檢查採樣輸出範圍
        assert torch.all(torch.isfinite(action))
        assert torch.all(torch.isfinite(log_prob))
        
        # 測試 log_prob 計算
        log_prob_calc = actor.log_prob(state, action)
        assert log_prob_calc.shape == (batch_size,)
        assert torch.all(torch.isfinite(log_prob_calc))
        
        # 測試熵計算
        entropy = actor.entropy(state)
        assert entropy.shape == (batch_size,)
        assert torch.all(torch.isfinite(entropy))
    
    def test_aligned_classical_actor_shapes(self, state_dim, action_dim, batch_size):
        """測試對齊古典 Actor 的形狀。"""
        # 創建量子 Actor 來獲取目標參數數量
        quantum_actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        target_params = quantum_actor.count_parameters()
        
        # 創建對齊的古典 Actor
        actor = AlignedClassicalActor(
            state_dim=state_dim,
            action_dim=action_dim,
            target_params=target_params,
            device="cpu"
        )
        
        # 檢查參數數量是否接近目標
        actual_params = actor.count_parameters()
        param_ratio = actual_params / target_params
        
        # 參數數量應該在合理範圍內（0.8-1.2）
        assert 0.8 <= param_ratio <= 1.2, f"參數比例 {param_ratio:.3f} 超出範圍"
        
        # 創建測試輸入
        state = torch.randn(batch_size, state_dim)
        
        # 測試前向傳播
        mu, log_std = actor(state)
        
        # 檢查輸出形狀
        assert mu.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)
        
        # 檢查輸出範圍
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(log_std))
        
        # 測試採樣
        action, log_prob = actor.sample(state)
        
        # 檢查採樣輸出形狀
        assert action.shape == (batch_size, action_dim)
        assert log_prob.shape == (batch_size,)
        
        # 檢查採樣輸出範圍
        assert torch.all(torch.isfinite(action))
        assert torch.all(torch.isfinite(log_prob))
    
    def test_actor_parameter_counts(self, state_dim, action_dim):
        """測試 Actor 參數數量。"""
        # 創建量子 Actor
        quantum_actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        # 創建古典 Actor
        classical_actor = ClassicalActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        # 獲取參數數量
        quantum_params = quantum_actor.count_parameters()
        classical_params = classical_actor.count_parameters()
        
        # 檢查參數數量是否合理
        assert quantum_params > 0
        assert classical_params > 0
        
        # 量子 Actor 通常有較少的參數
        assert quantum_params <= classical_params * 2  # 允許一些變化
        
        print(f"量子 Actor 參數數量: {quantum_params}")
        print(f"古典 Actor 參數數量: {classical_params}")
        print(f"參數比例: {quantum_params / classical_params:.3f}")
    
    def test_actor_device_consistency(self, state_dim, action_dim, batch_size):
        """測試 Actor 設備一致性。"""
        # 測試 CPU
        actor_cpu = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        state_cpu = torch.randn(batch_size, state_dim)
        mu_cpu, log_std_cpu = actor_cpu(state_cpu)
        
        assert mu_cpu.device.type == "cpu"
        assert log_std_cpu.device.type == "cpu"
        
        # 如果 CUDA 可用，測試 GPU
        if torch.cuda.is_available():
            actor_cuda = QuantumActor(
                state_dim=state_dim,
                action_dim=action_dim,
                device="cuda"
            )
            
            state_cuda = torch.randn(batch_size, state_dim).cuda()
            mu_cuda, log_std_cuda = actor_cuda(state_cuda)
            
            assert mu_cuda.device.type == "cuda"
            assert log_std_cuda.device.type == "cuda"
    
    def test_actor_gradient_flow(self, state_dim, action_dim, batch_size):
        """測試 Actor 梯度流。"""
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        # 創建測試輸入
        state = torch.randn(batch_size, state_dim, requires_grad=True)
        
        # 前向傳播
        mu, log_std = actor(state)
        
        # 計算損失
        loss = mu.mean() + log_std.mean()
        
        # 反向傳播
        loss.backward()
        
        # 檢查梯度
        assert state.grad is not None
        assert torch.all(torch.isfinite(state.grad))
        
        # 檢查模型參數梯度
        for param in actor.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad))
    
    def test_actor_output_ranges(self, state_dim, action_dim, batch_size):
        """測試 Actor 輸出範圍。"""
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        # 創建測試輸入
        state = torch.randn(batch_size, state_dim)
        
        # 測試多個樣本
        for _ in range(10):
            mu, log_std = actor(state)
            
            # 檢查 mu 範圍（通常應該在合理範圍內）
            assert torch.all(torch.isfinite(mu))
            
            # 檢查 log_std 範圍（通常應該為負值）
            assert torch.all(torch.isfinite(log_std))
            
            # 檢查採樣
            action, log_prob = actor.sample(state)
            
            assert torch.all(torch.isfinite(action))
            assert torch.all(torch.isfinite(log_prob))
            
            # log_prob 通常應該為負值
            assert torch.all(log_prob <= 0)
    
    def test_actor_batch_consistency(self, state_dim, action_dim):
        """測試 Actor 批次一致性。"""
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        # 測試不同批次大小
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            state = torch.randn(batch_size, state_dim)
            mu, log_std = actor(state)
            
            assert mu.shape == (batch_size, action_dim)
            assert log_std.shape == (batch_size, action_dim)
            
            action, log_prob = actor.sample(state)
            
            assert action.shape == (batch_size, action_dim)
            assert log_prob.shape == (batch_size,)


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
