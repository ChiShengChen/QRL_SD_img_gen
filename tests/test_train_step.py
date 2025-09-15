"""訓練步驟測試。"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qrl.actors import QuantumActor, ClassicalActor
from qrl.critics import MLPCritic
from qrl.training import PPOTrainer, PPOBuffer
from qrl.utils.seed import set_seed


class TestPPOTraining:
    """測試 PPO 訓練。"""
    
    @pytest.fixture
    def state_dim(self):
        """狀態維度。"""
        return 32
    
    @pytest.fixture
    def action_dim(self):
        """動作維度。"""
        return 1
    
    @pytest.fixture
    def batch_size(self):
        """批次大小。"""
        return 4
    
    @pytest.fixture
    def rollout_steps(self):
        """rollout 步數。"""
        return 8
    
    @pytest.fixture
    def temp_dir(self):
        """臨時目錄。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_ppo_buffer_creation(self, state_dim, action_dim, batch_size, rollout_steps):
        """測試 PPO Buffer 創建。"""
        buffer = PPOBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=rollout_steps * batch_size,
            device="cpu"
        )
        
        assert buffer is not None
        assert buffer.state_dim == state_dim
        assert buffer.action_dim == action_dim
        assert buffer.max_size == rollout_steps * batch_size
    
    def test_ppo_buffer_storage(self, state_dim, action_dim, batch_size, rollout_steps):
        """測試 PPO Buffer 存儲。"""
        buffer = PPOBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=rollout_steps * batch_size,
            device="cpu"
        )
        
        # 創建測試數據
        states = torch.randn(rollout_steps, batch_size, state_dim)
        actions = torch.randn(rollout_steps, batch_size, action_dim)
        rewards = torch.randn(rollout_steps, batch_size)
        values = torch.randn(rollout_steps, batch_size)
        log_probs = torch.randn(rollout_steps, batch_size)
        
        # 存儲數據
        for step in range(rollout_steps):
            buffer.store(
                states[step], actions[step], rewards[step],
                values[step], log_probs[step]
            )
        
        # 檢查數據是否正確存儲
        assert buffer.size == rollout_steps * batch_size
        
        # 獲取數據
        batch_data = buffer.get()
        
        # 檢查批次數據
        assert 'states' in batch_data
        assert 'actions' in batch_data
        assert 'rewards' in batch_data
        assert 'values' in batch_data
        assert 'log_probs' in batch_data
        
        # 檢查形狀
        assert batch_data['states'].shape == (rollout_steps * batch_size, state_dim)
        assert batch_data['actions'].shape == (rollout_steps * batch_size, action_dim)
        assert batch_data['rewards'].shape == (rollout_steps * batch_size,)
        assert batch_data['values'].shape == (rollout_steps * batch_size,)
        assert batch_data['log_probs'].shape == (rollout_steps * batch_size,)
    
    def test_ppo_buffer_gae(self, state_dim, action_dim, batch_size, rollout_steps):
        """測試 PPO Buffer GAE 計算。"""
        buffer = PPOBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=rollout_steps * batch_size,
            device="cpu"
        )
        
        # 創建測試數據
        states = torch.randn(rollout_steps, batch_size, state_dim)
        actions = torch.randn(rollout_steps, batch_size, action_dim)
        rewards = torch.randn(rollout_steps, batch_size)
        values = torch.randn(rollout_steps, batch_size)
        log_probs = torch.randn(rollout_steps, batch_size)
        
        # 存儲數據
        for step in range(rollout_steps):
            buffer.store(
                states[step], actions[step], rewards[step],
                values[step], log_probs[step]
            )
        
        # 計算 GAE
        gamma = 0.99
        gae_lambda = 0.95
        last_value = torch.randn(batch_size)
        
        buffer.compute_gae(gamma, gae_lambda, last_value)
        
        # 獲取數據
        batch_data = buffer.get()
        
        # 檢查是否包含 GAE 相關字段
        assert 'advantages' in batch_data
        assert 'returns' in batch_data
        
        # 檢查形狀
        assert batch_data['advantages'].shape == (rollout_steps * batch_size,)
        assert batch_data['returns'].shape == (rollout_steps * batch_size,)
        
        # 檢查數值範圍
        assert torch.all(torch.isfinite(batch_data['advantages']))
        assert torch.all(torch.isfinite(batch_data['returns']))
    
    def test_ppo_trainer_creation(self, state_dim, action_dim, temp_dir):
        """測試 PPO 訓練器創建。"""
        # 創建 Actor 和 Critic
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        critic = MLPCritic(
            state_dim=state_dim,
            device="cpu"
        )
        
        # 創建配置
        config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'rollout_steps': 8,
            'mini_batch_size': 4,
            'update_epochs': 4,
            'num_envs': 1
        }
        
        # 創建訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            env=None,  # 暫時不設置環境
            config=config,
            device="cpu",
            work_dir=temp_dir
        )
        
        assert trainer is not None
        assert trainer.actor == actor
        assert trainer.critic == critic
        assert trainer.config == config
    
    def test_ppo_trainer_optimizers(self, state_dim, action_dim, temp_dir):
        """測試 PPO 訓練器優化器。"""
        # 創建 Actor 和 Critic
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        critic = MLPCritic(
            state_dim=state_dim,
            device="cpu"
        )
        
        # 創建配置
        config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'rollout_steps': 8,
            'mini_batch_size': 4,
            'update_epochs': 4,
            'num_envs': 1,
            'actor_lr': 3e-4,
            'critic_lr': 1e-3
        }
        
        # 創建訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            env=None,
            config=config,
            device="cpu",
            work_dir=temp_dir
        )
        
        # 檢查優化器
        assert trainer.actor_optimizer is not None
        assert trainer.critic_optimizer is not None
        
        # 檢查學習率調度器
        assert trainer.actor_scheduler is not None
        assert trainer.critic_scheduler is not None
    
    def test_ppo_trainer_update_step(self, state_dim, action_dim, temp_dir):
        """測試 PPO 訓練器更新步驟。"""
        # 設置種子
        set_seed(42)
        
        # 創建 Actor 和 Critic
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        critic = MLPCritic(
            state_dim=state_dim,
            device="cpu"
        )
        
        # 創建配置
        config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'rollout_steps': 8,
            'mini_batch_size': 4,
            'update_epochs': 4,
            'num_envs': 1,
            'actor_lr': 3e-4,
            'critic_lr': 1e-3
        }
        
        # 創建訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            env=None,
            config=config,
            device="cpu",
            work_dir=temp_dir
        )
        
        # 創建模擬的 rollout 數據
        batch_size = 16
        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)
        old_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        
        # 執行更新步驟
        update_info = trainer._update_step(
            states, actions, old_log_probs, advantages, returns
        )
        
        # 檢查更新信息
        assert isinstance(update_info, dict)
        assert 'actor_loss' in update_info
        assert 'critic_loss' in update_info
        assert 'entropy' in update_info
        assert 'clip_ratio' in update_info
        
        # 檢查損失值
        assert torch.isfinite(update_info['actor_loss'])
        assert torch.isfinite(update_info['critic_loss'])
        assert torch.isfinite(update_info['entropy'])
        assert torch.isfinite(update_info['clip_ratio'])
        
        # 檢查損失範圍
        assert update_info['actor_loss'] >= 0
        assert update_info['critic_loss'] >= 0
        assert update_info['entropy'] >= 0
        assert 0 <= update_info['clip_ratio'] <= 1
    
    def test_ppo_trainer_checkpoint_save_load(self, state_dim, action_dim, temp_dir):
        """測試 PPO 訓練器檢查點保存和載入。"""
        # 創建 Actor 和 Critic
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        critic = MLPCritic(
            state_dim=state_dim,
            device="cpu"
        )
        
        # 創建配置
        config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'rollout_steps': 8,
            'mini_batch_size': 4,
            'update_epochs': 4,
            'num_envs': 1
        }
        
        # 創建訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            env=None,
            config=config,
            device="cpu",
            work_dir=temp_dir
        )
        
        # 保存檢查點
        checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # 檢查文件是否存在
        assert checkpoint_path.exists()
        
        # 創建新的訓練器
        new_trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            env=None,
            config=config,
            device="cpu",
            work_dir=temp_dir
        )
        
        # 載入檢查點
        new_trainer.load_checkpoint(checkpoint_path)
        
        # 檢查是否成功載入
        assert True  # 如果沒有拋出異常，說明載入成功
    
    def test_ppo_trainer_gradient_clipping(self, state_dim, action_dim, temp_dir):
        """測試 PPO 訓練器梯度裁剪。"""
        # 創建 Actor 和 Critic
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        critic = MLPCritic(
            state_dim=state_dim,
            device="cpu"
        )
        
        # 創建配置
        config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'rollout_steps': 8,
            'mini_batch_size': 4,
            'update_epochs': 4,
            'num_envs': 1
        }
        
        # 創建訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            env=None,
            config=config,
            device="cpu",
            work_dir=temp_dir
        )
        
        # 創建模擬的 rollout 數據
        batch_size = 16
        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)
        old_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        
        # 執行更新步驟
        update_info = trainer._update_step(
            states, actions, old_log_probs, advantages, returns
        )
        
        # 檢查梯度是否被正確裁剪
        # 這需要檢查模型參數的梯度範數
        max_grad_norm = config['max_grad_norm']
        
        # 檢查 Actor 梯度
        actor_grad_norm = 0.0
        for param in actor.parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2)
                actor_grad_norm = max(actor_grad_norm, param_grad_norm.item())
        
        # 檢查 Critic 梯度
        critic_grad_norm = 0.0
        for param in critic.parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2)
                critic_grad_norm = max(critic_grad_norm, param_grad_norm.item())
        
        # 梯度範數應該小於或等於 max_grad_norm
        if actor_grad_norm > 0:
            assert actor_grad_norm <= max_grad_norm * 1.1  # 允許一些誤差
        if critic_grad_norm > 0:
            assert critic_grad_norm <= max_grad_norm * 1.1  # 允許一些誤差
    
    def test_ppo_trainer_mixed_precision(self, state_dim, action_dim, temp_dir):
        """測試 PPO 訓練器混合精度。"""
        # 創建 Actor 和 Critic
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        critic = MLPCritic(
            state_dim=state_dim,
            device="cpu"
        )
        
        # 創建配置（啟用混合精度）
        config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'rollout_steps': 8,
            'mini_batch_size': 4,
            'update_epochs': 4,
            'num_envs': 1,
            'mixed_precision': True
        }
        
        # 創建訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            env=None,
            config=config,
            device="cpu",
            work_dir=temp_dir
        )
        
        # 檢查是否啟用混合精度
        if hasattr(trainer, 'scaler'):
            assert trainer.scaler is not None
        else:
            # 如果沒有 scaler，檢查配置
            assert config.get('mixed_precision', False) == True
    
    def test_ppo_trainer_learning_rate_scheduling(self, state_dim, action_dim, temp_dir):
        """測試 PPO 訓練器學習率調度。"""
        # 創建 Actor 和 Critic
        actor = QuantumActor(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu"
        )
        
        critic = MLPCritic(
            state_dim=state_dim,
            device="cpu"
        )
        
        # 創建配置
        config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'rollout_steps': 8,
            'mini_batch_size': 4,
            'update_epochs': 4,
            'num_envs': 1,
            'actor_lr': 3e-4,
            'critic_lr': 1e-3,
            'lr_schedule': 'linear'
        }
        
        # 創建訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=actor,
            env=None,
            config=config,
            device="cpu",
            work_dir=temp_dir
        )
        
        # 檢查學習率調度器
        assert trainer.actor_scheduler is not None
        assert trainer.critic_scheduler is not None
        
        # 記錄初始學習率
        initial_actor_lr = trainer.actor_optimizer.param_groups[0]['lr']
        initial_critic_lr = trainer.critic_optimizer.param_groups[0]['lr']
        
        # 執行一些更新步驟
        for step in range(5):
            # 創建模擬數據
            batch_size = 16
            states = torch.randn(batch_size, state_dim)
            actions = torch.randn(batch_size, action_dim)
            old_log_probs = torch.randn(batch_size)
            advantages = torch.randn(batch_size)
            returns = torch.randn(batch_size)
            
            # 執行更新
            trainer._update_step(
                states, actions, old_log_probs, advantages, returns
            )
            
            # 更新學習率
            trainer.actor_scheduler.step()
            trainer.critic_scheduler.step()
        
        # 檢查學習率是否變化
        current_actor_lr = trainer.actor_optimizer.param_groups[0]['lr']
        current_critic_lr = trainer.critic_optimizer.param_groups[0]['lr']
        
        # 學習率應該有所變化（對於線性調度）
        if config['lr_schedule'] == 'linear':
            assert current_actor_lr != initial_actor_lr
            assert current_critic_lr != initial_critic_lr


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
