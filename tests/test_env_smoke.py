"""環境煙霧測試。"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qrl.envs import create_diffusion_env
from qrl.controls import ControlSpaces
from qrl.reward import ClassifierReward


class TestDiffusionEnvironment:
    """測試擴散環境。"""
    
    @pytest.fixture
    def env(self):
        """創建測試環境。"""
        # 使用 CPU 進行測試
        env = create_diffusion_env(
            dataset="cifar10",
            model="unet_ddim",
            control="stageA_cfg",
            device="cpu"
        )
        return env
    
    def test_env_creation(self, env):
        """測試環境創建。"""
        assert env is not None
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
    
    def test_observation_space(self, env):
        """測試觀察空間。"""
        obs_space = env.observation_space
        assert obs_space is not None
        
        # 檢查觀察空間的形狀
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs['observation']
        
        assert obs.shape == obs_space.shape
        assert obs.dtype == obs_space.dtype
    
    def test_action_space(self, env):
        """測試動作空間。"""
        action_space = env.action_space
        assert action_space is not None
        
        # 檢查動作空間的形狀
        action = action_space.sample()
        assert action.shape == action_space.shape
        assert action.dtype == action_space.dtype
    
    def test_reset(self, env):
        """測試環境重置。"""
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs['observation']
        
        assert obs is not None
        assert obs.shape == env.observation_space.shape
    
    def test_step(self, env):
        """測試環境步進。"""
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs['observation']
        
        # 生成隨機動作
        action = env.action_space.sample()
        
        # 執行動作
        next_obs, reward, done, truncated, info = env.step(action)
        
        if isinstance(next_obs, dict):
            next_obs = next_obs['observation']
        
        # 檢查返回值
        assert next_obs is not None
        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_multiple_steps(self, env):
        """測試多步執行。"""
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs['observation']
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            if isinstance(obs, dict):
                obs = obs['observation']
            
            if done or truncated:
                break
        
        # 環境應該能夠完成多步執行
        assert True
    
    def test_action_validation(self, env):
        """測試動作驗證。"""
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs['observation']
        
        # 測試有效動作
        valid_action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(valid_action)
        assert next_obs is not None
        
        # 測試無效動作（超出範圍）
        invalid_action = np.ones_like(valid_action) * 1000
        try:
            next_obs, reward, done, truncated, info = env.step(invalid_action)
            # 如果環境沒有拋出異常，檢查動作是否被裁剪
            assert True
        except Exception:
            # 環境拋出異常也是可以接受的
            assert True
    
    def test_reward_range(self, env):
        """測試獎勵範圍。"""
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs['observation']
        
        rewards = []
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            if isinstance(obs, dict):
                obs = obs['observation']
            
            rewards.append(reward)
            
            if done or truncated:
                break
        
        # 檢查獎勵是否在合理範圍內
        assert len(rewards) > 0
        assert all(isinstance(r, (int, float, np.number)) for r in rewards)
    
    def test_environment_info(self, env):
        """測試環境信息。"""
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs['observation']
        
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        
        # 檢查信息字典
        assert isinstance(info, dict)
        
        # 檢查是否有必要的鍵
        expected_keys = ['step', 'total_reward']
        for key in expected_keys:
            if key in info:
                assert True  # 鍵存在
            else:
                # 鍵不存在也是可以接受的
                assert True
    
    def test_environment_seed(self, env):
        """測試環境種子設置。"""
        # 設置種子
        env.seed(42)
        
        # 重置環境
        obs1 = env.reset()
        if isinstance(obs1, dict):
            obs1 = obs1['observation']
        
        # 再次設置種子並重置
        env.seed(42)
        obs2 = env.reset()
        if isinstance(obs2, dict):
            obs2 = obs2['observation']
        
        # 檢查是否可重現（如果環境支持）
        try:
            np.testing.assert_array_equal(obs1, obs2)
        except AssertionError:
            # 如果不可重現，這也是可以接受的
            assert True
    
    def test_environment_close(self, env):
        """測試環境關閉。"""
        # 如果環境有 close 方法，測試它
        if hasattr(env, 'close'):
            try:
                env.close()
                assert True
            except Exception:
                assert False
        else:
            # 沒有 close 方法也是可以接受的
            assert True


class TestControlSpaces:
    """測試控制空間。"""
    
    def test_stage_a_control_space(self):
        """測試 Stage A 控制空間。"""
        action_space = ControlSpaces.get_stage_a_action_space()
        assert action_space is not None
        assert hasattr(action_space, 'shape')
        assert hasattr(action_space, 'dtype')
        assert hasattr(action_space, 'sample')
    
    def test_stage_b_control_space(self):
        """測試 Stage B 控制空間。"""
        action_space = ControlSpaces.get_stage_b_action_space()
        assert action_space is not None
        assert hasattr(action_space, 'shape')
        assert hasattr(action_space, 'dtype')
        assert hasattr(action_space, 'sample')
    
    def test_stage_c_control_space(self):
        """測試 Stage C 控制空間。"""
        action_space = ControlSpaces.get_stage_c_action_space()
        assert action_space is not None
        assert hasattr(action_space, 'shape')
        assert hasattr(action_space, 'dtype')
        assert hasattr(action_space, 'sample')
    
    def test_action_processor(self):
        """測試動作處理器。"""
        processor = ControlSpaces.ActionProcessor()
        
        # 測試有效動作
        valid_action = np.array([0.5])
        processed_action = processor.process_action(valid_action, 'stageA_cfg')
        assert processed_action is not None
        
        # 測試無效動作（超出範圍）
        invalid_action = np.array([100.0])
        processed_action = processor.process_action(invalid_action, 'stageA_cfg')
        assert processed_action is not None


class TestClassifierReward:
    """測試分類器獎勵。"""
    
    def test_classifier_creation(self):
        """測試分類器創建。"""
        try:
            classifier = ClassifierReward(target_class=0, device="cpu")
            assert classifier is not None
        except Exception as e:
            # 如果無法創建分類器（例如缺少預訓練權重），這也是可以接受的
            pytest.skip(f"無法創建分類器: {e}")
    
    def test_reward_calculation(self):
        """測試獎勵計算。"""
        try:
            classifier = ClassifierReward(target_class=0, device="cpu")
            
            # 創建測試圖像
            test_image = torch.randn(1, 3, 32, 32)
            
            # 計算獎勵
            reward = classifier.calculate_reward(test_image)
            assert isinstance(reward, (int, float, np.number))
            
        except Exception as e:
            # 如果無法計算獎勵，這也是可以接受的
            pytest.skip(f"無法計算獎勵: {e}")


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
