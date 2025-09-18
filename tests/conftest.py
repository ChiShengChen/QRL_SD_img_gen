"""Pytest 配置文件。"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_device():
    """測試設備。"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@pytest.fixture(scope="session")
def test_seed():
    """測試種子。"""
    return 42


@pytest.fixture(scope="session")
def test_work_dir():
    """測試工作目錄。"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def temp_dir():
    """臨時目錄。"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_config():
    """測試配置。"""
    return {
        'seed': 42,
        'device': 'cpu',
        'batch_size': 4,
        'state_dim': 32,
        'action_dim': 1,
        'rollout_steps': 8,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'mini_batch_size': 4,
        'update_epochs': 4,
        'num_envs': 1,
        'actor_lr': 3e-4,
        'critic_lr': 1e-3
    }


@pytest.fixture(scope="function")
def mock_env():
    """模擬環境。"""
    class MockEnv:
        def __init__(self):
            self.observation_space = type('obj', (object,), {
                'shape': (32,),
                'dtype': np.float32
            })()
            self.action_space = type('obj', (object,), {
                'shape': (1,),
                'dtype': np.float32,
                'sample': lambda: np.random.randn(1)
            })()
        
        def reset(self):
            return np.random.randn(32)
        
        def step(self, action):
            obs = np.random.randn(32)
            reward = np.random.randn()
            done = False
            truncated = False
            info = {'step': 0, 'total_reward': 0.0}
            return obs, reward, done, truncated, info
        
        def seed(self, seed):
            np.random.seed(seed)
    
    return MockEnv()


# 設置測試環境
def pytest_configure(config):
    """配置 pytest。"""
    # 設置 NumPy 和 PyTorch 的隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 設置 PyTorch 為確定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pytest_collection_modifyitems(config, items):
    """修改測試項目。"""
    # 為慢速測試添加標記
    slow_tests = [
        "test_env_smoke.py::TestDiffusionEnvironment::test_multiple_steps",
        "test_train_step.py::TestPPOTraining::test_ppo_trainer_learning_rate_scheduling"
    ]
    
    for item in items:
        if any(slow_test in str(item.fspath) for slow_test in slow_tests):
            item.add_marker(pytest.mark.slow)


# 測試標記
pytest_plugins = []


# 自定義標記
def pytest_ini_setup(config):
    """設置 pytest.ini。"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
