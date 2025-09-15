"""測試模組。"""

# 測試配置
pytest_plugins = []

# 測試工具函數
def create_test_config():
    """創建測試配置。"""
    return {
        'seed': 42,
        'device': 'cpu',
        'batch_size': 4,
        'state_dim': 32,
        'action_dim': 1
    }


def create_test_tensors(batch_size, state_dim, action_dim):
    """創建測試張量。"""
    return {
        'states': torch.randn(batch_size, state_dim),
        'actions': torch.randn(batch_size, action_dim),
        'rewards': torch.randn(batch_size),
        'values': torch.randn(batch_size),
        'log_probs': torch.randn(batch_size)
    }


# 版本信息
__version__ = "0.1.0"
