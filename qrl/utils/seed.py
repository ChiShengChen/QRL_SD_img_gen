"""種子設置工具模組，確保實驗的可重現性。"""

import random
import numpy as np
import torch
import os
from typing import Optional, Union


def set_seed(seed: Union[int, Optional[int]] = None) -> int:
    """
    設置隨機種子以確保實驗可重現性。
    
    Args:
        seed: 種子值，如果為 None 則使用環境變數或隨機生成
        
    Returns:
        使用的種子值
    """
    if seed is None:
        seed = int(os.environ.get("QRL_SEED", random.randint(0, 2**32 - 1)))
    
    # 設置 Python 隨機種子
    random.seed(seed)
    
    # 設置 NumPy 隨機種子
    np.random.seed(seed)
    
    # 設置 PyTorch 隨機種子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情況
    
    # 設置 PyTorch 隨機數生成器種子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 設置環境變數
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["QRL_SEED"] = str(seed)
    
    print(f"種子已設置為: {seed}")
    return seed


def get_seed() -> int:
    """獲取當前種子值。"""
    return int(os.environ.get("QRL_SEED", 42))


def set_deterministic() -> None:
    """設置完全確定性模式（可能影響性能）。"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print("已啟用確定性模式")


def set_reproducible() -> None:
    """設置可重現模式（平衡性能和可重現性）。"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("已啟用可重現模式")


def seed_worker(worker_id: int) -> None:
    """為 DataLoader 工作進程設置種子。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """獲取 PyTorch 隨機數生成器。"""
    if seed is None:
        seed = get_seed()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
