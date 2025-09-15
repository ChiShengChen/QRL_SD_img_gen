"""
QRL Image Synthesis Package

Quantum Reinforcement Learning-Guided Image Synthesis via Hybrid Quantum-Classical Generative Model Architectures.
"""

__version__ = "0.1.0"
__author__ = "QRL Team"
__email__ = "team@qrl-synthesis.org"

# 主要模組導入
from . import (
    actors,
    critics,
    envs,
    controls,
    reward,
    training,
    models,
    metrics,
    utils,
)

# 版本信息
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "actors",
    "critics", 
    "envs",
    "controls",
    "reward",
    "training",
    "models",
    "metrics",
    "utils",
]
