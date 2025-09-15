"""獎勵模組包。"""

from .classifier_reward import (
    CIFAR10Classifier,
    ClassifierReward,
    create_classifier_reward,
)

from .diversity import (
    DiversityReward,
    create_diversity_reward,
)

__all__ = [
    # 分類器獎勵
    "CIFAR10Classifier",
    "ClassifierReward",
    "create_classifier_reward",
    
    # 多樣性獎勵
    "DiversityReward",
    "create_diversity_reward",
]
