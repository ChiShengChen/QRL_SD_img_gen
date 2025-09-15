"""分類器獎勵模組，實現基於分類器置信度的獎勵函數。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import os


class CIFAR10Classifier(nn.Module):
    """CIFAR-10 分類器，基於 ResNet-18。"""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        初始化分類器。
        
        Args:
            num_classes: 類別數量
            pretrained: 是否使用預訓練權重
        """
        super().__init__()
        
        # 載入預訓練 ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 修改第一層以適應 CIFAR-10 的 32x32 輸入
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # 移除最大池化層
        
        # 修改最後一層以適應類別數量
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # 凍結部分層以減少參數量
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 只訓練最後幾層
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        
        Args:
            x: 輸入圖像張量 [B, 3, 32, 32]
            
        Returns:
            torch.Tensor: 分類 logits [B, num_classes]
        """
        return self.backbone(x)
    
    def get_confidence(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        獲取目標類別的置信度。
        
        Args:
            x: 輸入圖像張量
            target_class: 目標類別
            
        Returns:
            torch.Tensor: 目標類別的置信度
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        confidence = probs[:, target_class]
        return confidence


class ClassifierReward:
    """分類器獎勵計算器。"""
    
    def __init__(
        self,
        classifier: Optional[nn.Module] = None,
        target_class: int = 3,
        alpha: float = 1.0,
        beta: float = 0.2,
        action_penalty: float = 0.005,
        tv_penalty: float = 0.0,
        device: str = "cuda"
    ):
        """
        初始化分類器獎勵。
        
        Args:
            classifier: 預訓練分類器
            target_class: 目標類別
            alpha: 終端獎勵係數
            beta: 步間獎勵係數
            action_penalty: 動作正則化係數
            tv_penalty: 總變分正則化係數
            device: 設備
        """
        self.target_class = target_class
        self.alpha = alpha
        self.beta = beta
        self.action_penalty = action_penalty
        self.tv_penalty = tv_penalty
        self.device = device
        
        # 初始化分類器
        if classifier is None:
            self.classifier = self._load_default_classifier()
        else:
            self.classifier = classifier
        
        self.classifier.to(device)
        self.classifier.eval()
        
        # 記錄歷史置信度
        self.confidence_history = []
    
    def _load_default_classifier(self) -> nn.Module:
        """載入預設分類器。"""
        classifier = CIFAR10Classifier(pretrained=True)
        
        # 嘗試載入微調後的權重
        weight_path = Path("checkpoints/cifar10_classifier.pth")
        if weight_path.exists():
            print(f"載入預訓練分類器權重: {weight_path}")
            classifier.load_state_dict(torch.load(weight_path, map_location="cpu"))
        else:
            print("使用 ImageNet 預訓練權重，建議在 CIFAR-10 上微調")
        
        return classifier
    
    def terminal_reward(self, image: torch.Tensor) -> torch.Tensor:
        """
        計算終端獎勵（基於分類器置信度）。
        
        Args:
            image: 生成的圖像 [B, 3, 32, 32]
            
        Returns:
            torch.Tensor: 終端獎勵 [B]
        """
        with torch.no_grad():
            confidence = self.classifier.get_confidence(image, self.target_class)
            # 使用對數置信度作為獎勵
            reward = self.alpha * torch.log(confidence + 1e-8)
            
            # 記錄置信度
            self.confidence_history.append(confidence.mean().item())
            
            return reward
    
    def step_reward(
        self,
        prev_confidence: torch.Tensor,
        curr_confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        計算步間獎勵（置信度增量）。
        
        Args:
            prev_confidence: 上一步置信度
            curr_confidence: 當前步置信度
            
        Returns:
            torch.Tensor: 步間獎勵
        """
        confidence_diff = curr_confidence - prev_confidence
        reward = self.beta * confidence_diff
        return reward
    
    def action_regularization(self, action: torch.Tensor) -> torch.Tensor:
        """
        計算動作正則化獎勵。
        
        Args:
            action: 動作張量
            
        Returns:
            torch.Tensor: 正則化獎勵
        """
        # L2 正則化
        l2_penalty = torch.norm(action, dim=-1) ** 2
        reward = -self.action_penalty * l2_penalty
        return reward
    
    def tv_regularization(self, image: torch.Tensor) -> torch.Tensor:
        """
        計算總變分正則化獎勵。
        
        Args:
            image: 圖像張量 [B, 3, H, W]
            
        Returns:
            torch.Tensor: TV 正則化獎勵
        """
        if self.tv_penalty <= 0:
            return torch.zeros(image.size(0), device=image.device)
        
        # 計算水平和垂直梯度
        h_gradient = image[:, :, :, 1:] - image[:, :, :, :-1]
        v_gradient = image[:, :, 1:, :] - image[:, :, :-1, :]
        
        # 計算總變分
        tv = torch.sum(torch.abs(h_gradient)) + torch.sum(torch.abs(v_gradient))
        reward = -self.tv_penalty * tv
        
        return reward
    
    def compute_reward(
        self,
        image: torch.Tensor,
        action: torch.Tensor,
        prev_confidence: Optional[torch.Tensor] = None,
        curr_confidence: Optional[torch.Tensor] = None,
        is_terminal: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        計算完整的獎勵。
        
        Args:
            image: 當前圖像
            action: 當前動作
            prev_confidence: 上一步置信度
            curr_confidence: 當前步置信度
            is_terminal: 是否為終端狀態
            
        Returns:
            Dict: 包含各種獎勵組件的字典
        """
        rewards = {}
        
        # 終端獎勵
        if is_terminal:
            rewards["terminal"] = self.terminal_reward(image)
        else:
            rewards["terminal"] = torch.zeros(image.size(0), device=image.device)
        
        # 步間獎勵
        if prev_confidence is not None and curr_confidence is not None:
            rewards["step"] = self.step_reward(prev_confidence, curr_confidence)
        else:
            rewards["step"] = torch.zeros(image.size(0), device=image.device)
        
        # 動作正則化
        rewards["action_reg"] = self.action_regularization(action)
        
        # TV 正則化
        rewards["tv_reg"] = self.tv_regularization(image)
        
        # 總獎勵
        total_reward = sum(rewards.values())
        rewards["total"] = total_reward
        
        return rewards
    
    def get_confidence_history(self) -> list:
        """獲取置信度歷史記錄。"""
        return self.confidence_history.copy()
    
    def reset_history(self) -> None:
        """重置置信度歷史記錄。"""
        self.confidence_history.clear()


# 便捷函數
def create_classifier_reward(
    target_class: int = 3,
    alpha: float = 1.0,
    beta: float = 0.2,
    **kwargs
) -> ClassifierReward:
    """創建分類器獎勵的便捷函數。"""
    return ClassifierReward(
        target_class=target_class,
        alpha=alpha,
        beta=beta,
        **kwargs
    )
