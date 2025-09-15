"""多樣性獎勵模組，實現基於圖像多樣性的獎勵函數。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from scipy.spatial.distance import pdist, squareform


class DiversityReward:
    """多樣性獎勵計算器。"""
    
    def __init__(
        self,
        diversity_weight: float = 0.1,
        lpips_weight: float = 0.05,
        perceptual_weight: float = 0.02,
        device: str = "cuda"
    ):
        """
        初始化多樣性獎勵。
        
        Args:
            diversity_weight: 多樣性獎勵權重
            lpips_weight: LPIPS 多樣性權重
            perceptual_weight: 感知多樣性權重
            device: 設備
        """
        self.diversity_weight = diversity_weight
        self.lpips_weight = lpips_weight
        self.perceptual_weight = perceptual_weight
        self.device = device
        
        # 初始化 LPIPS 模型（如果可用）
        self.lpips_model = self._init_lpips()
        
        # 記錄歷史多樣性分數
        self.diversity_history = []
    
    def _init_lpips(self) -> Optional[nn.Module]:
        """初始化 LPIPS 模型。"""
        try:
            import lpips
            model = lpips.LPIPS(net='alex', verbose=False)
            model.to(self.device)
            return model
        except ImportError:
            print("警告：LPIPS 未安裝，將禁用 LPIPS 多樣性計算")
            return None
    
    def pixel_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        計算像素級多樣性。
        
        Args:
            images: 圖像批次 [B, C, H, W]
            
        Returns:
            torch.Tensor: 多樣性分數
        """
        # 將圖像展平為向量
        batch_size = images.size(0)
        flattened = images.view(batch_size, -1)
        
        # 計算成對歐氏距離
        distances = torch.cdist(flattened, flattened)
        
        # 排除對角線元素（自己與自己的距離）
        mask = ~torch.eye(batch_size, dtype=bool, device=images.device)
        valid_distances = distances[mask]
        
        # 計算平均距離作為多樣性分數
        diversity_score = valid_distances.mean()
        
        return diversity_score
    
    def lpips_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        計算 LPIPS 多樣性。
        
        Args:
            images: 圖像批次 [B, C, H, W]
            
        Returns:
            torch.Tensor: LPIPS 多樣性分數
        """
        if self.lpips_model is None:
            return torch.tensor(0.0, device=images.device)
        
        batch_size = images.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=images.device)
        
        # 計算成對 LPIPS 距離
        total_distance = 0.0
        count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                with torch.no_grad():
                    distance = self.lpips_model(images[i:i+1], images[j:j+1])
                    total_distance += distance.item()
                    count += 1
        
        # 計算平均距離
        avg_distance = total_distance / count if count > 0 else 0.0
        
        return torch.tensor(avg_distance, device=images.device)
    
    def perceptual_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        計算感知多樣性（基於特徵空間）。
        
        Args:
            images: 圖像批次 [B, C, H, W]
            
        Returns:
            torch.Tensor: 感知多樣性分數
        """
        # 使用簡單的統計特徵作為感知表示
        # 計算每個圖像的統計特徵
        features = []
        
        for i in range(images.size(0)):
            img = images[i]
            # 計算均值、標準差、偏度等統計特徵
            mean = img.mean()
            std = img.std()
            # 簡化的偏度計算
            centered = img - mean
            skewness = (centered ** 3).mean() / (std ** 3 + 1e-8)
            
            feature = torch.stack([mean, std, skewness])
            features.append(feature)
        
        features = torch.stack(features)  # [B, 3]
        
        # 計算特徵空間中的多樣性
        distances = torch.cdist(features, features)
        mask = ~torch.eye(images.size(0), dtype=bool, device=images.device)
        valid_distances = distances[mask]
        
        diversity_score = valid_distances.mean()
        
        return diversity_score
    
    def compute_diversity_reward(
        self,
        images: torch.Tensor,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        計算多樣性獎勵。
        
        Args:
            images: 圖像批次
            normalize: 是否標準化多樣性分數
            
        Returns:
            Dict: 包含各種多樣性分數的字典
        """
        # 計算各種多樣性分數
        pixel_div = self.pixel_diversity(images)
        lpips_div = self.lpips_diversity(images)
        perceptual_div = self.perceptual_diversity(images)
        
        # 標準化（可選）
        if normalize:
            # 簡單的 min-max 標準化
            pixel_div = (pixel_div - pixel_div.min()) / (pixel_div.max() - pixel_div.min() + 1e-8)
            lpips_div = (lpips_div - lpips_div.min()) / (lpips_div.max() - lpips_div.min() + 1e-8)
            perceptual_div = (perceptual_div - perceptual_div.min()) / (perceptual_div.max() - perceptual_div.min() + 1e-8)
        
        # 計算加權總分
        total_diversity = (
            self.diversity_weight * pixel_div +
            self.lpips_weight * lpips_div +
            self.perceptual_weight * perceptual_div
        )
        
        # 記錄歷史
        self.diversity_history.append(total_diversity.item())
        
        return {
            "pixel_diversity": pixel_div,
            "lpips_diversity": lpips_div,
            "perceptual_diversity": perceptual_div,
            "total_diversity": total_diversity,
        }
    
    def compute_batch_diversity_reward(
        self,
        image_batches: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        計算批次間的多樣性獎勵。
        
        Args:
            image_batches: 圖像批次列表
            
        Returns:
            torch.Tensor: 批次間多樣性獎勵
        """
        if len(image_batches) < 2:
            return torch.tensor(0.0, device=image_batches[0].device)
        
        # 計算每個批次的代表性特徵（使用均值）
        batch_features = []
        
        for batch in image_batches:
            # 計算批次的統計特徵
            mean_feature = batch.mean(dim=0)  # [C, H, W]
            std_feature = batch.std(dim=0)    # [C, H, W]
            
            # 展平並連接特徵
            feature = torch.cat([mean_feature.flatten(), std_feature.flatten()])
            batch_features.append(feature)
        
        batch_features = torch.stack(batch_features)  # [num_batches, feature_dim]
        
        # 計算批次間距離
        distances = torch.cdist(batch_features, batch_features)
        mask = ~torch.eye(len(image_batches), dtype=bool, device=batch_features.device)
        valid_distances = distances[mask]
        
        # 計算平均距離作為批次間多樣性
        batch_diversity = valid_distances.mean()
        
        return batch_diversity
    
    def get_diversity_history(self) -> list:
        """獲取多樣性歷史記錄。"""
        return self.diversity_history.copy()
    
    def reset_history(self) -> None:
        """重置多樣性歷史記錄。"""
        self.diversity_history.clear()
    
    def update_weights(
        self,
        diversity_weight: Optional[float] = None,
        lpips_weight: Optional[float] = None,
        perceptual_weight: Optional[float] = None
    ) -> None:
        """更新獎勵權重。"""
        if diversity_weight is not None:
            self.diversity_weight = diversity_weight
        if lpips_weight is not None:
            self.lpips_weight = lpips_weight
        if perceptual_weight is not None:
            self.perceptual_weight = perceptual_weight


# 便捷函數
def create_diversity_reward(
    diversity_weight: float = 0.1,
    lpips_weight: float = 0.05,
    perceptual_weight: float = 0.02,
    **kwargs
) -> DiversityReward:
    """創建多樣性獎勵的便捷函數。"""
    return DiversityReward(
        diversity_weight=diversity_weight,
        lpips_weight=lpips_weight,
        perceptual_weight=perceptual_weight,
        **kwargs
    )
