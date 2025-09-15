"""Inception Score 指標計算模組。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from pathlib import Path
import os
from tqdm import tqdm
import warnings
from scipy.stats import entropy


class InceptionV3Classifier(nn.Module):
    """InceptionV3 分類器。"""
    
    def __init__(self, device: str = "cuda"):
        """
        初始化 InceptionV3 分類器。
        
        Args:
            device: 設備
        """
        super().__init__()
        
        try:
            import torchvision.models as models
            self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        except:
            # 如果無法載入預訓練模型，使用隨機初始化的模型
            warnings.warn("無法載入預訓練 InceptionV3，使用隨機初始化模型")
            self.inception = models.inception_v3(pretrained=False, aux_logits=False)
        
        # 保持分類層
        self.inception.to(device)
        self.inception.eval()
        
        self.device = device
        self.num_classes = 1000  # ImageNet 類別數
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        
        Args:
            x: 輸入圖像 [B, 3, H, W]
            
        Returns:
            torch.Tensor: 分類 logits [B, num_classes]
        """
        # 確保圖像在正確的範圍內 [0, 1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # 調整圖像大小為 299x299（InceptionV3 的標準輸入）
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 標準化（ImageNet 統計）
        x = (x - 0.5) * 2.0
        
        with torch.no_grad():
            logits = self.inception(x)
        
        return logits
    
    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        獲取分類概率。
        
        Args:
            x: 輸入圖像
            
        Returns:
            torch.Tensor: 分類概率
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs


class InceptionScoreCalculator:
    """Inception Score 計算器。"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
        splits: int = 10
    ):
        """
        初始化 Inception Score 計算器。
        
        Args:
            device: 設備
            batch_size: 批次大小
            splits: 分割數量
        """
        self.device = device
        self.batch_size = batch_size
        self.splits = splits
        
        self.classifier = InceptionV3Classifier(device)
    
    def calculate_inception_score(
        self,
        images: Union[str, List[str], torch.Tensor]
    ) -> Tuple[float, float]:
        """
        計算 Inception Score。
        
        Args:
            images: 圖像路徑列表或張量
            
        Returns:
            Tuple: (平均 Inception Score, 標準差)
        """
        # 提取分類概率
        probs = self._extract_classification_probs(images)
        
        # 計算 Inception Score
        scores = []
        n = len(probs)
        
        for i in range(self.splits):
            # 隨機分割
            start_idx = (i * n) // self.splits
            end_idx = ((i + 1) * n) // self.splits
            
            if start_idx == end_idx:
                continue
            
            split_probs = probs[start_idx:end_idx]
            
            # 計算平均概率
            mean_probs = split_probs.mean(axis=0)
            
            # 計算 KL 散度
            kl_divs = []
            for p in split_probs:
                # 避免零概率
                p = np.clip(p, 1e-8, 1.0)
                mean_probs_clipped = np.clip(mean_probs, 1e-8, 1.0)
                
                kl_div = entropy(p, mean_probs_clipped)
                kl_divs.append(kl_div)
            
            # 計算平均 KL 散度
            mean_kl = np.mean(kl_divs)
            
            # Inception Score = exp(mean_kl)
            score = np.exp(mean_kl)
            scores.append(score)
        
        # 計算平均值和標準差
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return mean_score, std_score
    
    def _extract_classification_probs(
        self,
        images: Union[str, List[str], torch.Tensor]
    ) -> np.ndarray:
        """提取分類概率。"""
        if isinstance(images, str):
            # 單個目錄路徑
            image_paths = list(Path(images).glob("*.jpg")) + list(Path(images).glob("*.png"))
            return self._extract_probs_from_paths(image_paths)
        elif isinstance(images, list):
            # 圖像路徑列表
            return self._extract_probs_from_paths(images)
        elif isinstance(images, torch.Tensor):
            # 圖像張量
            return self._extract_probs_from_tensor(images)
        else:
            raise ValueError(f"不支援的圖像類型: {type(images)}")
    
    def _extract_probs_from_paths(self, image_paths: List[str]) -> np.ndarray:
        """從圖像路徑提取分類概率。"""
        all_probs = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="提取分類概率"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    from PIL import Image
                    import torchvision.transforms as transforms
                    
                    # 載入圖像
                    image = Image.open(path).convert('RGB')
                    
                    # 轉換為張量
                    transform = transforms.Compose([
                        transforms.Resize((299, 299)),
                        transforms.ToTensor()
                    ])
                    
                    image_tensor = transform(image).unsqueeze(0)
                    batch_images.append(image_tensor)
                    
                except Exception as e:
                    warnings.warn(f"無法載入圖像 {path}: {e}")
                    continue
            
            if batch_images:
                batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                batch_probs = self.classifier.get_probs(batch_tensor)
                all_probs.append(batch_probs.cpu().numpy())
        
        if not all_probs:
            raise ValueError("沒有成功載入任何圖像")
        
        return np.concatenate(all_probs, axis=0)
    
    def _extract_probs_from_tensor(self, images: torch.Tensor) -> np.ndarray:
        """從圖像張量提取分類概率。"""
        all_probs = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size].to(self.device)
            batch_probs = self.classifier.get_probs(batch)
            all_probs.append(batch_probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def calculate_fid_and_is(
        self,
        real_images: Union[str, List[str], torch.Tensor],
        fake_images: Union[str, List[str], torch.Tensor]
    ) -> Dict[str, float]:
        """
        同時計算 FID 和 Inception Score。
        
        Args:
            real_images: 真實圖像
            fake_images: 生成圖像
            
        Returns:
            Dict: 包含 FID 和 IS 的字典
        """
        from .fid import FIDCalculator
        
        # 計算 FID
        fid_calculator = FIDCalculator(device=self.device, batch_size=self.batch_size)
        fid_score = fid_calculator.calculate_fid(real_images, fake_images)
        
        # 計算 Inception Score
        is_mean, is_std = self.calculate_inception_score(fake_images)
        
        return {
            'fid': fid_score,
            'inception_score_mean': is_mean,
            'inception_score_std': is_std
        }


class SimplifiedInceptionScore:
    """簡化的 Inception Score 計算器（使用預計算特徵）。"""
    
    def __init__(self, device: str = "cuda"):
        """
        初始化簡化 Inception Score 計算器。
        
        Args:
            device: 設備
        """
        self.device = device
    
    def calculate_simplified_is(
        self,
        features: torch.Tensor,
        splits: int = 10
    ) -> Tuple[float, float]:
        """
        使用預計算特徵計算簡化的 Inception Score。
        
        Args:
            features: 預計算的特徵 [N, num_classes]
            splits: 分割數量
            
        Returns:
            Tuple: (平均 Inception Score, 標準差)
        """
        # 轉換為概率
        probs = F.softmax(features, dim=1).cpu().numpy()
        
        # 計算 Inception Score
        scores = []
        n = len(probs)
        
        for i in range(splits):
            # 隨機分割
            start_idx = (i * n) // splits
            end_idx = ((i + 1) * n) // splits
            
            if start_idx == end_idx:
                continue
            
            split_probs = probs[start_idx:end_idx]
            
            # 計算平均概率
            mean_probs = split_probs.mean(axis=0)
            
            # 計算 KL 散度
            kl_divs = []
            for p in split_probs:
                # 避免零概率
                p = np.clip(p, 1e-8, 1.0)
                mean_probs_clipped = np.clip(mean_probs, 1e-8, 1.0)
                
                kl_div = entropy(p, mean_probs_clipped)
                kl_divs.append(kl_div)
            
            # 計算平均 KL 散度
            mean_kl = np.mean(kl_divs)
            
            # Inception Score = exp(mean_kl)
            score = np.exp(mean_kl)
            scores.append(score)
        
        # 計算平均值和標準差
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return mean_score, std_score


# 便捷函數
def calculate_inception_score(
    images: Union[str, List[str], torch.Tensor],
    device: str = "cuda",
    batch_size: int = 32,
    splits: int = 10
) -> Tuple[float, float]:
    """計算 Inception Score 的便捷函數。"""
    calculator = InceptionScoreCalculator(device=device, batch_size=batch_size, splits=splits)
    return calculator.calculate_inception_score(images)


def create_inception_calculator(
    device: str = "cuda",
    batch_size: int = 32,
    splits: int = 10
) -> InceptionScoreCalculator:
    """創建 Inception Score 計算器的便捷函數。"""
    return InceptionScoreCalculator(
        device=device,
        batch_size=batch_size,
        splits=splits
    )


def calculate_fid_and_is(
    real_images: Union[str, List[str], torch.Tensor],
    fake_images: Union[str, List[str], torch.Tensor],
    device: str = "cuda",
    batch_size: int = 32
) -> Dict[str, float]:
    """同時計算 FID 和 Inception Score 的便捷函數。"""
    calculator = InceptionScoreCalculator(device=device, batch_size=batch_size)
    return calculator.calculate_fid_and_is(real_images, fake_images)
