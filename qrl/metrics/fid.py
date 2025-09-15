"""FID (Fréchet Inception Distance) 指標計算模組。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple
from pathlib import Path
import os
from tqdm import tqdm
import warnings

try:
    from pytorch_fid import fid_score
    PYTORCH_FID_AVAILABLE = True
except ImportError:
    PYTORCH_FID_AVAILABLE = False
    warnings.warn("pytorch-fid 未安裝，將使用備用 FID 計算方法")


class InceptionV3Features(nn.Module):
    """InceptionV3 特徵提取器。"""
    
    def __init__(self, device: str = "cuda"):
        """
        初始化 InceptionV3 特徵提取器。
        
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
        
        # 移除最後的分類層
        self.inception.fc = nn.Identity()
        self.inception.avgpool = nn.Identity()
        
        self.inception.to(device)
        self.inception.eval()
        
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特徵。
        
        Args:
            x: 輸入圖像 [B, 3, H, W]
            
        Returns:
            torch.Tensor: 特徵向量 [B, 2048]
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
            features = self.inception(x)
        
        return features


class FIDCalculator:
    """FID 計算器。"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
        use_pytorch_fid: bool = True
    ):
        """
        初始化 FID 計算器。
        
        Args:
            device: 設備
            batch_size: 批次大小
            use_pytorch_fid: 是否使用 pytorch-fid
        """
        self.device = device
        self.batch_size = batch_size
        self.use_pytorch_fid = use_pytorch_fid and PYTORCH_FID_AVAILABLE
        
        if not self.use_pytorch_fid:
            self.feature_extractor = InceptionV3Features(device)
            warnings.warn("使用備用 FID 計算方法，結果可能與標準 FID 略有差異")
    
    def calculate_fid(
        self,
        real_images: Union[str, List[str], torch.Tensor],
        fake_images: Union[str, List[str], torch.Tensor]
    ) -> float:
        """
        計算 FID 分數。
        
        Args:
            real_images: 真實圖像路徑列表或張量
            fake_images: 生成圖像路徑列表或張量
            
        Returns:
            float: FID 分數
        """
        if self.use_pytorch_fid:
            return self._calculate_fid_pytorch_fid(real_images, fake_images)
        else:
            return self._calculate_fid_custom(real_images, fake_images)
    
    def _calculate_fid_pytorch_fid(
        self,
        real_images: Union[str, List[str], torch.Tensor],
        fake_images: Union[str, List[str], torch.Tensor]
    ) -> float:
        """使用 pytorch-fid 計算 FID。"""
        if isinstance(real_images, str):
            real_path = real_images
        elif isinstance(real_images, list):
            real_path = real_images[0] if real_images else ""
        else:
            # 如果是張量，保存到臨時目錄
            real_path = self._save_tensor_to_dir(real_images, "real_temp")
        
        if isinstance(fake_images, str):
            fake_path = fake_images
        elif isinstance(fake_images, list):
            fake_path = fake_images[0] if fake_images else ""
        else:
            # 如果是張量，保存到臨時目錄
            fake_path = self._save_tensor_to_dir(fake_images, "fake_temp")
        
        try:
            fid_score_value = fid_score.calculate_fid_given_paths(
                [real_path, fake_path],
                batch_size=self.batch_size,
                device=self.device
            )
            return fid_score_value
        except Exception as e:
            warnings.warn(f"pytorch-fid 計算失敗: {e}，使用備用方法")
            return self._calculate_fid_custom(real_images, fake_images)
    
    def _calculate_fid_custom(
        self,
        real_images: Union[str, List[str], torch.Tensor],
        fake_images: Union[str, List[str], torch.Tensor]
    ) -> float:
        """使用自定義方法計算 FID。"""
        # 提取特徵
        real_features = self._extract_features(real_images)
        fake_features = self._extract_features(fake_images)
        
        # 計算統計量
        real_mean = real_features.mean(dim=0)
        real_cov = self._compute_covariance(real_features)
        
        fake_mean = fake_features.mean(dim=0)
        fake_cov = self._compute_covariance(fake_features)
        
        # 計算 FID
        fid_score = self._compute_fid_score(
            real_mean, real_cov, fake_mean, fake_cov
        )
        
        return fid_score
    
    def _extract_features(
        self,
        images: Union[str, List[str], torch.Tensor]
    ) -> torch.Tensor:
        """提取圖像特徵。"""
        if isinstance(images, str):
            # 單個目錄路徑
            image_paths = list(Path(images).glob("*.jpg")) + list(Path(images).glob("*.png"))
            return self._extract_features_from_paths(image_paths)
        elif isinstance(images, list):
            # 圖像路徑列表
            return self._extract_features_from_paths(images)
        elif isinstance(images, torch.Tensor):
            # 圖像張量
            return self._extract_features_from_tensor(images)
        else:
            raise ValueError(f"不支援的圖像類型: {type(images)}")
    
    def _extract_features_from_paths(self, image_paths: List[str]) -> torch.Tensor:
        """從圖像路徑提取特徵。"""
        features = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="提取特徵"):
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
                batch_features = self.feature_extractor(batch_tensor)
                features.append(batch_features.cpu())
        
        if not features:
            raise ValueError("沒有成功載入任何圖像")
        
        return torch.cat(features, dim=0)
    
    def _extract_features_from_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """從圖像張量提取特徵。"""
        features = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size].to(self.device)
            batch_features = self.feature_extractor(batch)
            features.append(batch_features.cpu())
        
        return torch.cat(features, dim=0)
    
    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """計算協方差矩陣。"""
        mean = features.mean(dim=0)
        centered = features - mean.unsqueeze(0)
        cov = torch.matmul(centered.T, centered) / (len(features) - 1)
        return cov
    
    def _compute_fid_score(
        self,
        real_mean: torch.Tensor,
        real_cov: torch.Tensor,
        fake_mean: torch.Tensor,
        fake_cov: torch.Tensor
    ) -> float:
        """計算 FID 分數。"""
        # 計算均值差異的平方
        mean_diff = real_mean - fake_mean
        mean_diff_sq = torch.dot(mean_diff, mean_diff)
        
        # 計算協方差矩陣的平方根
        cov_sum = real_cov + fake_cov
        
        # 使用 SVD 計算矩陣平方根
        try:
            U, S, V = torch.svd(cov_sum)
            sqrt_cov_sum = torch.matmul(U, torch.matmul(torch.diag(torch.sqrt(S)), V.T))
            
            # 計算 trace
            trace_term = torch.trace(real_cov + fake_cov - 2 * sqrt_cov_sum)
            
            fid_score = mean_diff_sq + trace_term
            return fid_score.item()
            
        except Exception as e:
            warnings.warn(f"協方差矩陣計算失敗: {e}，使用簡化方法")
            # 簡化方法：只計算均值差異
            return mean_diff_sq.item()
    
    def _save_tensor_to_dir(self, images: torch.Tensor, dir_name: str) -> str:
        """將張量保存到目錄。"""
        import tempfile
        from PIL import Image
        import torchvision.transforms as transforms
        
        # 創建臨時目錄
        temp_dir = Path(tempfile.mkdtemp()) / dir_name
        temp_dir.mkdir(exist_ok=True)
        
        # 轉換為 PIL 圖像並保存
        to_pil = transforms.ToPILImage()
        
        for i in range(len(images)):
            image = images[i]
            if image.max() > 1.0:
                image = image / 255.0
            
            pil_image = to_pil(image)
            pil_image.save(temp_dir / f"image_{i:04d}.png")
        
        return str(temp_dir)


# 便捷函數
def calculate_fid(
    real_images: Union[str, List[str], torch.Tensor],
    fake_images: Union[str, List[str], torch.Tensor],
    device: str = "cuda",
    batch_size: int = 32
) -> float:
    """計算 FID 分數的便捷函數。"""
    calculator = FIDCalculator(device=device, batch_size=batch_size)
    return calculator.calculate_fid(real_images, fake_images)


def create_fid_calculator(
    device: str = "cuda",
    batch_size: int = 32,
    use_pytorch_fid: bool = True
) -> FIDCalculator:
    """創建 FID 計算器的便捷函數。"""
    return FIDCalculator(
        device=device,
        batch_size=batch_size,
        use_pytorch_fid=use_pytorch_fid
    )
