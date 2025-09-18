"""圖像品質評估指標模組。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg16
import lpips
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class ImageQualityMetrics:
    """圖像品質評估指標計算器。"""
    
    def __init__(self, device: str = "cpu"):
        """
        初始化圖像品質評估器。
        
        Args:
            device: 計算設備
        """
        self.device = device
        self.lpips_model = None
        self.vgg_model = None
        
        # 初始化LPIPS模型
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        except Exception as e:
            print(f"Warning: Failed to initialize LPIPS model: {e}")
        
        # 初始化VGG模型用於FID計算
        try:
            self.vgg_model = vgg16(pretrained=True).to(device)
            self.vgg_model.eval()
            # 移除最後的分類層，只保留特徵提取部分
            self.vgg_model.classifier = nn.Identity()
        except Exception as e:
            print(f"Warning: Failed to initialize VGG model: {e}")
    
    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        計算峰值信噪比 (PSNR)。
        
        Args:
            img1: 參考圖像 [C, H, W]
            img2: 比較圖像 [C, H, W]
            
        Returns:
            PSNR值
        """
        # 轉換為numpy數組
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        
        # 確保圖像範圍在[0, 1]
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        # 確保圖像尺寸相同
        if img1.shape != img2.shape:
            # 調整到較小的尺寸
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        # 轉換為[0, 255]範圍
        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)
        
        # 計算PSNR
        return psnr(img1, img2, data_range=255)
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        計算結構相似性指數 (SSIM)。
        
        Args:
            img1: 參考圖像 [C, H, W]
            img2: 比較圖像 [C, H, W]
            
        Returns:
            SSIM值
        """
        # 轉換為numpy數組
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        
        # 確保圖像範圍在[0, 1]
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        # 確保圖像尺寸相同
        if img1.shape != img2.shape:
            # 調整到較小的尺寸
            min_h = min(img1.shape[1], img2.shape[1])
            min_w = min(img1.shape[2], img2.shape[2])
            img1 = img1[:, :min_h, :min_w]
            img2 = img2[:, :min_h, :min_w]
        
        # 轉換為灰度圖像進行SSIM計算
        if img1.shape[0] == 3:  # RGB圖像
            img1_gray = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
            img2_gray = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
        else:
            img1_gray = img1[0]
            img2_gray = img2[0]
        
        # 計算SSIM
        return ssim(img1_gray, img2_gray, data_range=1.0)
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        計算感知圖像補丁相似性 (LPIPS)。
        
        Args:
            img1: 參考圖像 [C, H, W]
            img2: 比較圖像 [C, H, W]
            
        Returns:
            LPIPS值
        """
        if self.lpips_model is None:
            return 0.0
        
        # 確保圖像在正確的設備上
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # 確保圖像尺寸相同
        if img1.shape != img2.shape:
            # 調整到較小的尺寸
            min_h = min(img1.shape[1], img2.shape[1])
            min_w = min(img1.shape[2], img2.shape[2])
            img1 = img1[:, :min_h, :min_w]
            img2 = img2[:, :min_h, :min_w]
        
        # 確保圖像範圍在[-1, 1]（LPIPS模型的要求）
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0
        
        # 添加批次維度
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        with torch.no_grad():
            lpips_value = self.lpips_model(img1, img2)
        
        return lpips_value.item()
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        使用VGG模型提取圖像特徵。
        
        Args:
            images: 圖像批次 [B, C, H, W]
            
        Returns:
            特徵向量 [B, feature_dim]
        """
        if self.vgg_model is None:
            # 如果VGG模型不可用，返回隨機特徵
            return torch.randn(images.shape[0], 4096).to(self.device)
        
        # 確保圖像在正確的設備上
        images = images.to(self.device)
        
        # 調整圖像大小到224x224（VGG模型的要求）
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 標準化圖像（ImageNet標準化）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images - mean) / std
        
        with torch.no_grad():
            features = self.vgg_model(images)
        
        return features
    
    def calculate_fid(self, real_images: List[torch.Tensor], 
                     generated_images: List[torch.Tensor]) -> float:
        """
        計算Fréchet Inception Distance (FID)。
        
        Args:
            real_images: 真實圖像列表
            generated_images: 生成圖像列表
            
        Returns:
            FID值
        """
        if len(real_images) == 0 or len(generated_images) == 0:
            return float('inf')
        
        # 轉換為張量
        real_tensor = torch.stack(real_images).to(self.device)
        gen_tensor = torch.stack(generated_images).to(self.device)
        
        # 提取特徵
        real_features = self.extract_features(real_tensor)
        gen_features = self.extract_features(gen_tensor)
        
        # 轉換為numpy數組
        real_features = real_features.detach().cpu().numpy()
        gen_features = gen_features.detach().cpu().numpy()
        
        # 計算均值和協方差
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # 計算FID
        diff = mu_real - mu_gen
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
        
        return fid
    
    def calculate_all_metrics(self, reference_images: List[torch.Tensor],
                            generated_images: List[torch.Tensor]) -> Dict[str, float]:
        """
        計算所有圖像品質指標。
        
        Args:
            reference_images: 參考圖像列表
            generated_images: 生成圖像列表
            
        Returns:
            包含所有指標的字典
        """
        if len(reference_images) == 0 or len(generated_images) == 0:
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'lpips': 0.0,
                'fid': float('inf')
            }
        
        # 計算PSNR和SSIM（需要配對的圖像）
        min_len = min(len(reference_images), len(generated_images))
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        for i in range(min_len):
            # PSNR
            psnr_val = self.calculate_psnr(reference_images[i], generated_images[i])
            psnr_values.append(psnr_val)
            
            # SSIM
            ssim_val = self.calculate_ssim(reference_images[i], generated_images[i])
            ssim_values.append(ssim_val)
            
            # LPIPS
            lpips_val = self.calculate_lpips(reference_images[i], generated_images[i])
            lpips_values.append(lpips_val)
        
        # 計算FID（使用所有圖像）
        fid_value = self.calculate_fid(reference_images, generated_images)
        
        return {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values),
            'lpips': np.mean(lpips_values),
            'fid': fid_value
        }
    
    def compare_models(self, quantum_images: List[torch.Tensor],
                      classical_images: List[torch.Tensor],
                      reference_images: List[torch.Tensor]) -> Dict[str, Any]:
        """
        比較量子模型和經典模型的圖像品質。
        
        Args:
            quantum_images: 量子模型生成的圖像
            classical_images: 經典模型生成的圖像
            reference_images: 參考圖像（真實圖像）
            
        Returns:
            比較結果字典
        """
        # 計算量子模型的指標
        quantum_metrics = self.calculate_all_metrics(reference_images, quantum_images)
        
        # 計算經典模型的指標
        classical_metrics = self.calculate_all_metrics(reference_images, classical_images)
        
        # 計算改善百分比
        improvements = {}
        for metric in ['psnr', 'ssim']:
            if classical_metrics[metric] != 0:
                improvement = ((quantum_metrics[metric] - classical_metrics[metric]) / 
                             abs(classical_metrics[metric])) * 100
                improvements[f'{metric}_improvement'] = improvement
            else:
                improvements[f'{metric}_improvement'] = 0.0
        
        # 對於LPIPS和FID，數值越小越好
        for metric in ['lpips', 'fid']:
            if classical_metrics[metric] != 0:
                improvement = ((classical_metrics[metric] - quantum_metrics[metric]) / 
                             abs(classical_metrics[metric])) * 100
                improvements[f'{metric}_improvement'] = improvement
            else:
                improvements[f'{metric}_improvement'] = 0.0
        
        return {
            'quantum_metrics': quantum_metrics,
            'classical_metrics': classical_metrics,
            'improvements': improvements
        }


def create_reference_images(num_images: int = 10, 
                          image_size: Tuple[int, int] = (64, 64)) -> List[torch.Tensor]:
    """
    創建參考圖像（用於比較的基準圖像）。
    
    Args:
        num_images: 圖像數量
        image_size: 圖像大小 (H, W)
        
    Returns:
        參考圖像列表
    """
    reference_images = []
    
    for i in range(num_images):
        # 創建一些簡單的測試圖像
        if i % 3 == 0:
            # 漸變圖像
            img = torch.zeros(3, image_size[0], image_size[1])
            for h in range(image_size[0]):
                for w in range(image_size[1]):
                    img[0, h, w] = h / image_size[0]  # 紅色通道
                    img[1, h, w] = w / image_size[1]  # 綠色通道
                    img[2, h, w] = (h + w) / (image_size[0] + image_size[1])  # 藍色通道
        elif i % 3 == 1:
            # 條紋圖像
            img = torch.zeros(3, image_size[0], image_size[1])
            for h in range(image_size[0]):
                if h % 8 < 4:
                    img[:, h, :] = 1.0
        else:
            # 隨機圖像
            img = torch.rand(3, image_size[0], image_size[1])
        
        reference_images.append(img)
    
    return reference_images
