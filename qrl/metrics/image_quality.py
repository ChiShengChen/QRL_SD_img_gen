"""圖像質量評估指標。"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
import torchvision.transforms as transforms
from torchvision.models import inception_v3, resnet18
from scipy import linalg
import lpips
from PIL import Image


class ImageQualityMetrics:
    """圖像質量評估指標。"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self._init_models()
        
        # 初始化LPIPS
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        
        # 圖像預處理
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception V3 輸入尺寸
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.cifar_transform = transforms.Compose([
            transforms.Resize((32, 32)),  # CIFAR-10 尺寸
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_models(self):
        """初始化評估模型。"""
        # Inception V3 for FID and IS
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.eval()
        self.inception_model.to(self.device)
        
        # ResNet18 for classification
        self.classifier = resnet18(pretrained=True)
        self.classifier.eval()
        self.classifier.to(self.device)
        
        # 移除最後的分類層
        self.inception_features = torch.nn.Sequential(*list(self.inception_model.children())[:-1])
        self.inception_features.eval()
    
    def compute_all(self, generated_images: List[torch.Tensor]) -> Dict[str, float]:
        """計算所有圖像質量指標。"""
        if not generated_images:
            return {}
        
        # 轉換為張量
        images_tensor = self._prepare_images(generated_images)
        
        metrics = {}
        
        # 分類準確率
        metrics['classification_accuracy'] = self.compute_classification_accuracy(images_tensor)
        
        # FID分數
        metrics['fid_score'] = self.compute_fid_score(images_tensor)
        
        # IS分數
        metrics['inception_score'] = self.compute_inception_score(images_tensor)
        
        # LPIPS (需要參考圖像，這裡使用隨機參考)
        metrics['lpips'] = self.compute_lpips(images_tensor)
        
        return metrics
    
    def _prepare_images(self, images: List[torch.Tensor]) -> torch.Tensor:
        """準備圖像張量。"""
        processed_images = []
        
        for img in images:
            if isinstance(img, torch.Tensor):
                # 確保圖像在正確的範圍內
                if img.max() > 1.0:
                    img = img / 255.0
                
                # 確保是3通道
                if img.dim() == 2:
                    img = img.unsqueeze(0).repeat(3, 1, 1)
                elif img.dim() == 3 and img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                elif img.dim() == 3 and img.shape[2] == 3:
                    img = img.permute(2, 0, 1)
                
                # 調整尺寸
                if img.shape[1] != 32 or img.shape[2] != 32:
                    img = F.interpolate(img.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
                
                processed_images.append(img)
        
        if not processed_images:
            return torch.empty(0, 3, 32, 32)
        
        return torch.stack(processed_images).to(self.device)
    
    def compute_classification_accuracy(self, images: torch.Tensor) -> float:
        """計算分類準確率。"""
        if images.numel() == 0:
            return 0.0
        
        with torch.no_grad():
            # 使用CIFAR-10預處理
            images_cifar = self.cifar_transform(images)
            
            # 獲取分類結果
            outputs = self.classifier(images_cifar)
            _, predicted = torch.max(outputs, 1)
            
            # 計算準確率 (這裡簡化為隨機準確率，實際應該使用CIFAR-10標籤)
            # 在實際應用中，需要真實的標籤來計算準確率
            accuracy = torch.mean((predicted == 3).float()).item()  # 假設目標類別是3 (貓)
        
        return accuracy
    
    def compute_fid_score(self, images: torch.Tensor) -> float:
        """計算FID分數。"""
        if images.numel() == 0:
            return float('inf')
        
        with torch.no_grad():
            # 調整到Inception V3輸入尺寸
            images_inception = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            images_inception = self.transform(images_inception)
            
            # 獲取特徵
            features = self.inception_features(images_inception)
            features = features.squeeze()
            
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # 計算統計量
            mu = torch.mean(features, dim=0)
            sigma = torch.cov(features.T)
            
            # 簡化的FID計算 (實際應該與真實圖像分布比較)
            # 這裡使用隨機分布作為參考
            real_mu = torch.randn_like(mu)
            real_sigma = torch.eye(mu.shape[0]).to(self.device)
            
            # 計算FID
            diff = mu - real_mu
            covmean, _ = linalg.sqrtm(sigma.cpu().numpy() @ real_sigma.cpu().numpy(), disp=False)
            covmean = torch.tensor(covmean).to(self.device)
            
            if torch.is_complex(covmean):
                covmean = covmean.real
            
            fid = torch.sum(diff * diff) + torch.trace(sigma) + torch.trace(real_sigma) - 2 * torch.trace(covmean)
            
            return fid.item()
    
    def compute_inception_score(self, images: torch.Tensor) -> float:
        """計算Inception Score。"""
        if images.numel() == 0:
            return 0.0
        
        with torch.no_grad():
            # 調整到Inception V3輸入尺寸
            images_inception = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            images_inception = self.transform(images_inception)
            
            # 獲取預測概率
            outputs = self.inception_model(images_inception)
            probabilities = F.softmax(outputs, dim=1)
            
            # 計算IS
            py = torch.mean(probabilities, dim=0)
            scores = []
            
            for i in range(probabilities.shape[0]):
                pyx = probabilities[i]
                score = torch.sum(pyx * torch.log(pyx / py))
                scores.append(score.item())
            
            is_score = np.exp(np.mean(scores))
            
            return is_score
    
    def compute_lpips(self, images: torch.Tensor) -> float:
        """計算LPIPS分數。"""
        if images.numel() == 0:
            return 0.0
        
        with torch.no_grad():
            # 調整到LPIPS輸入尺寸
            images_lpips = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
            
            # 創建參考圖像 (隨機)
            ref_images = torch.randn_like(images_lpips)
            
            # 計算LPIPS
            lpips_scores = []
            for i in range(images_lpips.shape[0]):
                score = self.lpips_model(images_lpips[i:i+1], ref_images[i:i+1])
                lpips_scores.append(score.item())
            
            return np.mean(lpips_scores)
