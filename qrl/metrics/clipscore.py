"""CLIP Score 指標計算模組。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from pathlib import Path
import os
from tqdm import tqdm
import warnings


class CLIPScoreCalculator:
    """CLIP Score 計算器。"""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        初始化 CLIP Score 計算器。
        
        Args:
            model_name: CLIP 模型名稱
            device: 設備
            batch_size: 批次大小
        """
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        
        # 初始化 CLIP 模型
        self._init_clip_model()
    
    def _init_clip_model(self):
        """初始化 CLIP 模型。"""
        try:
            import open_clip
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
                self.model_name, pretrained='openai'
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.use_open_clip = True
        except ImportError:
            try:
                import clip
                self.model, self.preprocess = clip.load(self.model_name, device=self.device)
                self.use_open_clip = False
            except ImportError:
                warnings.warn("open_clip 和 clip 都未安裝，將使用備用方法")
                self.model = None
                self.preprocess = None
                self.use_open_clip = False
        
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
    
    def calculate_clip_score(
        self,
        images: Union[str, List[str], torch.Tensor],
        text_prompts: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[float, List[float]]:
        """
        計算 CLIP Score。
        
        Args:
            images: 圖像路徑列表或張量
            text_prompts: 文本提示
            normalize: 是否標準化分數
            
        Returns:
            Union[float, List[float]]: CLIP Score
        """
        if self.model is None:
            warnings.warn("CLIP 模型未載入，返回隨機分數")
            if isinstance(images, (list, str)):
                return np.random.random(len(images) if isinstance(images, list) else 1)
            else:
                return np.random.random(len(images))
        
        # 處理文本提示
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts] * self._get_image_count(images)
        elif len(text_prompts) != self._get_image_count(images):
            raise ValueError("文本提示數量與圖像數量不匹配")
        
        # 提取圖像特徵
        image_features = self._extract_image_features(images)
        
        # 提取文本特徵
        text_features = self._extract_text_features(text_prompts)
        
        # 計算相似度
        scores = self._compute_similarity(image_features, text_features)
        
        # 標準化（可選）
        if normalize:
            scores = self._normalize_scores(scores)
        
        return scores
    
    def _get_image_count(self, images: Union[str, List[str], torch.Tensor]) -> int:
        """獲取圖像數量。"""
        if isinstance(images, str):
            # 目錄路徑
            image_paths = list(Path(images).glob("*.jpg")) + list(Path(images).glob("*.png"))
            return len(image_paths)
        elif isinstance(images, list):
            return len(images)
        elif isinstance(images, torch.Tensor):
            return len(images)
        else:
            raise ValueError(f"不支援的圖像類型: {type(images)}")
    
    def _extract_image_features(
        self,
        images: Union[str, List[str], torch.Tensor]
    ) -> torch.Tensor:
        """提取圖像特徵。"""
        if isinstance(images, str):
            # 目錄路徑
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
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="提取圖像特徵"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    from PIL import Image
                    
                    # 載入圖像
                    image = Image.open(path).convert('RGB')
                    
                    # 預處理
                    if self.use_open_clip:
                        image_tensor = self.preprocess(image).unsqueeze(0)
                    else:
                        image_tensor = self.preprocess(image).unsqueeze(0)
                    
                    batch_images.append(image_tensor)
                    
                except Exception as e:
                    warnings.warn(f"無法載入圖像 {path}: {e}")
                    continue
            
            if batch_images:
                batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                batch_features = self._encode_images(batch_tensor)
                all_features.append(batch_features.cpu())
        
        if not all_features:
            raise ValueError("沒有成功載入任何圖像")
        
        return torch.cat(all_features, dim=0)
    
    def _extract_features_from_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """從圖像張量提取特徵。"""
        all_features = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size].to(self.device)
            
            # 預處理張量
            if batch.max() > 1.0:
                batch = batch / 255.0
            
            # 調整大小（如果需要）
            if batch.shape[-2:] != (224, 224):
                batch = F.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
            
            batch_features = self._encode_images(batch)
            all_features.append(batch_features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """編碼圖像。"""
        with torch.no_grad():
            if self.use_open_clip:
                features = self.model.encode_image(images)
            else:
                features = self.model.encode_image(images)
        
        # 標準化特徵
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def _extract_text_features(self, text_prompts: List[str]) -> torch.Tensor:
        """提取文本特徵。"""
        all_features = []
        
        for i in range(0, len(text_prompts), self.batch_size):
            batch_texts = text_prompts[i:i + self.batch_size]
            
            # 標記化
            if self.use_open_clip:
                tokens = self.tokenizer(batch_texts).to(self.device)
            else:
                tokens = clip.tokenize(batch_texts).to(self.device)
            
            # 編碼文本
            with torch.no_grad():
                if self.use_open_clip:
                    features = self.model.encode_text(tokens)
                else:
                    features = self.model.encode_text(tokens)
            
            # 標準化特徵
            features = F.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def _compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> np.ndarray:
        """計算相似度分數。"""
        # 計算餘弦相似度
        similarity = torch.matmul(image_features, text_features.T)
        
        # 轉換為 numpy 陣列
        scores = similarity.diagonal().cpu().numpy()
        
        return scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """標準化分數。"""
        # 將分數從 [-1, 1] 轉換到 [0, 1]
        normalized_scores = (scores + 1) / 2
        return normalized_scores
    
    def calculate_class_conditional_clip_score(
        self,
        images: Union[str, List[str], torch.Tensor],
        target_class: int,
        dataset: str = "cifar10"
    ) -> Dict[str, float]:
        """
        計算類條件 CLIP Score。
        
        Args:
            images: 圖像
            target_class: 目標類別
            dataset: 數據集名稱
            
        Returns:
            Dict: 包含各種類別提示的 CLIP Score
        """
        # 根據數據集和類別生成提示
        prompts = self._generate_class_prompts(target_class, dataset)
        
        # 計算每個提示的 CLIP Score
        scores = {}
        for prompt_name, prompt in prompts.items():
            score = self.calculate_clip_score(images, prompt)
            scores[f"clip_score_{prompt_name}"] = score
        
        # 計算平均分數
        all_scores = np.concatenate(list(scores.values()))
        scores["clip_score_mean"] = np.mean(all_scores)
        scores["clip_score_std"] = np.std(all_scores)
        
        return scores
    
    def _generate_class_prompts(self, target_class: int, dataset: str) -> Dict[str, str]:
        """生成類別提示。"""
        if dataset.lower() == "cifar10":
            class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
            
            if 0 <= target_class < len(class_names):
                class_name = class_names[target_class]
            else:
                class_name = "object"
            
            prompts = {
                "simple": f"a photo of a {class_name}",
                "detailed": f"a high-quality photograph of a {class_name}",
                "artistic": f"an artistic image of a {class_name}",
                "realistic": f"a realistic photo of a {class_name}"
            }
        else:
            # 通用提示
            prompts = {
                "simple": "a photo of an object",
                "detailed": "a high-quality photograph of an object",
                "artistic": "an artistic image of an object",
                "realistic": "a realistic photo of an object"
            }
        
        return prompts


class SimplifiedCLIPScore:
    """簡化的 CLIP Score 計算器（使用預計算特徵）。"""
    
    def __init__(self, device: str = "cuda"):
        """
        初始化簡化 CLIP Score 計算器。
        
        Args:
            device: 設備
        """
        self.device = device
    
    def calculate_simplified_clip_score(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> np.ndarray:
        """
        使用預計算特徵計算簡化的 CLIP Score。
        
        Args:
            image_features: 預計算的圖像特徵 [N, D]
            text_features: 預計算的文本特徵 [N, D]
            
        Returns:
            np.ndarray: CLIP Score
        """
        # 標準化特徵
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 計算相似度
        similarity = torch.matmul(image_features, text_features.T)
        scores = similarity.diagonal().cpu().numpy()
        
        # 標準化到 [0, 1]
        normalized_scores = (scores + 1) / 2
        
        return normalized_scores


# 便捷函數
def calculate_clip_score(
    images: Union[str, List[str], torch.Tensor],
    text_prompts: Union[str, List[str]],
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    **kwargs
) -> Union[float, List[float]]:
    """計算 CLIP Score 的便捷函數。"""
    calculator = CLIPScoreCalculator(model_name=model_name, device=device, **kwargs)
    return calculator.calculate_clip_score(images, text_prompts)


def create_clip_calculator(
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    batch_size: int = 32
) -> CLIPScoreCalculator:
    """創建 CLIP Score 計算器的便捷函數。"""
    return CLIPScoreCalculator(
        model_name=model_name,
        device=device,
        batch_size=batch_size
    )


def calculate_class_conditional_clip_score(
    images: Union[str, List[str], torch.Tensor],
    target_class: int,
    dataset: str = "cifar10",
    **kwargs
) -> Dict[str, float]:
    """計算類條件 CLIP Score 的便捷函數。"""
    calculator = CLIPScoreCalculator(**kwargs)
    return calculator.calculate_class_conditional_clip_score(images, target_class, dataset)
