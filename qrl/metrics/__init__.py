"""指標計算模組。"""

from .fid import (
    InceptionV3Features,
    FIDCalculator,
    calculate_fid,
    create_fid_calculator
)

from .inception import (
    InceptionV3Classifier,
    InceptionScoreCalculator,
    SimplifiedInceptionScore,
    calculate_inception_score,
    create_inception_calculator,
    calculate_fid_and_is
)

from .clipscore import (
    CLIPScoreCalculator,
    SimplifiedCLIPScore,
    calculate_clip_score,
    create_clip_calculator,
    calculate_class_conditional_clip_score
)

# 主要指標計算器
__all__ = [
    # FID 相關
    "InceptionV3Features",
    "FIDCalculator",
    "calculate_fid",
    "create_fid_calculator",
    
    # Inception Score 相關
    "InceptionV3Classifier",
    "InceptionScoreCalculator",
    "SimplifiedInceptionScore",
    "calculate_inception_score",
    "create_inception_calculator",
    "calculate_fid_and_is",
    
    # CLIP Score 相關
    "CLIPScoreCalculator",
    "SimplifiedCLIPScore",
    "calculate_clip_score",
    "create_clip_calculator",
    "calculate_class_conditional_clip_score",
]

# 便捷函數：同時計算多個指標
def calculate_all_metrics(
    real_images,
    fake_images,
    target_class: int = None,
    dataset: str = "cifar10",
    device: str = "cuda",
    batch_size: int = 32
):
    """
    同時計算所有指標。
    
    Args:
        real_images: 真實圖像
        fake_images: 生成圖像
        target_class: 目標類別（用於 CLIP Score）
        dataset: 數據集名稱
        device: 設備
        batch_size: 批次大小
        
    Returns:
        Dict: 包含所有指標的字典
    """
    results = {}
    
    # 計算 FID
    try:
        fid_calculator = FIDCalculator(device=device, batch_size=batch_size)
        results['fid'] = fid_calculator.calculate_fid(real_images, fake_images)
    except Exception as e:
        print(f"FID 計算失敗: {e}")
        results['fid'] = float('nan')
    
    # 計算 Inception Score
    try:
        is_calculator = InceptionScoreCalculator(device=device, batch_size=batch_size)
        is_mean, is_std = is_calculator.calculate_inception_score(fake_images)
        results['inception_score_mean'] = is_mean
        results['inception_score_std'] = is_std
    except Exception as e:
        print(f"Inception Score 計算失敗: {e}")
        results['inception_score_mean'] = float('nan')
        results['inception_score_std'] = float('nan')
    
    # 計算 CLIP Score（如果指定了目標類別）
    if target_class is not None:
        try:
            clip_calculator = CLIPScoreCalculator(device=device, batch_size=batch_size)
            clip_scores = clip_calculator.calculate_class_conditional_clip_score(
                fake_images, target_class, dataset
            )
            results.update(clip_scores)
        except Exception as e:
            print(f"CLIP Score 計算失敗: {e}")
            results['clip_score_mean'] = float('nan')
            results['clip_score_std'] = float('nan')
    
    return results


def create_metrics_calculator(
    device: str = "cuda",
    batch_size: int = 32
):
    """
    創建統一的指標計算器。
    
    Args:
        device: 設備
        batch_size: 批次大小
        
    Returns:
        Dict: 包含所有計算器的字典
    """
    return {
        'fid': FIDCalculator(device=device, batch_size=batch_size),
        'inception': InceptionScoreCalculator(device=device, batch_size=batch_size),
        'clip': CLIPScoreCalculator(device=device, batch_size=batch_size)
    }


# 版本信息
__version__ = "0.1.0"
