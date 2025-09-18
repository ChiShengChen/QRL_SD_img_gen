#!/usr/bin/env python3
"""
調試腳本：檢查生成圖像和真實圖像的尺寸
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# 添加項目根目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_improved_comparison_v2 import load_cifar10_images, CIFAR10_CLASSES, GENERATED_DIR

def check_image_dimensions():
    """檢查圖像尺寸"""
    print("🔍 檢查圖像尺寸...")
    
    # 加載真實CIFAR-10圖像
    print("\n📥 加載真實CIFAR-10圖像...")
    real_images = load_cifar10_images(2)  # 只加載2張圖像進行測試
    
    for class_name in CIFAR10_CLASSES[:2]:  # 只檢查前2個類別
        if class_name in real_images and len(real_images[class_name]) > 0:
            real_img = real_images[class_name][0]
            print(f"  真實 {class_name}: {real_img.shape} (dtype: {real_img.dtype})")
    
    # 檢查生成的圖像
    print("\n🎨 檢查生成的圖像...")
    
    for model_type in ['quantum', 'classical']:
        print(f"\n  {model_type} 模型:")
        for class_name in CIFAR10_CLASSES[:2]:  # 只檢查前2個類別
            class_dir = GENERATED_DIR / model_type / class_name
            if class_dir.exists():
                pt_files = list(class_dir.glob("*.pt"))
                if pt_files:
                    try:
                        gen_img = torch.load(pt_files[0], map_location=torch.device('cpu'))
                        print(f"    生成 {class_name}: {gen_img.shape} (dtype: {gen_img.dtype})")
                        
                        # 檢查圖像範圍
                        print(f"      值範圍: [{gen_img.min():.4f}, {gen_img.max():.4f}]")
                        
                        # 檢查是否需要轉換
                        if gen_img.dim() == 4:
                            print(f"      4D tensor detected, first image shape: {gen_img[0].shape}")
                        elif gen_img.dim() == 3:
                            print(f"      3D tensor detected")
                        else:
                            print(f"      Unexpected tensor dimensions: {gen_img.dim()}")
                            
                    except Exception as e:
                        print(f"    ❌ 無法加載 {class_name}: {e}")
                else:
                    print(f"    ❌ 沒有找到 {class_name} 的生成圖像")
            else:
                print(f"    ❌ 目錄不存在: {class_dir}")

def test_metrics_calculation():
    """測試指標計算"""
    print("\n🧮 測試指標計算...")
    
    # 創建測試圖像
    real_img = torch.randn(3, 32, 32)  # CIFAR-10 尺寸
    gen_img = torch.randn(3, 32, 32)   # 相同尺寸
    
    print(f"  測試圖像尺寸: real={real_img.shape}, gen={gen_img.shape}")
    
    # 測試灰度轉換
    def to_grayscale(img):
        if len(img.shape) == 4:
            img = img[0]
        if len(img.shape) == 3:
            if img.shape[0] == 3:
                return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            else:
                return img[0]
        return img
    
    try:
        real_gray = to_grayscale(real_img.numpy())
        gen_gray = to_grayscale(gen_img.numpy())
        
        print(f"  灰度圖像尺寸: real={real_gray.shape}, gen={gen_gray.shape}")
        
        # 測試PSNR計算
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        
        real_uint8 = (real_gray * 255).astype(np.uint8)
        gen_uint8 = (gen_gray * 255).astype(np.uint8)
        
        psnr_val = psnr(real_uint8, gen_uint8, data_range=255)
        ssim_val = ssim(real_gray, gen_gray, data_range=1.0)
        
        print(f"  PSNR: {psnr_val:.2f}")
        print(f"  SSIM: {ssim_val:.4f}")
        
    except Exception as e:
        print(f"  ❌ 指標計算失敗: {e}")

if __name__ == "__main__":
    check_image_dimensions()
    test_metrics_calculation()
