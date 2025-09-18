#!/usr/bin/env python3
"""
快速圖像品質評估測試腳本 - 跳過耗時操作
"""

import sys
import os
sys.path.append('qrl_image_synthesis')

import torch
import numpy as np
import time
from qrl.evaluation import ImageQualityMetrics, create_reference_images

def test_image_quality_metrics_fast():
    """快速測試圖像品質評估功能 - 跳過FID和LPIPS"""
    print("🧪 開始快速圖像品質評估測試...")
    
    try:
        # 初始化評估器（跳過耗時模型）
        print("⚡ 初始化評估器（跳過VGG和LPIPS模型）...")
        quality_evaluator = ImageQualityMetrics(device="cpu")
        
        # 手動設置模型為None以跳過耗時操作
        quality_evaluator.vgg_model = None
        quality_evaluator.lpips_model = None
        print("✅ 評估器初始化完成（跳過耗時模型）")
        
        # 創建測試圖像
        print("📸 創建測試圖像...")
        
        # 創建較小的測試圖像以加快處理
        quantum_images = [
            torch.rand(3, 32, 32),  # 使用32x32而不是64x64
            torch.rand(3, 32, 32),
            torch.rand(3, 32, 32),
        ]
        
        classical_images = [
            torch.rand(3, 32, 32),
            torch.rand(3, 32, 32), 
            torch.rand(3, 32, 32),
        ]
        
        # 創建參考圖像（使用相同尺寸）
        reference_images = create_reference_images(num_images=3, image_size=(32, 32))
        
        print(f"✅ 創建了 {len(quantum_images)} 個量子圖像 (32x32)")
        print(f"✅ 創建了 {len(classical_images)} 個經典圖像 (32x32)")
        print(f"✅ 創建了 {len(reference_images)} 個參考圖像 (32x32)")
        
        # 測試單個指標計算（只測試PSNR和SSIM）
        print("\n🔍 測試快速指標計算...")
        
        # 測試PSNR
        start_time = time.time()
        try:
            psnr_val = quality_evaluator.calculate_psnr(reference_images[0], quantum_images[0])
            psnr_time = time.time() - start_time
            print(f"✅ PSNR 計算成功: {psnr_val:.4f} (耗時: {psnr_time:.2f}秒)")
        except Exception as e:
            print(f"❌ PSNR 計算失敗: {e}")
        
        # 測試SSIM
        start_time = time.time()
        try:
            ssim_val = quality_evaluator.calculate_ssim(reference_images[0], quantum_images[0])
            ssim_time = time.time() - start_time
            print(f"✅ SSIM 計算成功: {ssim_val:.4f} (耗時: {ssim_time:.2f}秒)")
        except Exception as e:
            print(f"❌ SSIM 計算失敗: {e}")
        
        # 跳過LPIPS測試（因為模型未初始化）
        print("⏭️  跳過LPIPS測試（模型未初始化）")
        
        # 測試快速比較（只計算PSNR和SSIM）
        print("\n🔄 測試快速模型比較...")
        start_time = time.time()
        
        try:
            # 手動計算快速比較結果
            quantum_psnr_values = []
            quantum_ssim_values = []
            classical_psnr_values = []
            classical_ssim_values = []
            
            min_len = min(len(reference_images), len(quantum_images), len(classical_images))
            
            for i in range(min_len):
                # 量子模型指標
                q_psnr = quality_evaluator.calculate_psnr(reference_images[i], quantum_images[i])
                q_ssim = quality_evaluator.calculate_ssim(reference_images[i], quantum_images[i])
                quantum_psnr_values.append(q_psnr)
                quantum_ssim_values.append(q_ssim)
                
                # 經典模型指標
                c_psnr = quality_evaluator.calculate_psnr(reference_images[i], classical_images[i])
                c_ssim = quality_evaluator.calculate_ssim(reference_images[i], classical_images[i])
                classical_psnr_values.append(c_psnr)
                classical_ssim_values.append(c_ssim)
            
            # 計算平均值
            quantum_metrics = {
                'psnr': np.mean(quantum_psnr_values),
                'ssim': np.mean(quantum_ssim_values),
                'lpips': 0.0,  # 跳過
                'fid': 0.0     # 跳過
            }
            
            classical_metrics = {
                'psnr': np.mean(classical_psnr_values),
                'ssim': np.mean(classical_ssim_values),
                'lpips': 0.0,  # 跳過
                'fid': 0.0     # 跳過
            }
            
            # 計算改善百分比
            improvements = {}
            for metric in ['psnr', 'ssim']:
                if classical_metrics[metric] != 0:
                    improvement = ((quantum_metrics[metric] - classical_metrics[metric]) / 
                                 abs(classical_metrics[metric])) * 100
                    improvements[metric] = improvement
                else:
                    improvements[metric] = 0.0
            
            comparison_time = time.time() - start_time
            
            print("✅ 快速模型比較成功!")
            print(f"⏱️  總耗時: {comparison_time:.2f}秒")
            print("\n📊 比較結果:")
            
            print(f"量子模型 PSNR: {quantum_metrics['psnr']:.4f}")
            print(f"經典模型 PSNR: {classical_metrics['psnr']:.4f}")
            print(f"PSNR 改善: {improvements['psnr']:+.1f}%")
            
            print(f"量子模型 SSIM: {quantum_metrics['ssim']:.4f}")
            print(f"經典模型 SSIM: {classical_metrics['ssim']:.4f}")
            print(f"SSIM 改善: {improvements['ssim']:+.1f}%")
            
            print("⏭️  LPIPS 和 FID 已跳過（需要預訓練模型）")
            
        except Exception as e:
            print(f"❌ 快速模型比較失敗: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n🎉 快速圖像品質評估測試完成!")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

def test_with_timeout():
    """帶超時的測試"""
    print("⏰ 開始帶超時的測試...")
    
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("測試超時")
    
    # 設置30秒超時
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        test_image_quality_metrics_fast()
        signal.alarm(0)  # 取消超時
    except TimeoutError:
        print("⏰ 測試超時（30秒）")
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        signal.alarm(0)  # 取消超時

if __name__ == "__main__":
    print("🚀 開始快速圖像品質評估測試")
    print("=" * 60)
    print("⚠️  注意：此版本跳過了耗時的FID和LPIPS計算")
    print("=" * 60)
    
    # 快速測試
    test_image_quality_metrics_fast()
    
    print("\n" + "=" * 60)
    print("🏁 快速測試完成!")
    print("💡 提示：如果需要完整測試，請確保網絡連接正常以下載預訓練模型")
