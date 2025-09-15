#!/usr/bin/env python3
"""評估圖像生成指標腳本。"""

import os
import sys
import argparse
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qrl.utils.seed import set_seed, set_deterministic
from qrl.utils.logging import QRLLogger
from qrl.metrics import (
    calculate_all_metrics,
    create_metrics_calculator,
    FIDCalculator,
    InceptionScoreCalculator,
    CLIPScoreCalculator
)


def evaluate_metrics(
    real_images_path: str,
    fake_images_path: str,
    target_class: int = None,
    dataset: str = "cifar10",
    device: str = "cuda",
    batch_size: int = 32,
    output_dir: str = "eval_results"
):
    """
    評估圖像生成指標。
    
    Args:
        real_images_path: 真實圖像路徑
        fake_images_path: 生成圖像路徑
        target_class: 目標類別
        dataset: 數據集名稱
        device: 設備
        batch_size: 批次大小
        output_dir: 輸出目錄
    """
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化日誌記錄器
    logger = QRLLogger(
        name="eval_metrics",
        log_dir=output_path / "logs",
        use_tensorboard=False
    )
    
    logger.info("開始評估圖像生成指標...")
    logger.info(f"真實圖像路徑: {real_images_path}")
    logger.info(f"生成圖像路徑: {fake_images_path}")
    logger.info(f"目標類別: {target_class}")
    logger.info(f"數據集: {dataset}")
    logger.info(f"設備: {device}")
    logger.info(f"輸出目錄: {output_path}")
    
    try:
        # 檢查路徑
        real_path = Path(real_images_path)
        fake_path = Path(fake_images_path)
        
        if not real_path.exists():
            raise ValueError(f"真實圖像路徑不存在: {real_path}")
        if not fake_path.exists():
            raise ValueError(f"生成圖像路徑不存在: {fake_path}")
        
        # 創建指標計算器
        logger.info("創建指標計算器...")
        calculators = create_metrics_calculator(device=device, batch_size=batch_size)
        
        # 計算所有指標
        logger.info("計算所有指標...")
        all_metrics = calculate_all_metrics(
            real_images=real_images_path,
            fake_images=fake_images_path,
            target_class=target_class,
            dataset=dataset,
            device=device,
            batch_size=batch_size
        )
        
        # 詳細指標計算
        detailed_metrics = {}
        
        # FID 計算
        logger.info("計算 FID...")
        try:
            fid_calculator = calculators['fid']
            fid_score = fid_calculator.calculate_fid(real_images_path, fake_images_path)
            detailed_metrics['fid'] = fid_score
            logger.info(f"FID Score: {fid_score:.4f}")
        except Exception as e:
            logger.error(f"FID 計算失敗: {e}")
            detailed_metrics['fid'] = float('nan')
        
        # Inception Score 計算
        logger.info("計算 Inception Score...")
        try:
            is_calculator = calculators['inception']
            is_mean, is_std = is_calculator.calculate_inception_score(fake_images_path)
            detailed_metrics['inception_score_mean'] = is_mean
            detailed_metrics['inception_score_std'] = is_std
            logger.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        except Exception as e:
            logger.error(f"Inception Score 計算失敗: {e}")
            detailed_metrics['inception_score_mean'] = float('nan')
            detailed_metrics['inception_score_std'] = float('nan')
        
        # CLIP Score 計算
        if target_class is not None:
            logger.info("計算 CLIP Score...")
            try:
                clip_calculator = calculators['clip']
                clip_scores = clip_calculator.calculate_class_conditional_clip_score(
                    fake_images_path, target_class, dataset
                )
                detailed_metrics.update(clip_scores)
                logger.info(f"CLIP Score: {clip_scores.get('clip_score_mean', 0.0):.4f}")
            except Exception as e:
                logger.error(f"CLIP Score 計算失敗: {e}")
                detailed_metrics['clip_score_mean'] = float('nan')
                detailed_metrics['clip_score_std'] = float('nan')
        
        # 合併指標
        final_metrics = {**all_metrics, **detailed_metrics}
        
        # 保存指標
        metrics_file = output_path / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # 創建指標摘要
        summary = create_metrics_summary(final_metrics)
        summary_file = output_path / "metrics_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # 創建指標可視化
        create_metrics_visualization(final_metrics, output_path / "metrics_plot.png")
        
        # 保存評估配置
        config = {
            'real_images_path': real_images_path,
            'fake_images_path': fake_images_path,
            'target_class': target_class,
            'dataset': dataset,
            'device': device,
            'batch_size': batch_size,
            'evaluation_time': datetime.now().isoformat()
        }
        
        config_file = output_path / "evaluation_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("指標評估完成！")
        logger.info(f"指標文件: {metrics_file}")
        logger.info(f"摘要文件: {summary_file}")
        logger.info(f"可視化文件: {output_path / 'metrics_plot.png'}")
        
        # 打印摘要
        print("\n" + "="*50)
        print("指標評估結果摘要")
        print("="*50)
        print(summary)
        print("="*50)
        
        return final_metrics
        
    except Exception as e:
        logger.error(f"指標評估過程中發生錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_metrics_summary(metrics: dict) -> str:
    """創建指標摘要。"""
    summary_lines = []
    summary_lines.append("圖像生成指標評估摘要")
    summary_lines.append("=" * 40)
    summary_lines.append("")
    
    # 主要指標
    if 'fid' in metrics and not np.isnan(metrics['fid']):
        summary_lines.append(f"FID Score: {metrics['fid']:.4f}")
        summary_lines.append("  (越低越好，理想值 < 10)")
        summary_lines.append("")
    
    if 'inception_score_mean' in metrics and not np.isnan(metrics['inception_score_mean']):
        summary_lines.append(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics.get('inception_score_std', 0.0):.4f}")
        summary_lines.append("  (越高越好，理想值 > 8)")
        summary_lines.append("")
    
    # CLIP Score
    clip_scores = {k: v for k, v in metrics.items() if k.startswith('clip_score_') and not np.isnan(v)}
    if clip_scores:
        summary_lines.append("CLIP Score:")
        for name, score in clip_scores.items():
            if isinstance(score, (list, np.ndarray)):
                score = np.mean(score)
            summary_lines.append(f"  {name}: {score:.4f}")
        summary_lines.append("  (越高越好，理想值 > 0.7)")
        summary_lines.append("")
    
    # 其他指標
    other_metrics = {k: v for k, v in metrics.items() 
                    if k not in ['fid', 'inception_score_mean', 'inception_score_std'] 
                    and not k.startswith('clip_score_')}
    
    if other_metrics:
        summary_lines.append("其他指標:")
        for name, value in other_metrics.items():
            if isinstance(value, (int, float)):
                summary_lines.append(f"  {name}: {value:.4f}")
            else:
                summary_lines.append(f"  {name}: {value}")
        summary_lines.append("")
    
    # 評估建議
    summary_lines.append("評估建議:")
    if 'fid' in metrics and metrics['fid'] < 10:
        summary_lines.append("  ✅ FID 分數優秀")
    elif 'fid' in metrics and metrics['fid'] < 20:
        summary_lines.append("  ⚠️  FID 分數良好，有改進空間")
    else:
        summary_lines.append("  ❌ FID 分數需要改進")
    
    if 'inception_score_mean' in metrics and metrics['inception_score_mean'] > 8:
        summary_lines.append("  ✅ Inception Score 優秀")
    elif 'inception_score_mean' in metrics and metrics['inception_score_mean'] > 6:
        summary_lines.append("  ⚠️  Inception Score 良好，有改進空間")
    else:
        summary_lines.append("  ❌ Inception Score 需要改進")
    
    return "\n".join(summary_lines)


def create_metrics_visualization(metrics: dict, output_path: Path):
    """創建指標可視化。"""
    try:
        # 準備數據
        metric_names = []
        metric_values = []
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metric_names.append(name.replace('_', ' ').title())
                metric_values.append(value)
        
        if not metric_names:
            return
        
        # 創建柱狀圖
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(metric_names)), metric_values, color='skyblue', alpha=0.7)
        
        # 設置標籤
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Image Generation Metrics Evaluation')
        plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right')
        
        # 添加數值標籤
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"創建可視化失敗: {e}")


def compare_models(
    model_paths: list,
    real_images_path: str,
    target_class: int = None,
    dataset: str = "cifar10",
    device: str = "cuda",
    batch_size: int = 32,
    output_dir: str = "model_comparison"
):
    """
    比較多個模型的性能。
    
    Args:
        model_paths: 模型路徑列表
        real_images_path: 真實圖像路徑
        target_class: 目標類別
        dataset: 數據集名稱
        device: 設備
        batch_size: 批次大小
        output_dir: 輸出目錄
    """
    logger = QRLLogger("model_comparison")
    logger.info("開始模型比較...")
    
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).name
        logger.info(f"評估模型: {model_name}")
        
        try:
            metrics = evaluate_metrics(
                real_images_path=real_images_path,
                fake_images_path=model_path,
                target_class=target_class,
                dataset=dataset,
                device=device,
                batch_size=batch_size,
                output_dir=output_path / model_name
            )
            
            if metrics:
                comparison_results[model_name] = metrics
                
        except Exception as e:
            logger.error(f"評估模型 {model_name} 時發生錯誤: {e}")
            comparison_results[model_name] = {'error': str(e)}
    
    # 創建比較表格
    if comparison_results:
        create_comparison_table(comparison_results, output_path / "comparison_table.csv")
        create_comparison_plot(comparison_results, output_path / "comparison_plot.png")
    
    return comparison_results


def create_comparison_table(results: dict, output_path: Path):
    """創建比較表格。"""
    try:
        # 準備數據
        data = []
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                row = {'Model': model_name}
                row.update(metrics)
                data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"比較表格已保存到: {output_path}")
            
    except Exception as e:
        print(f"創建比較表格失敗: {e}")


def create_comparison_plot(results: dict, output_path: Path):
    """創建比較圖表。"""
    try:
        # 準備數據
        models = []
        fid_scores = []
        is_scores = []
        
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                models.append(model_name)
                fid_scores.append(metrics.get('fid', float('nan')))
                is_scores.append(metrics.get('inception_score_mean', float('nan')))
        
        if not models:
            return
        
        # 創建子圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # FID 比較
        ax1.bar(models, fid_scores, color='lightcoral', alpha=0.7)
        ax1.set_title('FID Score Comparison (Lower is Better)')
        ax1.set_ylabel('FID Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Inception Score 比較
        ax2.bar(models, is_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Inception Score Comparison (Higher is Better)')
        ax2.set_ylabel('Inception Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"創建比較圖表失敗: {e}")


def main():
    """主函數。"""
    parser = argparse.ArgumentParser(description="評估圖像生成指標")
    parser.add_argument(
        "--real-images",
        type=str,
        required=True,
        help="真實圖像路徑"
    )
    parser.add_argument(
        "--fake-images",
        type=str,
        required=True,
        help="生成圖像路徑"
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="目標類別"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="數據集名稱"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="設備"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批次大小"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="輸出目錄"
    )
    parser.add_argument(
        "--compare-models",
        nargs='+',
        help="要比較的模型路徑列表"
    )
    
    args = parser.parse_args()
    
    if args.compare_models:
        # 模型比較模式
        results = compare_models(
            model_paths=args.compare_models,
            real_images_path=args.real_images,
            target_class=args.target_class,
            dataset=args.dataset,
            device=args.device,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    else:
        # 單一評估模式
        results = evaluate_metrics(
            real_images_path=args.real_images,
            fake_images_path=args.fake_images,
            target_class=args.target_class,
            dataset=args.dataset,
            device=args.device,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    
    if results:
        print("✅ 評估完成")
        sys.exit(0)
    else:
        print("❌ 評估失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
