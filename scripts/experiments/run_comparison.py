#!/usr/bin/env python3
"""完整的比較實驗運行腳本，自動執行訓練和評估。"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
from datetime import datetime
import json
import shutil


def run_training(config_path: str, output_dir: str) -> bool:
    """運行訓練。"""
    print(f"🚀 開始訓練: {config_path}")
    
    cmd = [
        sys.executable, "train_comparison.py",
        "--config", config_path,
        "--override", f"output.save_dir={output_dir}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小時超時
        
        if result.returncode == 0:
            print(f"✅ 訓練完成: {config_path}")
            return True
        else:
            print(f"❌ 訓練失敗: {config_path}")
            print(f"錯誤輸出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 訓練超時: {config_path}")
        return False
    except Exception as e:
        print(f"❌ 訓練異常: {config_path}, 錯誤: {e}")
        return False


def run_evaluation(model_configs: list, output_dir: str) -> bool:
    """運行評估。"""
    print("📊 開始模型評估...")
    
    # 構建模型參數
    models_str = ",".join([f"{config['name']}:{config['path']}:{config['type']}" 
                          for config in model_configs])
    
    cmd = [
        sys.executable, "eval_comparison.py",
        "--models", models_str,
        "--output", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分鐘超時
        
        if result.returncode == 0:
            print("✅ 評估完成")
            return True
        else:
            print("❌ 評估失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 評估超時")
        return False
    except Exception as e:
        print(f"❌ 評估異常: {e}")
        return False


def find_best_checkpoint(model_dir: str) -> str:
    """找到最佳檢查點。"""
    model_path = Path(model_dir)
    
    # 優先順序: ckpt_best.pt > ckpt_final.pt > 最新的檢查點
    best_checkpoint = model_path / "ckpt_best.pt"
    if best_checkpoint.exists():
        return str(best_checkpoint)
    
    final_checkpoint = model_path / "ckpt_final.pt"
    if final_checkpoint.exists():
        return str(final_checkpoint)
    
    # 尋找最新的檢查點
    checkpoints = list(model_path.glob("ckpt_*.pt"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        return str(latest_checkpoint)
    
    return None


def main():
    """主函數。"""
    parser = argparse.ArgumentParser(description="運行完整的比較實驗")
    parser.add_argument("--output", "-o", default="comparison_experiment", 
                       help="輸出目錄")
    parser.add_argument("--skip-training", action="store_true", 
                       help="跳過訓練，直接評估")
    parser.add_argument("--models-only", nargs="+", 
                       help="只訓練指定的模型 (quantum, classical, aligned_classical)")
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 實驗配置
    experiment_configs = {
        "quantum": {
            "name": "Quantum Actor",
            "config": "configs/quantum_config.yaml",
            "type": "quantum"
        },
        "classical": {
            "name": "Classical Actor", 
            "config": "configs/classical_config.yaml",
            "type": "classical"
        },
        "aligned_classical": {
            "name": "Aligned Classical Actor",
            "config": "configs/aligned_classical_config.yaml", 
            "type": "aligned_classical"
        }
    }
    
    # 如果指定了特定模型，只運行這些模型
    if args.models_only:
        experiment_configs = {k: v for k, v in experiment_configs.items() 
                            if k in args.models_only}
    
    print("=" * 60)
    print("🧪 量子 vs 經典比較實驗")
    print("=" * 60)
    print(f"輸出目錄: {output_dir}")
    print(f"實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"將要訓練的模型: {list(experiment_configs.keys())}")
    print()
    
    # 記錄實驗開始時間
    start_time = time.time()
    
    # 訓練階段
    trained_models = {}
    
    if not args.skip_training:
        print("🎯 階段 1: 模型訓練")
        print("-" * 40)
        
        for model_key, config in experiment_configs.items():
            model_output_dir = output_dir / f"{model_key}_training"
            
            success = run_training(config["config"], str(model_output_dir))
            
            if success:
                # 找到最佳檢查點
                checkpoint_path = find_best_checkpoint(str(model_output_dir))
                if checkpoint_path:
                    trained_models[model_key] = {
                        "name": config["name"],
                        "path": checkpoint_path,
                        "type": config["type"],
                        "training_dir": str(model_output_dir)
                    }
                    print(f"✅ {config['name']} 訓練成功")
                else:
                    print(f"⚠️ {config['name']} 訓練完成但未找到檢查點")
            else:
                print(f"❌ {config['name']} 訓練失敗")
            
            print()
    else:
        print("⏭️ 跳過訓練階段")
        print()
    
    # 評估階段
    if trained_models:
        print("📊 階段 2: 模型評估")
        print("-" * 40)
        
        eval_output_dir = output_dir / "evaluation_results"
        success = run_evaluation(list(trained_models.values()), str(eval_output_dir))
        
        if success:
            print("✅ 評估完成")
        else:
            print("❌ 評估失敗")
    else:
        print("⚠️ 沒有成功訓練的模型，跳過評估")
    
    # 生成實驗報告
    print("\n📋 階段 3: 生成實驗報告")
    print("-" * 40)
    
    generate_experiment_report(trained_models, output_dir, start_time)
    
    # 實驗完成
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 實驗完成！")
    print(f"總耗時: {total_time/3600:.2f} 小時")
    print(f"結果保存在: {output_dir}")
    print("=" * 60)


def generate_experiment_report(trained_models: dict, output_dir: Path, start_time: float):
    """生成實驗報告。"""
    report = {
        "experiment_info": {
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration": time.time() - start_time,
            "models_trained": len(trained_models)
        },
        "models": trained_models,
        "results_summary": {}
    }
    
    # 檢查評估結果
    eval_dir = output_dir / "evaluation_results"
    if eval_dir.exists():
        # 讀取比較結果
        results_file = eval_dir / "comparison_results.csv"
        if results_file.exists():
            import pandas as pd
            df = pd.read_csv(results_file, index_col=0)
            report["results_summary"] = df.to_dict()
        
        # 讀取統計報告
        stats_file = eval_dir / "statistical_report.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                report["statistical_analysis"] = stats
    
    # 保存報告
    report_file = output_dir / "experiment_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 生成簡要報告
    summary_file = output_dir / "experiment_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("QUANTUM VS CLASSICAL COMPARISON EXPERIMENT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Experiment Time: {report['experiment_info']['start_time']}\n")
        f.write(f"Duration: {report['experiment_info']['total_duration']/3600:.2f} hours\n")
        f.write(f"Models Trained: {report['experiment_info']['models_trained']}\n\n")
        
        f.write("TRAINED MODELS:\n")
        f.write("-" * 30 + "\n")
        for model_key, model_info in trained_models.items():
            f.write(f"{model_info['name']}: {model_info['path']}\n")
        
        if report["results_summary"]:
            f.write("\nRESULTS SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for metric, values in report["results_summary"].items():
                f.write(f"{metric}:\n")
                for model, value in values.items():
                    f.write(f"  {model}: {value:.4f}\n")
                f.write("\n")
    
    print(f"✅ 實驗報告已生成: {report_file}")
    print(f"✅ 簡要報告已生成: {summary_file}")


if __name__ == "__main__":
    main()
