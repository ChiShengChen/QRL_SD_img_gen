#!/usr/bin/env python3
"""快速開始腳本 - 幫助用戶快速測試和運行QRL圖像合成"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """運行命令並顯示描述"""
    print(f"\n🔄 {description}...")
    print(f"執行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失敗")
        print(f"錯誤: {e.stderr}")
        return False

def check_dependencies():
    """檢查依賴"""
    print("🔍 檢查依賴...")
    try:
        # 先檢查基本導入
        import torch
        import numpy as np
        import gymnasium
        print("✅ 基本依賴檢查通過")
        
        # 嘗試檢查量子相關依賴
        try:
            import pennylane as qml
            print("✅ PennyLane 可用")
        except Exception as e:
            print(f"⚠️  PennyLane 問題: {e}")
            print("   建議: pip install pennylane-lightning")
        
        return True
    except ImportError as e:
        print(f"❌ 缺少依賴: {e}")
        return False

def quick_train(episodes=2, rollout_steps=4):
    """快速訓練"""
    cmd = f"python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes={episodes} algo.ppo.rollout_steps={rollout_steps}"
    return run_command(cmd, f"快速訓練 ({episodes} episodes)")

def generate_images(num_images=5, target_class=3):
    """生成圖像"""
    # 找到最新的檢查點
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("❌ 沒有找到訓練結果，請先運行訓練")
        return False
    
    # 找到最新的ckpt_best.pt
    checkpoints = list(runs_dir.glob("*/ckpt_best.pt"))
    if not checkpoints:
        print("❌ 沒有找到檢查點文件")
        return False
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"📁 使用檢查點: {latest_checkpoint}")
    
    cmd = f"python generate_images.py --checkpoint {latest_checkpoint} --num-images {num_images} --target-class {target_class}"
    return run_command(cmd, f"生成圖像 ({num_images} 張)")

def show_results():
    """顯示結果"""
    print("\n📊 查看結果...")
    
    # 檢查生成的圖像
    generated_dir = Path("generated_images")
    if generated_dir.exists():
        images = list(generated_dir.glob("*.png"))
        print(f"✅ 生成圖像: {len(images)} 張")
        for img in images[:5]:  # 只顯示前5張
            print(f"   - {img.name}")
        if len(images) > 5:
            print(f"   ... 還有 {len(images) - 5} 張圖像")
    else:
        print("❌ 沒有找到生成的圖像")
    
    # 檢查訓練日誌
    runs_dir = Path("runs")
    if runs_dir.exists():
        log_dirs = list(runs_dir.glob("*/logs"))
        if log_dirs:
            print(f"✅ 訓練日誌: {len(log_dirs)} 個")
            for log_dir in log_dirs:
                print(f"   - {log_dir.parent.name}/logs/")

def main():
    parser = argparse.ArgumentParser(description="QRL圖像合成快速開始腳本")
    parser.add_argument("--test-only", action="store_true", help="只測試依賴")
    parser.add_argument("--train-only", action="store_true", help="只運行訓練")
    parser.add_argument("--generate-only", action="store_true", help="只生成圖像")
    parser.add_argument("--episodes", type=int, default=2, help="訓練episodes數量")
    parser.add_argument("--rollout-steps", type=int, default=4, help="rollout步數")
    parser.add_argument("--num-images", type=int, default=5, help="生成圖像數量")
    parser.add_argument("--target-class", type=int, default=3, help="目標類別")
    
    args = parser.parse_args()
    
    print("🚀 QRL圖像合成快速開始")
    print("=" * 50)
    
    # 檢查依賴
    if not check_dependencies():
        print("\n❌ 依賴檢查失敗，請先安裝必要的依賴")
        return 1
    
    if args.test_only:
        print("\n✅ 依賴檢查完成")
        return 0
    
    # 訓練
    if not args.generate_only:
        if not quick_train(args.episodes, args.rollout_steps):
            print("\n❌ 訓練失敗")
            return 1
    
    if args.train_only:
        print("\n✅ 訓練完成")
        return 0
    
    # 生成圖像
    if not generate_images(args.num_images, args.target_class):
        print("\n❌ 圖像生成失敗")
        return 1
    
    # 顯示結果
    show_results()
    
    print("\n🎉 快速開始完成！")
    print("📖 查看 QUICK_START.md 獲取更多詳細指令")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
