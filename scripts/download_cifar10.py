#!/usr/bin/env python3
"""下載 CIFAR-10 數據集腳本。"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import json

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qrl.utils.logging import QRLLogger


def download_cifar10(
    data_dir: str = "data",
    train_size: int = 45000,
    val_size: int = 5000,
    test_size: int = 10000,
    image_size: int = 32,
    download: bool = True
):
    """
    下載並準備 CIFAR-10 數據集。
    
    Args:
        data_dir: 數據目錄
        train_size: 訓練集大小
        val_size: 驗證集大小
        test_size: 測試集大小
        image_size: 圖像大小
        download: 是否下載數據
    """
    logger = QRLLogger("download_cifar10")
    logger.info("開始下載 CIFAR-10 數據集...")
    
    # 創建數據目錄
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 數據轉換
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        # 下載訓練集
        logger.info("下載訓練集...")
        trainset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=download,
            transform=transform
        )
        
        # 下載測試集
        logger.info("下載測試集...")
        testset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            download=download,
            transform=transform
        )
        
        logger.info(f"訓練集大小: {len(trainset)}")
        logger.info(f"測試集大小: {len(testset)}")
        
        # 分割訓練集為訓練和驗證
        if train_size + val_size <= len(trainset):
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, train_size + val_size))
            
            # 創建子集
            train_subset = torch.utils.data.Subset(trainset, train_indices)
            val_subset = torch.utils.data.Subset(trainset, val_indices)
            
            logger.info(f"訓練子集大小: {len(train_subset)}")
            logger.info(f"驗證子集大小: {len(val_subset)}")
            
            # 保存分割信息
            split_info = {
                'train_size': len(train_subset),
                'val_size': len(val_subset),
                'test_size': len(testset),
                'total_size': len(trainset) + len(testset),
                'image_size': image_size,
                'num_classes': 10,
                'class_names': [
                    'airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'
                ]
            }
            
            split_file = data_path / "split_info.json"
            with open(split_file, 'w') as f:
                json.dump(split_info, f, indent=2)
            
            logger.info(f"分割信息已保存到: {split_file}")
            
        else:
            logger.warning(f"請求的訓練集大小 ({train_size}) 超過可用數據 ({len(trainset)})")
        
        # 創建數據加載器
        train_loader = torch.utils.data.DataLoader(
            train_subset if 'train_subset' in locals() else trainset,
            batch_size=64,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_subset if 'val_subset' in locals() else testset,
            batch_size=64,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=64,
            shuffle=False,
            num_workers=4
        )
        
        # 測試數據加載
        logger.info("測試數據加載...")
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="訓練集")):
            if batch_idx >= 2:  # 只測試前幾個批次
                break
        
        for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="驗證集")):
            if batch_idx >= 2:
                break
        
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="測試集")):
            if batch_idx >= 2:
                break
        
        logger.info("數據加載測試完成")
        
        # 保存數據統計信息
        stats = {
            'data_dir': str(data_path.absolute()),
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'image_shape': list(data.shape[1:]),
            'num_classes': 10,
            'class_distribution': {}
        }
        
        # 計算類別分布
        class_counts = {}
        for _, target in train_loader.dataset:
            # target 可能是 tensor 或 int，需要處理兩種情況
            if hasattr(target, 'item'):
                target_value = target.item()
            else:
                target_value = int(target)
            class_counts[target_value] = class_counts.get(target_value, 0) + 1
        
        stats['class_distribution'] = class_counts
        
        stats_file = data_path / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"數據集統計信息已保存到: {stats_file}")
        logger.info("CIFAR-10 數據集下載和準備完成！")
        
        return True
        
    except Exception as e:
        logger.error(f"下載 CIFAR-10 數據集時發生錯誤: {e}")
        return False


def verify_dataset(data_dir: str = "data"):
    """驗證數據集完整性。"""
    logger = QRLLogger("verify_dataset")
    logger.info("驗證數據集完整性...")
    
    data_path = Path(data_dir)
    
    # 檢查必要文件
    required_files = [
        "cifar-10-batches-py/data_batch_1",
        "cifar-10-batches-py/data_batch_2",
        "cifar-10-batches-py/data_batch_3",
        "cifar-10-batches-py/data_batch_4",
        "cifar-10-batches-py/data_batch_5",
        "cifar-10-batches-py/test_batch",
        "cifar-10-batches-py/batches.meta"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = data_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"缺少以下文件: {missing_files}")
        return False
    
    # 檢查分割信息
    split_file = data_path / "split_info.json"
    if not split_file.exists():
        logger.warning("未找到分割信息文件")
    else:
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        logger.info(f"分割信息: {split_info}")
    
    # 檢查統計信息
    stats_file = data_path / "dataset_stats.json"
    if not stats_file.exists():
        logger.warning("未找到統計信息文件")
    else:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        logger.info(f"數據集統計: {stats}")
    
    logger.info("數據集驗證完成")
    return True


def main():
    """主函數。"""
    parser = argparse.ArgumentParser(description="下載 CIFAR-10 數據集")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="數據目錄 (默認: data)"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=45000,
        help="訓練集大小 (默認: 45000)"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="驗證集大小 (默認: 5000)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10000,
        help="測試集大小 (默認: 10000)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="圖像大小 (默認: 32)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="不下載數據，只驗證現有數據集"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="驗證數據集完整性"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_dataset(args.data_dir)
    elif args.no_download:
        success = verify_dataset(args.data_dir)
    else:
        success = download_cifar10(
            data_dir=args.data_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            image_size=args.image_size,
            download=True
        )
    
    if success:
        print("✅ 操作完成")
        sys.exit(0)
    else:
        print("❌ 操作失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
