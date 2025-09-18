# 🚀 QRL 圖像合成快速開始指南

## ⚡ 30秒快速測試

```bash
# 1. 測試依賴
python test_imports.py

# 2. 快速訓練（2個episodes，約2分鐘）
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes=2 algo.ppo.rollout_steps=4

# 3. 生成圖像（5張）
python generate_images.py --checkpoint runs/*/ckpt_best.pt --num-images 5 --target-class 3

# 4. 查看結果
ls generated_images/
```

## 🎯 核心指令

### 訓練指令

#### 快速測試訓練（2個episodes）
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes=2 algo.ppo.rollout_steps=4
```

#### 完整訓練（1000個episodes）
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes=1000
```

#### 自定義參數訓練
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    training.num_episodes=500 \
    algo.ppo.rollout_steps=32 \
    algo.ppo.lr_actor=0.0003 \
    algo.ppo.lr_critic=0.001 \
    control.environment.max_steps=100 \
    dataset.cifar10.target_class=5
```

### 生成指令

#### 簡單生成（推薦）
```bash
# 基本生成（10張圖像）
python generate_images.py --checkpoint runs/*/ckpt_best.pt --num-images 10 --target-class 3

# 自定義參數生成
python generate_images.py \
    --checkpoint runs/20250915_132757/ckpt_best.pt \
    --num-images 20 \
    --target-class 5 \
    --output-dir my_generated_images \
    --device cuda
```

#### 完整生成（使用Hydra配置）
```bash
python scripts/sample_qrl.py --config-path ../qrl/configs --config-name main \
    sampling.num_samples=100 \
    sampling.num_steps=50 \
    sampling.batch_size=16
```

## 📊 參數說明

### 訓練參數

| 參數 | 說明 | 默認值 | 建議值 |
|------|------|--------|--------|
| `training.num_episodes` | 訓練episodes數量 | 1000 | 2-1000 |
| `algo.ppo.rollout_steps` | 每次rollout的步數 | 32 | 4-64 |
| `algo.ppo.lr_actor` | Actor學習率 | 0.0001 | 0.00005-0.001 |
| `algo.ppo.lr_critic` | Critic學習率 | 0.001 | 0.0005-0.002 |
| `control.environment.max_steps` | 環境最大步數 | 50 | 25-100 |
| `dataset.cifar10.target_class` | 目標類別 (0-9) | 3 | 0-9 |

### 生成參數

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--checkpoint` | 檢查點文件路徑 | 必需 |
| `--num-images` | 生成圖像數量 | 10 |
| `--target-class` | 目標類別 (0-9) | 3 |
| `--output-dir` | 輸出目錄 | generated_images |
| `--device` | 設備 | cuda |

## 📈 訓練監控

訓練過程中會顯示詳細的epoch級別日誌：

```
=== Episode 1/1000 ===
  收集經驗數據...
  收集完成，數據大小: 32
  開始策略更新...
    Epoch 1/4 開始更新...
    Epoch 1/4 完成 | Policy Loss: 1.165920 | Value Loss: 0.142671 | Entropy: -1.786369 | Clip Fraction: 1.0000
    Epoch 2/4 開始更新...
    Epoch 2/4 完成 | Policy Loss: -0.354277 | Value Loss: 0.503881 | Entropy: -1.790026 | Clip Fraction: 1.0000
  策略更新完成
  開始策略評估...
  評估完成，獎勵: -3.6600
  🎉 新的最佳模型！獎勵: -3.6600
Episode 1/1000 完成 | Eval Reward: -3.6600 | Actor Loss: 0.4058 | Critic Loss: 0.3233 | 時間: 192.13s
```

## 🔧 故障排除

### 常見問題

#### 1. 依賴問題
```bash
# 檢查依賴是否安裝
python test_imports.py

# 如果缺少依賴，安裝
pip install gymnasium diffusers hydra-core
```

#### 2. CUDA問題
```bash
# 檢查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 如果沒有CUDA，使用CPU
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main device=cpu
```

#### 3. 檢查點問題
```bash
# 檢查檢查點文件
ls -la runs/*/ckpt_*.pt

# 使用最新檢查點
python generate_images.py --checkpoint $(ls -t runs/*/ckpt_best.pt | head -1) --num-images 5
```

#### 4. 內存不足
```bash
# 減少批次大小
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    algo.ppo.rollout_steps=16 \
    algo.ppo.mini_batch_size=4

# 使用CPU
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main device=cpu
```

## 📁 輸出文件

### 訓練輸出
```
runs/YYYYMMDD_HHMMSS/
├── ckpt_best.pt          # 最佳模型
├── ckpt_final.pt         # 最終模型
├── ckpt_episode_*.pt     # 定期檢查點
├── logs/                 # 訓練日誌
└── tensorboard/          # TensorBoard日誌
```

### 生成輸出
```
generated_images/
├── generated_000.png
├── generated_001.png
├── ...
└── grid.png              # 網格可視化
```

## 🎨 目標類別說明

CIFAR-10 數據集的10個類別：

| 類別 | 名稱 | 類別 | 名稱 |
|------|------|------|------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

## 🚀 進階用法

### 1. 多類別訓練
```bash
# 訓練多個類別
for class in 0 1 2 3 4; do
    python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
        training.num_episodes=100 \
        dataset.cifar10.target_class=$class \
        work_dir=runs/class_$class
done
```

### 2. 批量生成
```bash
# 為所有類別生成圖像
for class in 0 1 2 3 4 5 6 7 8 9; do
    python generate_images.py \
        --checkpoint runs/*/ckpt_best.pt \
        --num-images 10 \
        --target-class $class \
        --output-dir generated_class_$class
done
```

### 3. 性能監控
```bash
# 使用TensorBoard監控訓練
tensorboard --logdir runs/*/tensorboard

# 查看訓練日誌
tail -f runs/*/logs/train_qrl_*.log
```

## 📞 獲取幫助

如果遇到問題，請：

1. 檢查 `python test_imports.py` 是否通過
2. 查看訓練日誌文件
3. 確認檢查點文件存在
4. 檢查CUDA/內存使用情況

祝您訓練愉快！🎉
