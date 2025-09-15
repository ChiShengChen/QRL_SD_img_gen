# 🚀 QRL 圖像合成訓練與生成指南

## 📋 目錄
- [環境準備](#環境準備)
- [數據準備](#數據準備)
- [訓練方法](#訓練方法)
- [生成方法](#生成方法)
- [配置說明](#配置說明)
- [故障排除](#故障排除)

## 🔧 環境準備

### 1. 安裝依賴
```bash
# 進入項目目錄
cd /media/meow/Transcend/QRL_ddpm_image_gen/qrl_image_synthesis

# 安裝依賴
pip install -r requirements.txt
# 或者使用 conda
conda env create -f environment.yml
conda activate qrl_image_synthesis
```

### 2. 下載數據集
```bash
# 下載 CIFAR-10 數據集
python scripts/download_cifar10.py
```

## 🎯 訓練方法

### 1. 快速測試訓練 (2 episodes)
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    training.num_episodes=2 \
    algo.ppo.rollout_steps=4
```

### 2. 完整訓練 (1000 episodes)
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    training.num_episodes=1000 \
    algo.ppo.rollout_steps=32
```

### 3. 自定義參數訓練
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    training.num_episodes=500 \
    algo.ppo.rollout_steps=32 \
    algo.ppo.lr_actor=0.0003 \
    algo.ppo.lr_critic=0.001 \
    control.environment.max_steps=100 \
    dataset.cifar10.target_class=5
```

### 4. 訓練參數說明
| 參數 | 說明 | 默認值 |
|------|------|--------|
| `training.num_episodes` | 訓練episodes數量 | 1000 |
| `algo.ppo.rollout_steps` | 每次rollout的步數 | 32 |
| `algo.ppo.lr_actor` | Actor學習率 | 0.0001 |
| `algo.ppo.lr_critic` | Critic學習率 | 0.001 |
| `control.environment.max_steps` | 環境最大步數 | 50 |
| `dataset.cifar10.target_class` | 目標類別 (0-9) | 3 |

### 5. 監控訓練進度
訓練過程中會生成：
- **日誌文件**: `runs/YYYYMMDD_HHMMSS/logs/`
- **檢查點**: `runs/YYYYMMDD_HHMMSS/ckpt_*.pt`
- **配置**: `runs/YYYYMMDD_HHMMSS/training_config.yaml`
- **TensorBoard**: `runs/YYYYMMDD_HHMMSS/tensorboard/`

## 🎨 生成方法

### 1. 使用簡單生成腳本 (推薦)
```bash
# 基本生成 (10張圖像)
python generate_images.py --checkpoint runs/20250915_131017/ckpt_best.pt

# 自定義參數生成
python generate_images.py \
    --checkpoint runs/20250915_131017/ckpt_best.pt \
    --num-images 20 \
    --target-class 5 \
    --output-dir my_generated_images \
    --device cuda
```

### 2. 使用完整採樣腳本
```bash
# 使用 Hydra 配置
python scripts/sample_qrl.py --config-path ../qrl/configs --config-name main \
    sampling.num_samples=100 \
    sampling.num_steps=50 \
    sampling.batch_size=16
```

### 3. 生成參數說明
| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--checkpoint` | 檢查點文件路徑 | 必需 |
| `--num-images` | 生成圖像數量 | 10 |
| `--target-class` | 目標類別 (0-9) | 3 |
| `--output-dir` | 輸出目錄 | generated_images |
| `--device` | 設備 | cuda |

## ⚙️ 配置說明

### 1. 主要配置文件
- `qrl/configs/main.yaml` - 主配置文件
- `qrl/configs/base.yaml` - 基礎配置
- `qrl/configs/dataset/cifar10.yaml` - 數據集配置
- `qrl/configs/control/stageA_cfg.yaml` - 控制配置
- `qrl/configs/algo/ppo.yaml` - 算法配置
- `qrl/configs/model/unet_ddim.yaml` - 模型配置

### 2. 控制階段說明
- **Stage A**: ΔCFG 控制 (推薦用於快速測試)
- **Stage B**: Attention Gating 控制
- **Stage C**: 完整控制

### 3. 模型類型
- **Quantum Actor**: 使用量子電路的 Actor
- **Classical Actor**: 使用經典神經網絡的 Actor
- **Aligned Classical Actor**: 與量子 Actor 參數對齊的古典 Actor

## 🔍 故障排除

### 1. 常見錯誤
```bash
# 檢查依賴是否安裝
python test_imports.py

# 檢查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 檢查檢查點文件
ls -la runs/*/ckpt_*.pt
```

### 2. 內存不足
```bash
# 減少批次大小
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    algo.ppo.rollout_steps=16 \
    algo.ppo.mini_batch_size=4

# 使用 CPU
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    device=cpu
```

### 3. 訓練不穩定
```bash
# 降低學習率
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    algo.ppo.lr_actor=0.00005 \
    algo.ppo.lr_critic=0.0005

# 增加熵係數
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    algo.ppo.entropy_coef=0.02
```

## 📊 評估指標

### 1. 訓練指標
- **Actor Loss**: Actor 網絡損失
- **Critic Loss**: Critic 網絡損失
- **Terminal Reward**: 終端獎勵
- **Episode Length**: Episode 長度

### 2. 生成指標
- **FID**: Fréchet Inception Distance
- **IS**: Inception Score
- **CLIP Score**: CLIP 相似度分數
- **Diversity**: 生成圖像多樣性

## 🎯 最佳實踐

### 1. 訓練建議
- 從 Stage A 開始，逐步嘗試 Stage B 和 Stage C
- 使用較小的 `rollout_steps` 進行快速測試
- 監控 TensorBoard 日誌
- 定期保存檢查點

### 2. 生成建議
- 使用訓練好的最佳檢查點
- 嘗試不同的目標類別
- 調整生成步數以平衡質量和速度
- 使用網格可視化檢查生成結果

## 📁 文件結構
```
qrl_image_synthesis/
├── scripts/
│   ├── train_qrl.py          # 訓練腳本
│   ├── sample_qrl.py         # 完整採樣腳本
│   └── download_cifar10.py   # 數據下載腳本
├── generate_images.py        # 簡單生成腳本
├── test_imports.py          # 測試腳本
├── qrl/                     # 核心代碼
│   ├── configs/             # 配置文件
│   ├── actors/              # Actor 實現
│   ├── critics/             # Critic 實現
│   ├── envs/                # 環境實現
│   ├── training/            # 訓練器實現
│   └── ...
└── runs/                    # 訓練輸出
    └── YYYYMMDD_HHMMSS/    # 訓練運行目錄
        ├── ckpt_*.pt       # 檢查點文件
        ├── logs/           # 日誌文件
        └── tensorboard/    # TensorBoard 日誌
```

## 🚀 快速開始

1. **準備環境**:
   ```bash
   cd /media/meow/Transcend/QRL_ddpm_image_gen/qrl_image_synthesis
   python test_imports.py
   ```

2. **快速訓練**:
   ```bash
   python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes=2
   ```

3. **生成圖像**:
   ```bash
   python generate_images.py --checkpoint runs/*/ckpt_best.pt --num-images 10
   ```

4. **查看結果**:
   ```bash
   ls generated_images/
   ```

祝您訓練愉快！🎉
