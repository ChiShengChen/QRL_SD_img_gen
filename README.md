# 🧩 QRL Image Synthesis

**Quantum Reinforcement Learning-Guided Image Synthesis via Hybrid Quantum-Classical Generative Model Architectures**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.36+-purple.svg)](https://pennylane.ai/)

## ⚡ 快速開始

### 1. 一鍵快速開始（推薦）
```bash
# 自動完成：依賴檢查 + 快速訓練 + 生成圖像
python quick_start.py

# 只測試依賴
python quick_start.py --test-only

# 自定義參數
python quick_start.py --episodes 5 --num-images 10 --target-class 5
```

### 2. 手動步驟（2分鐘）
```bash
# 測試依賴
python test_imports.py

# 快速訓練（2個episodes）
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes=2 algo.ppo.rollout_steps=4

# 生成圖像
python generate_images.py --checkpoint runs/*/ckpt_best.pt --num-images 5 --target-class 3
```

### 3. 完整訓練（30分鐘+）
```bash
# 完整訓練（1000個episodes）
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes=1000

# 生成更多圖像
python generate_images.py --checkpoint runs/*/ckpt_best.pt --num-images 20 --target-class 3 --output-dir my_images
```

### 4. 自定義訓練
```bash
# 自定義參數
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    training.num_episodes=500 \
    algo.ppo.rollout_steps=32 \
    algo.ppo.lr_actor=0.0003 \
    algo.ppo.lr_critic=0.001 \
    dataset.cifar10.target_class=5
```

### 5. 查看結果
```bash
# 查看生成的圖像
ls generated_images/
ls my_images/

# 查看訓練日誌
ls runs/*/logs/
```

> 📖 **詳細指南**: 查看 [QUICK_START.md](QUICK_START.md) 獲取完整的訓練和生成指令說明

## 📖 專案簡介

本專案實現了一個**量子-經典混合**的強化學習框架，用於優化擴散模型的採樣過程。通過訓練**量子策略網路（Actor）**來動態控制擴散採樣中的關鍵參數，實現更智能、更高效的圖像生成。

### 🎯 核心目標

1. **第一階段（Stage A）**：使用 PPO 訓練量子 Actor 控制 **CFG 增量（ΔCFG）**
2. **第二階段（Stage B）**：擴展到注意力門控控制
3. **第三階段（Stage C）**：加入步長因子動態調整

### 🏗️ 架構概覽

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CIFAR-10     │    │  Quantum Actor  │    │   Diffusion     │
│   Environment  │◄──►│  (VQC + MLP)    │◄──►│   Sampler       │
│                │    │                 │    │   (UNet+DDIM)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PPO Trainer  │    │   MLP Critic    │    │   Generated     │
│   (GAE + Clip) │    │   (Value Net)   │    │   Images        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 快速開始

### 環境設置

```bash
# 克隆專案
git clone <your-repo-url>
cd qrl_image_synthesis

# 建立 conda 環境
make env
conda activate qrl_image_synthesis

# 下載資料與預訓練模型
make data
```

### 🎯 核心指令

#### 1. 快速測試訓練（2個episodes）
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes=2 algo.ppo.rollout_steps=4
```

#### 2. 完整訓練（1000個episodes）
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main training.num_episodes=1000
```

#### 3. 自定義參數訓練
```bash
python scripts/train_qrl.py --config-path ../qrl/configs --config-name main \
    training.num_episodes=500 \
    algo.ppo.rollout_steps=32 \
    algo.ppo.lr_actor=0.0003 \
    algo.ppo.lr_critic=0.001 \
    control.environment.max_steps=100 \
    dataset.cifar10.target_class=5
```

#### 4. 生成圖像（簡單方式）
```bash
# 使用最新訓練的模型生成圖像
python generate_images.py --checkpoint runs/*/ckpt_best.pt --num-images 10 --target-class 3

# 自定義參數生成
python generate_images.py \
    --checkpoint runs/20250915_132757/ckpt_best.pt \
    --num-images 20 \
    --target-class 5 \
    --output-dir my_generated_images \
    --device cuda
```

#### 5. 生成圖像（完整方式）
```bash
# 使用 Hydra 配置
python scripts/sample_qrl.py --config-path ../qrl/configs --config-name main \
    sampling.num_samples=100 \
    sampling.num_steps=50 \
    sampling.batch_size=16
```

#### 6. 評估指標
```bash
# 計算 FID、IS、LPIPS、CLIPScore
python scripts/eval_metrics.py \
    --real ./data/cifar10/real \
    --fake ./runs/20250915_132757/samples
```

### 📊 訓練參數說明

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `training.num_episodes` | 訓練episodes數量 | 1000 |
| `algo.ppo.rollout_steps` | 每次rollout的步數 | 32 |
| `algo.ppo.lr_actor` | Actor學習率 | 0.0001 |
| `algo.ppo.lr_critic` | Critic學習率 | 0.001 |
| `control.environment.max_steps` | 環境最大步數 | 50 |
| `dataset.cifar10.target_class` | 目標類別 (0-9) | 3 |

### 🎨 生成參數說明

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--checkpoint` | 檢查點文件路徑 | 必需 |
| `--num-images` | 生成圖像數量 | 10 |
| `--target-class` | 目標類別 (0-9) | 3 |
| `--output-dir` | 輸出目錄 | generated_images |
| `--device` | 設備 | cuda |

### 📈 訓練監控

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

### 🔧 故障排除

```bash
# 檢查依賴是否安裝
python test_imports.py

# 檢查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 檢查檢查點文件
ls -la runs/*/ckpt_*.pt
```

## 📁 專案結構

```
qrl_image_synthesis/
├─ README.md                 # 專案說明
├─ LICENSE                   # MIT 授權
├─ .gitignore               # Git 忽略檔案
├─ Makefile                 # 建置指令
├─ pyproject.toml           # Python 專案配置
├─ environment.yml          # Conda 環境配置
├─ Dockerfile               # Docker 容器配置
├─ scripts/                 # 執行腳本
│  ├─ download_cifar10.py  # 資料下載
│  ├─ train_qrl.py         # 訓練腳本
│  ├─ sample_qrl.py        # 採樣腳本
│  └─ eval_metrics.py      # 指標評估
├─ qrl/                     # 核心模組
│  ├─ configs/              # 配置檔案
│  ├─ utils/                # 工具函數
│  ├─ envs/                 # 環境定義
│  ├─ controls/             # 控制邏輯
│  ├─ reward/               # 獎勵函數
│  ├─ actors/               # 策略網路
│  ├─ critics/              # 價值網路
│  ├─ training/             # 訓練邏輯
│  ├─ models/               # 模型定義
│  └─ metrics/              # 評估指標
├─ tests/                   # 測試檔案
└─ assets/                  # 輸出資源
   └─ samples/              # 生成圖像
```

## ⚙️ 配置說明

### 訓練配置

- **PPO 參數**：`gamma=0.995`, `gae_lambda=0.95`, `clip_ratio=0.1`
- **量子 Actor**：8 qubits, 4 layers, RY-RZ + ring entanglement
- **採樣器**：DDIM, 50 steps, base CFG=5.0
- **控制空間**：ΔCFG ∈ [-2.0, +2.0]

### 環境配置

- **狀態空間**：12-16 維向量（步數、潛在變數、預測噪聲、代理置信度等）
- **動作空間**：連續標量（ΔCFG）
- **獎勵設計**：分類器置信度增量 + 動作正則化

## 🔬 技術細節

### 量子策略網路

- **架構**：Variational Quantum Circuit (VQC) + MLP 頭
- **量子比特**：8 qubits
- **層數**：4 layers
- **參數化**：RY-RZ 旋轉門 + ring entanglement
- **輸出**：高斯策略的均值 μ 和對數標準差 logσ

### 經典對比

- **參數對齊**：經典 MLP Actor 的參數量與量子 Actor + MLP 頭匹配
- **公平比較**：確保量子優勢來自於量子特性而非參數數量

### 獎勵函數

- **終端獎勵**：`α * log p(y|x)` （分類器置信度）
- **步間獎勵**：`β * (conf_t - conf_{t-1})` （置信度增量）
- **正則化**：L2 動作懲罰 + 可選 TV 正則

## 📊 預期結果

### 訓練曲線

- **PPO 回報**：隨訓練穩定上升
- **分類器置信度**：採樣過程中逐步增長
- **動作正則化**：ΔCFG 變化趨於平滑

### 生成品質

- **FID 分數**：優於固定 CFG 基準
- **多樣性**：LPIPS 分數保持合理範圍
- **語義一致性**：CLIPScore 反映目標類別匹配度

## 🚧 後續擴展

### Stage B：注意力門控控制

```yaml
# 在 configs/control/stageB_attn.yaml 中啟用
enable_attn_gating: true
enable_step_scale: false
attn_gate_min: 0.0
attn_gate_max: 1.0
```

### Stage C：步長因子控制

```yaml
# 在 configs/control/stageC_full.yaml 中啟用
enable_attn_gating: true
enable_step_scale: true
step_scale_min: 0.5
step_scale_max: 2.0
```

## 🧪 測試

```bash
# 運行所有測試
make test

# 或指定測試
pytest tests/ -v
```

## 📝 開發指南

### 添加新的控制參數

1. 在 `qrl/controls/control_spaces.py` 中定義新的動作空間
2. 在 `qrl/controls/apply_controls.py` 中實現控制邏輯
3. 更新配置檔案和環境狀態空間

### 擴展到其他資料集

1. 在 `qrl/configs/dataset/` 中添加新配置
2. 實現對應的資料載入器
3. 調整獎勵函數和評估指標

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本專案採用 MIT 授權條款，詳見 [LICENSE](LICENSE) 檔案。

## 🙏 致謝

- [Diffusers](https://github.com/huggingface/diffusers) - 擴散模型實現
- [PennyLane](https://pennylane.ai/) - 量子機器學習框架
- [Gymnasium](https://gymnasium.farama.org/) - 強化學習環境
- [PyTorch](https://pytorch.org/) - 深度學習框架
