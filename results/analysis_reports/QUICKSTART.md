# QRL 圖像合成 - 快速開始指南

## 🚀 5分鐘快速開始

### 1. 克隆項目
```bash
git clone <your-repo-url>
cd qrl_image_synthesis
```

### 2. 創建環境
```bash
# 使用 conda 創建環境
conda env create -f environment.yml
conda activate qrl_image_synthesis

# 安裝項目
pip install -e .
```

### 3. 下載數據
```bash
# 下載 CIFAR-10 數據集
python scripts/download_cifar10.py
```

### 4. 運行測試
```bash
# 測試核心功能
python test_core_imports.py

# 運行單元測試
pytest tests/test_actor_shapes.py -v
```

## 🔧 環境要求

### 系統要求
- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (推薦)
- **RAM**: 16GB+ system memory

### 依賴包
- **PyTorch**: 2.0+ with CUDA 12.1
- **PennyLane**: 0.36+
- **HuggingFace**: diffusers 0.30+
- **其他**: 見 `environment.yml`

## 📚 核心概念

### 量子強化學習 (QRL)
- **量子 Actor**: 使用量子變分電路 (VQC)
- **古典 Critic**: 評估狀態價值
- **PPO 算法**: 策略優化

### 控制階段
- **Stage A**: ΔCFG 控制 (當前實現)
- **Stage B**: 注意力門控 (計劃中)
- **Stage C**: 完整控制 (計劃中)

## 🎯 使用示例

### 基本訓練
```python
from qrl.actors import QuantumActor
from qrl.critics import MLPCritic
from qrl.training import PPOTrainer

# 創建量子 Actor
actor = QuantumActor(state_dim=64, action_dim=1, device="cuda")

# 創建 Critic
critic = MLPCritic(state_dim=64, device="cuda")

# 創建訓練器
trainer = PPOTrainer(actor, critic, env, config, device="cuda")

# 開始訓練
trainer.train()
```

### 圖像採樣
```python
from qrl.envs import create_diffusion_env

# 創建環境
env = create_diffusion_env(
    dataset="cifar10",
    model="unet_ddim",
    control="stageA_cfg"
)

# 採樣圖像
obs = env.reset()
for step in range(50):
    action = actor.sample(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

# 獲取最終圖像
final_image = env.get_final_image()
```

### 指標評估
```python
from qrl.metrics import calculate_all_metrics

# 計算所有指標
metrics = calculate_all_metrics(
    real_images="data/cifar10/test",
    fake_images="runs/latest/samples",
    target_class=0,
    dataset="cifar10"
)

print(f"FID: {metrics['fid']:.4f}")
print(f"Inception Score: {metrics['inception_score_mean']:.4f}")
```

## 🛠️ 故障排除

### 常見問題

#### 1. 導入錯誤
```bash
# 檢查環境
conda list | grep qrl
pip list | grep qrl

# 重新安裝
pip install -e . --force-reinstall
```

#### 2. CUDA 錯誤
```bash
# 檢查 CUDA 版本
nvidia-smi
python -c "import torch; print(torch.version.cuda)"

# 使用 CPU 模式
export CUDA_VISIBLE_DEVICES=""
```

#### 3. 內存不足
```bash
# 減少批次大小
python scripts/train_qrl.py algo.mini_batch_size=2

# 使用混合精度
python scripts/train_qrl.py precision=float16
```

### 調試模式
```bash
# 啟用詳細日誌
export QRL_LOG_LEVEL=DEBUG

# 運行單步測試
python -c "
from qrl.actors import QuantumActor
actor = QuantumActor(state_dim=32, action_dim=1, device='cpu')
print('Actor created successfully')
"
```

## 📖 深入學習

### 文檔結構
```
qrl_image_synthesis/
├── README.md              # 項目概述
├── PROJECT_STATUS.md      # 詳細狀態
├── QUICKSTART.md          # 本文件
├── qrl/                   # 核心模組
│   ├── actors/           # Actor 網絡
│   ├── critics/          # Critic 網絡
│   ├── training/         # 訓練算法
│   ├── envs/             # 環境定義
│   ├── controls/         # 控制系統
│   ├── reward/           # 獎勵函數
│   ├── metrics/          # 評估指標
│   └── models/           # 模型定義
├── scripts/               # 實用腳本
├── tests/                 # 測試代碼
└── configs/               # 配置文件
```

### 關鍵文件
- **`qrl/actors/quantum_actor.py`**: 量子 Actor 實現
- **`qrl/training/ppo_trainer.py`**: PPO 訓練器
- **`qrl/envs/diffusion_env.py`**: 擴散環境
- **`scripts/train_qrl.py`**: 訓練腳本

### 配置系統
- **Hydra**: 配置管理框架
- **YAML**: 配置文件格式
- **命令行**: 參數覆蓋

## 🎉 下一步

### 立即嘗試
1. ✅ 運行導入測試
2. ✅ 檢查環境設置
3. ✅ 下載數據集
4. 🔄 開始訓練
5. 🔄 生成樣本
6. 🔄 評估結果

### 進階功能
- [ ] 自定義量子電路
- [ ] 新的獎勵函數
- [ ] 多 GPU 訓練
- [ ] 模型部署

### 貢獻指南
1. Fork 項目
2. 創建功能分支
3. 編寫測試
4. 提交 PR

## 📞 獲取幫助

- **文檔**: 查看 `README.md` 和 `PROJECT_STATUS.md`
- **測試**: 運行 `python test_core_imports.py`
- **問題**: 檢查錯誤日誌和常見問題
- **支持**: 提交 Issue 或討論

---

**快速開始完成！** 🎯

現在你已經了解了項目的基本結構和使用方法。建議從運行測試開始，然後逐步嘗試訓練和採樣功能。

祝你好運！ 🚀
