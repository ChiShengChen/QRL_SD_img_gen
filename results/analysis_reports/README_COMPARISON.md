# 量子 vs 經典比較實驗

本項目實現了量子機器學習與經典機器學習在圖像生成控制任務中的全面比較實驗。

## 🎯 實驗目標

證明量子機器學習在圖像生成控制任務中的優勢，評估量子方法的實際應用價值。

## 🧪 實驗設計

### 對照組設置

| 方法 | 架構 | 參數數量 | 特點 |
|------|------|----------|------|
| **量子 Actor** | 8量子比特VQC + 經典後處理 | 2,850 | 量子糾纏 + 經典優化 |
| **經典 Actor** | 2層MLP [128,64] | ~10,000 | 純經典神經網路 |
| **對齊經典 Actor** | 參數匹配的MLP | 2,850 | 公平參數比較 |

### 基線對比
- **固定CFG=7.5**：無動態控制
- **隨機CFG**：隨機控制策略
- **線性CFG**：線性變化策略

## 📊 評估指標

### A. 生成質量指標
- **分類準確率**：ResNet18在CIFAR-10上的準確率
- **FID分數**：生成圖像與真實圖像的分布距離
- **IS分數**：生成圖像的質量和多樣性
- **LPIPS**：感知相似度，評估視覺質量

### B. 控制性能指標
- **回合總獎勵**：訓練過程中的累積獎勵
- **收斂速度**：達到穩定性能所需的episode數
- **CFG平滑度**：動態CFG變化的平滑度
- **策略穩定性**：策略的穩定性

### C. 效率指標
- **參數效率**：性能/參數數量比率
- **訓練效率**：達到目標性能的訓練時間
- **推理效率**：單次推理的計算時間
- **記憶體效率**：模型大小和運行記憶體

### D. 量子特定指標
- **表達能力**：相同參數下的模型容量
- **泛化能力**：跨類別的泛化性能
- **魯棒性**：對噪聲和擾動的抵抗能力
- **可解釋性**：量子電路的可解釋性

## 🚀 快速開始

### 1. 環境準備

```bash
# 安裝依賴
pip install -r requirements.txt

# 或使用conda
conda env create -f environment.yml
conda activate qrl_image_synthesis
```

### 2. 運行完整比較實驗

```bash
# 運行所有模型的訓練和評估
python run_comparison.py --output comparison_results

# 只訓練特定模型
python run_comparison.py --models-only quantum classical

# 跳過訓練，直接評估現有模型
python run_comparison.py --skip-training
```

### 3. 單獨訓練模型

```bash
# 訓練量子模型
python train_comparison.py --config configs/quantum_config.yaml

# 訓練經典模型
python train_comparison.py --config configs/classical_config.yaml

# 訓練對齊經典模型
python train_comparison.py --config configs/aligned_classical_config.yaml
```

### 4. 評估模型

```bash
# 評估多個模型
python eval_comparison.py \
    --models "quantum:runs/quantum_comparison/ckpt_best.pt:quantum,classical:runs/classical_comparison/ckpt_best.pt:classical" \
    --output evaluation_results
```

## 📁 項目結構

```
qrl_image_synthesis/
├── configs/                          # 配置文件
│   ├── quantum_config.yaml          # 量子模型配置
│   ├── classical_config.yaml        # 經典模型配置
│   └── aligned_classical_config.yaml # 對齊經典模型配置
├── qrl/
│   ├── metrics/                      # 評估指標
│   │   ├── image_quality.py         # 圖像質量指標
│   │   ├── control_performance.py   # 控制性能指標
│   │   └── efficiency.py            # 效率指標
│   └── ...
├── train_comparison.py              # 訓練腳本
├── eval_comparison.py               # 評估腳本
├── run_comparison.py                # 完整實驗腳本
└── README_COMPARISON.md             # 本文檔
```

## 🔬 實驗流程

### 1. 預訓練階段
- 所有模型使用相同種子初始化
- 相同的訓練環境和超參數
- 記錄訓練過程中的所有指標

### 2. 評估階段
- 在10個CIFAR-10類別上分別測試
- 每個類別生成100張圖像
- 計算所有評估指標

### 3. 統計分析
- 多次運行取平均值和標準差
- 統計顯著性檢驗 (t-test)
- 效應量分析 (Cohen's d)

## 📈 結果分析

### 成功標準

量子方法需要證明以下至少一項：
- ✅ **參數效率提升 > 20%**
- ✅ **生成質量提升 > 5%**
- ✅ **收斂速度提升 > 15%**
- ✅ **泛化能力提升 > 10%**

### 結果可視化

實驗完成後會生成以下可視化結果：

1. **主要指標對比圖** (`main_metrics_comparison.png`)
   - 分類準確率
   - FID分數
   - 參數效率
   - 回合獎勵

2. **雷達圖** (`radar_chart.png`)
   - 綜合性能比較
   - 多維度評估

3. **熱力圖** (`performance_heatmap.png`)
   - 詳細指標對比
   - 標準化分數

4. **統計報告** (`statistical_report.json`)
   - 詳細統計分析
   - 最佳表現者

## 🎯 預期結果

### 量子方法可能的優勢

1. **參數效率**：更少的參數達到相同性能
2. **表達能力**：量子糾纏提供更豐富的表達
3. **泛化能力**：量子結構的內在泛化性
4. **可解釋性**：量子電路結構更易理解

### 可能的挑戰

1. **訓練穩定性**：量子梯度可能不穩定
2. **計算開銷**：量子電路模擬的額外成本
3. **超參數敏感**：量子電路對超參數更敏感

## 🔧 自定義配置

### 修改模型參數

編輯對應的配置文件：

```yaml
# configs/quantum_config.yaml
quantum_actor:
  n_qubits: 8          # 量子比特數量
  n_layers: 4          # 變分層數
  backend: "default.qubit"  # 量子後端
```

### 修改評估指標

在配置文件中添加或移除指標：

```yaml
evaluation:
  metrics:
    - classification_accuracy
    - fid_score
    - inception_score
    - lpips
    - parameter_efficiency
    - convergence_speed
```

## 📊 結果解讀

### 關鍵指標說明

- **FID分數**：越低越好，表示生成圖像與真實圖像越相似
- **IS分數**：越高越好，表示生成圖像質量越高且多樣性越好
- **參數效率**：越高越好，表示用更少參數達到更好性能
- **收斂速度**：越高越好，表示訓練收斂越快

### 統計顯著性

實驗結果會包含統計顯著性檢驗，幫助判斷量子方法的優勢是否具有統計學意義。

## 🐛 故障排除

### 常見問題

1. **記憶體不足**
   ```bash
   # 減少批次大小
   python train_comparison.py --config configs/quantum_config.yaml --override algo.ppo.mini_batch_size=4
   ```

2. **訓練不穩定**
   ```bash
   # 降低學習率
   python train_comparison.py --config configs/quantum_config.yaml --override algo.ppo.lr_actor=5e-5
   ```

3. **評估失敗**
   ```bash
   # 檢查模型路徑
   ls -la runs/*/ckpt_*.pt
   ```

## 📚 參考文獻

1. Quantum Machine Learning for Image Generation
2. Variational Quantum Circuits for Reinforcement Learning
3. Stable Diffusion with Dynamic Guidance Control
4. PPO Algorithm for Quantum-Enhanced Control

## 🤝 貢獻

歡迎提交Issue和Pull Request來改進這個比較實驗框架。

## 📄 許可證

MIT License
