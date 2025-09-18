# 量子 vs 經典模型比較指標

本文檔列出了可用於比較量子與經典模型的所有評估指標。

## 📊 基本性能指標

### 1. 性能指標
- **平均獎勵** (Average Reward): 模型在任務中的平均表現
- **獎勵標準差** (Reward Std): 表現的穩定性
- **收斂速度** (Convergence Speed): 達到穩定性能的速度
- **策略穩定性** (Policy Stability): 輸出的一致性

### 2. 效率指標
- **參數數量** (Parameter Count): 模型總參數數
- **參數效率** (Parameter Efficiency): 性能/參數比
- **推理速度** (Inference Speed): 單次推理時間
- **記憶體效率** (Memory Efficiency): 記憶體使用量
- **訓練效率** (Training Efficiency): 學習速度

## 📸 圖像質量指標

### 1. 真實性指標
- **FID分數** (Fréchet Inception Distance): 生成圖像與真實圖像的距離
- **IS分數** (Inception Score): 圖像多樣性和質量
- **分類準確率** (Classification Accuracy): 目標類別識別準確性

### 2. 感知指標
- **LPIPS** (Learned Perceptual Image Patch Similarity): 感知相似性
- **SSIM** (Structural Similarity Index): 結構相似性
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比

## 🎮 控制性能指標

### 1. 控制軌跡指標
- **CFG平滑度** (CFG Smoothness): 控制軌跡的平滑性
- **CFG效率** (CFG Efficiency): 接近目標CFG的程度
- **CFG範圍** (CFG Range): CFG值的變化範圍
- **控制一致性** (Control Consistency): 不同軌跡間的相似性

### 2. 學習動態指標
- **收斂速度** (Convergence Speed): 達到穩定性能的速度
- **學習穩定性** (Learning Stability): 學習過程的穩定性
- **動作多樣性** (Action Diversity): 動作的多樣性

## 🔬 量子特定指標

### 1. 量子特性指標
- **量子相干性** (Quantum Coherence): 量子態的相干性保持
- **量子糾纏** (Quantum Entanglement): 量子電路中的糾纏程度
- **量子門效率** (Quantum Gate Efficiency): 量子門的使用效率
- **量子表達能力** (Quantum Expressibility): 量子模型的表達能力

### 2. 量子優勢指標
- **量子優勢** (Quantum Advantage): 相對於經典模型的優勢程度
- **量子噪聲魯棒性** (Quantum Noise Robustness): 對量子噪聲的抵抗能力
- **量子梯度質量** (Quantum Gradient Quality): 量子梯度的質量和穩定性
- **量子學習動態** (Quantum Learning Dynamics): 量子模型的學習動態

### 3. 量子效率指標
- **量子參數比例** (Quantum Parameter Ratio): 量子參數佔總參數的比例
- **量子計算效率** (Quantum Computational Efficiency): 量子計算的效率
- **量子能源效率** (Quantum Energy Efficiency): 量子計算的能源效率

## ⚖️ 比較指標

### 1. 性能比較
- **性能改善** (Performance Improvement): 量子模型相對於經典模型的性能提升
- **參數效率比** (Parameter Efficiency Ratio): 參數效率的比較
- **速度改善** (Speed Improvement): 推理速度的比較
- **記憶體效率比** (Memory Efficiency Ratio): 記憶體效率的比較

### 2. 可擴展性指標
- **可擴展性** (Scalability): 不同批次大小的處理能力
- **吞吐量** (Throughput): 單位時間處理的樣本數
- **並行效率** (Parallel Efficiency): 並行處理的效率

## 🚀 使用方法

### 1. 快速比較
```bash
python run_quick_comparison.py
```

### 2. 全面比較
```bash
python run_comprehensive_comparison.py
```

### 3. 測試新指標
```bash
python test_new_metrics.py
```

## 📈 指標解釋

### 數值越大越好的指標
- 平均獎勵、分類準確率、IS分數
- 參數效率、推理效率、記憶體效率
- 量子相干性、量子糾纏、量子表達能力
- 性能改善、速度改善

### 數值越小越好的指標
- FID分數、LPIPS、獎勵標準差
- 推理時間、記憶體使用量
- 收斂時間

### 範圍在0-1的指標
- 分類準確率、CFG效率、控制一致性
- 量子門效率、量子參數比例
- 各種效率指標

## 🎯 建議的評估流程

1. **基本性能**: 先比較平均獎勵和穩定性
2. **效率分析**: 比較參數效率和推理速度
3. **圖像質量**: 評估生成圖像的質量
4. **控制性能**: 分析控制軌跡的質量
5. **量子特性**: 評估量子特定的優勢
6. **綜合比較**: 綜合所有指標得出結論

## 📝 注意事項

1. 某些指標需要特定的硬體支持（如量子計算機）
2. 指標計算可能需要較長時間，建議使用較小的測試集
3. 不同指標的權重可以根據具體應用場景調整
4. 建議多次運行取平均值以提高結果的可靠性
