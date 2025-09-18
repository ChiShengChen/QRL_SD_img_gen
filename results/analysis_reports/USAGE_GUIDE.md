# 量子 vs 經典模型比較使用指南

## 🚀 快速開始

### 1. 測試指標功能
```bash
python test_metrics_only.py
```
**用途**: 驗證所有指標計算功能是否正常
**時間**: ~30秒
**輸出**: 22個指標的測試結果

### 2. 快速比較 (原有功能)
```bash
python run_quick_comparison.py
```
**用途**: 基本性能比較
**時間**: ~5分鐘
**輸出**: 4個基本指標

### 3. 全面比較 (新功能)
```bash
python run_comprehensive_comparison.py
```
**用途**: 完整的28個指標比較
**時間**: ~15分鐘
**要求**: CUDA支持
**輸出**: 全面的比較結果

### 4. 簡化版全面比較 (新功能)
```bash
python run_simple_comprehensive_comparison.py
```
**用途**: 16個核心指標比較
**時間**: ~10分鐘
**要求**: CPU即可
**輸出**: 核心比較結果

## 📊 指標說明

### 基本性能指標
- **參數數量**: 模型總參數數 (越少越好)
- **平均獎勵**: 任務表現 (越高越好)
- **平均時間/Episode**: 推理速度 (越快越好)
- **穩定性**: 輸出一致性 (標準差越小越好)

### 控制性能指標
- **收斂速度**: 達到穩定性能的速度 (越快越好)
- **CFG平滑度**: 控制軌跡的平滑性 (越高越好)
- **策略穩定性**: 學習過程的穩定性 (越高越好)
- **CFG效率**: 接近目標CFG的程度 (越高越好)

### 效率指標
- **參數效率**: 性能/參數比 (越高越好)
- **推理效率**: 推理速度效率 (越高越好)
- **記憶體效率**: 記憶體使用效率 (越高越好)
- **訓練效率**: 學習速度效率 (越高越好)

### 量子特定指標
- **量子相干性**: 量子態的相干性保持 (越高越好)
- **量子優勢**: 相對於經典模型的優勢程度 (越高越好)
- **量子糾纏**: 量子電路中的糾纏程度 (越高越好)
- **量子門效率**: 量子門的使用效率 (越高越好)

## 🎯 結果解讀

### 性能改善百分比
- **正值**: 量子模型優於經典模型
- **負值**: 經典模型優於量子模型
- **0%**: 兩者性能相同

### 效率比
- **>1**: 量子模型效率更高
- **<1**: 經典模型效率更高
- **=1**: 兩者效率相同

### 量子特定指標
- **0.0000**: 該特性在當前配置下未激活
- **>0**: 該特性有貢獻
- **N/A**: 經典模型不適用

## 📈 典型結果示例

```
📊 量子 vs 經典模型全面比較結果
================================================================================

🎯 基本性能指標
------------------------------------------------------------
指標                     量子模型          經典模型          改善            
------------------------------------------------------------
參數數量                 2498            2594            量子 1.04x       
平均獎勵                 -2.9000         -4.1000         +29.3%        
平均時間/Episode         1.47s           1.37s           量子 0.93x       
穩定性 (std)            0.1732          0.1732          相同       

🎮 控制性能指標
------------------------------------------------------------
CFG平滑度                0.8333          0.8000          量子           
策略穩定性               0.8524          0.8200          量子           

⚡ 效率指標
------------------------------------------------------------
參數效率                 -0.0012         -0.0016         量子           
推理效率                 29.4266         25.0000         量子           
記憶體效率                104.9414        100.0000        量子           

🔬 量子特定指標
------------------------------------------------------------
量子相干性                0.0000          N/A             量子專有       
量子優勢                  0.4361          N/A             量子專有       
量子糾纏                  0.0000          N/A             量子專有       

🎯 總結:
✅ 量子模型在基本性能上優於經典模型
✅ 量子模型參數效率更高
```

## 🔧 自定義配置

### 修改實驗參數
在腳本中修改以下參數：

```python
# 實驗回合數
num_episodes = 50  # 減少以節省時間

# 每回合最大步數
max_steps = 50

# 目標類別
target_class = 3  # CIFAR-10 類別

# 計算設備
device = "cuda"  # 或 "cpu"
```

### 修改模型參數
```python
# 量子模型
quantum_actor = QuantumActor(
    state_dim=6,
    action_dim=1,
    n_qubits=4,      # 量子比特數
    n_layers=2,      # 電路層數
    device="cuda"
)

# 經典模型
classical_actor = ClassicalActor(
    state_dim=6,
    action_dim=1,
    hidden_dims=[64, 32],  # 隱藏層維度
    device="cuda"
)
```

## 📁 輸出文件

### 結果文件
- `quick_comparison_results.json`: 快速比較結果
- `comprehensive_comparison_results.json`: 全面比較結果
- `simple_comprehensive_comparison_results.json`: 簡化比較結果

### 日誌文件
- `logs/diffusion_env_*.log`: 環境運行日誌

## 🐛 常見問題

### 1. CUDA錯誤
```
AssertionError: Torch not compiled with CUDA enabled
```
**解決方案**: 使用CPU版本或安裝CUDA版本的PyTorch

### 2. 參數錯誤
```
TypeError: __init__() got an unexpected keyword argument
```
**解決方案**: 檢查模型初始化參數是否正確

### 3. 記憶體不足
```
RuntimeError: CUDA out of memory
```
**解決方案**: 減少batch_size或使用CPU版本

### 4. 依賴缺失
```
ModuleNotFoundError: No module named 'xxx'
```
**解決方案**: 安裝缺失的依賴包

## 📊 數據分析

### 使用Python分析結果
```python
import json
import matplotlib.pyplot as plt

# 載入結果
with open('comprehensive_comparison_results.json', 'r') as f:
    results = json.load(f)

# 提取數據
quantum_rewards = results['quantum']['rewards']
classical_rewards = results['classical']['rewards']

# 繪製比較圖
plt.figure(figsize=(10, 6))
plt.plot(quantum_rewards, label='Quantum', alpha=0.7)
plt.plot(classical_rewards, label='Classical', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.title('Quantum vs Classical Performance')
plt.show()
```

## 🎯 最佳實踐

### 1. 實驗設計
- 使用相同的隨機種子確保可重現性
- 多次運行取平均值
- 記錄環境配置和硬體信息

### 2. 結果分析
- 關注統計顯著性
- 考慮多個指標的綜合評估
- 分析量子特定指標的貢獻

### 3. 報告撰寫
- 明確說明實驗設置
- 提供完整的指標結果
- 討論量子優勢的具體表現

## 📚 參考文檔

- `COMPARISON_METRICS.md`: 詳細指標說明
- `COMPARISON_SUMMARY.md`: 功能總結
- `README.md`: 項目概述

## 🤝 貢獻

歡迎提交Issue和Pull Request來改進比較功能！

## 📄 許可證

請查看項目根目錄的LICENSE文件。
