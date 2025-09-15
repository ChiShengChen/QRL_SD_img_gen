#!/usr/bin/env python3
"""測試基本功能，不依賴 gymnasium。"""

import sys
import traceback
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name, description=""):
    """測試模組導入。"""
    try:
        __import__(module_name)
        print(f"✅ {module_name}: {description}")
        return True
    except Exception as e:
        print(f"❌ {module_name}: {description} - {e}")
        return False

def main():
    """主函數。"""
    print("=" * 60)
    print("QRL 圖像合成項目 - 基本功能測試")
    print("=" * 60)
    
    # 測試核心依賴
    print("\n🔧 測試核心依賴:")
    core_deps = [
        ("torch", "PyTorch 深度學習框架"),
        ("numpy", "數值計算"),
        ("scipy", "科學計算"),
        ("pennylane", "量子機器學習"),
        ("transformers", "HuggingFace 變換器"),
        ("omegaconf", "配置管理"),
        ("hydra", "配置框架"),
    ]
    
    core_success = True
    for module, desc in core_deps:
        if not test_import(module, desc):
            core_success = False
    
    # 測試項目核心模組（不依賴 gymnasium）
    print("\n📦 測試項目核心模組:")
    project_modules = [
        ("qrl.utils", "工具模組"),
        ("qrl.metrics.fid", "FID 計算"),
        ("qrl.metrics.inception", "Inception Score"),
        ("qrl.metrics.clipscore", "CLIP Score"),
    ]
    
    project_success = True
    for module, desc in project_modules:
        if not test_import(module, desc):
            project_success = False
    
    # 測試基本功能
    print("\n🧪 測試基本功能:")
    try:
        from qrl.utils import set_seed, get_seed
        set_seed(42)
        seed = get_seed()
        print(f"✅ 種子設置: {seed}")
    except Exception as e:
        print(f"❌ 種子設置: {e}")
        project_success = False
    
    try:
        from qrl.metrics.fid import FIDCalculator
        fid_calc = FIDCalculator()
        print(f"✅ FID 計算器: 創建成功")
    except Exception as e:
        print(f"❌ FID 計算器: {e}")
        project_success = False
    
    try:
        from qrl.metrics.inception import InceptionScoreCalculator
        is_calc = InceptionScoreCalculator()
        print(f"✅ Inception Score 計算器: 創建成功")
    except Exception as e:
        print(f"❌ Inception Score 計算器: {e}")
        project_success = False
    
    try:
        from qrl.metrics.clipscore import CLIPScoreCalculator
        clip_calc = CLIPScoreCalculator()
        print(f"✅ CLIP Score 計算器: 創建成功")
    except Exception as e:
        print(f"❌ CLIP Score 計算器: {e}")
        project_success = False
    
    # 測試量子 Actor（不依賴 gymnasium）
    print("\n⚛️ 測試量子 Actor:")
    try:
        from qrl.actors.quantum_actor import QuantumActor
        import torch
        
        # 創建量子 Actor
        quantum_actor = QuantumActor(
            state_dim=6,
            action_dim=1,
            device="cpu"
        )
        print(f"✅ 量子 Actor: {quantum_actor.num_params} 參數")
        
        # 測試前向傳播
        state = torch.randn(2, 6)
        mu, log_std = quantum_actor(state)
        print(f"✅ 前向傳播: mu.shape={mu.shape}, log_std.shape={log_std.shape}")
        
        # 測試採樣
        action = quantum_actor.sample(state)
        print(f"✅ 採樣: action.shape={action.shape}")
        
        # 測試對數概率
        log_prob = quantum_actor.log_prob(state, action)
        print(f"✅ 對數概率: log_prob.shape={log_prob.shape}")
        
        # 測試熵
        entropy = quantum_actor.entropy(state)
        print(f"✅ 熵: entropy.shape={entropy.shape}")
        
    except Exception as e:
        print(f"❌ 量子 Actor 測試: {e}")
        traceback.print_exc()
        project_success = False
    
    # 測試經典 Actor
    print("\n🤖 測試經典 Actor:")
    try:
        from qrl.actors.classical_actor import ClassicalActor, AlignedClassicalActor
        import torch
        
        # 測試經典 Actor
        classical_actor = ClassicalActor(
            state_dim=6,
            action_dim=1,
            hidden_dims=[64, 32]
        )
        print(f"✅ 經典 Actor: {classical_actor.num_params} 參數")
        
        # 測試對齊的經典 Actor
        aligned_actor = AlignedClassicalActor(
            state_dim=6,
            action_dim=1,
            target_params=quantum_actor.num_params
        )
        print(f"✅ 對齊經典 Actor: {aligned_actor.num_params} 參數")
        
        # 測試前向傳播
        state = torch.randn(2, 6)
        mu, log_std = classical_actor(state)
        print(f"✅ 經典 Actor 前向傳播: mu.shape={mu.shape}, log_std.shape={log_std.shape}")
        
    except Exception as e:
        print(f"❌ 經典 Actor 測試: {e}")
        traceback.print_exc()
        project_success = False
    
    # 測試 Critic
    print("\n🎯 測試 Critic:")
    try:
        from qrl.critics.mlp_critic import MLPCritic
        import torch
        
        # 創建 Critic
        critic = MLPCritic(
            state_dim=6,
            hidden_dims=[64, 32]
        )
        print(f"✅ MLP Critic: {critic.num_params} 參數")
        
        # 測試前向傳播
        state = torch.randn(2, 6)
        value = critic(state)
        print(f"✅ Critic 前向傳播: value.shape={value.shape}")
        
    except Exception as e:
        print(f"❌ Critic 測試: {e}")
        traceback.print_exc()
        project_success = False
    
    # 總結
    print("\n" + "=" * 60)
    print("測試總結:")
    print("=" * 60)
    
    if core_success:
        print("✅ 核心依賴: 正常")
    else:
        print("❌ 核心依賴: 異常")
    
    if project_success:
        print("✅ 項目模組: 正常")
    else:
        print("❌ 項目模組: 異常")
    
    # 建議
    print("\n📋 建議:")
    print("=" * 60)
    
    if not core_success:
        print("1. 安裝核心依賴:")
        print("   pip install torch torchvision numpy scipy pennylane transformers omegaconf hydra-core")
    
    if not project_success:
        print("2. 檢查項目結構是否完整")
        print("3. 確保所有 __init__.py 文件存在")
    
    print("4. 安裝完整環境:")
    print("   conda env create -f environment.yml")
    print("   conda activate qrl_image_synthesis")
    
    print("5. 安裝缺失的依賴:")
    print("   pip install gymnasium diffusers")
    
    return core_success and project_success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 基本功能測試通過！")
        print("現在可以嘗試運行: python check_dependencies.py")
    else:
        print("\n⚠️  部分功能測試失敗，請檢查依賴和項目結構。")
        sys.exit(1)