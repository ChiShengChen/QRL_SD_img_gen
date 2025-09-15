#!/usr/bin/env python3
"""測試核心模組導入。"""

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
    print("QRL 圖像合成項目 - 核心模組導入測試")
    print("=" * 60)
    
    # 測試核心依賴
    print("\n🔧 測試核心依賴:")
    core_deps = [
        ("torch", "PyTorch 深度學習框架"),
        ("numpy", "數值計算"),
        ("scipy", "科學計算"),
    ]
    
    core_success = True
    for module, desc in core_deps:
        if not test_import(module, desc):
            core_success = False
    
    # 測試項目核心模組
    print("\n📦 測試項目核心模組:")
    project_modules = [
        ("qrl.utils", "工具模組"),
        ("qrl.controls.control_spaces", "控制空間"),
        ("qrl.controls.apply_controls", "控制應用"),
        ("qrl.reward.classifier_reward", "分類器獎勵"),
        ("qrl.actors.quantum_actor", "量子 Actor"),
        ("qrl.actors.classical_actor", "經典 Actor"),
        ("qrl.critics.mlp_critic", "MLP Critic"),
        ("qrl.training.buffers", "經驗緩衝區"),
        ("qrl.training.ppo_trainer", "PPO 訓練器"),
        ("qrl.metrics.fid", "FID 計算"),
        ("qrl.metrics.inception", "Inception Score"),
        ("qrl.metrics.clipscore", "CLIP Score"),
    ]
    
    project_success = True
    for module, desc in project_modules:
        if not test_import(module, desc):
            project_success = False
    
    # 測試可選依賴
    print("\n🔍 測試可選依賴:")
    optional_deps = [
        ("gymnasium", "強化學習環境"),
        ("diffusers", "HuggingFace 擴散模型"),
        ("pennylane", "量子機器學習"),
        ("transformers", "HuggingFace 變換器"),
        ("omegaconf", "配置管理"),
        ("hydra", "配置框架"),
    ]
    
    for module, desc in optional_deps:
        test_import(module, desc)
    
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
        from qrl.controls.control_spaces import ControlSpaces
        action_space = ControlSpaces.stageA_cfg()
        print(f"✅ 動作空間: {action_space}")
    except Exception as e:
        print(f"❌ 動作空間: {e}")
        project_success = False
    
    try:
        from qrl.actors.quantum_actor import QuantumActor
        from qrl.actors.classical_actor import ClassicalActor, AlignedClassicalActor
        
        # 測試量子 Actor
        quantum_actor = QuantumActor(
            state_dim=6,
            action_dim=1,
            device="cpu"
        )
        print(f"✅ 量子 Actor: {quantum_actor.num_params} 參數")
        
        # 測試經典 Actor
        classical_actor = ClassicalActor(
            state_dim=6,
            action_dim=1,
            hidden_dim=64
        )
        print(f"✅ 經典 Actor: {classical_actor.num_params} 參數")
        
        # 測試對齊的經典 Actor
        aligned_actor = AlignedClassicalActor(
            state_dim=6,
            action_dim=1,
            target_params=quantum_actor.num_params
        )
        print(f"✅ 對齊經典 Actor: {aligned_actor.num_params} 參數")
        
    except Exception as e:
        print(f"❌ Actor 測試: {e}")
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
        print("   pip install torch torchvision numpy scipy")
    
    if not project_success:
        print("2. 檢查項目結構是否完整")
        print("3. 確保所有 __init__.py 文件存在")
    
    print("4. 安裝完整環境:")
    print("   conda env create -f environment.yml")
    print("   conda activate qrl_image_synthesis")
    
    return core_success and project_success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 核心模組測試通過！")
        print("現在可以嘗試運行: python check_dependencies.py")
    else:
        print("\n⚠️  部分模組測試失敗，請檢查依賴和項目結構。")
        sys.exit(1)