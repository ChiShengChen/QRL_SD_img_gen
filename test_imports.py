#!/usr/bin/env python3
"""測試所有模組的導入。"""

import sys
from pathlib import Path

def test_imports():
    """測試所有模組的導入。"""
    print("開始測試模組導入...")
    
    try:
        # 測試基礎模組
        print("測試基礎模組...")
        from qrl.utils import seed, logging
        print("✅ 基礎模組導入成功")
        
        # 測試控制模組
        print("測試控制模組...")
        from qrl.controls import ControlSpaces, ControlApplier
        print("✅ 控制模組導入成功")
        
        # 測試獎勵模組
        print("測試獎勵模組...")
        from qrl.reward import ClassifierReward, DiversityReward
        print("✅ 獎勵模組導入成功")
        
        # 測試 Actor 模組
        print("測試 Actor 模組...")
        from qrl.actors import QuantumActor, ClassicalActor, AlignedClassicalActor
        print("✅ Actor 模組導入成功")
        
        # 測試 Critic 模組
        print("測試 Critic 模組...")
        from qrl.critics import MLPCritic
        print("✅ Critic 模組導入成功")
        
        # 測試訓練模組
        print("測試訓練模組...")
        from qrl.training import PPOTrainer, PPOBuffer
        print("✅ 訓練模組導入成功")
        
        # 測試環境模組
        print("測試環境模組...")
        from qrl.envs import create_diffusion_env
        print("✅ 環境模組導入成功")
        
        # 測試模型模組
        print("測試模型模組...")
        from qrl.models import UNetLite
        print("✅ 模型模組導入成功")
        
        # 測試指標模組
        print("測試指標模組...")
        from qrl.metrics import (
            FIDCalculator,
            InceptionScoreCalculator,
            CLIPScoreCalculator
        )
        print("✅ 指標模組導入成功")
        
        print("\n🎉 所有模組導入成功！")
        return True
        
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他錯誤: {e}")
        return False


def test_basic_functionality():
    """測試基本功能。"""
    print("\n開始測試基本功能...")
    
    try:
        # 測試種子設置
        from qrl.utils.seed import set_seed
        set_seed(42)
        print("✅ 種子設置成功")
        
        # 測試日誌記錄器
        from qrl.utils.logging import QRLLogger
        logger = QRLLogger("test")
        logger.info("測試日誌記錄")
        print("✅ 日誌記錄器創建成功")
        
        # 測試控制空間
        from qrl.controls import ControlSpaces
        action_space = ControlSpaces.stageA_cfg()
        print(f"✅ Stage A 動作空間創建成功: {action_space}")
        
        # 測試量子 Actor
        from qrl.actors import QuantumActor
        actor = QuantumActor(state_dim=32, action_dim=1, device="cpu")
        print(f"✅ 量子 Actor 創建成功，參數數量: {actor.total_params}")
        
        # 測試古典 Actor
        from qrl.actors import ClassicalActor
        classical_actor = ClassicalActor(state_dim=32, action_dim=1, device="cpu")
        print(f"✅ 古典 Actor 創建成功，參數數量: {classical_actor.total_params}")
        
        # 測試對齊古典 Actor
        from qrl.actors import AlignedClassicalActor
        target_params = actor.total_params
        aligned_actor = AlignedClassicalActor(
            state_dim=32, action_dim=1, target_params=target_params, device="cpu"
        )
        print(f"✅ 對齊古典 Actor 創建成功，參數數量: {aligned_actor.total_params}")
        
        # 測試 Critic
        from qrl.critics import MLPCritic
        critic = MLPCritic(state_dim=32, device="cpu")
        print(f"✅ Critic 創建成功，參數數量: {critic.total_params}")
        
        # 測試 PPO Buffer
        from qrl.training import PPOBuffer
        buffer = PPOBuffer(state_dim=32, action_dim=1, device="cpu")
        print("✅ PPO Buffer 創建成功")
        
        print("\n🎉 所有基本功能測試成功！")
        return True
        
    except Exception as e:
        print(f"❌ 基本功能測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函數。"""
    print("=" * 50)
    print("QRL 圖像合成項目模組導入測試")
    print("=" * 50)
    
    # 測試導入
    import_success = test_imports()
    
    if import_success:
        # 測試基本功能
        functionality_success = test_basic_functionality()
        
        if functionality_success:
            print("\n" + "=" * 50)
            print("✅ 所有測試通過！項目準備就緒。")
            print("=" * 50)
            return True
        else:
            print("\n" + "=" * 50)
            print("❌ 基本功能測試失敗。")
            print("=" * 50)
            return False
    else:
        print("\n" + "=" * 50)
        print("❌ 模組導入測試失敗。")
        print("=" * 50)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
