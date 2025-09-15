#!/usr/bin/env python3
"""檢查項目依賴。"""

import importlib
import sys
from pathlib import Path

def check_dependency(package_name, import_name=None, min_version=None):
    """檢查依賴包是否可用。"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安裝")
        return False

def check_optional_dependency(package_name, import_name=None, description=""):
    """檢查可選依賴包。"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version} (可選)")
        return True
    except ImportError:
        print(f"⚠️  {package_name}: 未安裝 {description}")
        return False

def main():
    """主函數。"""
    print("=" * 60)
    print("QRL 圖像合成項目依賴檢查")
    print("=" * 60)
    
    # 核心依賴
    print("\n🔧 核心依賴:")
    core_deps = [
        ("torch", "torch", "PyTorch 深度學習框架"),
        ("torchvision", "torchvision", "PyTorch 計算機視覺"),
        ("numpy", "numpy", "數值計算"),
        ("scipy", "scipy", "科學計算"),
    ]
    
    core_available = True
    for package, import_name, desc in core_deps:
        if not check_dependency(package, import_name):
            core_available = False
    
    # 量子計算依賴
    print("\n⚛️  量子計算依賴:")
    quantum_deps = [
        ("pennylane", "pennylane", "量子機器學習"),
    ]
    
    quantum_available = True
    for package, import_name, desc in quantum_deps:
        if not check_dependency(package, import_name):
            quantum_available = False
    
    # 擴散模型依賴
    print("\n🎨 擴散模型依賴:")
    diffusion_deps = [
        ("diffusers", "diffusers", "HuggingFace 擴散模型"),
        ("transformers", "transformers", "HuggingFace 變換器"),
    ]
    
    diffusion_available = True
    for package, import_name, desc in diffusion_deps:
        if not check_dependency(package, import_name):
            diffusion_available = False
    
    # 強化學習依賴
    print("\n🤖 強化學習依賴:")
    rl_deps = [
        ("gymnasium", "gymnasium", "強化學習環境"),
    ]
    
    rl_available = True
    for package, import_name, desc in rl_deps:
        if not check_dependency(package, import_name):
            rl_available = False
    
    # 指標計算依賴
    print("\n📊 指標計算依賴:")
    metrics_deps = [
        ("pytorch_fid", "pytorch_fid", "FID 計算"),
        ("open_clip_torch", "open_clip_torch", "CLIP 模型"),
        ("lpips", "lpips", "LPIPS 距離"),
    ]
    
    for package, import_name, desc in metrics_deps:
        check_optional_dependency(package, import_name, desc)
    
    # 配置管理依賴
    print("\n⚙️  配置管理依賴:")
    config_deps = [
        ("omegaconf", "omegaconf", "配置管理"),
        ("hydra", "hydra", "配置框架"),
    ]
    
    config_available = True
    for package, import_name, desc in config_deps:
        if not check_dependency(package, import_name):
            config_available = False
    
    # 開發工具依賴
    print("\n🛠️  開發工具依賴:")
    dev_deps = [
        ("pytest", "pytest", "測試框架"),
        ("black", "black", "代碼格式化"),
        ("isort", "isort", "導入排序"),
        ("flake8", "flake8", "代碼檢查"),
    ]
    
    for package, import_name, desc in dev_deps:
        check_optional_dependency(package, import_name, desc)
    
    # 總結
    print("\n" + "=" * 60)
    print("依賴檢查總結:")
    print("=" * 60)
    
    if core_available:
        print("✅ 核心依賴: 已安裝")
    else:
        print("❌ 核心依賴: 缺失")
    
    if quantum_available:
        print("✅ 量子計算依賴: 已安裝")
    else:
        print("❌ 量子計算依賴: 缺失")
    
    if diffusion_available:
        print("✅ 擴散模型依賴: 已安裝")
    else:
        print("❌ 擴散模型依賴: 缺失")
    
    if rl_available:
        print("✅ 強化學習依賴: 已安裝")
    else:
        print("❌ 強化學習依賴: 缺失")
    
    if config_available:
        print("✅ 配置管理依賴: 已安裝")
    else:
        print("❌ 配置管理依賴: 缺失")
    
    # 安裝建議
    print("\n📋 安裝建議:")
    print("=" * 60)
    
    if not core_available:
        print("1. 安裝 PyTorch:")
        print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    
    if not quantum_available:
        print("2. 安裝 PennyLane:")
        print("   pip install pennylane")
    
    if not diffusion_available:
        print("3. 安裝 HuggingFace 擴散模型:")
        print("   pip install diffusers transformers")
    
    if not rl_available:
        print("4. 安裝 Gymnasium:")
        print("   pip install gymnasium")
    
    if not config_available:
        print("5. 安裝配置管理工具:")
        print("   pip install omegaconf hydra-core")
    
    print("\n6. 安裝可選依賴:")
    print("   pip install pytorch-fid open-clip-torch lpips")
    
    print("\n7. 安裝開發工具:")
    print("   pip install pytest black isort flake8")
    
    # 快速安裝命令
    print("\n🚀 快速安裝 (推薦):")
    print("=" * 60)
    print("conda env create -f environment.yml")
    print("conda activate qrl_image_synthesis")
    print("pip install -e .")
    
    return all([core_available, quantum_available, diffusion_available, rl_available, config_available])

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 所有核心依賴已安裝！")
        print("現在可以運行: python test_core_imports.py")
    else:
        print("\n⚠️  部分依賴缺失，請按照建議安裝。")
        sys.exit(1)
