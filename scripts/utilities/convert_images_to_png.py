#!/usr/bin/env python3
"""
將 .pt 格式的圖像轉換為 .png 格式
用於可視化生成的圖像和真實圖像
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def convert_tensor_to_pil(tensor):
    """將PyTorch tensor轉換為PIL Image"""
    # 確保tensor在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 移除batch dimension如果存在
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # 確保值在[0,1]範圍內
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    # 轉換為numpy array
    if len(tensor.shape) == 3:
        # RGB圖像
        array = tensor.permute(1, 2, 0).numpy()
    else:
        # 灰度圖像
        array = tensor.numpy()
    
    # 確保值在[0,1]範圍內
    array = np.clip(array, 0, 1)
    
    # 轉換為PIL Image
    if len(array.shape) == 3:
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)
    else:
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array, mode='L')

def convert_images_in_directory(input_dir, output_dir, prefix="img"):
    """轉換目錄中的所有.pt文件為.png文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    
    for pt_file in input_path.glob("*.pt"):
        try:
            # 加載tensor
            tensor = torch.load(pt_file, map_location='cpu')
            
            # 轉換為PIL Image
            pil_image = convert_tensor_to_pil(tensor)
            
            # 保存為PNG
            png_filename = pt_file.stem + ".png"
            png_path = output_path / png_filename
            pil_image.save(png_path)
            
            converted_count += 1
            print(f"✅ 轉換: {pt_file.name} -> {png_filename}")
            
        except Exception as e:
            print(f"❌ 轉換失敗 {pt_file.name}: {e}")
    
    print(f"📊 完成轉換 {converted_count} 個文件")
    return converted_count

def create_comparison_grid(quantum_dir, classical_dir, real_dir, output_path, class_name, num_images=4):
    """Create comparison grid for quantum vs classical vs real images"""
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    fig.suptitle(f'{class_name} Class Comparison: Quantum vs Classical vs Real', fontsize=16)
    
    # Row titles
    row_titles = ['Quantum Model', 'Classical Model', 'Real Images']
    for i, title in enumerate(row_titles):
        axes[i, 0].text(-0.1, 0.5, title, transform=axes[i, 0].transAxes, 
                       rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
    
    # Load and display images
    for col in range(num_images):
        # Quantum images
        quantum_path = Path(quantum_dir) / f"gen_{col}.png"
        if quantum_path.exists():
            quantum_img = Image.open(quantum_path)
            axes[0, col].imshow(quantum_img)
        axes[0, col].set_title(f'Quantum {col}')
        axes[0, col].axis('off')
        
        # Classical images
        classical_path = Path(classical_dir) / f"gen_{col}.png"
        if classical_path.exists():
            classical_img = Image.open(classical_path)
            axes[1, col].imshow(classical_img)
        axes[1, col].set_title(f'Classical {col}')
        axes[1, col].axis('off')
        
        # Real images
        real_path = Path(real_dir) / f"real_{col}.png"
        if real_path.exists():
            real_img = Image.open(real_path)
            axes[2, col].imshow(real_img)
        axes[2, col].set_title(f'Real {col}')
        axes[2, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 比較網格已保存: {output_path}")

def main():
    """主函數"""
    print("🖼️ 開始轉換 .pt 圖像為 .png 格式...")
    print("="*60)
    
    # 設置路徑
    base_dir = Path("improved_comparison_results")
    generated_dir = base_dir / "generated_images"
    real_dir = base_dir / "real_images"
    
    # 輸出目錄
    png_output_dir = base_dir / "png_images"
    quantum_png_dir = png_output_dir / "quantum"
    classical_png_dir = png_output_dir / "classical"
    real_png_dir = png_output_dir / "real"
    comparison_dir = png_output_dir / "comparisons"
    
    # 創建輸出目錄
    for dir_path in [quantum_png_dir, classical_png_dir, real_png_dir, comparison_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # CIFAR-10類別
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    total_converted = 0
    
    # 轉換每個類別的圖像
    for class_name in cifar10_classes:
        print(f"\n🔄 處理 {class_name} 類別...")
        
        # 轉換量子模型圖像
        quantum_input = generated_dir / "quantum" / class_name
        quantum_output = quantum_png_dir / class_name
        quantum_output.mkdir(exist_ok=True)
        if quantum_input.exists():
            count = convert_images_in_directory(quantum_input, quantum_output, "quantum")
            total_converted += count
        
        # 轉換經典模型圖像
        classical_input = generated_dir / "classical" / class_name
        classical_output = classical_png_dir / class_name
        classical_output.mkdir(exist_ok=True)
        if classical_input.exists():
            count = convert_images_in_directory(classical_input, classical_output, "classical")
            total_converted += count
        
        # 轉換真實圖像
        real_input = real_dir / class_name
        real_output = real_png_dir / class_name
        real_output.mkdir(exist_ok=True)
        if real_input.exists():
            count = convert_images_in_directory(real_input, real_output, "real")
            total_converted += count
        
        # 創建比較網格
        try:
            comparison_path = comparison_dir / f"{class_name}_comparison.png"
            create_comparison_grid(
                quantum_output, classical_output, real_output, 
                comparison_path, class_name, num_images=4
            )
        except Exception as e:
            print(f"❌ 創建 {class_name} 比較網格失敗: {e}")
    
    # Create overall comparison grid
    print(f"\n📊 Creating overall comparison grid...")
    try:
        fig, axes = plt.subplots(3, 10, figsize=(20, 6))
        fig.suptitle('All Classes Comparison: Quantum vs Classical vs Real', fontsize=16)
        
        for i, class_name in enumerate(cifar10_classes):
            # Quantum images
            quantum_path = quantum_png_dir / class_name / "gen_0.png"
            if quantum_path.exists():
                quantum_img = Image.open(quantum_path)
                axes[0, i].imshow(quantum_img)
            axes[0, i].set_title(f'{class_name}\nQuantum', fontsize=8)
            axes[0, i].axis('off')
            
            # Classical images
            classical_path = classical_png_dir / class_name / "gen_0.png"
            if classical_path.exists():
                classical_img = Image.open(classical_path)
                axes[1, i].imshow(classical_img)
            axes[1, i].set_title(f'{class_name}\nClassical', fontsize=8)
            axes[1, i].axis('off')
            
            # Real images
            real_path = real_png_dir / class_name / "real_0.png"
            if real_path.exists():
                real_img = Image.open(real_path)
                axes[2, i].imshow(real_img)
            axes[2, i].set_title(f'{class_name}\nReal', fontsize=8)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        overall_comparison_path = comparison_dir / "overall_comparison.png"
        plt.savefig(overall_comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 總體比較網格已保存: {overall_comparison_path}")
        
    except Exception as e:
        print(f"❌ 創建總體比較網格失敗: {e}")
    
    print(f"\n✅ 轉換完成！")
    print(f"📊 總共轉換了 {total_converted} 個圖像")
    print(f"📁 PNG圖像保存在: {png_output_dir}")
    print(f"📁 比較網格保存在: {comparison_dir}")

if __name__ == "__main__":
    main()
