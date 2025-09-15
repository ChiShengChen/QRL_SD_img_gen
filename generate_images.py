#!/usr/bin/env python3
"""簡單的圖像生成腳本。"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qrl.utils.seed import set_seed
from qrl.envs import create_diffusion_env
from qrl.actors import QuantumActor
from qrl.critics import MLPCritic
from qrl.training import QuantumPPOTrainer


def generate_images(
    checkpoint_path: str,
    num_images: int = 10,
    target_class: int = 3,
    output_dir: str = "generated_images",
    device: str = "cuda"
):
    """生成圖像。
    
    Args:
        checkpoint_path: 檢查點文件路徑
        num_images: 生成圖像數量
        target_class: 目標類別 (0-9)
        output_dir: 輸出目錄
        device: 設備
    """
    # 設置種子
    set_seed(42)
    
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 設置設備
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    try:
        # 創建環境
        print("創建擴散環境...")
        env = create_diffusion_env(
            model_id="runwayml/stable-diffusion-v1-5",
            target_class=target_class,
            max_steps=50,
            stage="A",
            device=device,
            reward_config={
                'alpha_cls': 1.0,
                'beta_step': 0.2,
                'action_penalty': 0.005,
                'tv_penalty': 0.0
            },
            control_config={
                'stage': 'A',
                'delta_cfg': {'min': -2.0, 'max': 2.0, 'base_cfg': 5.0},
                'environment': {'max_steps': 50}
            }
        )
        
        # 創建 Actor 和 Critic
        print("創建模型...")
        actor = QuantumActor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device
        )
        
        critic = MLPCritic(
            state_dim=env.observation_space.shape[0],
            device=device
        )
        
        # 創建訓練器
        trainer = QuantumPPOTrainer(
            actor=actor,
            critic=critic,
            device=device,
            lr_actor=0.0001,
            lr_critic=0.001,
            gamma=0.995,
            gae_lambda=0.95,
            clip_ratio=0.1,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=4,
            mini_batch_size=8,
            rollout_steps=4,
            num_envs=8,
            mixed_precision=True,
            use_quantum_gradients=False
        )
        
        # 載入檢查點
        print(f"載入檢查點: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        
        # 設置為評估模式
        actor.eval()
        critic.eval()
        
        # 生成圖像
        print(f"開始生成 {num_images} 張圖像...")
        generated_images = []
        
        with torch.no_grad():
            for i in range(num_images):
                print(f"生成圖像 {i+1}/{num_images}...")
                
                # 重置環境
                obs, _ = env.reset()
                
                # 生成過程
                for step in range(50):  # 最大50步
                    # 獲取動作
                    action = actor.sample(torch.FloatTensor(obs).unsqueeze(0).to(device))
                    action = action.cpu().numpy()[0]
                    
                    # 執行動作
                    obs, reward, done, truncated, info = env.step(action)
                    
                    if done or truncated:
                        break
                
                # 獲取最終圖像
                final_image = env._decode_latent()
                if final_image is not None:
                    generated_images.append(final_image)
        
        # 保存圖像
        print(f"保存圖像到 {output_path}...")
        for i, image in enumerate(generated_images):
            # 轉換為 PIL 圖像
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            # 確保圖像在 [0, 1] 範圍內
            if image.max() > 1.0:
                image = image / 255.0
            
            # 處理不同的圖像格式
            if len(image.shape) == 4:  # [B, C, H, W]
                image = image[0]  # 取第一個批次
            
            if len(image.shape) == 3:
                if image.shape[0] == 3 or image.shape[0] == 1:  # CHW 格式
                    image = np.transpose(image, (1, 2, 0))  # 轉換為 HWC
                elif image.shape[2] == 3 or image.shape[2] == 1:  # 已經是 HWC 格式
                    pass
                else:
                    print(f"警告: 未知的圖像格式 {image.shape}")
                    continue
            
            # 確保圖像是3通道的
            if len(image.shape) == 2:  # 灰度圖
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:  # 單通道
                image = np.repeat(image, 3, axis=2)
            
            # 確保圖像在正確的範圍內
            image = np.clip(image, 0, 1)
            
            # 轉換為 PIL 圖像
            try:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
                
                # 保存圖像
                image_path = output_path / f"generated_{i:03d}.png"
                image_pil.save(image_path)
                print(f"保存: {image_path}")
            except Exception as e:
                print(f"保存圖像 {i} 失敗: {e}")
                print(f"圖像形狀: {image.shape}, 數據類型: {image.dtype}")
                continue
        
        # 創建網格可視化
        if len(generated_images) > 0:
            create_image_grid(generated_images, output_path / "grid.png")
            print(f"網格可視化保存到: {output_path / 'grid.png'}")
        
        print(f"成功生成 {len(generated_images)} 張圖像！")
        return True
        
    except Exception as e:
        print(f"生成過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_image_grid(images, output_path, grid_size=None):
    """創建圖像網格。"""
    n_images = len(images)
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(n_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, image in enumerate(images):
        if i < n_images:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            # 確保圖像在 [0, 1] 範圍內
            if image.max() > 1.0:
                image = image / 255.0
            
            # 處理不同的圖像格式
            if len(image.shape) == 4:  # [B, C, H, W]
                image = image[0]  # 取第一個批次
            
            if len(image.shape) == 3:
                if image.shape[0] == 3 or image.shape[0] == 1:  # CHW 格式
                    image = np.transpose(image, (1, 2, 0))  # 轉換為 HWC
                elif image.shape[2] == 3 or image.shape[2] == 1:  # 已經是 HWC 格式
                    pass
            
            # 確保圖像是3通道的
            if len(image.shape) == 2:  # 灰度圖
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 1:  # 單通道
                image = np.repeat(image, 3, axis=2)
            
            # 確保圖像在正確的範圍內
            image = np.clip(image, 0, 1)
            
            axes[i].imshow(image)
            axes[i].set_title(f"Image {i}")
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函數。"""
    parser = argparse.ArgumentParser(description="生成 QRL 圖像")
    parser.add_argument("--checkpoint", "-c", required=True, help="檢查點文件路徑")
    parser.add_argument("--num-images", "-n", type=int, default=10, help="生成圖像數量")
    parser.add_argument("--target-class", "-t", type=int, default=3, help="目標類別 (0-9)")
    parser.add_argument("--output-dir", "-o", default="generated_images", help="輸出目錄")
    parser.add_argument("--device", "-d", default="cuda", help="設備")
    
    args = parser.parse_args()
    
    # 檢查檢查點文件是否存在
    if not Path(args.checkpoint).exists():
        print(f"錯誤: 檢查點文件不存在: {args.checkpoint}")
        return 1
    
    # 生成圖像
    success = generate_images(
        checkpoint_path=args.checkpoint,
        num_images=args.num_images,
        target_class=args.target_class,
        output_dir=args.output_dir,
        device=args.device
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
