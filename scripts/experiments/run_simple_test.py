#!/usr/bin/env python3
"""
簡化測試版本 - 避免網絡連接問題
只測試基本的圖像生成和評估功能
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 導入項目模組
from qrl.envs import DiffusionEnvironment
from qrl.actors import QuantumActor, ClassicalActor
from qrl.critics import MLPCritic
from qrl.training import PPOTrainer

# =============================================================================
# 配置參數
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE_DIM = 6
ACTION_DIM = 1
N_QUBITS = 4
N_LAYERS = 2

# 簡化測試參數
NUM_EPISODES = 1
MAX_STEPS = 5
TARGET_CLASS = 3
STAGE = "A"
PPO_LEARNING_RATE = 3e-4
PPO_EPOCHS = 1
PPO_BATCH_SIZE = 16

# 測試類別（只測試1個類別）
TEST_CLASSES = ["cat"]
IMAGES_PER_CLASS = 2  # 每個類別生成2張圖像

# =============================================================================

def load_cifar10_sample(num_images_per_class: int = 2) -> Dict[str, List[torch.Tensor]]:
    """加載CIFAR-10樣本圖像"""
    print("📥 加載CIFAR-10樣本圖像...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    class_images = {}
    for i, class_name in enumerate(TEST_CLASSES):
        class_indices = [idx for idx, (_, label) in enumerate(testset) if label == i + 3]  # cat是類別3
        selected_indices = np.random.choice(class_indices, num_images_per_class, replace=False)
        
        class_images[class_name] = []
        for idx in selected_indices:
            image, _ = testset[idx]
            class_images[class_name].append(image)
    
    print(f"✅ 已加載樣本圖像，每個類別 {num_images_per_class} 張")
    return class_images

def train_quick_model(actor_type: str) -> Any:
    """快速訓練模型"""
    print(f"🔬 快速訓練{actor_type}模型...")
    
    try:
        # 初始化環境
        env = DiffusionEnvironment(
            model_id="runwayml/stable-diffusion-v1-5",
            target_class=TARGET_CLASS,
            max_steps=MAX_STEPS,
            device=DEVICE,
            stage=STAGE
        )
        
        # 初始化Actor
        if actor_type == "quantum":
            actor = QuantumActor(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                n_qubits=N_QUBITS,
                n_layers=N_LAYERS,
                device=DEVICE
            )
        else:
            actor = ClassicalActor(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                hidden_dims=[64, 32],
                device=DEVICE
            )
        
        # 初始化Critic
        critic = MLPCritic(
            state_dim=STATE_DIM,
            hidden_dims=[64, 32],
            device=DEVICE
        )
        
        # 初始化PPO訓練器
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            device=DEVICE,
            lr_actor=PPO_LEARNING_RATE,
            lr_critic=PPO_LEARNING_RATE,
            rollout_steps=3,  # 減少rollout步數
            num_envs=1,
            update_epochs=PPO_EPOCHS,
            mini_batch_size=PPO_BATCH_SIZE
        )
        
        # 快速訓練
        episode_rewards = []
        for episode in range(NUM_EPISODES):
            obs, _ = env.reset()
            rollouts = trainer.collect_rollouts(env, num_steps=3)
            episode_reward = sum(rollouts['rewards'].cpu().numpy())
            
            if len(rollouts['rewards']) > 0:
                trainer.update(rollouts)
            
            episode_rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: 獎勵={episode_reward:.4f}")
        
        avg_reward = np.mean(episode_rewards)
        print(f"✅ {actor_type}模型訓練完成 - 平均獎勵: {avg_reward:.4f}")
        
        env.close()
        return actor
        
    except Exception as e:
        print(f"❌ {actor_type}模型訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_test_images(actor, actor_type: str) -> Dict[str, List[torch.Tensor]]:
    """生成測試圖像"""
    print(f"🎨 使用{actor_type}模型生成測試圖像...")
    
    generated_images = {}
    
    for class_id, class_name in enumerate(TEST_CLASSES):
        print(f"  生成 {class_name} 類別圖像...")
        class_images = []
        
        for img_idx in range(IMAGES_PER_CLASS):
            try:
                # 創建環境
                env = DiffusionEnvironment(
                    model_id="runwayml/stable-diffusion-v1-5",
                    target_class=class_id + 3,  # cat是類別3
                    max_steps=MAX_STEPS,
                    device=DEVICE,
                    stage=STAGE
                )
                
                obs, _ = env.reset()
                
                # 簡化的rollout
                with torch.no_grad():
                    for step in range(3):
                        # 使用 sample 方法獲取動作，而不是直接調用 forward
                        action = actor.sample(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
                        action_np = action.cpu().numpy().flatten()
                        obs, reward, done, truncated, info = env.step(action_np)
                        if done or truncated:
                            break
                
                # 獲取圖像
                final_image = env._decode_latent()
                if final_image is not None:
                    # 確保圖像是torch.Tensor格式
                    if isinstance(final_image, tuple):
                        final_image = final_image[0]
                    
                    class_images.append(final_image)
                
                env.close()
                
            except Exception as e:
                print(f"    ❌ 生成 {class_name} 第 {img_idx} 張圖像失敗: {e}")
                continue
        
        generated_images[class_name] = class_images
        print(f"  ✅ {class_name}: 成功生成 {len(class_images)} 張圖像")
    
    return generated_images

def calculate_metrics(real_images: List[torch.Tensor], generated_images: List[torch.Tensor]) -> Dict[str, float]:
    """計算PSNR和SSIM指標"""
    if len(real_images) == 0 or len(generated_images) == 0:
        return {"psnr": 0.0, "ssim": 0.0}
    
    psnr_values = []
    ssim_values = []
    
    min_count = min(len(real_images), len(generated_images))
    
    for i in range(min_count):
        try:
            real_img = real_images[i].detach().cpu().numpy()
            gen_img = generated_images[i].detach().cpu().numpy()
            
            # 轉換為灰度圖像
            def to_grayscale(img):
                if len(img.shape) == 4:
                    img = img[0]
                if len(img.shape) == 3:
                    if img.shape[0] == 3:
                        return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                    else:
                        return img[0]
                return img
            
            real_gray = to_grayscale(real_img)
            gen_gray = to_grayscale(gen_img)
            
            # 確保尺寸相同
            if real_gray.shape != gen_gray.shape:
                min_h = min(real_gray.shape[0], gen_gray.shape[0])
                min_w = min(real_gray.shape[1], gen_gray.shape[1])
                real_gray = real_gray[:min_h, :min_w]
                gen_gray = gen_gray[:min_h, :min_w]
            
            # 計算PSNR
            real_uint8 = (real_gray * 255).astype(np.uint8)
            gen_uint8 = (gen_gray * 255).astype(np.uint8)
            psnr_val = psnr(real_uint8, gen_uint8, data_range=255)
            
            # 計算SSIM
            ssim_val = ssim(real_gray, gen_gray, data_range=1.0)
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            
        except Exception as e:
            print(f"    ❌ 計算第 {i} 張圖像指標失敗: {e}")
            continue
    
    return {
        "psnr": np.mean(psnr_values) if psnr_values else 0.0,
        "ssim": np.mean(ssim_values) if ssim_values else 0.0,
        "count": len(psnr_values)
    }

def main():
    """主函數"""
    print("🚀 開始簡化測試...")
    print("="*60)
    
    # 加載真實圖像
    real_images = load_cifar10_sample(IMAGES_PER_CLASS)
    
    # 訓練模型
    quantum_actor = train_quick_model("quantum")
    if quantum_actor is None:
        print("❌ 量子模型訓練失敗，退出")
        return
    
    classical_actor = train_quick_model("classical")
    if classical_actor is None:
        print("❌ 經典模型訓練失敗，退出")
        return
    
    # 生成圖像
    quantum_images = generate_test_images(quantum_actor, "quantum")
    classical_images = generate_test_images(classical_actor, "classical")
    
    # 評估
    print("\n🖼️ 開始圖像品質評估...")
    
    for class_name in TEST_CLASSES:
        print(f"  評估 {class_name} 類別...")
        
        real_imgs = real_images.get(class_name, [])
        quantum_imgs = quantum_images.get(class_name, [])
        classical_imgs = classical_images.get(class_name, [])
        
        # 計算指標
        quantum_metrics = calculate_metrics(real_imgs, quantum_imgs)
        classical_metrics = calculate_metrics(real_imgs, classical_imgs)
        
        print(f"    量子: PSNR={quantum_metrics['psnr']:.2f}, SSIM={quantum_metrics['ssim']:.4f}")
        print(f"    經典: PSNR={classical_metrics['psnr']:.2f}, SSIM={classical_metrics['ssim']:.4f}")
    
    print("\n✅ 簡化測試完成！")

if __name__ == "__main__":
    main()
