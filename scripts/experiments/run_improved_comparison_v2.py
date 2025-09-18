#!/usr/bin/env python3
"""
改進的量子 vs 經典模型比較實驗 V2
- 使用與 generate_images.py 相同的環境配置
- 增加生成步數以改善圖像品質
- 訓練100 episodes後保存模型
- 為CIFAR-10所有類別生成圖像
- 與真實CIFAR-10圖像進行PSNR/SSIM比較
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 導入項目模組
from qrl.envs import create_diffusion_env
from qrl.actors import QuantumActor, ClassicalActor
from qrl.critics import MLPCritic
from qrl.training import QuantumPPOTrainer, PPOTrainer

# =============================================================================
# 配置參數
# =============================================================================

# 設備配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型配置
STATE_DIM = 6
ACTION_DIM = 1
N_QUBITS = 4
N_LAYERS = 2

# 訓練參數
NUM_EPISODES = 100
MAX_STEPS = 50  # 增加最大步數
TARGET_CLASS = 3  # 訓練時使用cat類別
STAGE = "A"

# PPO參數
PPO_LEARNING_RATE = 3e-4
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 64

# 生成參數
GENERATE_IMAGES_PER_CLASS = 16
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 輸出目錄
OUTPUT_DIR = Path("improved_comparison_results_v2")
MODELS_DIR = OUTPUT_DIR / "models"
GENERATED_DIR = OUTPUT_DIR / "generated_images"
REAL_DIR = OUTPUT_DIR / "real_images"

# =============================================================================

def setup_directories():
    """創建輸出目錄"""
    for dir_path in [OUTPUT_DIR, MODELS_DIR, GENERATED_DIR, REAL_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 為每個類別創建子目錄
    for class_name in CIFAR10_CLASSES:
        (GENERATED_DIR / "quantum" / class_name).mkdir(parents=True, exist_ok=True)
        (GENERATED_DIR / "classical" / class_name).mkdir(parents=True, exist_ok=True)
        (REAL_DIR / class_name).mkdir(parents=True, exist_ok=True)

def load_cifar10_images(num_images_per_class: int = 16) -> Dict[str, List[torch.Tensor]]:
    """加載CIFAR-10真實圖像"""
    print("📥 加載CIFAR-10真實圖像...")
    
    # 數據轉換 - 保持原始解析度
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 不調整尺寸，保持原始解析度
    ])
    
    # 加載CIFAR-10測試集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 按類別分組
    class_images = {class_name: [] for class_name in CIFAR10_CLASSES}
    
    for class_id in range(10):
        class_name = CIFAR10_CLASSES[class_id]
        class_indices = [i for i, (_, label) in enumerate(testset) if label == class_id]
        
        # 隨機選擇指定數量的圖像
        selected_indices = np.random.choice(class_indices, 
                                          min(num_images_per_class, len(class_indices)), 
                                          replace=False)
        
        for idx in selected_indices:
            image, _ = testset[idx]
            class_images[class_name].append(image)
            
            # 保存圖像
            torch.save(image, REAL_DIR / class_name / f"real_{len(class_images[class_name])-1}.pt")
    
    print(f"✅ 已加載CIFAR-10圖像，每個類別 {num_images_per_class} 張")
    return class_images

def create_environment(target_class: int, device: str):
    """創建與 generate_images.py 相同的環境配置"""
    return create_diffusion_env(
        model_id="runwayml/stable-diffusion-v1-5",
        target_class=target_class,
        max_steps=MAX_STEPS,
        stage=STAGE,
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
            'environment': {'max_steps': MAX_STEPS}
        }
    )

def train_model(actor_type: str, num_episodes: int = NUM_EPISODES) -> Dict[str, Any]:
    """訓練量子或經典模型"""
    print(f"🔬 開始{actor_type}模型訓練實驗...")
    
    try:
        # 初始化環境
        env = create_environment(TARGET_CLASS, DEVICE)
        
        # 初始化Actor
        if actor_type == "quantum":
            actor = QuantumActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                n_qubits=N_QUBITS,
                n_layers=N_LAYERS,
                device=DEVICE
            )
        else:
            actor = ClassicalActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                hidden_dims=[64, 32],
                device=DEVICE
            )
        
        # 初始化Critic
        critic = MLPCritic(
            state_dim=env.observation_space.shape[0],
            hidden_dims=[64, 32],
            device=DEVICE
        )
        
        # 初始化訓練器 - 使用與 generate_images.py 相同的配置
        if actor_type == "quantum":
            trainer = QuantumPPOTrainer(
                actor=actor,
                critic=critic,
                device=DEVICE,
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
        else:
            trainer = PPOTrainer(
                actor=actor,
                critic=critic,
                device=DEVICE,
                lr_actor=PPO_LEARNING_RATE,
                lr_critic=PPO_LEARNING_RATE,
                rollout_steps=10,
                num_envs=1,
                update_epochs=PPO_EPOCHS,
                mini_batch_size=PPO_BATCH_SIZE
            )
        
        # 訓練
        episode_rewards = []
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 重置環境
            obs, _ = env.reset()
            
            # 收集rollout數據
            if actor_type == "quantum":
                rollouts = trainer.collect_rollouts(env, num_steps=4)
            else:
                rollouts = trainer.collect_rollouts(env, num_steps=10)
            
            # 計算episode獎勵
            episode_reward = sum(rollouts['rewards'].cpu().numpy())
            
            # 執行PPO更新
            if len(rollouts['rewards']) > 0:
                update_stats = trainer.update(rollouts)
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}: 獎勵={episode_reward:.4f}, 時間={episode_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # 計算統計
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # 保存模型
        model_path = MODELS_DIR / f"{actor_type}_actor.pt"
        torch.save(actor.state_dict(), model_path)
        
        print(f"✅ {actor_type}模型訓練完成 - 平均獎勵: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"💾 模型已保存到: {model_path}")
        
        return {
            "actor": actor,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "episode_rewards": episode_rewards,
            "total_time": total_time,
            "model_path": model_path
        }
        
    except Exception as e:
        print(f"❌ {actor_type}模型訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_images_for_all_classes(actor, actor_type: str, num_images_per_class: int = GENERATE_IMAGES_PER_CLASS):
    """為所有CIFAR-10類別生成圖像"""
    print(f"🎨 使用{actor_type}模型為所有類別生成圖像...")
    
    generated_images = {}
    
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        print(f"  生成 {class_name} 類別圖像...")
        class_images = []
        
        for img_idx in range(num_images_per_class):
            try:
                # 創建新的環境實例，使用當前類別
                env = create_environment(class_id, DEVICE)
                
                # 重置環境
                obs, _ = env.reset()
                
                # 使用訓練好的actor進行rollout - 使用與 generate_images.py 相同的步數
                with torch.no_grad():
                    for step in range(50):  # 與 generate_images.py 相同的步數
                        # 使用 sample 方法獲取動作
                        action = actor.sample(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
                        action_np = action.cpu().numpy().flatten()
                        obs, reward, done, truncated, info = env.step(action_np)
                        if done or truncated:
                            break
                
                # 獲取最終圖像
                final_image = env._decode_latent()
                if final_image is not None:
                    # 確保圖像是torch.Tensor格式
                    if isinstance(final_image, tuple):
                        final_image = final_image[0]  # 如果是tuple，取第一個元素
                    
                    class_images.append(final_image)
                    
                    # 保存圖像到CPU
                    final_image_cpu = final_image.cpu() if final_image.is_cuda else final_image
                    torch.save(final_image_cpu, GENERATED_DIR / actor_type / class_name / f"gen_{img_idx}.pt")
                
                env.close()
                
            except Exception as e:
                print(f"    ❌ 生成 {class_name} 第 {img_idx} 張圖像失敗: {e}")
                continue
        
        generated_images[class_name] = class_images
        print(f"  ✅ {class_name}: 成功生成 {len(class_images)} 張圖像")
    
    return generated_images

def calculate_metrics_batch(real_images: List[torch.Tensor], generated_images: List[torch.Tensor]) -> Dict[str, float]:
    """批量計算PSNR和SSIM指標"""
    if len(real_images) == 0 or len(generated_images) == 0:
        return {"psnr": 0.0, "ssim": 0.0, "count": 0}
    
    psnr_values = []
    ssim_values = []
    
    # 確保圖像數量匹配
    min_count = min(len(real_images), len(generated_images))
    
    for i in range(min_count):
        try:
            real_img = real_images[i].detach().cpu().numpy()
            gen_img = generated_images[i].detach().cpu().numpy()
            
            # 處理生成圖像的4D tensor (batch_size=1)
            if len(gen_img.shape) == 4 and gen_img.shape[0] == 1:
                gen_img = gen_img[0]  # 移除batch維度
            
            # 將生成圖像調整為與真實圖像相同的尺寸
            if real_img.shape != gen_img.shape:
                # 使用雙線性插值調整生成圖像尺寸
                from PIL import Image
                import torch.nn.functional as F
                
                # 轉換為PIL圖像進行調整
                gen_tensor = torch.from_numpy(gen_img).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                real_tensor = torch.from_numpy(real_img).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                
                # 調整生成圖像尺寸以匹配真實圖像
                target_size = (real_tensor.shape[2], real_tensor.shape[3])  # (H, W)
                gen_resized = F.interpolate(gen_tensor, size=target_size, mode='bilinear', align_corners=False)
                gen_img = gen_resized.squeeze(0).permute(1, 2, 0).numpy()  # 轉回 [H, W, C]
            
            # 轉換為灰度圖像
            def to_grayscale(img):
                if len(img.shape) == 3:
                    if img.shape[2] == 3:  # [H, W, C]
                        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                    elif img.shape[0] == 3:  # [C, H, W]
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

def evaluate_image_quality(real_images: Dict[str, List[torch.Tensor]], 
                          quantum_images: Dict[str, List[torch.Tensor]], 
                          classical_images: Dict[str, List[torch.Tensor]]) -> Dict[str, Any]:
    """評估圖像品質"""
    print("\n🖼️ 開始圖像品質評估...")
    
    results = {
        "class_results": {},
        "overall_quantum": {"psnr": [], "ssim": []},
        "overall_classical": {"psnr": [], "ssim": []}
    }
    
    for class_name in CIFAR10_CLASSES:
        print(f"  評估 {class_name} 類別...")
        
        real_imgs = real_images.get(class_name, [])
        quantum_imgs = quantum_images.get(class_name, [])
        classical_imgs = classical_images.get(class_name, [])
        
        # 計算量子模型指標
        quantum_metrics = calculate_metrics_batch(real_imgs, quantum_imgs)
        
        # 計算經典模型指標
        classical_metrics = calculate_metrics_batch(real_imgs, classical_imgs)
        
        results["class_results"][class_name] = {
            "quantum": quantum_metrics,
            "classical": classical_metrics
        }
        
        # 收集總體統計
        if quantum_metrics["count"] > 0:
            results["overall_quantum"]["psnr"].append(quantum_metrics["psnr"])
            results["overall_quantum"]["ssim"].append(quantum_metrics["ssim"])
        
        if classical_metrics["count"] > 0:
            results["overall_classical"]["psnr"].append(classical_metrics["psnr"])
            results["overall_classical"]["ssim"].append(classical_metrics["ssim"])
        
        print(f"    量子: PSNR={quantum_metrics['psnr']:.2f}, SSIM={quantum_metrics['ssim']:.4f}")
        print(f"    經典: PSNR={classical_metrics['psnr']:.2f}, SSIM={classical_metrics['ssim']:.4f}")
    
    # 計算總體平均
    results["overall_summary"] = {
        "quantum": {
            "psnr": np.mean(results["overall_quantum"]["psnr"]) if results["overall_quantum"]["psnr"] else 0.0,
            "ssim": np.mean(results["overall_quantum"]["ssim"]) if results["overall_quantum"]["ssim"] else 0.0
        },
        "classical": {
            "psnr": np.mean(results["overall_classical"]["psnr"]) if results["overall_classical"]["psnr"] else 0.0,
            "ssim": np.mean(results["overall_classical"]["ssim"]) if results["overall_classical"]["ssim"] else 0.0
        }
    }
    
    return results

def print_comparison_results(results: Dict[str, Any]):
    """打印比較結果"""
    print("\n" + "="*80)
    print("📊 量子 vs 經典模型圖像品質比較結果 (與真實CIFAR-10圖像)")
    print("="*80)
    
    # 總體結果
    overall = results["overall_summary"]
    print(f"\n🎯 總體結果:")
    print(f"  量子模型:  PSNR={overall['quantum']['psnr']:.2f}, SSIM={overall['quantum']['ssim']:.4f}")
    print(f"  經典模型:  PSNR={overall['classical']['psnr']:.2f}, SSIM={overall['classical']['ssim']:.4f}")
    
    # 改善百分比 - 避免除零錯誤
    if overall['classical']['psnr'] > 0:
        psnr_improvement = ((overall['quantum']['psnr'] - overall['classical']['psnr']) / overall['classical']['psnr']) * 100
    else:
        psnr_improvement = 0.0
    
    if overall['classical']['ssim'] > 0:
        ssim_improvement = ((overall['quantum']['ssim'] - overall['classical']['ssim']) / overall['classical']['ssim']) * 100
    else:
        ssim_improvement = 0.0
    
    print(f"\n📈 改善:")
    print(f"  PSNR: {psnr_improvement:+.1f}%")
    print(f"  SSIM: {ssim_improvement:+.1f}%")
    
    # 按類別詳細結果
    print(f"\n📋 按類別詳細結果:")
    print(f"{'類別':<12} {'量子PSNR':<10} {'經典PSNR':<10} {'量子SSIM':<10} {'經典SSIM':<10}")
    print("-" * 60)
    
    for class_name, class_result in results["class_results"].items():
        q_psnr = class_result["quantum"]["psnr"]
        c_psnr = class_result["classical"]["psnr"]
        q_ssim = class_result["quantum"]["ssim"]
        c_ssim = class_result["classical"]["ssim"]
        
        print(f"{class_name:<12} {q_psnr:<10.2f} {c_psnr:<10.2f} {q_ssim:<10.4f} {c_ssim:<10.4f}")

def main():
    """主函數"""
    print("🚀 開始改進的量子 vs 經典模型比較實驗 V2...")
    print("="*80)
    
    # 設置目錄
    setup_directories()
    
    # 加載真實CIFAR-10圖像
    real_images = load_cifar10_images(GENERATE_IMAGES_PER_CLASS)
    
    # 訓練量子模型
    quantum_results = train_model("quantum", NUM_EPISODES)
    if quantum_results is None:
        print("❌ 量子模型訓練失敗，退出")
        return
    
    # 訓練經典模型
    classical_results = train_model("classical", NUM_EPISODES)
    if classical_results is None:
        print("❌ 經典模型訓練失敗，退出")
        return
    
    # 生成圖像
    print("\n🎨 開始生成圖像...")
    quantum_images = generate_images_for_all_classes(
        quantum_results["actor"], "quantum", GENERATE_IMAGES_PER_CLASS
    )
    
    classical_images = generate_images_for_all_classes(
        classical_results["actor"], "classical", GENERATE_IMAGES_PER_CLASS
    )
    
    # 評估圖像品質
    evaluation_results = evaluate_image_quality(real_images, quantum_images, classical_images)
    
    # 打印結果
    print_comparison_results(evaluation_results)
    
    # 保存結果
    final_results = {
        "training_results": {
            "quantum": {
                "avg_reward": quantum_results["avg_reward"],
                "std_reward": quantum_results["std_reward"],
                "total_time": quantum_results["total_time"]
            },
            "classical": {
                "avg_reward": classical_results["avg_reward"],
                "std_reward": classical_results["std_reward"],
                "total_time": classical_results["total_time"]
            }
        },
        "evaluation_results": evaluation_results,
        "config": {
            "num_episodes": NUM_EPISODES,
            "images_per_class": GENERATE_IMAGES_PER_CLASS,
            "target_class": TARGET_CLASS,
            "max_steps": MAX_STEPS
        }
    }
    
    results_file = OUTPUT_DIR / "improved_comparison_results_v2.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n💾 結果已保存到: {results_file}")
    print("✅ 實驗完成！")

if __name__ == "__main__":
    main()
