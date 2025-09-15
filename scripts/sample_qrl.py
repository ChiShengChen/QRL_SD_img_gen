#!/usr/bin/env python3
"""量子強化學習圖像合成採樣腳本。"""

import os
import sys
import argparse
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qrl.utils.seed import set_seed, set_deterministic
from qrl.utils.logging import QRLLogger
from qrl.envs import create_diffusion_env
from qrl.actors import QuantumActor, ClassicalActor, AlignedClassicalActor
from qrl.critics import MLPCritic
from qrl.training import PPOTrainer, QuantumPPOTrainer
from qrl.controls import ControlSpaces
from qrl.reward import ClassifierReward, DiversityReward
from qrl.metrics import calculate_all_metrics


@hydra.main(version_base=None, config_path="../qrl/configs", config_name="base")
def main(cfg: DictConfig):
    """主採樣函數。"""
    # 設置種子和確定性
    set_seed(cfg.seed)
    if cfg.deterministic:
        set_deterministic()
    
    # 創建工作目錄
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建採樣目錄
    sample_dir = work_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置設備
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    
    # 初始化日誌記錄器
    logger = QRLLogger(
        name="sample_qrl",
        log_dir=work_dir / "logs",
        use_tensorboard=False
    )
    
    logger.info("開始 QRL 圖像合成採樣...")
    logger.info(f"配置: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"設備: {device}")
    logger.info(f"工作目錄: {work_dir}")
    logger.info(f"採樣目錄: {sample_dir}")
    
    try:
        # 創建環境
        logger.info("創建擴散環境...")
        env = create_diffusion_env(
            model_id=cfg.model.unet_ddim.model_id,
            target_class=cfg.dataset.cifar10.target_class,
            max_steps=cfg.control.environment.max_steps,
            stage=cfg.control.stage,
            device=device,
            reward_config=cfg.control.reward,
            control_config=cfg.control
        )
        
        # 創建 Actor
        logger.info("創建 Actor...")
        if cfg.algo.ppo.actor_type == "quantum":
            actor = QuantumActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                device=device
            )
        else:
            # 創建對齊的古典 Actor
            quantum_actor = QuantumActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                device=device
            )
            target_params = quantum_actor.total_params
            
            actor = AlignedClassicalActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                target_params=target_params,
                device=device
            )
        
        # 創建 Critic
        logger.info("創建 Critic...")
        critic = MLPCritic(
            state_dim=env.observation_space.shape[0],
            device=device
        )
        
        # 創建訓練器（用於載入檢查點）
        logger.info("創建訓練器...")
        if cfg.algo.ppo.actor_type == "quantum":
            trainer = QuantumPPOTrainer(
                actor=actor,
                critic=critic,
                device=device,
                lr_actor=cfg.algo.ppo.lr_actor,
                lr_critic=cfg.algo.ppo.lr_critic,
                gamma=cfg.algo.ppo.gamma,
                gae_lambda=cfg.algo.ppo.gae_lambda,
                clip_ratio=cfg.algo.ppo.clip_ratio,
                entropy_coef=cfg.algo.ppo.entropy_coef,
                value_coef=cfg.algo.ppo.value_coef,
                max_grad_norm=cfg.algo.ppo.max_grad_norm,
                update_epochs=cfg.algo.ppo.update_epochs,
                mini_batch_size=cfg.algo.ppo.mini_batch_size,
                rollout_steps=cfg.algo.ppo.rollout_steps,
                num_envs=cfg.algo.ppo.num_envs,
                mixed_precision=cfg.algo.ppo.mixed_precision,
                use_quantum_gradients=cfg.algo.ppo.use_quantum_gradients
            )
        else:
            trainer = PPOTrainer(
                actor=actor,
                critic=critic,
                device=device,
                lr_actor=cfg.algo.ppo.lr_actor,
                lr_critic=cfg.algo.ppo.lr_critic,
                gamma=cfg.algo.ppo.gamma,
                gae_lambda=cfg.algo.ppo.gae_lambda,
                clip_ratio=cfg.algo.ppo.clip_ratio,
                entropy_coef=cfg.algo.ppo.entropy_coef,
                value_coef=cfg.algo.ppo.value_coef,
                max_grad_norm=cfg.algo.ppo.max_grad_norm,
                update_epochs=cfg.algo.ppo.update_epochs,
                mini_batch_size=cfg.algo.ppo.mini_batch_size,
                rollout_steps=cfg.algo.ppo.rollout_steps,
                num_envs=cfg.algo.ppo.num_envs,
                mixed_precision=cfg.algo.ppo.mixed_precision
            )
        
        # 載入檢查點
        checkpoint_path = work_dir / "ckpt_best.pt"
        if not checkpoint_path.exists():
            logger.error(f"檢查點文件不存在: {checkpoint_path}")
            return False
        
        logger.info(f"載入檢查點: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        
        # 設置為評估模式
        actor.eval()
        critic.eval()
        
        # 開始採樣
        logger.info("開始採樣...")
        num_samples = cfg.sampling.num_samples
        batch_size = cfg.sampling.batch_size
        
        all_images = []
        all_actions = []
        all_rewards = []
        all_metrics = []
        
        for batch_idx in tqdm(range(0, num_samples, batch_size), desc="採樣批次"):
            batch_size_actual = min(batch_size, num_samples - batch_idx)
            
            # 重置環境
            obs, _ = env.reset()
            
            # 採樣過程
            images_batch = []
            actions_batch = []
            rewards_batch = []
            
            for step in range(cfg.sampling.num_steps):
                # 獲取動作
                with torch.no_grad():
                    action, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0).to(device))
                    action = action.cpu().numpy()[0]
                
                # 執行動作
                obs, reward, done, truncated, info = env.step(action)
                
                # 記錄
                actions_batch.append(action)
                rewards_batch.append(reward)
                
                if done or truncated:
                    break
            
            # 獲取最終圖像
            final_image = env.get_final_image()
            if final_image is not None:
                images_batch.append(final_image)
            
            # 保存批次結果
            all_images.extend(images_batch)
            all_actions.extend(actions_batch)
            all_rewards.extend(rewards_batch)
            
            # 保存圖像
            for i, image in enumerate(images_batch):
                sample_idx = batch_idx + i
                if sample_idx < num_samples:
                    # 轉換為 PIL 圖像
                    if isinstance(image, torch.Tensor):
                        image = image.cpu().numpy()
                    
                    # 確保圖像在 [0, 1] 範圍內
                    if image.max() > 1.0:
                        image = image / 255.0
                    
                    # 轉換為 PIL 圖像
                    if image.shape[0] == 3:  # CHW 格式
                        image = np.transpose(image, (1, 2, 0))
                    
                    image_pil = Image.fromarray((image * 255).astype(np.uint8))
                    
                    # 保存圖像
                    image_path = sample_dir / f"sample_{sample_idx:06d}.png"
                    image_pil.save(image_path)
        
        # 計算指標
        logger.info("計算採樣指標...")
        if len(all_images) > 0:
            # 轉換圖像為張量
            image_tensors = []
            for image in all_images:
                if isinstance(image, np.ndarray):
                    if image.shape[0] == 3:  # CHW 格式
                        image = np.transpose(image, (1, 2, 0))
                    image_tensor = torch.from_numpy(image).float().unsqueeze(0)
                    image_tensors.append(image_tensor)
            
            if image_tensors:
                fake_images = torch.cat(image_tensors, dim=0)
                
                # 計算指標
                metrics = calculate_all_metrics(
                    real_images=None,  # 暫時不計算 FID
                    fake_images=fake_images,
                    target_class=cfg.dataset.target_class,
                    dataset=cfg.dataset.name,
                    device=device,
                    batch_size=cfg.sampling.batch_size
                )
                
                # 添加自定義指標
                metrics['num_samples'] = len(all_images)
                metrics['mean_reward'] = np.mean(all_rewards) if all_rewards else 0.0
                metrics['std_reward'] = np.std(all_rewards) if all_rewards else 0.0
                
                all_metrics.append(metrics)
                
                # 保存指標
                metrics_file = sample_dir / "sampling_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                logger.info(f"採樣指標: {metrics}")
        
        # 保存採樣摘要
        summary = {
            'num_samples': len(all_images),
            'num_steps': cfg.sampling.num_steps,
            'batch_size': cfg.sampling.batch_size,
            'actor_type': cfg.algo.actor_type,
            'checkpoint': str(checkpoint_path),
            'sampling_time': time.time(),
            'metrics': all_metrics
        }
        
        summary_file = sample_dir / "sampling_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 創建採樣可視化
        if len(all_images) > 0:
            create_sampling_visualization(
                all_images[:min(16, len(all_images))],
                sample_dir / "sampling_grid.png"
            )
        
        logger.info("採樣完成！")
        logger.info(f"生成圖像數量: {len(all_images)}")
        logger.info(f"採樣目錄: {sample_dir}")
        logger.info(f"採樣摘要已保存到: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"採樣過程中發生錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def create_sampling_visualization(images, output_path):
    """創建採樣可視化網格。"""
    try:
        n_images = len(images)
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
                
                # 轉換格式
                if image.shape[0] == 3:  # CHW 格式
                    image = np.transpose(image, (1, 2, 0))
                
                axes[i].imshow(image)
                axes[i].set_title(f"Sample {i}")
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"創建可視化失敗: {e}")


def create_sampling_config(
    dataset: str = "cifar10",
    control: str = "stageA_cfg",
    algo: str = "ppo",
    model: str = "unet_ddim",
    **kwargs
) -> DictConfig:
    """
    創建採樣配置。
    
    Args:
        dataset: 數據集配置
        control: 控制配置
        algo: 算法配置
        model: 模型配置
        **kwargs: 其他配置參數
        
    Returns:
        DictConfig: 採樣配置
    """
    # 基礎配置
    base_config = {
        'seed': 42,
        'work_dir': 'runs/latest',
        'device': 'cuda',
        'precision': 'float32',
        'deterministic': False,
        'sampling': {
            'num_samples': 100,
            'num_steps': 50,
            'batch_size': 16
        }
    }
    
    # 合併配置
    config = OmegaConf.create(base_config)
    
    # 載入子配置
    if dataset:
        dataset_config = OmegaConf.load(f"../qrl/configs/dataset/{dataset}.yaml")
        config.dataset = dataset_config
    
    if control:
        control_config = OmegaConf.load(f"../qrl/configs/control/{control}.yaml")
        config.control = control_config
    
    if algo:
        algo_config = OmegaConf.load(f"../qrl/configs/algo/{algo}.yaml")
        config.algo = algo_config
    
    if model:
        model_config = OmegaConf.load(f"../qrl/configs/model/{model}.yaml")
        config.model = model_config
    
    # 更新自定義參數
    for key, value in kwargs.items():
        OmegaConf.update(config, key, value)
    
    return config


if __name__ == "__main__":
    # 如果直接運行，使用默認配置
    if len(sys.argv) == 1:
        # 創建默認配置
        config = create_sampling_config()
        main(config)
    else:
        # 使用 Hydra 配置
        main()
