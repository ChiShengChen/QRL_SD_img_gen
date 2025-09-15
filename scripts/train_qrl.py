#!/usr/bin/env python3
"""量子強化學習圖像合成訓練腳本。"""

import os
import sys
import argparse
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import time
from datetime import datetime

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qrl.utils.seed import set_seed, set_deterministic
from qrl.utils.logging import QRLLogger, MetricsLogger
from qrl.envs import create_diffusion_env
from qrl.actors import QuantumActor, ClassicalActor, AlignedClassicalActor
from qrl.critics import MLPCritic
from qrl.training import PPOTrainer, QuantumPPOTrainer
from qrl.controls import ControlSpaces
from qrl.reward import ClassifierReward, DiversityReward


@hydra.main(version_base=None, config_path="../qrl/configs", config_name="base")
def main(cfg: DictConfig):
    """主訓練函數。"""
    # 設置種子和確定性
    set_seed(cfg.seed)
    if cfg.deterministic:
        set_deterministic()
    
    # 創建工作目錄
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置設備
    if torch.cuda.is_available() and cfg.device == "cuda":
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # 初始化日誌記錄器
    logger = QRLLogger(
        name="train_qrl",
        log_dir=work_dir / "logs",
        tensorboard=cfg.logging.use_tensorboard
    )
    
    metrics_logger = MetricsLogger(work_dir / "metrics")
    
    logger.info("開始 QRL 圖像合成訓練...")
    logger.info(f"配置: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"設備: {device}")
    logger.info(f"工作目錄: {work_dir}")
    
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
            logger.info(f"量子 Actor 參數數量: {actor.total_params}")
        else:
            # 創建對齊的古典 Actor
            quantum_actor = QuantumActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                device=device
            )
            target_params = quantum_actor.count_parameters()
            
            actor = AlignedClassicalActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                target_params=target_params,
                device=device
            )
            logger.info(f"古典 Actor 參數數量: {actor.total_params}")
            logger.info(f"目標參數數量: {target_params}")
        
        # 創建 Critic
        logger.info("創建 Critic...")
        critic = MLPCritic(
            state_dim=env.observation_space.shape[0],
            device=device
        )
        logger.info(f"Critic 參數數量: {sum(p.numel() for p in critic.parameters())}")
        
        # 創建獎勵函數
        logger.info("創建獎勵函數...")
        classifier_reward = ClassifierReward(
            target_class=cfg.dataset.cifar10.target_class,
            device=device
        )
        
        diversity_reward = DiversityReward(device=device)
        
        # 創建訓練器
        logger.info("創建 PPO 訓練器...")
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
                mixed_precision=cfg.algo.ppo.mixed_precision,
                use_quantum_gradients=cfg.algo.ppo.use_quantum_gradients
            )
        
        # 訓練循環
        logger.info("開始訓練...")
        best_reward = float('-inf')
        best_episode = 0
        
        for episode in range(cfg.training.num_episodes):
            episode_start_time = time.time()
            logger.info(f"=== Episode {episode+1}/{cfg.training.num_episodes} ===")
            
            # 收集經驗
            logger.info("  收集經驗數據...")
            rollout_data = trainer.collect_rollouts(env, cfg.algo.ppo.rollout_steps)
            logger.info(f"  收集完成，數據大小: {len(rollout_data['states'])}")
            
            # 更新策略
            logger.info("  開始策略更新...")
            update_info = trainer.update(rollout_data)
            logger.info("  策略更新完成")
            
            # 評估
            if episode % cfg.training.eval_freq == 0:
                logger.info("  開始策略評估...")
                eval_reward = trainer.evaluate(env, num_episodes=cfg.training.eval_episodes)
                logger.info(f"  評估完成，獎勵: {eval_reward:.4f}")
                
                # 記錄指標
                metrics = {
                    'episode': episode,
                    'eval_reward': eval_reward,
                    'actor_loss': update_info.get('actor_loss', 0.0),
                    'critic_loss': update_info.get('critic_loss', 0.0),
                    'entropy': update_info.get('entropy', 0.0),
                    'clip_ratio': update_info.get('clip_ratio', 0.0),
                    'value_loss': update_info.get('value_loss', 0.0),
                    'episode_time': time.time() - episode_start_time
                }
                
                metrics_logger.log_metrics(metrics, step=episode)
                
                # 檢查最佳模型
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    best_episode = episode
                    trainer.save_checkpoint(work_dir / "ckpt_best.pt")
                    logger.info(f"  🎉 新的最佳模型！獎勵: {best_reward:.4f}")
                else:
                    logger.info(f"  當前最佳獎勵: {best_reward:.4f}")
                
                # 早停檢查
                if cfg.training.early_stopping.enabled:
                    if episode - best_episode >= cfg.training.early_stopping.patience:
                        logger.info(f"  早停觸發！{cfg.training.early_stopping.patience} 個 episode 無改善")
                        break
            
            # 定期保存檢查點
            if episode % cfg.checkpoint.save_freq == 0:
                trainer.save_checkpoint(work_dir / f"ckpt_episode_{episode}.pt")
                logger.info(f"  檢查點已保存: ckpt_episode_{episode}.pt")
            
            # 記錄進度
            if episode % cfg.logging.log_freq == 0:
                episode_time = time.time() - episode_start_time
                if episode % cfg.training.eval_freq == 0:
                    logger.info(
                        f"Episode {episode+1}/{cfg.training.num_episodes} 完成 | "
                        f"Eval Reward: {eval_reward:.4f} | "
                        f"Actor Loss: {update_info.get('actor_loss', 0.0):.4f} | "
                        f"Critic Loss: {update_info.get('critic_loss', 0.0):.4f} | "
                        f"時間: {episode_time:.2f}s"
                    )
                else:
                    logger.info(
                        f"Episode {episode+1}/{cfg.training.num_episodes} 完成 | "
                        f"Actor Loss: {update_info.get('actor_loss', 0.0):.4f} | "
                        f"Critic Loss: {update_info.get('critic_loss', 0.0):.4f} | "
                        f"時間: {episode_time:.2f}s"
                    )
        
        # 保存最終模型
        trainer.save_checkpoint(work_dir / "ckpt_final.pt")
        
        # 保存訓練配置
        config_file = work_dir / "training_config.yaml"
        with open(config_file, 'w') as f:
            OmegaConf.save(config=cfg, f=f)
        
        # 保存訓練摘要
        summary = {
            'best_reward': best_reward,
            'best_episode': best_episode,
            'total_episodes': episode + 1,
            'final_reward': eval_reward,
            'training_time': time.time() - episode_start_time,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }
        
        summary_file = work_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("訓練完成！")
        logger.info(f"最佳獎勵: {best_reward:.4f} (Episode {best_episode})")
        logger.info(f"最終獎勵: {eval_reward:.4f}")
        logger.info(f"訓練配置已保存到: {config_file}")
        logger.info(f"訓練摘要已保存到: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"訓練過程中發生錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def create_training_config(
    dataset: str = "cifar10",
    control: str = "stageA_cfg",
    algo: str = "ppo",
    model: str = "unet_ddim",
    **kwargs
) -> DictConfig:
    """
    創建訓練配置。
    
    Args:
        dataset: 數據集配置
        control: 控制配置
        algo: 算法配置
        model: 模型配置
        **kwargs: 其他配置參數
        
    Returns:
        DictConfig: 訓練配置
    """
    # 基礎配置
    base_config = {
        'seed': 42,
        'work_dir': 'runs/training',
        'device': 'cuda',
        'precision': 'float32',
        'deterministic': False,
        'logging': {
            'use_tensorboard': True,
            'log_freq': 10
        },
        'checkpointing': {
            'save_freq': 100
        }
    }
    
    # 合併配置
    config = OmegaConf.create(base_config)
    
    # 載入子配置
    config_dir = Path(__file__).parent.parent / "qrl" / "configs"
    
    if dataset:
        dataset_config = OmegaConf.load(config_dir / "dataset" / f"{dataset}.yaml")
        config.dataset = dataset_config
    
    if control:
        control_config = OmegaConf.load(config_dir / "control" / f"{control}.yaml")
        config.control = control_config
    
    if algo:
        algo_config = OmegaConf.load(config_dir / "algo" / f"{algo}.yaml")
        config.algo = algo_config
    
    if model:
        model_config = OmegaConf.load(config_dir / "model" / f"{model}.yaml")
        config.model = model_config
    
    # 更新自定義參數
    for key, value in kwargs.items():
        OmegaConf.update(config, key, value)
    
    return config


if __name__ == "__main__":
    # 如果直接運行，使用默認配置
    if len(sys.argv) == 1:
        # 創建默認配置
        config = create_training_config()
        main(config)
    else:
        # 使用 Hydra 配置
        main()
