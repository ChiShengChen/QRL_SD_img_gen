"""擴散環境模組，實現 Gymnasium 風格的擴散序列決策環境。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Tuple, List
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import gymnasium as gym

from diffusers import (
    UNet2DConditionModel,
    DDIMScheduler,
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer

from ..controls import ControlApplier
from ..reward import ClassifierReward
from ..utils.logging import get_logger


class DiffusionEnvironment(Env):
    """擴散序列決策環境。"""
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        target_class: int = 3,
        max_steps: int = 50,
        device: str = "cuda",
        stage: str = "A",
        reward_config: Optional[Dict] = None,
        control_config: Optional[Dict] = None
    ):
        """
        初始化擴散環境。
        
        Args:
            model_id: 預訓練模型 ID
            target_class: 目標類別
            max_steps: 最大步數
            device: 設備
            stage: 控制階段 ("A", "B", "C")
            reward_config: 獎勵配置
            control_config: 控制配置
        """
        super().__init__()
        
        self.model_id = model_id
        self.target_class = target_class
        self.max_steps = max_steps
        self.device = device
        self.stage = stage
        
        # 初始化 logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化模型組件
        self._init_models()
        
        # 初始化控制應用器
        self.control_applier = ControlApplier(stage=stage)
        
        # 初始化獎勵函數
        reward_params = reward_config or {}
        # 映射配置參數到 ClassifierReward 參數
        mapped_params = {
            'target_class': target_class,
            'device': device,
            'alpha': reward_params.get('alpha_cls', 1.0),
            'beta': reward_params.get('beta_step', 0.2),
            'action_penalty': reward_params.get('action_penalty', 0.005),
            'tv_penalty': reward_params.get('tv_penalty', 0.0)
        }
        self.reward_fn = ClassifierReward(**mapped_params)
        
        # 設置動作和觀察空間
        self._setup_spaces()
        
        # 環境狀態
        self.reset()
        
        # 日誌記錄器
        self.logger = get_logger("diffusion_env")
        
        self.logger.info(f"擴散環境初始化完成，階段: {stage}")
    
    def _init_models(self):
        """初始化擴散模型組件。"""
        # 載入 UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device)
        
        # 載入 VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae"
        ).to(self.device)
        
        # 載入文本編碼器
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )
        
        # 載入調度器
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        
        # 設置調度器時間步
        self.scheduler.set_timesteps(self.max_steps)
        
        # 凍結模型參數
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        self.logger.info("擴散模型組件載入完成")
    
    def _setup_spaces(self):
        """設置動作和觀察空間。"""
        # 動作空間：根據階段設置
        if self.stage == "A":
            # 僅 CFG 控制
            self.action_space = Box(
                low=np.array([-2.0]),
                high=np.array([2.0]),
                dtype=np.float32
            )
        elif self.stage == "B":
            # CFG + 注意力門控
            self.action_space = Box(
                low=np.array([-2.0, 0.0]),
                high=np.array([2.0, 1.0]),
                dtype=np.float32
            )
        elif self.stage == "C":
            # 完整控制
            self.action_space = Box(
                low=np.array([-2.0, 0.0, 0.5]),
                high=np.array([2.0, 1.0, 2.0]),
                dtype=np.float32
            )
        else:
            raise ValueError(f"未知的階段: {self.stage}")
        
        # 觀察空間：狀態向量
        # [timestep, latent_norm, noise_norm, latent_noise_dot, prev_action, proxy_confidence]
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, -2.0, 0.0]),
            high=np.array([1.0, 10.0, 10.0, 1.0, 2.0, 1.0]),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置環境。
        
        Args:
            seed: 隨機種子
            
        Returns:
            Tuple: (觀察, 信息)
        """
        super().reset(seed=seed)
        
        # 重置環境狀態
        self.current_step = 0
        self.prev_action = np.zeros(self.action_space.shape[0])
        self.proxy_confidence = 0.0
        
        # 初始化潛在變數
        self.latent = torch.randn(
            1, 4, 64, 64,  # [batch, channels, height, width]
            device=self.device
        )
        
        # 設置文本提示
        self.text_prompt = self._get_class_prompt(self.target_class)
        self.text_embeddings = self._encode_text(self.text_prompt)
        
        # 計算初始觀察
        observation = self._compute_observation()
        
        info = {
            'step': self.current_step,
            'target_class': self.target_class,
            'text_prompt': self.text_prompt
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        執行動作。
        
        Args:
            action: 動作向量
            
        Returns:
            Tuple: (觀察, 獎勵, 終止, 截斷, 信息)
        """
        # 驗證動作
        if not self.action_space.contains(action):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 應用控制
        control_result = self.control_applier.apply_controls(
            action=action,
            base_cfg=7.5,  # 基礎 CFG 值
            attention_weights=None,  # 暫時不使用注意力門控
            step_size=1.0  # 暫時不使用步長縮放
        )
        
        # 執行擴散步驟
        self._diffusion_step(control_result)
        
        # 更新狀態
        self.current_step += 1
        self.prev_action = action.copy()
        
        # 計算獎勵
        reward = self._compute_reward(action)
        
        # 檢查是否終止
        done = self.current_step >= self.max_steps
        
        # 計算觀察
        observation = self._compute_observation()
        
        # 信息
        info = {
            'step': self.current_step,
            'action': action,
            'control_result': control_result,
            'reward': reward,
            'done': done
        }
        
        return observation, reward, done, False, info
    
    def _diffusion_step(self, control_result: Dict[str, Any]):
        """執行擴散步驟。"""
        # 獲取當前時間步
        timestep = self.scheduler.timesteps[self.current_step]
        
        # 預測噪聲
        with torch.no_grad():
            noise_pred = self.unet(
                self.latent,
                timestep,
                encoder_hidden_states=self.text_embeddings
            ).sample
        
        # 應用 CFG
        cfg = control_result['cfg']
        if cfg > 1.0:
            # 無條件預測
            uncond_embeddings = self._encode_text("")
            uncond_noise_pred = self.unet(
                self.latent,
                timestep,
                encoder_hidden_states=uncond_embeddings
            ).sample
            
            # 應用 CFG
            noise_pred = uncond_noise_pred + cfg * (noise_pred - uncond_noise_pred)
        
        # 去噪步驟
        self.latent = self.scheduler.step(
            noise_pred,
            timestep,
            self.latent
        ).prev_sample
        
        # 更新代理分類器置信度
        self._update_proxy_confidence()
    
    def _compute_observation(self) -> np.ndarray:
        """計算觀察向量。"""
        # 計算潛在變數範數
        latent_norm = torch.norm(self.latent).item()
        
        # 計算噪聲預測範數（簡化）
        noise_norm = torch.norm(self.latent).item() * 0.1  # 簡化計算
        
        # 計算潛在變數與噪聲的點積（簡化）
        latent_noise_dot = torch.sum(self.latent * self.latent * 0.1).item()
        
        # 標準化時間步
        timestep_norm = self.current_step / self.max_steps
        
        # 構建觀察向量
        observation = np.array([
            timestep_norm,           # 當前步數 / 總步數
            latent_norm,             # 潛在變數 L2 範數
            noise_norm,              # 預測噪聲 L2 範數
            latent_noise_dot,        # 潛在變數與噪聲的點積
            self.prev_action[0],     # 上一步動作（假設單一動作）
            self.proxy_confidence    # 代理分類器置信度
        ], dtype=np.float32)
        
        # 標準化觀察
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        
        return observation
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """計算獎勵。"""
        # 解碼圖像
        image = self._decode_latent()
        
        # 計算獎勵
        rewards = self.reward_fn.compute_reward(
            image=image,
            action=torch.tensor(action, device=self.device),
            is_terminal=(self.current_step >= self.max_steps - 1)
        )
        
        return rewards['total'].item()
    
    def _decode_latent(self) -> torch.Tensor:
        """解碼潛在變數為圖像。"""
        with torch.no_grad():
            # 縮放潛在變數
            latent_scaled = 1 / 0.18215 * self.latent
            
            # VAE 解碼
            image = self.vae.decode(latent_scaled).sample
            
            # 轉換到 [0, 1] 範圍
            image = (image / 2 + 0.5).clamp(0, 1)
            
            return image
    
    def _get_class_prompt(self, target_class: int) -> str:
        """根據目標類別獲取文本提示。"""
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        if 0 <= target_class < len(class_names):
            return f"a photo of a {class_names[target_class]}"
        else:
            return "a photo of an object"
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """編碼文本為嵌入向量。"""
        # 標記化
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 編碼
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                tokens.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def _update_proxy_confidence(self):
        """更新代理分類器置信度。"""
        # 簡化：使用潛在變數的統計特性作為代理置信度
        latent_mean = torch.mean(self.latent).item()
        latent_std = torch.std(self.latent).item()
        
        # 基於潛在變數的穩定性和結構計算置信度
        stability = 1.0 / (1.0 + abs(latent_mean))
        structure = 1.0 / (1.0 + latent_std)
        
        self.proxy_confidence = (stability + structure) / 2.0
    
    def render(self):
        """渲染當前狀態。"""
        image = self._decode_latent()
        return image
    
    def close(self):
        """關閉環境。"""
        # 清理資源
        del self.unet, self.vae, self.text_encoder, self.scheduler
        torch.cuda.empty_cache()
    
    def get_state_info(self) -> Dict[str, Any]:
        """獲取環境狀態信息。"""
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'target_class': self.target_class,
            'stage': self.stage,
            'device': self.device,
            'model_id': self.model_id
        }


# 便捷函數
def create_diffusion_env(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    target_class: int = 3,
    stage: str = "A",
    **kwargs
) -> DiffusionEnvironment:
    """創建擴散環境的便捷函數。"""
    return DiffusionEnvironment(
        model_id=model_id,
        target_class=target_class,
        stage=stage,
        **kwargs
    )
