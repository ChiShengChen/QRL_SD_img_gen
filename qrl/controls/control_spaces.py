"""控制空間定義模組，定義不同階段的動作空間。"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from gymnasium.spaces import Box, Space
import pennylane as qml


class ControlSpaces:
    """控制空間定義類別。"""
    
    @staticmethod
    def stageA_cfg() -> Box:
        """
        Stage A: CFG 增量控制空間。
        
        Returns:
            Box: 連續動作空間，範圍 [-2.0, 2.0]
        """
        return Box(
            low=np.array([-2.0]),
            high=np.array([2.0]),
            dtype=np.float32,
            shape=(1,)
        )
    
    @staticmethod
    def stageB_attn() -> Box:
        """
        Stage B: 注意力門控控制空間。
        
        Returns:
            Box: 連續動作空間，範圍 [0.0, 1.0]
        """
        return Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32,
            shape=(1,)
        )
    
    @staticmethod
    def stageC_full() -> Box:
        """
        Stage C: 完整控制空間（CFG + 注意力門控 + 步長因子）。
        
        Returns:
            Box: 連續動作空間
        """
        return Box(
            low=np.array([-2.0, 0.0, 0.5]),  # [delta_cfg, attn_gate, step_scale]
            high=np.array([2.0, 1.0, 2.0]),
            dtype=np.float32,
            shape=(3,)
        )
    
    @staticmethod
    def get_stage_space(stage: str) -> Box:
        """
        根據階段獲取對應的控制空間。
        
        Args:
            stage: 階段名稱 ("A", "B", "C")
            
        Returns:
            Box: 對應的控制空間
        """
        stage_mapping = {
            "A": ControlSpaces.stageA_cfg,
            "B": ControlSpaces.stageB_attn,
            "C": ControlSpaces.stageC_full,
        }
        
        if stage not in stage_mapping:
            raise ValueError(f"未知的階段: {stage}，支援的階段: {list(stage_mapping.keys())}")
        
        return stage_mapping[stage]()
    
    @staticmethod
    def get_action_dim(stage: str) -> int:
        """
        獲取指定階段的動作維度。
        
        Args:
            stage: 階段名稱
            
        Returns:
            int: 動作維度
        """
        space = ControlSpaces.get_stage_space(stage)
        return space.shape[0]
    
    @staticmethod
    def get_action_bounds(stage: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        獲取指定階段的動作邊界。
        
        Args:
            stage: 階段名稱
            
        Returns:
            Tuple: (low_bound, high_bound)
        """
        space = ControlSpaces.get_stage_space(stage)
        return space.low, space.high


class ActionProcessor:
    """動作處理器，用於處理和驗證動作。"""
    
    def __init__(self, stage: str):
        """
        初始化動作處理器。
        
        Args:
            stage: 當前階段
        """
        self.stage = stage
        self.space = ControlSpaces.get_stage_space(stage)
        self.action_dim = ControlSpaces.get_action_dim(stage)
    
    def process_action(self, action: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        處理動作，包括裁剪和類型轉換。
        
        Args:
            action: 原始動作
            
        Returns:
            np.ndarray: 處理後的動作
        """
        # 轉換為 numpy 陣列
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # 確保形狀正確
        if action.shape != (self.action_dim,):
            raise ValueError(f"動作形狀錯誤: {action.shape}，期望: ({self.action_dim},)")
        
        # 裁剪到有效範圍
        processed_action = np.clip(action, self.space.low, self.space.high)
        
        return processed_action.astype(np.float32)
    
    def validate_action(self, action: np.ndarray) -> bool:
        """
        驗證動作是否在有效範圍內。
        
        Args:
            action: 動作
            
        Returns:
            bool: 是否有效
        """
        return np.all((action >= self.space.low) & (action <= self.space.high))
    
    def get_action_info(self) -> Dict:
        """獲取動作空間信息。"""
        return {
            "stage": self.stage,
            "dimension": self.action_dim,
            "low_bound": self.space.low,
            "high_bound": self.space.high,
            "dtype": self.space.dtype,
        }


# 便捷函數
def get_control_space(stage: str) -> Box:
    """獲取控制空間的便捷函數。"""
    return ControlSpaces.get_stage_space(stage)


def get_action_processor(stage: str) -> ActionProcessor:
    """獲取動作處理器的便捷函數。"""
    return ActionProcessor(stage)
