"""控制應用模組，實現不同控制參數的應用邏輯。"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from .control_spaces import ActionProcessor


def apply_cfg_delta(base_cfg: float, delta: Union[float, np.ndarray, torch.Tensor]) -> float:
    """
    應用 CFG 增量控制。
    
    Args:
        base_cfg: 基礎 CFG 值
        delta: CFG 增量
        
    Returns:
        float: 應用後的 CFG 值，裁剪到 [1, 12] 範圍
    """
    # 轉換為 float
    if isinstance(delta, (np.ndarray, torch.Tensor)):
        delta = float(delta.item() if delta.numel() == 1 else delta[0])
    
    # 計算新的 CFG 值
    new_cfg = base_cfg + delta
    
    # 裁剪到有效範圍 [1, 12]
    new_cfg = np.clip(new_cfg, 1.0, 12.0)
    
    return float(new_cfg)


def apply_attention_gating(
    attention_weights: torch.Tensor,
    gate_value: Union[float, np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """
    應用注意力門控控制。
    
    Args:
        attention_weights: 注意力權重張量
        gate_value: 門控值 [0, 1]
        
    Returns:
        torch.Tensor: 應用門控後的注意力權重
    """
    # 轉換為 float
    if isinstance(gate_value, (np.ndarray, torch.Tensor)):
        gate_value = float(gate_value.item() if gate_value.numel() == 1 else gate_value[0])
    
    # 裁剪到 [0, 1] 範圍
    gate_value = np.clip(gate_value, 0.0, 1.0)
    
    # 應用門控
    gated_weights = attention_weights * gate_value
    
    return gated_weights


def apply_step_scaling(
    step_size: float,
    scale_factor: Union[float, np.ndarray, torch.Tensor]
) -> float:
    """
    應用步長因子控制。
    
    Args:
        step_size: 原始步長
        scale_factor: 縮放因子 [0.5, 2.0]
        
    Returns:
        float: 應用縮放後的步長
    """
    # 轉換為 float
    if isinstance(scale_factor, (np.ndarray, torch.Tensor)):
        scale_factor = float(scale_factor.item() if scale_factor.numel() == 1 else scale_factor[0])
    
    # 裁剪到 [0.5, 2.0] 範圍
    scale_factor = np.clip(scale_factor, 0.5, 2.0)
    
    # 應用縮放
    new_step_size = step_size * scale_factor
    
    return float(new_step_size)


class ControlApplier:
    """控制應用器，統一管理所有控制參數的應用。"""
    
    def __init__(self, stage: str = "A"):
        """
        初始化控制應用器。
        
        Args:
            stage: 當前階段
        """
        self.stage = stage
        self.action_processor = ActionProcessor(stage)
        
        # 根據階段設置啟用的控制
        self.enabled_controls = self._get_enabled_controls()
    
    def _get_enabled_controls(self) -> Dict[str, bool]:
        """根據階段獲取啟用的控制參數。"""
        if self.stage == "A":
            return {"delta_cfg": True, "attn_gating": False, "step_scale": False}
        elif self.stage == "B":
            return {"delta_cfg": True, "attn_gating": True, "step_scale": False}
        elif self.stage == "C":
            return {"delta_cfg": True, "attn_gating": True, "step_scale": True}
        else:
            raise ValueError(f"未知的階段: {self.stage}")
    
    def apply_controls(
        self,
        action: Union[np.ndarray, torch.Tensor],
        base_cfg: float = 5.0,
        attention_weights: Optional[torch.Tensor] = None,
        step_size: float = 1.0,
    ) -> Dict[str, Any]:
        """
        應用控制參數。
        
        Args:
            action: 動作向量
            base_cfg: 基礎 CFG 值
            attention_weights: 注意力權重（如果需要門控）
            step_size: 原始步長（如果需要縮放）
            
        Returns:
            Dict: 應用後的參數字典
        """
        # 處理動作
        processed_action = self.action_processor.process_action(action)
        
        # 初始化結果
        result = {
            "cfg": base_cfg,
            "attention_weights": attention_weights,
            "step_size": step_size,
            "applied_controls": {}
        }
        
        # 根據階段應用不同的控制
        if self.stage == "A":
            # 僅應用 CFG 控制
            if self.enabled_controls["delta_cfg"]:
                delta_cfg = processed_action[0]
                new_cfg = apply_cfg_delta(base_cfg, delta_cfg)
                result["cfg"] = new_cfg
                result["applied_controls"]["delta_cfg"] = delta_cfg
                
        elif self.stage == "B":
            # 應用 CFG 和注意力門控
            if self.enabled_controls["delta_cfg"]:
                delta_cfg = processed_action[0]
                new_cfg = apply_cfg_delta(base_cfg, delta_cfg)
                result["cfg"] = new_cfg
                result["applied_controls"]["delta_cfg"] = delta_cfg
            
            if self.enabled_controls["attn_gating"] and attention_weights is not None:
                gate_value = processed_action[1]
                gated_weights = apply_attention_gating(attention_weights, gate_value)
                result["attention_weights"] = gated_weights
                result["applied_controls"]["attn_gating"] = gate_value
                
        elif self.stage == "C":
            # 應用所有控制
            if self.enabled_controls["delta_cfg"]:
                delta_cfg = processed_action[0]
                new_cfg = apply_cfg_delta(base_cfg, delta_cfg)
                result["cfg"] = new_cfg
                result["applied_controls"]["delta_cfg"] = delta_cfg
            
            if self.enabled_controls["attn_gating"] and attention_weights is not None:
                gate_value = processed_action[1]
                gated_weights = apply_attention_gating(attention_weights, gate_value)
                result["attention_weights"] = gated_weights
                result["applied_controls"]["attn_gating"] = gate_value
            
            if self.enabled_controls["step_scale"]:
                scale_factor = processed_action[2]
                new_step_size = apply_step_scaling(step_size, scale_factor)
                result["step_size"] = new_step_size
                result["applied_controls"]["step_scale"] = scale_factor
        
        return result
    
    def get_control_info(self) -> Dict[str, Any]:
        """獲取控制信息。"""
        return {
            "stage": self.stage,
            "enabled_controls": self.enabled_controls,
            "action_info": self.action_processor.get_action_info(),
        }


# 便捷函數
def apply_stage_a_controls(
    delta_cfg: Union[float, np.ndarray, torch.Tensor],
    base_cfg: float = 5.0
) -> float:
    """便捷函數：應用 Stage A 的 CFG 控制。"""
    return apply_cfg_delta(base_cfg, delta_cfg)


def create_control_applier(stage: str = "A") -> ControlApplier:
    """便捷函數：創建控制應用器。"""
    return ControlApplier(stage)
