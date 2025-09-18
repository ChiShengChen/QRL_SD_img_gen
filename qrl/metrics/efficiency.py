"""效率評估指標。"""

import torch
import numpy as np
import time
from typing import List, Dict, Any, Union
import psutil
import os


class EfficiencyMetrics:
    """效率評估指標。"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_all(self, model: torch.nn.Module, generated_images: List[torch.Tensor], 
                   episode_rewards: List[float]) -> Dict[str, float]:
        """計算所有效率指標。"""
        metrics = {}
        
        # 參數效率
        metrics['parameter_efficiency'] = self.compute_parameter_efficiency(model, episode_rewards)
        
        # 推理效率
        metrics['inference_efficiency'] = self.compute_inference_efficiency(model)
        
        # 記憶體效率
        metrics['memory_efficiency'] = self.compute_memory_efficiency(model)
        
        # 訓練效率 (需要訓練時間，這裡使用獎勵作為代理)
        metrics['training_efficiency'] = self.compute_training_efficiency(episode_rewards)
        
        # 計算效率
        metrics['computational_efficiency'] = self.compute_computational_efficiency(model, generated_images)
        
        return metrics
    
    def compute_parameter_efficiency(self, model: torch.nn.Module, episode_rewards: List[float]) -> float:
        """計算參數效率。"""
        # 計算模型參數數量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 計算平均獎勵
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        
        # 參數效率 = 性能 / 參數數量
        if total_params > 0:
            efficiency = avg_reward / total_params
        else:
            efficiency = 0.0
        
        return efficiency
    
    def compute_inference_efficiency(self, model: torch.nn.Module) -> float:
        """計算推理效率。"""
        model.eval()
        
        # 創建測試輸入
        batch_size = 8
        state_dim = 6  # 使用實際的狀態維度
        test_input = torch.randn(batch_size, state_dim).to(self.device)
        
        # 測量推理時間
        num_runs = 100
        times = []
        
        with torch.no_grad():
            # 預熱
            for _ in range(10):
                _ = model(test_input)
            
            # 測量時間
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(test_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        # 計算平均推理時間
        avg_time = np.mean(times)
        
        # 推理效率 = 1 / 推理時間
        efficiency = 1.0 / (avg_time + 1e-8)
        
        return efficiency
    
    def compute_memory_efficiency(self, model: torch.nn.Module) -> float:
        """計算記憶體效率。"""
        # 計算模型大小
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size / (1024 * 1024)  # 轉換為MB
        
        # 測量運行時記憶體使用
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # 運行模型
        model.eval()
        test_input = torch.randn(1, 6).to(self.device)
        
        with torch.no_grad():
            _ = model(test_input)
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        runtime_memory = memory_after - memory_before
        
        # 記憶體效率 = 1 / (模型大小 + 運行記憶體)
        total_memory = model_size_mb + runtime_memory
        efficiency = 1.0 / (total_memory + 1e-8)
        
        return efficiency
    
    def compute_training_efficiency(self, episode_rewards: List[float]) -> float:
        """計算訓練效率。"""
        if not episode_rewards:
            return 0.0
        
        # 計算獎勵改善速度
        if len(episode_rewards) < 2:
            return 0.0
        
        # 計算獎勵梯度
        rewards_array = np.array(episode_rewards)
        
        # 使用移動平均平滑
        window_size = min(10, len(rewards_array) // 4)
        if window_size < 2:
            return 0.0
        
        smoothed_rewards = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
        
        # 計算改善速度
        if len(smoothed_rewards) < 2:
            return 0.0
        
        improvement_rate = (smoothed_rewards[-1] - smoothed_rewards[0]) / len(smoothed_rewards)
        
        # 訓練效率 = 改善速度
        efficiency = max(0.0, improvement_rate)
        
        return efficiency
    
    def compute_computational_efficiency(self, model: torch.nn.Module, generated_images: List[torch.Tensor]) -> float:
        """計算計算效率。"""
        if not generated_images:
            return 0.0
        
        # 計算模型參數數量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 計算生成的圖像數量
        num_images = len(generated_images)
        
        # 計算效率 = 生成圖像數量 / 參數數量
        if total_params > 0:
            efficiency = num_images / total_params
        else:
            efficiency = 0.0
        
        return efficiency
    
    def compute_quantum_efficiency(self, model: torch.nn.Module) -> float:
        """計算量子效率 (僅適用於量子模型)。"""
        # 檢查是否為量子模型
        if not hasattr(model, 'quantum_weights'):
            return 0.0
        
        # 計算量子參數數量
        quantum_params = model.quantum_weights.numel()
        total_params = sum(p.numel() for p in model.parameters())
        
        # 量子效率 = 量子參數比例
        if total_params > 0:
            quantum_ratio = quantum_params / total_params
        else:
            quantum_ratio = 0.0
        
        return quantum_ratio
    
    def compute_energy_efficiency(self, model: torch.nn.Module, inference_time: float) -> float:
        """計算能源效率。"""
        # 計算模型大小
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size / (1024 * 1024)
        
        # 能源效率 = 模型大小 / 推理時間
        if inference_time > 0:
            efficiency = model_size_mb / inference_time
        else:
            efficiency = 0.0
        
        return efficiency
    
    def compute_scalability(self, model: torch.nn.Module, batch_sizes: List[int] = [1, 4, 8, 16]) -> float:
        """計算可擴展性。"""
        model.eval()
        state_dim = 6
        
        scalability_scores = []
        
        for batch_size in batch_sizes:
            try:
                test_input = torch.randn(batch_size, state_dim).to(self.device)
                
                # 測量推理時間
                times = []
                with torch.no_grad():
                    for _ in range(10):
                        start_time = time.time()
                        _ = model(test_input)
                        end_time = time.time()
                        times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                
                # 計算吞吐量 (樣本/秒)
                throughput = batch_size / avg_time
                scalability_scores.append(throughput)
                
            except Exception as e:
                # 如果批次大小太大導致記憶體不足，跳過
                break
        
        if not scalability_scores:
            return 0.0
        
        # 可擴展性 = 最大吞吐量 / 最小吞吐量
        max_throughput = max(scalability_scores)
        min_throughput = min(scalability_scores)
        
        if min_throughput > 0:
            scalability = max_throughput / min_throughput
        else:
            scalability = 0.0
        
        return scalability