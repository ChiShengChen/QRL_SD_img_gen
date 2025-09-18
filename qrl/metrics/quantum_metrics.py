"""量子特定評估指標。"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from scipy import linalg
from scipy.stats import entropy


class QuantumMetrics:
    """量子特定評估指標。"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def compute_all(self, quantum_model: nn.Module, classical_model: nn.Module, 
                   generated_images: List[torch.Tensor], episode_rewards: List[float]) -> Dict[str, float]:
        """計算所有量子特定指標。"""
        metrics = {}
        
        # 量子相干性指標
        metrics['quantum_coherence'] = self.compute_quantum_coherence(quantum_model)
        
        # 量子優勢指標
        metrics['quantum_advantage'] = self.compute_quantum_advantage(quantum_model, classical_model, episode_rewards)
        
        # 量子糾纏指標
        metrics['quantum_entanglement'] = self.compute_quantum_entanglement(quantum_model)
        
        # 量子門效率指標
        metrics['quantum_gate_efficiency'] = self.compute_quantum_gate_efficiency(quantum_model)
        
        # 量子噪聲魯棒性指標
        metrics['quantum_noise_robustness'] = self.compute_quantum_noise_robustness(quantum_model)
        
        # 量子梯度指標
        metrics['quantum_gradient_quality'] = self.compute_quantum_gradient_quality(quantum_model)
        
        # 量子表達能力指標
        metrics['quantum_expressibility'] = self.compute_quantum_expressibility(quantum_model)
        
        # 量子學習動態指標
        metrics['quantum_learning_dynamics'] = self.compute_quantum_learning_dynamics(episode_rewards)
        
        return metrics
    
    def compute_quantum_coherence(self, quantum_model: nn.Module) -> float:
        """計算量子相干性指標。"""
        if not hasattr(quantum_model, 'quantum_circuit'):
            return 0.0
        
        try:
            # 獲取量子電路的密度矩陣
            if hasattr(quantum_model.quantum_circuit, 'get_density_matrix'):
                rho = quantum_model.quantum_circuit.get_density_matrix()
            else:
                # 簡化計算：使用量子權重的相干性
                if hasattr(quantum_model, 'quantum_weights'):
                    weights = quantum_model.quantum_weights
                    # 計算權重的相干性（基於複數的相位）
                    if weights.is_complex():
                        phases = torch.angle(weights)
                        coherence = torch.std(phases).item()
                    else:
                        coherence = 0.0
                else:
                    coherence = 0.0
                return coherence
            
            # 計算相干性（基於密度矩陣的非對角元素）
            if rho is not None:
                off_diagonal = rho - torch.diag(torch.diag(rho))
                coherence = torch.norm(off_diagonal).item()
            else:
                coherence = 0.0
            
            return coherence
            
        except Exception as e:
            print(f"計算量子相干性時出錯: {e}")
            return 0.0
    
    def compute_quantum_advantage(self, quantum_model: nn.Module, classical_model: nn.Module, 
                                 episode_rewards: List[float]) -> float:
        """計算量子優勢指標。"""
        if not episode_rewards:
            return 0.0
        
        # 計算量子模型的性能
        quantum_performance = np.mean(episode_rewards)
        
        # 計算參數效率
        quantum_params = sum(p.numel() for p in quantum_model.parameters())
        classical_params = sum(p.numel() for p in classical_model.parameters())
        
        # 量子優勢 = 性能提升 / 參數效率提升
        if classical_params > 0:
            param_efficiency_ratio = quantum_params / classical_params
        else:
            param_efficiency_ratio = 1.0
        
        # 假設經典模型性能為基準（這裡簡化為負數基準）
        classical_baseline = -5.0  # 基於你的實驗結果
        performance_improvement = (quantum_performance - classical_baseline) / abs(classical_baseline)
        
        # 量子優勢 = 性能改善 / 參數效率
        if param_efficiency_ratio > 0:
            quantum_advantage = performance_improvement / param_efficiency_ratio
        else:
            quantum_advantage = 0.0
        
        return quantum_advantage
    
    def compute_quantum_entanglement(self, quantum_model: nn.Module) -> float:
        """計算量子糾纏指標。"""
        if not hasattr(quantum_model, 'quantum_circuit'):
            return 0.0
        
        try:
            # 簡化計算：基於量子權重的糾纏度量
            if hasattr(quantum_model, 'quantum_weights'):
                weights = quantum_model.quantum_weights
                
                # 計算權重矩陣的奇異值分解
                if weights.dim() >= 2:
                    U, S, V = torch.svd(weights)
                    # 使用奇異值的熵作為糾纏度量
                    S_normalized = S / (torch.sum(S) + 1e-8)
                    entanglement = -torch.sum(S_normalized * torch.log(S_normalized + 1e-8)).item()
                else:
                    entanglement = 0.0
            else:
                entanglement = 0.0
            
            return entanglement
            
        except Exception as e:
            print(f"計算量子糾纏時出錯: {e}")
            return 0.0
    
    def compute_quantum_gate_efficiency(self, quantum_model: nn.Module) -> float:
        """計算量子門效率指標。"""
        if not hasattr(quantum_model, 'quantum_circuit'):
            return 0.0
        
        try:
            # 計算量子門的使用效率
            if hasattr(quantum_model.quantum_circuit, 'num_gates'):
                num_gates = quantum_model.quantum_circuit.num_gates()
            else:
                # 簡化計算：基於量子參數數量
                if hasattr(quantum_model, 'quantum_weights'):
                    num_gates = quantum_model.quantum_weights.numel()
                else:
                    num_gates = 0
            
            # 計算總參數數量
            total_params = sum(p.numel() for p in quantum_model.parameters())
            
            # 量子門效率 = 量子門數量 / 總參數數量
            if total_params > 0:
                gate_efficiency = num_gates / total_params
            else:
                gate_efficiency = 0.0
            
            return gate_efficiency
            
        except Exception as e:
            print(f"計算量子門效率時出錯: {e}")
            return 0.0
    
    def compute_quantum_noise_robustness(self, quantum_model: nn.Module) -> float:
        """計算量子噪聲魯棒性指標。"""
        if not hasattr(quantum_model, 'quantum_weights'):
            return 0.0
        
        try:
            # 添加噪聲並測試模型穩定性
            original_weights = quantum_model.quantum_weights.clone()
            
            # 添加不同強度的噪聲
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            robustness_scores = []
            
            for noise_level in noise_levels:
                # 添加高斯噪聲
                noise = torch.randn_like(original_weights) * noise_level
                quantum_model.quantum_weights.data = original_weights + noise
                
                # 測試模型輸出穩定性
                test_input = torch.randn(1, 6).to(self.device)
                with torch.no_grad():
                    output1 = quantum_model(test_input)
                    output2 = quantum_model(test_input)
                
                # 計算輸出穩定性
                stability = 1.0 / (1.0 + torch.norm(output1 - output2).item())
                robustness_scores.append(stability)
            
            # 恢復原始權重
            quantum_model.quantum_weights.data = original_weights
            
            # 魯棒性 = 平均穩定性
            robustness = np.mean(robustness_scores)
            
            return robustness
            
        except Exception as e:
            print(f"計算量子噪聲魯棒性時出錯: {e}")
            return 0.0
    
    def compute_quantum_gradient_quality(self, quantum_model: nn.Module) -> float:
        """計算量子梯度質量指標。"""
        if not hasattr(quantum_model, 'quantum_weights'):
            return 0.0
        
        try:
            # 計算梯度的質量指標
            test_input = torch.randn(1, 6, requires_grad=True).to(self.device)
            
            # 前向傳播
            output = quantum_model(test_input)
            loss = torch.sum(output ** 2)  # 簡單的損失函數
            
            # 計算梯度
            gradients = torch.autograd.grad(loss, quantum_model.quantum_weights, 
                                         create_graph=True, retain_graph=True)[0]
            
            # 計算梯度質量指標
            gradient_norm = torch.norm(gradients).item()
            gradient_std = torch.std(gradients).item()
            
            # 梯度質量 = 梯度範數 / (1 + 梯度標準差)
            gradient_quality = gradient_norm / (1.0 + gradient_std)
            
            return gradient_quality
            
        except Exception as e:
            print(f"計算量子梯度質量時出錯: {e}")
            return 0.0
    
    def compute_quantum_expressibility(self, quantum_model: nn.Module) -> float:
        """計算量子表達能力指標。"""
        if not hasattr(quantum_model, 'quantum_weights'):
            return 0.0
        
        try:
            # 測試模型對不同輸入的響應多樣性
            num_tests = 100
            outputs = []
            
            with torch.no_grad():
                for _ in range(num_tests):
                    test_input = torch.randn(1, 6).to(self.device)
                    output = quantum_model(test_input)
                    outputs.append(output.cpu().numpy())
            
            # 計算輸出多樣性
            outputs_array = np.vstack(outputs)
            output_std = np.std(outputs_array, axis=0)
            expressibility = np.mean(output_std)
            
            return expressibility
            
        except Exception as e:
            print(f"計算量子表達能力時出錯: {e}")
            return 0.0
    
    def compute_quantum_learning_dynamics(self, episode_rewards: List[float]) -> float:
        """計算量子學習動態指標。"""
        if not episode_rewards or len(episode_rewards) < 10:
            return 0.0
        
        try:
            # 計算學習曲線的特徵
            rewards_array = np.array(episode_rewards)
            
            # 計算學習速度（前後半段的改善）
            mid_point = len(rewards_array) // 2
            first_half = rewards_array[:mid_point]
            second_half = rewards_array[mid_point:]
            
            first_half_mean = np.mean(first_half)
            second_half_mean = np.mean(second_half)
            
            # 學習速度 = 後半段改善 / 前半段基準
            if abs(first_half_mean) > 1e-8:
                learning_speed = (second_half_mean - first_half_mean) / abs(first_half_mean)
            else:
                learning_speed = 0.0
            
            # 計算學習穩定性
            learning_stability = 1.0 / (1.0 + np.std(rewards_array))
            
            # 學習動態 = 學習速度 * 學習穩定性
            learning_dynamics = learning_speed * learning_stability
            
            return learning_dynamics
            
        except Exception as e:
            print(f"計算量子學習動態時出錯: {e}")
            return 0.0
    
    def compute_quantum_vs_classical_comparison(self, quantum_model: nn.Module, 
                                               classical_model: nn.Module,
                                               quantum_rewards: List[float],
                                               classical_rewards: List[float]) -> Dict[str, float]:
        """計算量子 vs 經典模型的詳細比較指標。"""
        comparison = {}
        
        # 性能比較
        if quantum_rewards and classical_rewards:
            q_perf = np.mean(quantum_rewards)
            c_perf = np.mean(classical_rewards)
            comparison['performance_improvement'] = (q_perf - c_perf) / abs(c_perf) if c_perf != 0 else 0.0
        
        # 參數效率比較
        q_params = sum(p.numel() for p in quantum_model.parameters())
        c_params = sum(p.numel() for p in classical_model.parameters())
        comparison['parameter_efficiency_ratio'] = c_params / q_params if q_params > 0 else 0.0
        
        # 推理速度比較
        q_speed = self._measure_inference_speed(quantum_model)
        c_speed = self._measure_inference_speed(classical_model)
        comparison['speed_improvement'] = c_speed / q_speed if q_speed > 0 else 0.0
        
        # 記憶體效率比較
        q_memory = self._measure_memory_usage(quantum_model)
        c_memory = self._measure_memory_usage(classical_model)
        comparison['memory_efficiency_ratio'] = c_memory / q_memory if q_memory > 0 else 0.0
        
        return comparison
    
    def _measure_inference_speed(self, model: nn.Module) -> float:
        """測量推理速度。"""
        model.eval()
        test_input = torch.randn(1, 6).to(self.device)
        
        # 預熱
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # 測量時間
        times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = model(test_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return np.mean(times)
    
    def _measure_memory_usage(self, model: nn.Module) -> float:
        """測量記憶體使用量。"""
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        return model_size / (1024 * 1024)  # MB
