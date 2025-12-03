"""
Mamba 模型分析工具：分析 Mamba/SSM 架构的性能特性
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MambaProfile:
    """Mamba 模型性能分析数据"""
    name: str
    params: int
    flops: int
    memory_mb: float
    state_memory_mb: float  # 状态空间显存
    param_ratio: float
    flops_ratio: float


class MambaProfiler:
    """Mamba 模型性能分析器"""
    
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        """
        Args:
            d_model: 模型维度
            d_state: SSM 状态维度 (N)
            d_conv: 卷积核大小
            expand: 扩展因子（通常是 2）
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
    def count_parameters(self) -> Dict[str, int]:
        """统计 Mamba 块的参数量"""
        
        # 1. Input projection (x -> z, x_proj, dt_proj)
        in_proj_params = self.d_model * self.d_inner * 2  # z 和 x_proj
        
        # 2. Conv1d
        conv_params = self.d_inner * self.d_conv
        
        # 3. SSM parameters
        # Δ, A, B, C, D
        dt_proj_params = self.d_inner  # dt projection
        A_params = self.d_inner * self.d_state  # A matrix
        B_params = self.d_inner * self.d_state  # B projection
        C_params = self.d_inner * self.d_state  # C projection
        D_params = self.d_inner  # D (skip connection)
        
        ssm_params = dt_proj_params + A_params + B_params + C_params + D_params
        
        # 4. Output projection
        out_proj_params = self.d_inner * self.d_model
        
        # 5. Norm parameters
        norm_params = self.d_model
        
        total = in_proj_params + conv_params + ssm_params + out_proj_params + norm_params
        
        return {
            'input_projection': in_proj_params,
            'conv1d': conv_params,
            'ssm_parameters': ssm_params,
            '  - A_matrix': A_params,
            '  - B_projection': B_params,
            '  - C_projection': C_params,
            '  - dt_projection': dt_proj_params,
            '  - D_skip': D_params,
            'output_projection': out_proj_params,
            'norm': norm_params,
            'total_per_block': total
        }
    
    def estimate_flops(self, batch_size: int, seq_len: int) -> Dict[str, int]:
        """估算 Mamba 的 FLOPs
        
        Mamba 的核心优势：O(L) 复杂度而非 O(L²)
        """
        
        # 1. Input projection: B * L * d * (2 * d_inner)
        in_proj_flops = batch_size * seq_len * self.d_model * (2 * self.d_inner)
        
        # 2. Conv1d: B * L * d_inner * d_conv
        conv_flops = batch_size * seq_len * self.d_inner * self.d_conv
        
        # 3. SSM (选择性扫描 - 核心!)
        # Discretization + State update: O(BLdN)
        # B, C, Δ computation
        ssm_bc_flops = batch_size * seq_len * self.d_inner * self.d_state * 2
        
        # State evolution (selective scan): O(BLdN)
        # h_{t+1} = A_bar * h_t + B_bar * x_t
        state_update_flops = batch_size * seq_len * self.d_inner * self.d_state
        
        # Output: y_t = C * h_t + D * x_t
        output_flops = batch_size * seq_len * self.d_inner * self.d_state
        
        ssm_total_flops = ssm_bc_flops + state_update_flops + output_flops
        
        # 4. Output projection: B * L * d_inner * d
        out_proj_flops = batch_size * seq_len * self.d_inner * self.d_model
        
        # 5. Activation (SiLU/Swish): 约 5 * B * L * d_inner
        activation_flops = 5 * batch_size * seq_len * self.d_inner
        
        total_flops = (
            in_proj_flops + 
            conv_flops + 
            ssm_total_flops + 
            out_proj_flops + 
            activation_flops
        )
        
        return {
            'input_projection': in_proj_flops,
            'conv1d': conv_flops,
            'ssm_total': ssm_total_flops,
            '  - bc_computation': ssm_bc_flops,
            '  - state_update': state_update_flops,
            '  - output_computation': output_flops,
            'output_projection': out_proj_flops,
            'activation': activation_flops,
            'total_per_block': total_flops,
            'complexity': 'O(BLdN)'  # 线性复杂度！
        }
    
    def estimate_memory(self, batch_size: int, seq_len: int, 
                       dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """估算 Mamba 的显存占用
        
        关键优势：固定大小的状态空间，不随序列长度平方增长
        """
        
        bytes_per_element = 4 if dtype == torch.float32 else 2
        
        # 1. 参数显存
        param_counts = self.count_parameters()
        param_memory = param_counts['total_per_block'] * bytes_per_element / (1024**2)
        
        # 2. 激活显存（训练时需要保存用于反向传播）
        # Input projection
        in_proj_activation = batch_size * seq_len * self.d_inner * 2
        
        # Conv1d output
        conv_activation = batch_size * seq_len * self.d_inner
        
        # SSM states (关键！固定大小)
        # Hidden state: B * d_inner * d_state (不依赖于 L!)
        ssm_state = batch_size * self.d_inner * self.d_state
        
        # SSM 中间变量 (B, C, Δ)
        ssm_intermediate = batch_size * seq_len * self.d_inner * 3
        
        # Output
        output_activation = batch_size * seq_len * self.d_model
        
        total_activation = (
            in_proj_activation + 
            conv_activation + 
            ssm_state + 
            ssm_intermediate + 
            output_activation
        ) * bytes_per_element / (1024**2)
        
        # 3. 梯度显存
        gradient_memory = param_memory
        
        # 4. 优化器状态 (AdamW)
        optimizer_memory = 2 * param_memory
        
        # 5. KV Cache 对比（推理时）
        # Mamba: 只需要固定的 state
        mamba_cache = batch_size * self.d_inner * self.d_state * bytes_per_element / (1024**2)
        
        # Transformer: 需要存储所有的 K, V
        # 假设同样的 d_model
        transformer_kv_cache = 2 * batch_size * seq_len * self.d_model * bytes_per_element / (1024**2)
        
        return {
            'parameters_mb': param_memory,
            'activation_mb': total_activation,
            'ssm_state_mb': ssm_state * bytes_per_element / (1024**2),
            'gradients_mb': gradient_memory,
            'optimizer_mb': optimizer_memory,
            'total_training_mb': param_memory + total_activation + gradient_memory + optimizer_memory,
            'total_inference_mb': param_memory + total_activation,
            'inference_cache_mb': mamba_cache,
            'transformer_kv_cache_mb': transformer_kv_cache,
            'cache_savings': f"{(1 - mamba_cache/transformer_kv_cache)*100:.1f}%"
        }
    
    def compare_with_transformer(self, batch_size: int, seq_len: int, 
                                n_heads: int = 8) -> Dict[str, float]:
        """对比 Mamba 和 Transformer 的性能差异"""
        
        # Mamba FLOPs (线性)
        mamba_flops = self.estimate_flops(batch_size, seq_len)['total_per_block']
        
        # Transformer Attention FLOPs (平方)
        # QKV projection: 3 * B * L * d * d
        qkv_flops = 3 * batch_size * seq_len * self.d_model * self.d_model
        
        # Attention matrix: B * h * L * L * (d/h)
        attn_matrix_flops = batch_size * n_heads * seq_len * seq_len * (self.d_model // n_heads)
        
        # Attention output: B * h * L * L * (d/h)
        attn_out_flops = batch_size * n_heads * seq_len * seq_len * (self.d_model // n_heads)
        
        # Output projection: B * L * d * d
        out_proj_flops = batch_size * seq_len * self.d_model * self.d_model
        
        transformer_flops = qkv_flops + attn_matrix_flops + attn_out_flops + out_proj_flops
        
        # Mamba Memory
        mamba_mem = self.estimate_memory(batch_size, seq_len)
        
        # Transformer Memory (attention weights)
        transformer_attn_mem = batch_size * n_heads * seq_len * seq_len * 4 / (1024**2)  # FP32
        
        return {
            'mamba_flops': mamba_flops,
            'transformer_flops': transformer_flops,
            'flops_speedup': transformer_flops / mamba_flops,
            'mamba_memory_mb': mamba_mem['total_inference_mb'],
            'transformer_memory_mb': mamba_mem['total_inference_mb'] + transformer_attn_mem,
            'memory_savings': f"{(1 - mamba_mem['total_inference_mb']/(mamba_mem['total_inference_mb'] + transformer_attn_mem))*100:.1f}%",
            'mamba_cache_mb': mamba_mem['inference_cache_mb'],
            'transformer_cache_mb': mamba_mem['transformer_kv_cache_mb'],
            'cache_speedup': mamba_mem['transformer_kv_cache_mb'] / mamba_mem['inference_cache_mb'],
            'cache_savings': f"{(1 - mamba_mem['inference_cache_mb']/mamba_mem['transformer_kv_cache_mb'])*100:.1f}%"
        }
    
    def analyze_scaling(self, seq_lengths: List[int], batch_size: int = 8) -> Dict[str, List]:
        """分析 Mamba 在不同序列长度下的扩展性"""
        
        mamba_flops = []
        mamba_memory = []
        transformer_flops = []
        transformer_memory = []
        
        for L in seq_lengths:
            # Mamba: O(L)
            m_flops = self.estimate_flops(batch_size, L)['total_per_block']
            m_mem = self.estimate_memory(batch_size, L)['total_inference_mb']
            
            # Transformer: O(L²)
            t_flops = batch_size * L * L * self.d_model  # 简化估算
            t_mem = m_mem + (batch_size * 8 * L * L * 4 / (1024**2))  # attention weights
            
            mamba_flops.append(m_flops)
            mamba_memory.append(m_mem)
            transformer_flops.append(t_flops)
            transformer_memory.append(t_mem)
        
        return {
            'seq_lengths': seq_lengths,
            'mamba_flops': mamba_flops,
            'transformer_flops': transformer_flops,
            'mamba_memory': mamba_memory,
            'transformer_memory': transformer_memory,
            'speedup_ratio': [t/m for t, m in zip(transformer_flops, mamba_flops)]
        }
    
    def get_selective_scan_analysis(self) -> Dict[str, any]:
        """分析 Mamba 的核心：选择性扫描机制"""
        
        return {
            'mechanism': 'Selective State Space Model',
            'key_innovation': '输入依赖的参数（B, C, Δ）',
            'state_dimension': self.d_state,
            'complexity': 'O(BLdN)',
            'advantages': [
                '线性复杂度（vs Transformer 的平方）',
                '固定大小的状态空间（不随序列长度增长）',
                '选择性记忆（动态过滤信息）',
                '高效的长序列处理'
            ],
            'trade_offs': [
                '训练时难以完全并行（有序列依赖）',
                '需要特殊的 CUDA kernel 优化',
                '对短序列可能不如 Transformer'
            ],
            'optimal_scenarios': [
                '长序列任务（> 2048 tokens）',
                '推理阶段（KV Cache 优势明显）',
                '边缘设备部署（显存受限）',
                '实时应用（低延迟需求）'
            ]
        }


def create_sample_mamba(d_model: int = 512, d_state: int = 16, 
                       d_conv: int = 4, expand: int = 2) -> Dict:
    """创建示例 Mamba 配置用于分析"""
    
    return {
        'd_model': d_model,
        'd_state': d_state,
        'd_conv': d_conv,
        'expand': expand,
        'd_inner': d_model * expand
    }


if __name__ == "__main__":
    # 测试 Mamba 分析器
    profiler = MambaProfiler(d_model=512, d_state=16, d_conv=4, expand=2)
    
    print("=" * 60)
    print("Mamba 模型分析")
    print("=" * 60)
    
    # 参数统计
    params = profiler.count_parameters()
    print(f"\n✅ 参数统计:")
    print(f"   总参数: {params['total_per_block']:,}")
    print(f"   SSM 参数: {params['ssm_parameters']:,}")
    
    # FLOPs
    flops = profiler.estimate_flops(8, 128)
    print(f"\n✅ FLOPs (Batch=8, Seq=128):")
    print(f"   总计: {flops['total_per_block']/1e9:.2f} GFLOPs")
    print(f"   复杂度: {flops['complexity']}")
    
    # 对比
    comparison = profiler.compare_with_transformer(8, 128)
    print(f"\n✅ vs Transformer:")
    print(f"   FLOPs 加速: {comparison['flops_speedup']:.1f}x")
    print(f"   显存节省: {comparison['memory_savings']}")
    print(f"   Cache 加速: {comparison['cache_speedup']:.1f}x")
