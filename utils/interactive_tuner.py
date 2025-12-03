"""
交互式参数调节工具：实时看到参数变化对模型的影响
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import time


class InteractiveParameterTuner:
    """交互式参数调节器"""
    
    def __init__(self):
        """初始化参数调节器"""
        self.model_cache = {}
        
    def create_simple_model(self, d_model: int, n_heads: int, n_layers: int, 
                           activation: str, dropout: float) -> nn.Module:
        """创建简单的Transformer模型"""
        
        class SimpleTransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads, dropout, activation):
                super().__init__()
                self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.ReLU() if activation == 'relu' else nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout)
                )
                
            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                
                # FFN
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)
                
                return x
        
        class SimpleTransformer(nn.Module):
            def __init__(self, d_model, n_heads, n_layers, activation, dropout):
                super().__init__()
                self.layers = nn.ModuleList([
                    SimpleTransformerBlock(d_model, n_heads, dropout, activation)
                    for _ in range(n_layers)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return SimpleTransformer(d_model, n_heads, n_layers, activation, dropout)
    
    def calculate_model_metrics(self, model: nn.Module, seq_len: int, batch_size: int = 8) -> Dict:
        """计算模型指标"""
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估算计算量（简化版）
        d_model = model.layers[0].attention.embed_dim
        n_heads = model.layers[0].attention.num_heads
        n_layers = len(model.layers)
        
        # Self-attention FLOPs
        qkv_flops = 3 * batch_size * seq_len * d_model * d_model
        attn_flops = batch_size * n_heads * seq_len * seq_len * (d_model // n_heads)
        
        # FFN FLOPs
        ffn_flops = batch_size * seq_len * d_model * (4 * d_model) * 2
        
        total_flops = n_layers * (qkv_flops + attn_flops + ffn_flops)
        
        # 估算显存（简化版）
        param_memory = total_params * 4 / (1024**2)  # FP32
        activation_memory = batch_size * seq_len * d_model * 4 / (1024**2) * n_layers
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_flops': total_flops,
            'param_memory_mb': param_memory,
            'activation_memory_mb': activation_memory,
            'total_memory_mb': param_memory + activation_memory,
            'estimated_inference_time_ms': total_flops / 1e9 * 1000  # 假设1 TFLOPS
        }
    
    def simulate_forward_pass(self, model: nn.Module, seq_len: int, batch_size: int = 2) -> Dict:
        """模拟前向传播并收集统计信息"""
        model.eval()
        
        # 创建随机输入
        x = torch.randn(batch_size, seq_len, model.layers[0].attention.embed_dim)
        
        with torch.no_grad():
            start_time = time.time()
            output = model(x)
            end_time = time.time()
            
            # 收集各层的激活统计
            layer_stats = []
            for i, layer in enumerate(model.layers):
                # 计算该层的输出范数
                layer_output = layer(x)
                output_norm = layer_output.norm(dim=-1).mean().item()
                output_std = layer_output.std(dim=-1).mean().item()
                
                layer_stats.append({
                    'layer': i,
                    'output_norm': output_norm,
                    'output_std': output_std
                })
                
                x = layer_output
        
        return {
            'inference_time_ms': (end_time - start_time) * 1000,
            'output_shape': output.shape,
            'layer_stats': layer_stats
        }
    
    def create_parameter_impact_visualization(self, param_name: str, param_values: List[float], 
                                            base_config: Dict) -> go.Figure:
        """创建参数影响可视化"""
        results = []
        
        for value in param_values:
            config = base_config.copy()
            config[param_name] = value
            
            # 创建模型
            model = self.create_simple_model(
                config['d_model'], config['n_heads'], 
                config['n_layers'], config['activation'], config['dropout']
            )
            
            # 计算指标
            metrics = self.calculate_model_metrics(model, config['seq_len'])
            
            results.append({
                param_name: value,
                'params_millions': metrics['total_params'] / 1e6,
                'flops_gflops': metrics['total_flops'] / 1e9,
                'memory_mb': metrics['total_memory_mb'],
                'inference_time_ms': metrics['estimated_inference_time_ms']
            })
        
        # 创建可视化
        fig = go.Figure()
        
        # 添加参数量曲线
        fig.add_trace(go.Scatter(
            x=[r[param_name] for r in results],
            y=[r['params_millions'] for r in results],
            mode='lines+markers',
            name='参数量 (M)',
            yaxis='y'
        ))
        
        # 添加FLOPs曲线
        fig.add_trace(go.Scatter(
            x=[r[param_name] for r in results],
            y=[r['flops_gflops'] for r in results],
            mode='lines+markers',
            name='FLOPs (GFLOPs)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'{param_name} 对模型性能的影响',
            xaxis_title=param_name,
            yaxis=dict(title='参数量 (M)', side='left'),
            yaxis2=dict(title='FLOPs (GFLOPs)', side='right', overlaying='y'),
            height=500
        )
        
        return fig
    
    def create_attention_head_analysis(self, d_model: int, head_options: List[int]) -> go.Figure:
        """创建注意力头数分析"""
        results = []
        
        for n_heads in head_options:
            if d_model % n_heads != 0:
                continue
                
            model = self.create_simple_model(d_model, n_heads, 4, 'gelu', 0.1)
            metrics = self.calculate_model_metrics(model, 128)
            
            # 模拟前向传播
            forward_stats = self.simulate_forward_pass(model, 128)
            
            results.append({
                'n_heads': n_heads,
                'head_dim': d_model // n_heads,
                'params_millions': metrics['total_params'] / 1e6,
                'inference_time_ms': forward_stats['inference_time_ms'],
                'memory_mb': metrics['total_memory_mb']
            })
        
        # 创建可视化
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[r['n_heads'] for r in results],
            y=[r['params_millions'] for r in results],
            mode='lines+markers',
            name='参数量 (M)',
            text=[f"头维度: {r['head_dim']}" for r in results],
            hovertemplate='注意力头数: %{x}<br>参数量: %{y:.2f}M<br>%{text}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[r['n_heads'] for r in results],
            y=[r['inference_time_ms'] for r in results],
            mode='lines+markers',
            name='推理时间 (ms)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='注意力头数对模型性能的影响',
            xaxis_title='注意力头数',
            yaxis=dict(title='参数量 (M)', side='left'),
            yaxis2=dict(title='推理时间 (ms)', side='right', overlaying='y'),
            height=500
        )
        
        return fig
    
    def create_depth_vs_width_analysis(self) -> go.Figure:
        """创建深度vs宽度分析"""
        # 不同的配置
        configs = [
            {'n_layers': 2, 'd_model': 512, 'name': '浅层宽模型'},
            {'n_layers': 4, 'd_model': 256, 'name': '中层中等模型'},
            {'n_layers': 8, 'd_model': 128, 'name': '深层窄模型'},
            {'n_layers': 12, 'd_model': 86, 'name': '很深层很窄模型'},
        ]
        
        results = []
        
        for config in configs:
            model = self.create_simple_model(
                config['d_model'], 8, config['n_layers'], 'gelu', 0.1
            )
            
            metrics = self.calculate_model_metrics(model, 128)
            forward_stats = self.simulate_forward_pass(model, 128)
            
            results.append({
                'name': config['name'],
                'n_layers': config['n_layers'],
                'd_model': config['d_model'],
                'params_millions': metrics['total_params'] / 1e6,
                'inference_time_ms': forward_stats['inference_time_ms'],
                'memory_mb': metrics['total_memory_mb']
            })
        
        # 创建可视化
        fig = go.Figure()
        
        # 散点图：深度 vs 参数量
        fig.add_trace(go.Scatter(
            x=[r['n_layers'] for r in results],
            y=[r['params_millions'] for r in results],
            mode='markers+lines',
            name='参数量 (M)',
            marker=dict(size=[r['d_model']/20 for r in results]),
            text=[r['name'] for r in results],
            hovertemplate='层数: %{x}<br>参数量: %{y:.2f}M<br>模型维度: %{marker.size:.0f}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='深度 vs 宽度权衡分析',
            xaxis_title='层数 (深度)',
            yaxis_title='参数量 (M)',
            height=500
        )
        
        return fig
    
    def create_parameter_recommendations(self, config: Dict) -> Dict:
        """基于当前配置生成参数建议"""
        recommendations = []
        
        # 分析参数量
        if config['d_model'] * config['n_layers'] > 10000:
            if config['n_heads'] > 16:
                recommendations.append({
                    'type': '优化建议',
                    'message': f"模型较大，考虑减少注意力头数到 {config['d_model'] // 64} 以提高效率",
                    'priority': 'Medium'
                })
        
        # 分析序列长度
        if config['seq_len'] > 1024:
            recommendations.append({
                'type': '长序列优化',
                'message': "序列长度超过1024，考虑使用FlashAttention或稀疏注意力",
                'priority': 'High'
            })
        
        # 分析深度vs宽度
        depth_to_width_ratio = config['n_layers'] / (config['d_model'] / 64)
        if depth_to_width_ratio > 2:
            recommendations.append({
                'type': '架构平衡',
                'message': "模型相对较深，考虑增加维度以改善梯度流",
                'priority': 'Low'
            })
        elif depth_to_width_ratio < 0.5:
            recommendations.append({
                'type': '架构平衡',
                'message': "模型相对较宽，考虑增加层数以提高表达能力",
                'priority': 'Low'
            })
        
        # Dropout建议
        if config['dropout'] > 0.2:
            recommendations.append({
                'type': '正则化',
                'message': "Dropout较高，可能影响训练速度，确保有足够的数据",
                'priority': 'Low'
            })
        elif config['dropout'] < 0.05 and config['n_layers'] > 6:
            recommendations.append({
                'type': '正则化',
                'message': "深层模型建议增加Dropout到0.1-0.15以防止过拟合",
                'priority': 'Medium'
            })
        
        return {
            'recommendations': recommendations,
            'config_summary': {
                'total_params_estimate': config['d_model'] * config['d_model'] * config['n_layers'] * 8 / 1e6,
                'complexity_level': 'High' if config['n_layers'] * config['d_model'] > 100000 else 'Medium' if config['n_layers'] * config['d_model'] > 50000 else 'Low'
            }
        }


def create_interactive_tuning_page():
    """创建交互式调节页面"""
    st.set_page_config(page_title="交互式参数调节", page_icon="🎛️", layout="wide")
    
    st.title("🎛️ 交互式参数调节工具")
    st.markdown("### 实时看到参数变化对模型的影响")
    
    # 初始化调节器
    tuner = InteractiveParameterTuner()
    
    # 侧边栏参数配置
    with st.sidebar:
        st.header("🔧 模型参数")
        
        d_model = st.slider("模型维度 (d_model)", 64, 1024, 512, step=64)
        n_heads = st.selectbox("注意力头数", [2, 4, 8, 12, 16], index=2)
        n_layers = st.slider("层数", 1, 12, 6)
        activation = st.selectbox("激活函数", ["relu", "gelu"], index=1)
        dropout = st.slider("Dropout率", 0.0, 0.5, 0.1, step=0.05)
        seq_len = st.slider("序列长度", 64, 2048, 128, step=64)
        
        st.divider()
        
        st.header("📊 分析选项")
        analysis_type = st.selectbox(
            "选择分析类型",
            ["参数影响分析", "注意力头分析", "深度vs宽度分析", "实时性能评估"]
        )
    
    # 当前配置
    current_config = {
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'activation': activation,
        'dropout': dropout,
        'seq_len': seq_len
    }
    
    # 创建模型并计算指标
    model = tuner.create_simple_model(d_model, n_heads, n_layers, activation, dropout)
    metrics = tuner.calculate_model_metrics(model, seq_len)
    
    # 显示当前配置摘要
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("参数量", f"{metrics['total_params']:,}")
    with col2:
        st.metric("计算量", f"{metrics['total_flops']/1e9:.2f} GFLOPs")
    with col3:
        st.metric("显存", f"{metrics['total_memory_mb']:.0f} MB")
    with col4:
        st.metric("推理时间", f"{metrics['estimated_inference_time_ms']:.2f} ms")
    
    # 根据分析类型显示不同内容
    if analysis_type == "参数影响分析":
        st.header("📈 参数影响分析")
        
        param_name = st.selectbox("选择参数", ["d_model", "n_layers", "seq_len", "dropout"])
        
        if param_name == "d_model":
            param_values = [128, 256, 384, 512, 640, 768, 896, 1024]
        elif param_name == "n_layers":
            param_values = [1, 2, 4, 6, 8, 10, 12]
        elif param_name == "seq_len":
            param_values = [64, 128, 256, 512, 1024, 2048]
        else:  # dropout
            param_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        
        fig = tuner.create_parameter_impact_visualization(param_name, param_values, current_config)
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "注意力头分析":
        st.header("👁️ 注意力头数分析")
        
        head_options = [2, 4, 8, 12, 16, 20, 24]
        valid_heads = [h for h in head_options if d_model % h == 0]
        
        if valid_heads:
            fig = tuner.create_attention_head_analysis(d_model, valid_heads)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"当前维度 {d_model} 无法被标准头数整除，建议调整为 {d_model//8} 或 {d_model//16} 个头")
        
    elif analysis_type == "深度vs宽度分析":
        st.header("📊 深度 vs 宽度权衡")
        
        fig = tuner.create_depth_vs_width_analysis()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **关键观察**：
        - **浅层宽模型**：训练快，并行性好，适合简单任务
        - **深层窄模型**：表达能力强，可能梯度消失，需要精心设计
        - **平衡点**：通常层数和维度的比例在 1:4 到 1:8 之间效果较好
        """)
        
    elif analysis_type == "实时性能评估":
        st.header("⚡ 实时性能评估")
        
        if st.button("运行前向传播测试"):
            with st.spinner("运行前向传播..."):
                forward_stats = tuner.simulate_forward_pass(model, seq_len)
                
                st.success(f"前向传播完成！耗时: {forward_stats['inference_time_ms']:.2f} ms")
                
                # 显示层统计
                layer_stats = forward_stats['layer_stats']
                if layer_stats:
                    df_data = {
                        '层数': [s['layer'] for s in layer_stats],
                        '输出范数': [s['output_norm'] for s in layer_stats],
                        '输出标准差': [s['output_std'] for s in layer_stats]
                    }
                    
                    st.dataframe(df_data, use_container_width=True)
                    
                    # 可视化层统计
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[s['layer'] for s in layer_stats],
                        y=[s['output_norm'] for s in layer_stats],
                        mode='lines+markers',
                        name='输出范数'
                    ))
                    
                    fig.update_layout(
                        title='各层输出范数变化',
                        xaxis_title='层数',
                        yaxis_title='输出范数'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # 参数建议
    st.divider()
    st.header("💡 智能建议")
    
    recommendations = tuner.create_parameter_recommendations(current_config)
    
    if recommendations['recommendations']:
        for rec in recommendations['recommendations']:
            priority_color = {
                'High': '🔴',
                'Medium': '🟡', 
                'Low': '🟢'
            }
            
            st.markdown(f"{priority_color.get(rec['priority'], '⚪')} **{rec['type']}**: {rec['message']}")
    else:
        st.info("当前配置看起来很合理！没有特别的优化建议。")
    
    # 配置摘要
    st.markdown(f"""
    **配置摘要**：
    - 预估参数量: {recommendations['config_summary']['total_params_estimate']:.2f}M
    - 复杂度等级: {recommendations['config_summary']['complexity_level']}
    """)


if __name__ == "__main__":
    # 测试代码
    tuner = InteractiveParameterTuner()
    
    # 创建模型
    model = tuner.create_simple_model(512, 8, 6, 'gelu', 0.1)
    metrics = tuner.calculate_model_metrics(model, 128)
    
    print(f"模型参数量: {metrics['total_params']:,}")
    print(f"计算量: {metrics['total_flops']/1e9:.2f} GFLOPs")
    
    # 生成建议
    config = {'d_model': 512, 'n_heads': 8, 'n_layers': 6, 'activation': 'gelu', 'dropout': 0.1, 'seq_len': 128}
    recommendations = tuner.create_parameter_recommendations(config)
    print(f"生成了 {len(recommendations['recommendations'])} 条建议")