"""
Transformer分析页面：提供Transformer模型结构分析、参数热点分析等工具
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.model_profiler import TransformerProfiler, create_sample_transformer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Transformer架构分析", page_icon="🤖", layout="wide")

st.title("🤖 Transformer 架构分析工具")
st.markdown("### 深度剖析Transformer模型结构、参数分布和性能瓶颈")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 模型配置")
    
    d_model = st.slider("嵌入维度 (d_model)", 128, 2048, 512, step=128)
    n_heads = st.select_slider("注意力头数", [4, 8, 12, 16, 32], value=8)
    n_layers = st.slider("层数", 1, 24, 6)
    vocab_size = st.select_slider("词表大小", [10000, 30000, 50000, 100000], value=50000)
    
    st.divider()
    
    st.header("💾 推理配置")
    batch_size = st.slider("Batch Size", 1, 64, 8)
    seq_len = st.slider("序列长度", 32, 2048, 128, step=32)
    
    st.divider()
    
    precision = st.radio("精度", ["FP32", "FP16", "BF16"], index=2)
    dtype_map = {"FP32": 4, "FP16": 2, "BF16": 2}
    bytes_per_element = dtype_map[precision]

# 创建模型和分析器
model = create_sample_transformer(d_model, n_heads, n_layers, vocab_size)
profiler = TransformerProfiler(model, (batch_size, seq_len, d_model))

# 主界面标签页
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 参数分析",
    "⚡ 计算复杂度",
    "💾 显存分析",
    "🔥 性能热点"
])

# =================== TAB 1: 参数分析 ===================
with tab1:
    st.header("📊 模型参数分布分析")
    
    # 统计参数
    param_counts = profiler.count_parameters()
    total_params = param_counts['total']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总参数量", f"{total_params:,}", help="模型所有可训练参数")
    with col2:
        st.metric("参数量 (M)", f"{total_params/1e6:.2f}M")
    with col3:
        st.metric("参数量 (B)", f"{total_params/1e9:.4f}B")
    
    st.divider()
    
    # 层级参数分布
    st.subheader("🏗️ 各层参数分布")
    
    profiles = profiler.profile_layers(batch_size, seq_len, d_model, n_heads, n_layers)
    
    # 创建数据框
    layer_data = []
    for p in profiles:
        layer_data.append({
            "层名称": p.name,
            "参数量": p.params,
            "参数占比 (%)": p.param_ratio,
            "显存 (MB)": p.memory_mb
        })
    
    df = pd.DataFrame(layer_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 饼图：参数分布
        fig = px.pie(df, values='参数量', names='层名称', 
                     title='参数分布饼图',
                     hover_data=['参数占比 (%)'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 表格显示
        st.dataframe(
            df.style.format({
                '参数量': '{:,}',
                '参数占比 (%)': '{:.2f}',
                '显存 (MB)': '{:.2f}'
            }).background_gradient(subset=['参数占比 (%)'], cmap='YlOrRd'),
            height=400
        )
    
    st.divider()
    
    # 组件级参数分析
    st.subheader("🔍 组件级参数分析")
    
    # 计算各组件参数量
    embedding_params = vocab_size * d_model
    attention_params_per_layer = 4 * d_model * d_model
    ffn_params_per_layer = 2 * d_model * 4 * d_model
    output_params = vocab_size * d_model
    
    component_data = {
        "组件": ["Embedding", "Self-Attention (所有层)", "FFN (所有层)", "Output Layer"],
        "参数量": [
            embedding_params,
            attention_params_per_layer * n_layers,
            ffn_params_per_layer * n_layers,
            output_params
        ]
    }
    
    df_comp = pd.DataFrame(component_data)
    df_comp['参数占比 (%)'] = df_comp['参数量'] / df_comp['参数量'].sum() * 100
    
    fig = px.bar(df_comp, x='组件', y='参数量', 
                 color='参数占比 (%)',
                 text='参数量',
                 title='组件参数量对比')
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    **关键观察**：
    - **Embedding + Output**: {(embedding_params + output_params)/1e6:.2f}M 参数 
      ({(embedding_params + output_params)/total_params*100:.1f}%)
    - **Attention**: {attention_params_per_layer * n_layers/1e6:.2f}M 参数
      ({attention_params_per_layer * n_layers/total_params*100:.1f}%)
    - **FFN**: {ffn_params_per_layer * n_layers/1e6:.2f}M 参数
      ({ffn_params_per_layer * n_layers/total_params*100:.1f}%)
    
    💡 **优化建议**：
    - 词表相关层占据了大量参数，考虑使用 **词表压缩** 或 **权重共享**
    - FFN 参数量约为 Attention 的 2 倍，是优化重点
    """)

# =================== TAB 2: 计算复杂度 ===================
with tab2:
    st.header("⚡ 计算复杂度分析")
    
    # 估算 FLOPs
    flops_data = profiler.estimate_flops(batch_size, seq_len, d_model, n_heads, n_layers, vocab_size)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("单层 FLOPs", f"{flops_data['total_per_layer']/1e9:.2f} GFLOPs")
    with col2:
        st.metric("总 FLOPs", f"{flops_data['total_model']/1e9:.2f} GFLOPs")
    with col3:
        throughput = flops_data['total_model'] / 1e12  # TFLOPs
        st.metric("吞吐量需求", f"{throughput:.3f} TFLOPs")
    
    st.divider()
    
    # FLOPs 分解
    st.subheader("🔬 FLOPs 分解分析")
    
    flops_breakdown = {
        "操作": [
            "QKV 投影",
            "Attention 计算",
            "FFN",
            "其他"
        ],
        "FLOPs": [
            flops_data['qkv_projection'],
            flops_data['attention_matrix'],
            flops_data['ffn_total'],
            flops_data['total_per_layer'] - flops_data['qkv_projection'] - 
            flops_data['attention_matrix'] - flops_data['ffn_total']
        ]
    }
    
    df_flops = pd.DataFrame(flops_breakdown)
    df_flops['占比 (%)'] = df_flops['FLOPs'] / df_flops['FLOPs'].sum() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(df_flops, values='FLOPs', names='操作',
                     title='单层 FLOPs 分布')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df_flops, x='操作', y='FLOPs',
                     color='占比 (%)',
                     text='FLOPs',
                     title='FLOPs 对比')
        fig.update_traces(texttemplate='%{text:.2e}')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 序列长度影响分析
    st.subheader("📈 序列长度对复杂度的影响")
    
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    complexity_data = profiler.get_attention_complexity_comparison(seq_lengths, d_model)
    
    df_complexity = pd.DataFrame({
        '序列长度': seq_lengths,
        'Transformer (O(L²))': complexity_data['transformer_flops'],
        'Linear Attention (O(Ld²))': complexity_data['linear_attention_flops'],
        'Mamba (O(LdN))': complexity_data['mamba_flops']
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seq_lengths, y=df_complexity['Transformer (O(L²))'], 
                            mode='lines+markers', name='Transformer', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=seq_lengths, y=df_complexity['Linear Attention (O(Ld²))'], 
                            mode='lines+markers', name='Linear Attention'))
    fig.add_trace(go.Scatter(x=seq_lengths, y=df_complexity['Mamba (O(LdN))'], 
                            mode='lines+markers', name='Mamba'))
    
    fig.update_layout(
        title='不同架构的计算复杂度对比',
        xaxis_title='序列长度',
        yaxis_title='FLOPs',
        yaxis_type='log',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **关键结论**：
    - 🔴 **Transformer**: 复杂度随序列长度**平方**增长，长序列场景代价高昂
    - 🟡 **Linear Attention**: 线性复杂度，但需要近似
    - 🟢 **Mamba**: 线性复杂度且性能接近，长序列优势明显
    
    **临界点分析**：
    """)
    
    # 找到 Transformer vs Mamba 的交叉点
    for i, L in enumerate(seq_lengths):
        if complexity_data['transformer_flops'][i] > complexity_data['mamba_flops'][i] * 10:
            st.info(f"💡 当序列长度超过 **{L}** 时，Mamba 的计算优势达到 **10倍** 以上")
            break

st.markdown("---")
st.caption("🔬 Transformer Explorer - Model Analysis Tool | © 2025")

# =================== TAB 3: 显存分析 ===================
with tab3:
    st.header("💾 显存占用分析")
    
    # 估算显存
    import torch
    memory_data = profiler.estimate_memory(batch_size, seq_len, d_model, n_layers, 
                                          dtype=torch.float32 if precision == "FP32" else torch.float16)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("参数显存", f"{memory_data['parameters_mb']:.0f} MB")
    with col2:
        st.metric("激活显存", f"{memory_data['activation_total_mb']:.0f} MB")
    with col3:
        st.metric("训练总显存", f"{memory_data['total_training_mb']:.0f} MB", 
                 help="包含参数、激活、梯度、优化器状态")
    with col4:
        st.metric("推理显存", f"{memory_data['total_inference_mb']:.0f} MB")
    
    st.divider()
    
    # 显存分解
    st.subheader("📊 训练时显存分解")
    
    memory_breakdown = {
        "类型": ["模型参数", "激活值", "梯度", "优化器状态 (AdamW)"],
        "显存 (MB)": [
            memory_data['parameters_mb'],
            memory_data['activation_total_mb'],
            memory_data['gradients_mb'],
            memory_data['optimizer_mb']
        ]
    }
    
    df_mem = pd.DataFrame(memory_breakdown)
    df_mem['占比 (%)'] = df_mem['显存 (MB)'] / df_mem['显存 (MB)'].sum() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(df_mem, values='显存 (MB)', names='类型',
                     title=f'训练显存分布 ({precision})')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            df_mem.style.format({
                '显存 (MB)': '{:.1f}',
                '占比 (%)': '{:.1f}'
            }).background_gradient(subset=['占比 (%)'], cmap='Reds'),
            height=200
        )
        
        st.markdown(f'''
        **显存占用分析**：
        - 💾 优化器状态占据了 **{memory_data['optimizer_mb']/memory_data['total_training_mb']*100:.1f}%** 的显存
        - 🔥 激活值占 **{memory_data['activation_total_mb']/memory_data['total_training_mb']*100:.1f}%**
        - 📉 使用 **Gradient Checkpointing** 可减少激活显存至原来的 1/√L
        ''')
    
    st.divider()
    
    # 精度对比
    st.subheader("🎯 精度对显存的影响")
    
    precisions = ["FP32", "FP16", "BF16"]
    precision_memory = []
    
    for prec in precisions:
        dtype = torch.float32 if prec == "FP32" else torch.float16
        mem = profiler.estimate_memory(batch_size, seq_len, d_model, n_layers, dtype)
        precision_memory.append({
            "精度": prec,
            "训练显存 (GB)": mem['total_training_mb'] / 1024,
            "推理显存 (GB)": mem['total_inference_mb'] / 1024,
            "节省比例": "0%" if prec == "FP32" else "50%"
        })
    
    df_prec = pd.DataFrame(precision_memory)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='训练', x=df_prec['精度'], y=df_prec['训练显存 (GB)']))
    fig.add_trace(go.Bar(name='推理', x=df_prec['精度'], y=df_prec['推理显存 (GB)']))
    fig.update_layout(
        title='不同精度下的显存占用',
        xaxis_title='精度',
        yaxis_title='显存 (GB)',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f'''
    💡 **优化建议**：
    - 使用 **BF16** 混合精度训练可节省 **50%** 显存
    - 当前配置下，推荐的最小显卡显存：
      - FP32 训练: **{df_prec.iloc[0]['训练显存 (GB)']*1.2:.1f} GB** (含余量)
      - BF16 训练: **{df_prec.iloc[2]['训练显存 (GB)']*1.2:.1f} GB**
      - BF16 推理: **{df_prec.iloc[2]['推理显存 (GB)']*1.2:.1f} GB**
    ''')
    
    st.divider()
    
    # Batch Size 影响分析
    st.subheader("📦 Batch Size 对显存的影响")
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    batch_memory = []
    
    for bs in batch_sizes:
        mem = profiler.estimate_memory(bs, seq_len, d_model, n_layers, torch.float16)
        batch_memory.append(mem['total_training_mb'] / 1024)  # Convert to GB
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=batch_sizes, y=batch_memory, mode='lines+markers',
                            line=dict(width=3, color='blue')))
    fig.update_layout(
        title='Batch Size vs 显存占用',
        xaxis_title='Batch Size',
        yaxis_title='显存 (GB)',
        height=400
    )
    
    # 添加显卡显存线
    common_gpus = {
        'RTX 3090': 24,
        'A100 (40GB)': 40,
        'A100 (80GB)': 80,
        'H100': 80
    }
    
    for gpu_name, gpu_mem in common_gpus.items():
        fig.add_hline(y=gpu_mem, line_dash="dash", 
                     annotation_text=gpu_name, 
                     annotation_position="right")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 找到每个 GPU 的最大 batch size
    st.markdown("**各显卡推荐的最大 Batch Size**：")
    for gpu_name, gpu_mem in common_gpus.items():
        max_bs = 1
        for i, bs in enumerate(batch_sizes):
            if batch_memory[i] <= gpu_mem * 0.9:  # 留 10% 余量
                max_bs = bs
        st.write(f"- **{gpu_name}**: Batch Size ≤ **{max_bs}**")

# =================== TAB 4: 性能热点 ===================
with tab4:
    st.header("🔥 性能热点分析")
    
    st.info("💡 本页面帮助你识别训练过程中的瓶颈，找到最值得优化的部分")
    
    # 训练步时间分解
    st.subheader("⏱️ 单步训练时间分解")
    
    timing = profiler.simulate_training_step(batch_size, seq_len, d_model, n_layers)
    
    timing_data = {
        "阶段": ["Forward", "Attention", "FFN", "Backward", "Optimizer"],
        "时间 (ms)": [
            timing['forward_ms'],
            timing['attention_ms'],
            timing['ffn_ms'],
            timing['backward_ms'],
            timing['optimizer_step_ms']
        ]
    }
    
    df_timing = pd.DataFrame(timing_data)
    df_timing['占比 (%)'] = df_timing['时间 (ms)'] / timing['total_ms'] * 100
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(df_timing, x='阶段', y='时间 (ms)',
                     color='占比 (%)',
                     title='单步训练时间分布',
                     text='时间 (ms)')
        fig.update_traces(texttemplate='%{text:.2f}ms', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("总时间", f"{timing['total_ms']:.2f} ms")
        st.metric("吞吐量", f"{1000/timing['total_ms']:.1f} steps/s")
        st.metric("样本吞吐量", f"{batch_size*1000/timing['total_ms']:.0f} samples/s")
        
        st.markdown(f'''
        **关键观察**：
        - Backward 是 Forward 的 **{timing['backward_ms']/timing['forward_ms']:.1f}x**
        - Attention 占 Forward 的 **{timing['attention_ms']/timing['forward_ms']*100:.0f}%**
        ''')
    
    st.divider()
    
    # 参数更新热点（模拟）
    st.subheader("🎯 参数更新热点识别")
    
    st.markdown('''
    在实际训练中，不同层的参数更新幅度差异巨大。识别这些"热点"可以帮助：
    - 🎯 **针对性调整学习率**（Layer-wise LR）
    - 🔍 **发现训练问题**（某些层不更新）
    - ⚡ **优化训练策略**（冻结不重要的层）
    ''')
    
    # 模拟梯度热点数据
    np.random.seed(42)
    hotspot_data = []
    
    for i in range(n_layers):
        # 模拟：浅层更新慢，深层更新快
        update_ratio = 0.001 * (1 + i / n_layers) * np.random.uniform(0.5, 1.5)
        
        hotspot_data.append({
            "层": f"Layer {i+1} Attention",
            "梯度范数": np.random.uniform(0.1, 2.0),
            "参数范数": np.random.uniform(5.0, 15.0),
            "更新比例": update_ratio,
            "是否热点": "🔥" if update_ratio > 0.0015 else "❄️"
        })
        
        update_ratio_ffn = 0.0012 * (1 + i / n_layers) * np.random.uniform(0.5, 1.5)
        hotspot_data.append({
            "层": f"Layer {i+1} FFN",
            "梯度范数": np.random.uniform(0.1, 2.0),
            "参数范数": np.random.uniform(10.0, 25.0),
            "更新比例": update_ratio_ffn,
            "是否热点": "🔥" if update_ratio_ffn > 0.0015 else "❄️"
        })
    
    df_hotspot = pd.DataFrame(hotspot_data)
    
    # 热点可视化
    fig = px.bar(df_hotspot, x='层', y='更新比例',
                 color='是否热点',
                 title='各层参数更新热点分布',
                 color_discrete_map={"🔥": "#e74c3c", "❄️": "#3498db"})
    fig.add_hline(y=0.0015, line_dash="dash", 
                 annotation_text="热点阈值", 
                 line_color="red")
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # 表格显示（只显示前10行）
    st.dataframe(
        df_hotspot.head(10).style.format({
            '梯度范数': '{:.4f}',
            '参数范数': '{:.2f}',
            '更新比例': '{:.6f}'
        }).background_gradient(subset=['更新比例'], cmap='YlOrRd'),
        height=300
    )
    
    st.divider()
    
    # 优化建议
    st.subheader("💡 性能优化建议")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        ### 🚀 计算优化
        
        1. **Flash Attention**
           - 减少 Attention 显存访问
           - 加速 2-4x
           - 适用于长序列
        
        2. **Gradient Checkpointing**
           - 以计算换显存
           - 激活显存减少至 O(√L)
           - 训练时间增加 20-30%
        
        3. **Mixed Precision (BF16)**
           - 速度提升 2-3x
           - 显存节省 50%
           - 现代 GPU 必备
        ''')
    
    with col2:
        st.markdown('''
        ### 🎯 架构优化
        
        1. **Multi-Query Attention**
           - 减少 KV Cache
           - 推理加速 1.5-2x
           - PaLM/Falcon 使用
        
        2. **SwiGLU FFN**
           - 替代 ReLU/GELU
           - 性能提升 5-10%
           - LLaMA 系列采用
        
        3. **RoPE 位置编码**
           - 外推能力强
           - 无额外参数
           - 相对位置编码
        ''')
    
    # 实际收益评估
    st.markdown("### 📊 优化收益评估")
    
    optimizations = {
        "优化方法": [
            "Baseline",
            "+ Flash Attention",
            "+ BF16",
            "+ Gradient Checkpointing",
            "+ All"
        ],
        "训练时间 (相对)": [1.0, 0.7, 0.4, 0.5, 0.3],
        "显存占用 (相对)": [1.0, 0.9, 0.5, 0.3, 0.15]
    }
    
    df_opt = pd.DataFrame(optimizations)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='训练时间', x=df_opt['优化方法'], 
                        y=df_opt['训练时间 (相对)'], 
                        text=df_opt['训练时间 (相对)'],
                        texttemplate='%{text:.1f}x'))
    fig.add_trace(go.Bar(name='显存占用', x=df_opt['优化方法'], 
                        y=df_opt['显存占用 (相对)'],
                        text=df_opt['显存占用 (相对)'],
                        texttemplate='%{text:.2f}x'))
    
    fig.update_layout(
        title='优化方法效果对比（相对于 Baseline）',
        xaxis_title='优化组合',
        yaxis_title='相对值（越小越好）',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.success('''
    🎉 **综合优化效果**：
    - 训练速度提升：**3.3x**
    - 显存节省：**85%**
    - 可训练的最大模型规模提升：**6-7x**
    ''')

