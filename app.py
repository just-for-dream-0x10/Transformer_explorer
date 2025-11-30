import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ==========================================
# 页面全局配置
# ==========================================
st.set_page_config(
    page_title="Transformer & Mamba 深度解析",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 美化
st.markdown(
    """
<style>
    .math-box {
        background-color: #f8f9fa;
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .analogy-box {
        background-color: #e8f5e9;
        border-left: 5px solid #66bb6a;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🤖 Transformer vs 🐍 Mamba：核心机制全景透视")
st.markdown("### 从直观动画到硬核数学：深度学习架构完全解析")

# ==========================================
# 侧边栏：参数控制台
# ==========================================
with st.sidebar:
    st.header("🎛️ 交互实验室")

    st.subheader("1. 输入设置")
    user_input = st.text_input(
        "输入文本 (空格分隔)",
        "I love learning AI models",
    )
    tokens = user_input.split()
    seq_len = len(tokens)
    st.info(f"Token 数量: {seq_len}")

    st.divider()

    st.subheader("2. Transformer 参数")
    d_model = st.slider("嵌入维度 (d_model)", 4, 64, 16, step=4)
    n_heads = st.radio("多头数量 (Heads)", [1, 2, 4, 8], index=2)
    d_k = d_model // n_heads
    st.caption(f"每个头的维度: d_k = {d_k}")

    st.divider()

    st.subheader("3. Mamba 参数")
    d_state = st.slider("状态维度 (d_state)", 2, 16, 4)

    st.divider()

    st.subheader("4. 训练参数")
    learning_rate = st.select_slider(
        "学习率",
        options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        value=1e-3,
        format_func=lambda x: f"{x:.0e}"
    )
    temperature = st.slider("Temperature (采样)", 0.1, 2.0, 1.0, 0.1)

    st.markdown("---")
    st.caption("© 2025 Transformer Explorer | Powered by Manim & Streamlit")

# ==========================================
# 主界面：四大核心板块
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎥 视频影院", 
    "🧮 交互实验室", 
    "📊 可视化分析",
    "🔬 训练与优化"
])

# ==========================================
# TAB 1: 视频影院
# ==========================================
with tab1:
    st.info("💡 选择下方类别，观看对应的动画演示")
    
    video_category = st.selectbox(
        "选择视频类别",
        [
            "🏗️ Transformer 基础架构",
            "🔄 位置编码机制",
            "🎯 注意力机制详解",
            "🧬 前馈网络 (FFN)",
            "🎲 采样与分词",
            "🏋️ 训练与优化",
            "🐍 Mamba 架构",
            "⚔️ 架构对比"
        ]
    )
    
    col_video, col_notes = st.columns([2, 1])
    
    # Transformer 基础架构
    if video_category == "🏗️ Transformer 基础架构":
        video_choice = st.radio(
            "选择组件",
            ["Encoder 编码器", "Decoder 掩码", "Cross Attention 交叉注意力", "残差连接与层归一化"],
            horizontal=True
        )
        
        with col_video:
            if "Encoder" in video_choice:
                st.video("assets/EncoderFlow.mp4")
            elif "Decoder" in video_choice:
                st.video("assets/DecoderMasking.mp4")
            elif "Cross" in video_choice:
                st.video("assets/CrossAttentionFlow.mp4")
            elif "残差" in video_choice:
                st.video("assets/ResidualNorm.mp4")
        
        with col_notes:
            st.markdown("### 📝 核心要点")
            if "Encoder" in video_choice:
                st.markdown("""
                **Encoder 编码器**
                - 多层堆叠结构
                - Self-Attention + FFN
                - 残差连接 + LayerNorm
                - 并行处理所有位置
                """)
            elif "Decoder" in video_choice:
                st.markdown("""
                **Causal Mask 因果掩码**
                - 防止"看见未来"
                - 上三角矩阵设为 -∞
                - Softmax 后变为 0
                - 保证自回归特性
                """)
            elif "Cross" in video_choice:
                st.markdown("""
                **Cross-Attention**
                - Q 来自 Decoder
                - K, V 来自 Encoder
                - 编解码器协作
                - 机器翻译的核心
                """)
            elif "残差" in video_choice:
                st.markdown("""
                **Residual & LayerNorm**
                - 残差连接：梯度高速公路
                - LayerNorm：稳定训练
                - Post-LN vs Pre-LN
                - 深层网络的关键
                """)
    
    # 位置编码机制
    elif video_category == "🔄 位置编码机制":
        video_choice = st.radio(
            "选择编码方式",
            ["正弦位置编码 (Sinusoidal)", "RoPE 旋转位置编码"],
            horizontal=True
        )
        
        with col_video:
            if "正弦" in video_choice:
                st.video("assets/PositionalEncoding.mp4")
            else:
                st.video("assets/RoPEMath.mp4")
        
        with col_notes:
            st.markdown("### 📝 核心要点")
            if "正弦" in video_choice:
                st.markdown("""
                **Sinusoidal Positional Encoding**
                - 正弦/余弦函数组合
                - 不同频率捕捉不同尺度
                - 固定编码，无需学习
                - 可以外推到更长序列
                """)
                
                # 交互式演示
                st.markdown("#### 🎮 交互演示")
                pos = st.slider("位置", 0, 20, 5, key="pos_sin")
                pe_dim = st.slider("维度", 4, 32, 16, key="dim_sin")
                
                pe = np.zeros(pe_dim)
                for i in range(0, pe_dim, 2):
                    pe[i] = np.sin(pos / (10000 ** (2 * i / pe_dim)))
                    if i + 1 < pe_dim:
                        pe[i+1] = np.cos(pos / (10000 ** (2 * i / pe_dim)))
                
                st.write(f"位置 {pos} 的编码:")
                st.bar_chart(pe)
            else:
                st.markdown("""
                **RoPE (Rotary Position Embedding)**
                - 复数旋转机制
                - 相对位置编码
                - 点积自动包含位置信息
                - LLaMA/GPT-NeoX 使用
                """)
    
    # 注意力机制详解
    elif video_category == "🎯 注意力机制详解":
        with col_video:
            st.video("assets/MultiHeadDetailed.mp4")
        
        with col_notes:
            st.markdown("### 📝 核心要点")
            st.markdown("""
            **Multi-Head Attention**
            - 8 个独立的注意力头
            - 每个头关注不同模式
            - 并行计算，拼接输出
            - 多样性与表达能力
            """)
            
            # 交互式多头可视化
            st.markdown("#### 🎮 多头权重分配")
            head_weights = []
            for i in range(n_heads):
                weight = st.slider(f"Head {i+1}", 0.0, 1.0, 1.0/n_heads, 0.1, key=f"head_{i}")
                head_weights.append(weight)
            
            # 归一化
            total = sum(head_weights)
            if total > 0:
                head_weights = [w/total for w in head_weights]
            
            fig = go.Figure(data=[go.Pie(labels=[f"Head {i+1}" for i in range(n_heads)], 
                                         values=head_weights)])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # 前馈网络
    elif video_category == "🧬 前馈网络 (FFN)":
        with col_video:
            st.video("assets/FFNSwiGLU.mp4")
        
        with col_notes:
            st.markdown("### 📝 核心要点")
            st.markdown("""
            **FFN vs SwiGLU**
            - 传统 FFN: d → 4d → d
            - SwiGLU: 门控机制
            - Swish 激活函数
            - LLaMA/PaLM 使用
            """)
            
            # 参数量计算
            st.markdown("#### 🎮 参数量计算")
            ffn_params = 2 * d_model * (4 * d_model)
            swiglu_params = 3 * d_model * (8 * d_model // 3)
            
            st.metric("传统 FFN 参数量", f"{ffn_params:,}")
            st.metric("SwiGLU 参数量", f"{swiglu_params:,}")
            st.caption("两者参数量相近，但 SwiGLU 性能更好")
    
    # 采样与分词
    elif video_category == "🎲 采样与分词":
        with col_video:
            st.video("assets/BPEDetailed.mp4")
        
        with col_notes:
            st.markdown("### 📝 核心要点")
            st.markdown("""
            **BPE 分词算法**
            - 字节对编码
            - 迭代合并高频对
            - 子词级别分词
            - 为什么 GPT 数不清 'r'
            """)
            
            # 简单的 BPE 演示
            st.markdown("#### 🎮 BPE 演示")
            text = st.text_input("输入文本", "strawberry", key="bpe_text")
            st.write("字符级拆分:", list(text))
            st.caption("完整的 BPE 需要训练词表，这里仅展示概念")
    
    # 训练与优化
    elif video_category == "🏋️ 训练与优化":
        video_choice = st.radio(
            "选择主题",
            ["训练损失", "AdamW 优化器", "混合精度训练"],
            horizontal=True
        )
        
        with col_video:
            if "损失" in video_choice:
                st.video("assets/TrainingLoss.mp4")
            elif "AdamW" in video_choice:
                st.video("assets/AdamWOptimizer.mp4")
            else:
                st.video("assets/MixedPrecision.mp4")
        
        with col_notes:
            st.markdown("### 📝 核心要点")
            if "损失" in video_choice:
                st.markdown("""
                **训练损失函数**
                - Cross-Entropy Loss
                - Next Token Prediction
                - Teacher Forcing
                - 梯度裁剪
                """)
            elif "AdamW" in video_choice:
                st.markdown("""
                **AdamW 优化器**
                - Adam + 解耦权重衰减
                - 自适应学习率
                - Warmup + Cosine Decay
                - 现代 LLM 标配
                """)
            else:
                st.markdown("""
                **混合精度训练**
                - FP32/FP16/BF16
                - 加速训练 2-3倍
                - BF16 更稳定
                - 节省显存
                """)
    
    # Mamba 架构
    elif video_category == "🐍 Mamba 架构":
        video_choice = st.radio(
            "选择组件",
            ["Mamba 机制", "离散化过程"],
            horizontal=True
        )
        
        with col_video:
            if "机制" in video_choice:
                st.video("assets/MambaMechanism.mp4")
            else:
                st.video("assets/DiscretizationVisual.mp4")
        
        with col_notes:
            st.markdown("### 📝 核心要点")
            if "机制" in video_choice:
                st.markdown("""
                **Mamba 选择性机制**
                - 动态状态空间模型
                - 选择性遗忘/记忆
                - O(L) 复杂度
                - 长序列优势
                """)
            else:
                st.markdown("""
                **离散化过程**
                - 连续 ODE → 离散化
                - 动态步长 Δ
                - Zero-Order Hold
                - 数学桥梁
                """)
    
    # 架构对比
    elif video_category == "⚔️ 架构对比":
        with col_video:
            st.video("assets/TransformerVsMamba.mp4")
        
        with col_notes:
            st.markdown("### 📝 核心要点")
            st.markdown("""
            **Transformer vs Mamba**
            - 复杂度: O(L²) vs O(L)
            - 显存: KV Cache vs Fixed State
            - 推理: 慢 vs 快
            - 训练: 并行 vs 串行
            """)
            
            # 复杂度对比
            st.markdown("#### 🎮 复杂度对比")
            seq_lengths = [128, 512, 1024, 2048, 4096]
            transformer_cost = [l**2 for l in seq_lengths]
            mamba_cost = seq_lengths
            
            df = pd.DataFrame({
                "序列长度": seq_lengths,
                "Transformer (L²)": transformer_cost,
                "Mamba (L)": mamba_cost
            })
            
            fig = px.line(df, x="序列长度", y=["Transformer (L²)", "Mamba (L)"],
                         labels={"value": "计算复杂度", "variable": "模型"})
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: 交互实验室
# ==========================================
with tab2:
    st.header("🧮 交互式数学实验室")
    
    experiment = st.selectbox(
        "选择实验",
        [
            "Attention 计算过程",
            "Softmax 温度调节",
            "位置编码可视化",
            "多头注意力权重",
            "FFN 维度变换"
        ]
    )
    
    if experiment == "Attention 计算过程":
        st.markdown("### 🎯 Attention 完整计算")
        
        # 生成随机 Q, K, V
        np.random.seed(42)
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Step 1: Q × K^T")
            scores = Q @ K.T
            st.dataframe(pd.DataFrame(scores, 
                                     index=[f"Q{i}" for i in range(seq_len)],
                                     columns=[f"K{i}" for i in range(seq_len)]).style.format("{:.2f}"))
        
        with col2:
            st.markdown("#### Step 2: 除以 √d_k")
            scaled_scores = scores / np.sqrt(d_k)
            st.dataframe(pd.DataFrame(scaled_scores,
                                     index=[f"Q{i}" for i in range(seq_len)],
                                     columns=[f"K{i}" for i in range(seq_len)]).style.format("{:.2f}"))
        
        st.markdown("#### Step 3: Softmax")
        attention_weights = F.softmax(torch.tensor(scaled_scores), dim=-1).numpy()
        st.dataframe(pd.DataFrame(attention_weights,
                                 index=[f"Q{i}" for i in range(seq_len)],
                                 columns=[f"K{i}" for i in range(seq_len)]).style.format("{:.3f}").background_gradient(cmap="Blues"))
        
        st.markdown("#### Step 4: Attention × V")
        output = attention_weights @ V
        st.dataframe(pd.DataFrame(output,
                                 index=[f"Out{i}" for i in range(seq_len)],
                                 columns=[f"d{i}" for i in range(d_k)]).style.format("{:.2f}"))
    
    elif experiment == "Softmax 温度调节":
        st.markdown("### 🌡️ Temperature 对 Softmax 的影响")
        
        # 模拟 logits
        logits = np.array([3.0, 1.0, 0.5])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 原始 Logits")
            st.write(logits)
            
            st.markdown("#### 不同温度下的概率分布")
            temps = [0.5, 1.0, 2.0]
            results = {}
            for temp in temps:
                probs = F.softmax(torch.tensor(logits / temp), dim=0).numpy()
                results[f"T={temp}"] = probs
            
            df_temp = pd.DataFrame(results, index=["Token 1", "Token 2", "Token 3"])
            st.dataframe(df_temp.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=1))
        
        with col2:
            st.markdown("#### 可视化")
            fig = go.Figure()
            for temp in temps:
                probs = F.softmax(torch.tensor(logits / temp), dim=0).numpy()
                fig.add_trace(go.Bar(name=f"T={temp}", x=["Token 1", "Token 2", "Token 3"], y=probs))
            fig.update_layout(barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **观察**:
            - T < 1: 分布更尖锐（保守）
            - T = 1: 原始分布
            - T > 1: 分布更平滑（创造性）
            """)
    
    elif experiment == "位置编码可视化":
        st.markdown("### 📍 位置编码热力图")
        
        max_len = st.slider("最大序列长度", 10, 100, 50)
        pe_dim = st.slider("编码维度", 8, 64, 32)
        
        # 生成位置编码矩阵
        pe_matrix = np.zeros((max_len, pe_dim))
        for pos in range(max_len):
            for i in range(0, pe_dim, 2):
                pe_matrix[pos, i] = np.sin(pos / (10000 ** (2 * i / pe_dim)))
                if i + 1 < pe_dim:
                    pe_matrix[pos, i+1] = np.cos(pos / (10000 ** (2 * i / pe_dim)))
        
        fig = px.imshow(pe_matrix, 
                       labels=dict(x="维度", y="位置", color="值"),
                       color_continuous_scale="RdBu",
                       aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **观察**:
        - 不同维度有不同的频率
        - 低维度变化快（高频）
        - 高维度变化慢（低频）
        - 形成独特的位置指纹
        """)
    
    elif experiment == "多头注意力权重":
        st.markdown("### 🎯 多头注意力权重分布")
        
        # 为每个头生成随机权重
        head_attentions = []
        for h in range(n_heads):
            np.random.seed(h)
            attn = np.random.rand(seq_len, seq_len)
            attn = attn / attn.sum(axis=1, keepdims=True)  # 归一化
            head_attentions.append(attn)
        
        # 选择要查看的头
        selected_head = st.selectbox("选择注意力头", [f"Head {i+1}" for i in range(n_heads)])
        head_idx = int(selected_head.split()[1]) - 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {selected_head} 权重矩阵")
            fig = px.imshow(head_attentions[head_idx],
                           labels=dict(x="Key", y="Query", color="权重"),
                           color_continuous_scale="Blues",
                           aspect="auto")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 所有头的平均权重")
            avg_attn = np.mean(head_attentions, axis=0)
            fig = px.imshow(avg_attn,
                           labels=dict(x="Key", y="Query", color="权重"),
                           color_continuous_scale="Greens",
                           aspect="auto")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif experiment == "FFN 维度变换":
        st.markdown("### 🧬 FFN 维度变换过程")
        
        st.markdown(f"""
        **当前配置**:
        - 输入维度: {d_model}
        - 中间维度: {4 * d_model} (4倍扩展)
        - 输出维度: {d_model}
        """)
        
        # 可视化维度变换
        stages = ["输入", "W1 扩展", "激活函数", "W2 压缩", "输出"]
        dims = [d_model, 4*d_model, 4*d_model, d_model, d_model]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stages,
            y=dims,
            text=dims,
            textposition='auto',
            marker_color=['blue', 'green', 'orange', 'green', 'purple']
        ))
        fig.update_layout(
            title="FFN 维度变化",
            yaxis_title="维度大小",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 参数量计算
        w1_params = d_model * (4 * d_model)
        w2_params = (4 * d_model) * d_model
        total_params = w1_params + w2_params
        
        st.markdown(f"""
        **参数量分析**:
        - W1 参数: {d_model} × {4*d_model} = {w1_params:,}
        - W2 参数: {4*d_model} × {d_model} = {w2_params:,}
        - 总参数量: {total_params:,}
        """)

# ==========================================
# TAB 3: 可视化分析
# ==========================================
with tab3:
    st.header("📊 模型行为可视化分析")
    
    analysis_type = st.selectbox(
        "选择分析类型",
        [
            "注意力模式分析",
            "层级特征演化",
            "参数量对比",
            "计算复杂度分析"
        ]
    )
    
    if analysis_type == "注意力模式分析":
        st.markdown("### 🎯 注意力模式可视化")
        
        # 生成模拟的注意力模式
        patterns = {
            "局部注意力": np.eye(seq_len, k=0) + np.eye(seq_len, k=1) + np.eye(seq_len, k=-1),
            "全局注意力": np.ones((seq_len, seq_len)) / seq_len,
            "因果注意力": np.tril(np.ones((seq_len, seq_len)))
        }
        
        pattern_choice = st.radio("选择模式", list(patterns.keys()), horizontal=True)
        
        pattern = patterns[pattern_choice]
        pattern = pattern / pattern.sum(axis=1, keepdims=True)  # 归一化
        
        fig = px.imshow(pattern,
                       labels=dict(x="Key Position", y="Query Position", color="Attention"),
                       color_continuous_scale="Blues",
                       aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **{pattern_choice} 特点**:
        """)
        if "局部" in pattern_choice:
            st.markdown("- 每个位置只关注相邻位置\n- 适合捕捉局部特征\n- 计算效率高")
        elif "全局" in pattern_choice:
            st.markdown("- 每个位置平等关注所有位置\n- 捕捉全局依赖\n- 计算复杂度高")
        else:
            st.markdown("- 只能看到当前及之前的位置\n- 防止信息泄露\n- 自回归生成必需")
    
    elif analysis_type == "层级特征演化":
        st.markdown("### 🔄 特征在层间的演化")
        
        num_layers = st.slider("Transformer 层数", 1, 12, 6)
        
        # 模拟特征演化（随机游走）
        np.random.seed(42)
        features = [np.random.randn(d_model)]
        for _ in range(num_layers):
            features.append(features[-1] + np.random.randn(d_model) * 0.3)
        
        # 计算每层的统计量
        means = [f.mean() for f in features]
        stds = [f.std() for f in features]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(num_layers+1)), y=means, mode='lines+markers', name='均值'))
            fig.update_layout(title="特征均值演化", xaxis_title="层数", yaxis_title="均值", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(num_layers+1)), y=stds, mode='lines+markers', name='标准差', line=dict(color='red')))
            fig.update_layout(title="特征方差演化", xaxis_title="层数", yaxis_title="标准差", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "参数量对比":
        st.markdown("### 🔢 不同模型的参数量对比")
        
        models = {
            "BERT-Base": {"L": 12, "d": 768, "h": 12},
            "BERT-Large": {"L": 24, "d": 1024, "h": 16},
            "GPT-2": {"L": 12, "d": 768, "h": 12},
            "GPT-3 Small": {"L": 12, "d": 768, "h": 12},
            "LLaMA-7B": {"L": 32, "d": 4096, "h": 32},
        }
        
        params_list = []
        for name, config in models.items():
            L, d, h = config["L"], config["d"], config["h"]
            # 简化计算: 12d² per layer (Attention + FFN)
            params = L * 12 * (d ** 2)
            params_list.append({"模型": name, "参数量 (M)": params / 1e6, "层数": L, "维度": d})
        
        df = pd.DataFrame(params_list)
        
        fig = px.bar(df, x="模型", y="参数量 (M)", 
                    color="层数",
                    hover_data=["维度"],
                    title="模型参数量对比")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df.style.format({"参数量 (M)": "{:.1f}"}))
    
    elif analysis_type == "计算复杂度分析":
        st.markdown("### ⚡ Transformer vs Mamba 复杂度对比")
        
        seq_lengths = np.arange(128, 4096, 128)
        
        # Transformer: O(L²d)
        transformer_flops = seq_lengths ** 2 * d_model
        # Mamba: O(Ld)
        mamba_flops = seq_lengths * d_model * d_state
        
        df = pd.DataFrame({
            "序列长度": seq_lengths,
            "Transformer (L²d)": transformer_flops,
            "Mamba (Ld×N)": mamba_flops
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seq_lengths, y=transformer_flops, mode='lines', name='Transformer', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=seq_lengths, y=mamba_flops, mode='lines', name='Mamba', line=dict(color='green')))
        fig.update_layout(
            title="计算复杂度对比 (FLOPs)",
            xaxis_title="序列长度",
            yaxis_title="FLOPs",
            yaxis_type="log",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **关键观察**:
        - Transformer 随序列长度平方增长
        - Mamba 保持线性增长
        - 长序列场景下 Mamba 优势明显
        - 但 Transformer 训练更容易并行化
        """)

# ==========================================
# TAB 4: 训练与优化
# ==========================================
with tab4:
    st.header("🏋️ 训练与优化实验")
    
    training_topic = st.selectbox(
        "选择主题",
        [
            "学习率调度",
            "优化器对比",
            "混合精度效果",
            "损失函数分析"
        ]
    )
    
    if training_topic == "学习率调度":
        st.markdown("### 📈 学习率调度策略")
        
        schedule_type = st.radio(
            "选择调度策略",
            ["Warmup + Cosine Decay", "Step Decay", "Exponential Decay"],
            horizontal=True
        )
        
        warmup_steps = st.slider("Warmup 步数", 0, 1000, 100)
        total_steps = st.slider("总训练步数", 1000, 10000, 5000)
        max_lr = learning_rate
        
        steps = np.arange(total_steps)
        
        if schedule_type == "Warmup + Cosine Decay":
            lrs = []
            for step in steps:
                if step < warmup_steps:
                    lr = max_lr * step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    lr = max_lr * 0.5 * (1 + np.cos(np.pi * progress))
                lrs.append(lr)
        elif schedule_type == "Step Decay":
            decay_steps = total_steps // 3
            lrs = [max_lr * (0.1 ** (step // decay_steps)) for step in steps]
        else:  # Exponential
            decay_rate = 0.96
            lrs = [max_lr * (decay_rate ** (step / 100)) for step in steps]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=lrs, mode='lines', name='Learning Rate'))
        fig.update_layout(
            title=f"{schedule_type} 学习率调度",
            xaxis_title="训练步数",
            yaxis_title="学习率",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **当前配置**:
        - 最大学习率: {max_lr:.0e}
        - Warmup 步数: {warmup_steps}
        - 总步数: {total_steps}
        """)
    
    elif training_topic == "优化器对比":
        st.markdown("### ⚙️ 优化器性能对比")
        
        st.markdown("""
        | 优化器 | 自适应学习率 | 动量 | 权重衰减 | 适用场景 |
        |--------|------------|------|---------|---------|
        | SGD | ❌ | ✅ | ✅ | 简单任务 |
        | Adam | ✅ | ✅ | ⚠️ (耦合) | 通用 |
        | AdamW | ✅ | ✅ | ✅ (解耦) | LLM 训练 |
        | Lion | ✅ | ✅ | ✅ | 大模型 |
        """)
        
        # 模拟优化轨迹
        np.random.seed(42)
        steps = 100
        
        # SGD: 震荡较大
        sgd_loss = 2.0 * np.exp(-np.arange(steps) / 30) + np.random.randn(steps) * 0.1
        # Adam: 平滑下降
        adam_loss = 2.0 * np.exp(-np.arange(steps) / 20) + np.random.randn(steps) * 0.05
        # AdamW: 更快收敛
        adamw_loss = 2.0 * np.exp(-np.arange(steps) / 15) + np.random.randn(steps) * 0.03
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(steps)), y=sgd_loss, mode='lines', name='SGD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(steps)), y=adam_loss, mode='lines', name='Adam', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=list(range(steps)), y=adamw_loss, mode='lines', name='AdamW', line=dict(color='green')))
        fig.update_layout(
            title="优化器收敛对比 (模拟)",
            xaxis_title="训练步数",
            yaxis_title="损失",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif training_topic == "混合精度效果":
        st.markdown("### 🎯 混合精度训练效果")
        
        precision_type = st.radio(
            "选择精度",
            ["FP32 (全精度)", "FP16 (半精度)", "BF16 (Brain Float)"],
            horizontal=True
        )
        
        # 精度对比表
        precision_data = {
            "精度类型": ["FP32", "FP16", "BF16"],
            "指数位": [8, 5, 8],
            "尾数位": [23, 10, 7],
            "数值范围": ["±3.4e38", "±6.5e4", "±3.4e38"],
            "精度": ["高", "中", "中"],
            "速度提升": ["1x", "2-3x", "2-3x"],
            "显存节省": ["0%", "50%", "50%"]
        }
        
        df = pd.DataFrame(precision_data)
        st.dataframe(df)
        
        st.markdown(f"""
        **{precision_type} 特点**:
        """)
        if "FP32" in precision_type:
            st.markdown("- 标准精度，最稳定\n- 显存占用大\n- 训练速度慢")
        elif "FP16" in precision_type:
            st.markdown("- 速度快，显存省\n- 容易溢出\n- 需要 Loss Scaling")
        else:
            st.markdown("- 速度快，显存省\n- 数值范围大，不易溢出\n- 现代 LLM 首选")
        
        # 显存占用对比
        model_params = 1e9  # 1B 参数
        fp32_mem = model_params * 4 / 1e9  # GB
        fp16_mem = model_params * 2 / 1e9
        bf16_mem = model_params * 2 / 1e9
        
        mem_data = pd.DataFrame({
            "精度": ["FP32", "FP16", "BF16"],
            "显存占用 (GB)": [fp32_mem, fp16_mem, bf16_mem]
        })
        
        fig = px.bar(mem_data, x="精度", y="显存占用 (GB)", 
                    title=f"1B 参数模型显存占用对比",
                    color="精度")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    elif training_topic == "损失函数分析":
        st.markdown("### 📉 损失函数行为分析")
        
        # 模拟训练曲线
        epochs = np.arange(100)
        train_loss = 2.5 * np.exp(-epochs / 20) + 0.1 + np.random.randn(100) * 0.05
        val_loss = 2.5 * np.exp(-epochs / 25) + 0.2 + np.random.randn(100) * 0.08
        
        # 添加过拟合段
        train_loss[70:] = train_loss[70] - (epochs[70:] - 70) * 0.002
        val_loss[70:] = val_loss[70] + (epochs[70:] - 70) * 0.003
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='训练损失', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='验证损失', line=dict(color='red')))
        fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="开始过拟合")
        fig.update_layout(
            title="训练过程损失曲线",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **观察要点**:
        - 训练损失持续下降
        - 验证损失在 Epoch 70 后开始上升
        - 这是典型的过拟合信号
        - 应该在此处停止训练或增加正则化
        """)

# ==========================================
# 底部信息
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🧠 Transformer Explorer | 深度学习架构可视化平台</p>
    <p>基于 Manim 动画引擎 & Streamlit 交互框架</p>
    <p>© 2025 by Just For Dream Lab | 严谨 · 务实 · 深度</p>
</div>
""", unsafe_allow_html=True)
