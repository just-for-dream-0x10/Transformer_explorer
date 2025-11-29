import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="Transformer & Mamba 深度解析",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Transformer vs 🐍 Mamba：核心机制可视化")
st.markdown("""
> **"What I cannot create, I do not understand."** — Richard Feynman
>
> 本项目通过 **Manim 动画** (宏观原理) 与 **交互式推导** (微观数值) 的结合，带你显微镜式观察 Transformer 与 Mamba 的内部运作。
""")

# ==========================================
# 左侧边栏：统一参数控制
# ==========================================
with st.sidebar:
    st.header("⚙️ 实验室设置")
    
    st.subheader("1. Transformer 参数")
    d_model = st.slider("嵌入维度 (d_model)", 4, 16, 8, step=4, help="每个 Token 用多少维向量表示")
    n_heads = st.radio("多头数量 (Heads)", [1, 2, 4], help="将维度切分为几个头并行计算")
    d_k = d_model // n_heads
    
    st.divider()
    
    st.subheader("2. Mamba 参数")
    d_state = st.slider("状态维度 (d_state/N)", 2, 8, 4, help="SSM 隐状态水箱的大小")
    
    st.divider()

    st.subheader("3. 输入数据")
    user_input = st.text_input("输入文本 (空格分隔)", "I love LLM", help="尝试输入不同的句子长度")
    tokens = user_input.split()
    seq_len = len(tokens)

    st.success(f"📊 当前配置:\n- 序列长度: {seq_len}\n- 头维度: {d_k}")
    st.caption("Powered by Streamlit & Manim")

# ==========================================
# 主界面：选项卡
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎥 核心原理动画 (Manim)", "🧮 交互式计算实验室", "🧊 Attention 热力图"])

# -----------------------------------------------------------------------------
# Tab 1: Manim 动画影院 (完整收录 6 个场景)
# -----------------------------------------------------------------------------
with tab1:
    anim_choice = st.radio(
        "选择观测对象:",
        [
            "1. 基础注意力 (Dot-Product)", 
            "2. Encoder 架构 (Residual)", 
            "3. Decoder 掩码 (Masking)", 
            "4. 协作机制 (Cross-Attention)",
            "5. 巅峰对决 (O(L^2) vs O(L))",
            "6. Mamba 核心 (Selective Scan)" 
        ],
        horizontal=True
    )
    
    st.divider()
    col_video, col_text = st.columns([1.8, 1])

    # === 场景 1: 基础注意力 ===
    if "1." in anim_choice:
        with col_video:
            try: st.video("assets/Attention.mp4")
            except: st.error("请确保 assets/Attention.mp4 存在")
        with col_text:
            st.subheader("🔍 深度解析")
            st.markdown("""
            **Transformer 的原子操作：计算相关性。**
            1. **布局**: 左侧 $Q$ (Query)，上方 $K^T$ (Key)。
            2. **点积**: 视频中**黄色高亮**扫描处，计算向量夹角。夹角越小，分数越高。
            3. **Softmax**: 矩阵变红，代表概率分布。每一行概率和为 1。
            """)

    # === 场景 2: Encoder ===
    elif "2." in anim_choice:
        with col_video:
            try: st.video("assets/EncoderFlow.mp4")
            except: st.error("请确保 assets/EncoderFlow.mp4 存在")
        with col_text:
            st.subheader("🔍 深度解析")
            st.markdown("""
            **Encoder 的宏观数据流。**
            1. **多头分裂**: Input 分裂为 Q, K, V，再分裂为多个 Head。
            2. **残差连接**: 注意那条巨大的**黄色弧线**。它是梯度的“高速公路”，防止深层网络梯度消失。
            """)

    # === 场景 3: Decoder ===
    elif "3." in anim_choice:
        with col_video:
            try: st.video("assets/DecoderMasking.mp4")
            except: st.error("请确保 assets/DecoderMasking.mp4 存在")
        with col_text:
            st.subheader("🔍 深度解析")
            st.markdown("""
            **Decoder 的时间机器锁。**
            1. **Mask 降临**: 右上角变成 <font color='red'>红色 -inf</font>，代表“未来”。
            2. **Softmax 归零**: `-inf` 经过 Softmax 变为 **0** (黑色)。这物理切断了通向未来的视线，确保自回归生成。
            """, unsafe_allow_html=True)

    # === 场景 4: Cross-Attention ===
    elif "4." in anim_choice:
        with col_video:
            try: st.video("assets/CrossAttentionFlow.mp4")
            except: st.error("请确保 assets/CrossAttentionFlow.mp4 存在")
        with col_text:
            st.subheader("🔍 深度解析")
            st.markdown("""
            **Encoder 与 Decoder 的对话。**
            1. **角色**: 左侧 Encoder 提供知识库 (K, V)，右侧 Decoder 拿着问题 (Q)。
            2. **三步走**: Q 扫描 K $\\to$ 生成权重 $\\to$ 提取 V $\\to$ 融合。
            """)

    # === 场景 5: Transformer vs Mamba ===
    elif "5." in anim_choice:
        with col_video:
            try: st.video("assets/TransformerVsMamba.mp4")
            except: st.error("请确保 assets/TransformerVsMamba.mp4 存在")
        with col_text:
            st.subheader("⚔️ 巅峰对决：复杂度")
            st.markdown("""
            **$O(L^2)$ vs $O(L)$ 的直观差异。**
            1. **左侧 (Transformer)**: 随着序列变长，矩阵面积呈**平方级爆炸**。显存迅速耗尽。
            2. **右侧 (Mamba)**: 无论序列多长，它的高度 (State Dim) 是固定的！推理显存恒定。
            """)

    # === 场景 6: Mamba 核心 ===
    elif "6." in anim_choice:
        with col_video:
            try: st.video("assets/MambaMechanism.mp4")
            except: st.error("请确保 assets/MambaMechanism.mp4 存在")
        with col_text:
            st.subheader("🐍 Mamba: 选择性机制")
            st.markdown(r"""
            **核心公式**: $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$
            
            1. **动态阀门**: $\bar{A}, \bar{B}$ 是**随输入变化的**。
            2. **现象**:
               - <font color='red'>Noise</font>: 阀门关闭，记忆衰减。
               - <font color='green'>Key Info</font>: 阀门大开，强力写入。
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 2: 交互式计算 (双核驱动：Transformer + Mamba)
# -----------------------------------------------------------------------------
with tab2:
    st.header("🧮 算法内核推导")
    
    # 选择内核
    model_type = st.selectbox("选择模型内核:", ["Transformer (Self-Attention)", "Mamba (Selective Scan)"])
    
    # === Transformer 模块 (保留原逻辑) ===
    if model_type == "Transformer (Self-Attention)":
        st.subheader("1. Input Embedding (随机初始化)")
        torch.manual_seed(42)
        X = torch.randn(seq_len, d_model)
        
        df_x = pd.DataFrame(X.numpy(), index=tokens, columns=[f"d_{i}" for i in range(d_model)])
        st.dataframe(df_x.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
        
        st.subheader("2. Linear Projections")
        col_q, col_k = st.columns(2)
        
        W_q = torch.randn(d_model, d_k)
        W_k = torch.randn(d_model, d_k)
        
        Q = X @ W_q
        K = X @ W_k
        
        with col_q:
            st.markdown(f"**Query Matrix ($X \\times W_Q$)** shape: `{Q.shape}`")
            st.dataframe(pd.DataFrame(Q.numpy(), index=tokens).style.background_gradient(cmap="Reds", axis=None))
        with col_k:
            st.markdown(f"**Key Matrix ($X \\times W_K$)** shape: `{K.shape}`")
            st.dataframe(pd.DataFrame(K.numpy(), index=tokens).style.background_gradient(cmap="Greens", axis=None))

        st.subheader("3. Scaled Dot-Product Attention")
        latex_formula = r"Attention(Q, K) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)"
        st.latex(latex_formula)
        
        raw_scores = Q @ K.T
        scaled_scores = raw_scores / np.sqrt(d_k)
        
        st.write("**Scaled Scores Matrix** (Softmax 之前):")
        st.dataframe(pd.DataFrame(scaled_scores.numpy(), index=tokens, columns=tokens).style.background_gradient(cmap="coolwarm", axis=None))
        
        # 保存用于 Tab 3
        st.session_state['transformer_scores'] = scaled_scores

    # === Mamba 模块 (修复版) ===
    elif model_type == "Mamba (Selective Scan)":
        st.subheader("🐍 Mamba 递归扫描模拟")
        st.markdown(r"手动模拟 RNN 模式推理，观察 Hidden State ($h$) 的演变。")
        
        np.random.seed(42)
        inputs = np.random.randn(seq_len, d_model)
        
        # 简化的参数初始化
        A = -np.exp(np.random.randn(d_model, d_state)) # A < 0 保证稳定
        h_t = np.zeros((d_model, d_state)) # 初始状态
        
        history = []
        cols = st.columns(min(seq_len, 4)) # 最多展示前4步
        
        for t in range(seq_len):
            x_t = inputs[t] # shape: (d_model,)
            
            # 模拟参数生成 (Linear projections)
            # delta_t shape: (d_model,)
            delta_t = np.log(1 + np.exp(np.dot(x_t, np.random.randn(d_model, d_model)))) 
            
            # B_t shape: (d_model, d_state)
            B_t = np.random.randn(d_model, d_state)
            
            # === 修复核心逻辑 ===
            # 将 delta_t 变成列向量 (d_model, 1)，以便广播
            delta_t_col = delta_t[:, None]
            
            # 离散化 (Discretization)
            bar_A = np.exp(delta_t_col * A)        # (D, 1) * (D, N) -> (D, N)
            bar_B = delta_t_col * B_t              # (D, 1) * (D, N) -> (D, N)
            
            # 递归更新 (Recurrence)
            # x_t 需要变成 (D, 1) 才能乘到 bar_B 上
            h_t = bar_A * h_t + bar_B * x_t[:, None]
            
            history.append(h_t.flatten())
            
            # 可视化前几步
            if t < 4:
                with cols[t]:
                    # 安全获取 token
                    curr_token = tokens[t] if t < len(tokens) else f"T{t}"
                    st.caption(f"Time t={t+1}: '{curr_token}'")
                    
                    st.metric("Gate $\Delta$ (Mean)", f"{np.mean(delta_t):.2f}")
                    
                    fig_h = px.imshow(h_t, color_continuous_scale="Magma", title=f"State $h_{t}$")
                    fig_h.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0))
                    fig_h.update_xaxes(showticklabels=False)
                    fig_h.update_yaxes(showticklabels=False)
                    st.plotly_chart(fig_h, use_container_width=True)
        
        st.subheader("📊 记忆演变轨迹")
        st.caption("展示隐状态中前10个维度的数值变化。可以看到有些状态被保持，有些被遗忘。")
        
        # 转换数据以便绘图
        hist_data = np.array(history)[:, :10] # 取前10个特征
        chart_data = pd.DataFrame(
            hist_data, 
            index=[f"t={i+1}" for i in range(seq_len)]
        )
        st.line_chart(chart_data)
        
        st.success(f"""
        **关键结论**:
        无论序列长度 $L$ 是 {seq_len} 还是 10000:
        1. 我们只需要维护 **1 个** 矩阵 $h_t$ (尺寸 {d_model}x{d_state})。
        2. 下一步计算只依赖于 $h_t$ 和 $x_{{t+1}}$。
        3. **显存占用恒定为 $O(1)$** (与 $L$ 无关)。
        """)

# -----------------------------------------------------------------------------
# Tab 3: 热力图 (仅适用于 Transformer)
# -----------------------------------------------------------------------------
with tab3:
    st.header("🧊 最终注意力图 (Attention Map)")
    
    # 检查是否有 Transformer 的计算结果
    if 'transformer_scores' in st.session_state:
        scores = st.session_state['transformer_scores']
        attn_weights = F.softmax(scores, dim=-1)
        
        c1, c2 = st.columns([3, 1])
        with c1:
            fig = px.imshow(
                attn_weights.numpy(),
                x=tokens, y=tokens,
                labels=dict(x="Key (被关注)", y="Query (查询)", color="概率"),
                color_continuous_scale="Viridis",
                text_auto=".2f", aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.info("💡 **解读**")
            st.markdown("* 对角线颜色深：关注自己。\n* 每一行概率和为 1.0。")
    else:
        st.warning("⚠️ 请先在 Tab 2 中选择 'Transformer' 并运行一次计算。")