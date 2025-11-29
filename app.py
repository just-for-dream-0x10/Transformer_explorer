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
</style>
""",
    unsafe_allow_html=True,
)

st.title("🤖 Transformer vs 🐍 Mamba：核心机制可视化")
st.markdown("从宏观动画到微观数学：严谨还原算法细节。")

# ==========================================
# 左侧边栏：统一参数控制 (保留)
# ==========================================
with st.sidebar:
    st.header("⚙️ 实验室设置")

    st.subheader("1. Transformer 参数")
    d_model = st.slider("嵌入维度 (d_model)", 4, 32, 8, step=4)
    n_heads = st.radio("多头数量 (Heads)", [1, 2, 4], index=1)
    d_k = d_model // n_heads

    st.divider()

    st.subheader("2. Mamba 参数")
    d_state = st.slider("状态维度 (d_state/N)", 2, 16, 4, help="SSM 隐状态水箱的大小")

    st.divider()

    st.subheader("3. 输入数据")
    user_input = st.text_input(
        "输入文本 (空格分隔)", "I love LLM", help="尝试输入不同的句子长度"
    )
    tokens = user_input.split()
    seq_len = len(tokens)

    st.success(
        f"📊 当前配置:\n- 序列长度: {seq_len}\n- 头维度: {d_k}\n- 状态维度: {d_state}"
    )

# ==========================================
# 主界面：选项卡
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["🎥 核心原理动画 (Manim)", "🧮 交互式数学实验室", "🧊 Attention 热力图"]
)

# -----------------------------------------------------------------------------
# Tab 1: Manim 动画影院 (布局大重构：拆分为子标签页)
# -----------------------------------------------------------------------------
with tab1:
    st.info("💡 请点击下方子标签切换不同架构的演示。")

    # === 使用子标签页进行物理隔离 ===
    sub_tf, sub_vs, sub_mamba = st.tabs(
        ["🤖 Transformer 家族", "⚔️ 架构对比", "🐍 Mamba 家族"]
    )

    # --- 1. Transformer 专区 ---
    with sub_tf:
        tf_choice = st.radio(
            "选择 Transformer 组件:",
            [
                "1. 基础注意力 (Dot-Product)",
                "2. Encoder 架构 (Residual)",
                "3. Decoder 掩码 (Masking)",
                "4. 协作机制 (Cross-Attention)",
            ],
            horizontal=True,
        )
        st.divider()
        col_v, col_t = st.columns([1.8, 1])

        if "1." in tf_choice:
            with col_v:
                try:
                    st.video("assets/Attention.mp4")
                except:
                    st.error("缺文件: assets/Attention.mp4")
            with col_t:
                st.subheader("🔍 基础注意力解析")
                st.markdown(
                    """
                **Transformer 的原子操作：计算相关性。**
                1. **布局**: 左侧 $Q$ (Query)，上方 $K^T$ (Key)。
                2. **点积**: 视频中**黄色高亮**扫描处，计算向量夹角。
                3. **Softmax**: 矩阵变红，代表概率分布。
                """
                )
        elif "2." in tf_choice:
            with col_v:
                try:
                    st.video("assets/EncoderFlow.mp4")
                except:
                    st.error("缺文件: assets/EncoderFlow.mp4")
            with col_t:
                st.subheader("🔍 Encoder 解析")
                st.markdown(
                    """
                **宏观数据流向。**
                1. **多头分裂**: Input 分裂为 Q, K, V，再分裂为多个 Head。
                2. **残差连接**: 巨大的**黄色弧线**。它是梯度的“高速公路”。
                """
                )
        elif "3." in tf_choice:
            with col_v:
                try:
                    st.video("assets/DecoderMasking.mp4")
                except:
                    st.error("缺文件: assets/DecoderMasking.mp4")
            with col_t:
                st.subheader("🔍 Decoder Mask 解析")
                st.markdown(
                    """
                **时间机器锁。**
                1. **Mask 降临**: 右上角变成 <font color='red'>-inf</font>。
                2. **Softmax 归零**: 物理切断了通向未来的视线。
                """,
                    unsafe_allow_html=True,
                )
        elif "4." in tf_choice:
            with col_v:
                try:
                    st.video("assets/CrossAttentionFlow.mp4")
                except:
                    st.error("缺文件: assets/CrossAttentionFlow.mp4")
            with col_t:
                st.subheader("🔍 Cross-Attention 解析")
                st.markdown(
                    """
                **Encoder 与 Decoder 的对话。**
                1. **角色**: 左侧 Encoder 提供知识库 (K, V)，右侧 Decoder 拿着问题 (Q)。
                2. **流程**: Q 扫描 K $\\to$ 生成权重 $\\to$ 提取 V。
                """
                )

    # --- 2. 对比专区 ---
    with sub_vs:
        st.subheader("⚔️ 巅峰对决：复杂度可视化")
        col_v_vs, col_t_vs = st.columns([1.8, 1])
        with col_v_vs:
            try:
                st.video("assets/TransformerVsMamba.mp4")
            except:
                st.error("缺文件: assets/TransformerVsMamba.mp4")
        with col_t_vs:
            st.markdown(
                """
            **$O(L^2)$ vs $O(L)$ 的直观差异**
            
            **左侧 (Transformer)**: 
            * 随着序列变长，矩阵面积呈**平方级爆炸**。
            * 处理长文时显存迅速耗尽。
            
            **右侧 (Mamba)**: 
            * 无论序列多长，它的高度 (State Dim) 是固定的！
            * 推理显存恒定 $O(1)$。
            """
            )

    # --- 3. Mamba 专区 ---
    with sub_mamba:
        mamba_choice = st.radio(
            "选择 Mamba 组件:",
            ["1. Mamba 核心机制 (Selective Scan)", "2. 数学基础 (Discretization)"],
            horizontal=True,
        )
        st.divider()
        col_v_m, col_t_m = st.columns([1.8, 1])

        if "1." in mamba_choice:
            with col_v_m:
                try:
                    st.video("assets/MambaMechanism.mp4")
                except:
                    st.error("缺文件: assets/MambaMechanism.mp4")
            with col_t_m:
                st.subheader("🐍 选择性遗忘机制")
                st.markdown(
                    r"""
                **核心公式**: $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$
                
                **现象解析**:
                * **动态阀门**: $\bar{A}, \bar{B}$ 是随输入变化的。
                * **Noise (红)**: 阀门关闭，记忆衰减。
                * **Key Info (绿)**: 阀门大开，强力写入。
                """
                )
        elif "2." in mamba_choice:
            with col_v_m:
                try:
                    st.video("assets/DiscretizationVisual.mp4")
                except:
                    st.error("缺文件: assets/DiscretizationVisual.mp4")
            with col_t_m:
                st.subheader("🐍 数学桥梁：离散化")
                st.markdown(
                    r"""
                **从连续物理到数字信号**
                
                1. **ZOH (零阶保持)**: 假设在 $\Delta$ 时间内输入不变。
                2. **$\Delta$ (步长)**: 
                   * $\Delta$ 大 $\to$ 更多遗忘，更多写入。
                   * $\Delta$ 小 $\to$ 保持状态。
                """
                )

# -----------------------------------------------------------------------------
# Tab 2: 交互式计算 (保持原样，含公式推导)
# -----------------------------------------------------------------------------
with tab2:
    st.header("🧮 算法内核推导")
    st.caption("这里不仅有代码运行结果，还有每一步背后的数学公式。")

    # 选择内核
    model_type = st.selectbox(
        "选择模型内核进行推导:",
        ["Transformer (Self-Attention)", "Mamba (Selective Scan)"],
    )

    # =========================================================
    # Transformer 模块
    # =========================================================
    if model_type == "Transformer (Self-Attention)":

        # --- 数学原理区 ---
        st.markdown('<div class="math-box">', unsafe_allow_html=True)
        st.markdown("### 📐 核心公式：Scaled Dot-Product Attention")
        st.latex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
        )
        st.markdown(
            """
        * **Q (Query)**: 查询向量，代表当前 Token 想找什么。
        * **K (Key)**: 键向量，代表被查询 Token 的特征标签。
        * **V (Value)**: 值向量，代表实际包含的信息内容。
        * **$\sqrt{d_k}$**: 缩放因子，防止点积过大导致梯度消失。
        """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 代码交互区 ---
        st.subheader("1. Input Embedding (随机初始化)")
        torch.manual_seed(42)
        X = torch.randn(seq_len, d_model)
        df_x = pd.DataFrame(
            X.numpy(), index=tokens, columns=[f"d_{i}" for i in range(d_model)]
        )
        st.dataframe(
            df_x.style.background_gradient(cmap="Blues", axis=None), height=150
        )

        st.subheader("2. Linear Projections ($W_Q, W_K, W_V$)")
        col_q, col_k = st.columns(2)
        W_q = torch.randn(d_model, d_k)
        W_k = torch.randn(d_model, d_k)
        W_v = torch.randn(d_model, d_model)  # Fix: W_v added

        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v  # Fix: V added

        with col_q:
            st.markdown("**Query Matrix ($Q = XW_Q$)**")
            st.dataframe(
                pd.DataFrame(Q.numpy(), index=tokens).style.background_gradient(
                    cmap="Reds", axis=None
                ),
                height=150,
            )
        with col_k:
            st.markdown("**Key Matrix ($K = XW_K$)**")
            st.dataframe(
                pd.DataFrame(K.numpy(), index=tokens).style.background_gradient(
                    cmap="Greens", axis=None
                ),
                height=150,
            )

        st.subheader("3. Attention Scores & Softmax")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(r"**Raw Scores**: $S = QK^T$")
            raw_scores = Q @ K.T
            st.dataframe(
                pd.DataFrame(
                    raw_scores.numpy(), index=tokens, columns=tokens
                ).style.background_gradient(cmap="coolwarm", axis=None)
            )
        with c2:
            st.markdown(r"**Probabilities**: $P = \text{softmax}(S / \sqrt{d_k})$")
            scaled_scores = raw_scores / np.sqrt(d_k)
            attn_weights = F.softmax(scaled_scores, dim=-1)
            st.dataframe(
                pd.DataFrame(
                    attn_weights.numpy(), index=tokens, columns=tokens
                ).style.background_gradient(cmap="Oranges", axis=None)
            )

        # 保存用于 Tab 3
        st.session_state["transformer_scores"] = scaled_scores
        st.subheader("4. Final Output ($Z = P \cdot V$)")
        output_z = attn_weights @ V
        st.dataframe(
            pd.DataFrame(output_z.numpy(), index=tokens).style.background_gradient(
                cmap="Purples", axis=None
            )
        )

    # =========================================================
    # Mamba 模块
    # =========================================================
    elif model_type == "Mamba (Selective Scan)":

        # --- 数学原理区 ---
        st.markdown('<div class="math-box">', unsafe_allow_html=True)
        st.markdown("### 📐 核心公式：Selective SSM")
        st.markdown("**1. 连续系统 (Continuous)**")
        st.latex(r"h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)")

        st.markdown("**2. 离散化 (Discretization) - Zero Order Hold**")
        c_math1, c_math2 = st.columns(2)
        with c_math1:
            st.latex(r"\bar{A} = \exp(\Delta \cdot \mathbf{A})")
        with c_math2:
            st.latex(r"\bar{B} \approx \Delta \cdot \mathbf{B}")

        st.markdown("**3. 递归推理 (Recurrence)**")
        st.latex(r"h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t")
        st.warning("关键：$\Delta, \bar{B}$ 随输入 $x_t$ 变化！")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 代码交互区 ---
        st.subheader("🐍 Mamba 逐步递归模拟")

        np.random.seed(42)
        # 初始化参数
        A_fixed = -np.exp(np.random.randn(d_model, d_state))
        B_fixed = np.random.randn(d_model, d_state)
        W_delta = np.random.randn(d_model, d_model)

        inputs = np.random.randn(seq_len, d_model)
        h_t = np.zeros((d_model, d_state))

        history_h = []
        history_delta = []

        # 逐步展示
        cols = st.columns(min(seq_len, 4))

        for t in range(seq_len):
            x_t = inputs[t]

            # 1. 计算 Delta
            delta_val = np.log(1 + np.exp(x_t @ W_delta))

            # 2. 离散化
            delta_col = delta_val[:, None]
            bar_A = np.exp(delta_col * A_fixed)
            bar_B = delta_col * B_fixed

            # 3. 递归更新
            x_t_col = x_t[:, None]
            h_next = bar_A * h_t + bar_B * x_t_col

            history_h.append(h_next.flatten())
            history_delta.append(delta_val.mean())

            # 前几步的可视化
            if t < 4:
                with cols[t]:
                    curr_token = tokens[t] if t < len(tokens) else f"T{t}"
                    st.markdown(f"**Step {t+1}: {curr_token}**")
                    st.metric("Avg $\Delta$", f"{np.mean(delta_val):.2f}")

                    fig_cell = px.imshow(
                        h_next, color_continuous_scale="Magma", title="State $h_t$"
                    )
                    fig_cell.update_layout(
                        height=150,
                        margin=dict(l=0, r=0, t=30, b=0),
                        coloraxis_showscale=False,
                    )
                    fig_cell.update_xaxes(showticklabels=False)
                    fig_cell.update_yaxes(showticklabels=False)
                    st.plotly_chart(fig_cell, use_container_width=True)

            h_t = h_next

        st.subheader("📊 记忆演变与门控分析")
        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            st.markdown("**1. 隐状态 (Memory) 随时间变化**")
            hist_data = np.array(history_h)[:, :20]
            fig_hist = px.imshow(
                hist_data.T,
                aspect="auto",
                color_continuous_scale="Magma",
                labels=dict(x="Time", y="Dim"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_chart2:
            st.markdown("**2. $\Delta$ (步长) 波动**")
            st.caption("$\Delta$ 越大，代表当前 Token 越重要（被写入越多）。")
            st.line_chart(history_delta)

# -----------------------------------------------------------------------------
# Tab 3: 热力图 (保持原样)
# -----------------------------------------------------------------------------
with tab3:
    st.header("🧊 最终注意力图 (Attention Map)")

    if "transformer_scores" in st.session_state:
        scores = st.session_state["transformer_scores"]
        attn_weights = F.softmax(scores, dim=-1)

        c1, c2 = st.columns([3, 1])
        with c1:
            fig = px.imshow(
                attn_weights.numpy(),
                x=tokens,
                y=tokens,
                labels=dict(x="Key", y="Query", color="Prob"),
                color_continuous_scale="Viridis",
                text_auto=".2f",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.info("💡 **解读**")
            st.markdown("* 对角线颜色深：关注自己。\n* 每一行概率和为 1.0。")
    else:
        st.warning("⚠️ 请先在 Tab 2 中选择 'Transformer' 并运行一次计算。")

# -----------------------------------------------------------------------------
# Tab 4: 关于本项目
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 👨‍💻 关于本项目")
st.info(
    "本项目旨在通过可视化手段，"
    "直观对比 **Transformer** 与 **Mamba (SSM)** "
    "的底层机制差异。"
)
st.caption("© 2025 Just For Dream Lab")
