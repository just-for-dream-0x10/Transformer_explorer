# Transformer & Mamba 架构可视化项目

基于数学原理的深度学习架构解析工具，通过动画演示和交互式实验帮助理解 Transformer 和 Mamba (SSM) 的核心机制。

项目包含：[配套学习笔记](https://github.com/just-for-dream-0x10/beginML/tree/master/other/Self-Attention)

## 项目功能

### 1. 动画演示模块
使用 Manim 生成的数学原理动画，涵盖以下主题：

**Transformer 核心组件**
- Encoder 编码器流程：多头拆分、自注意力计算、残差连接
- Decoder 解码器：因果掩码机制、自回归生成
- Cross Attention：编码器-解码器交互机制
- Multi-Head Attention：多头注意力权重分配与计算
- Positional Encoding：正弦位置编码原理
- Residual & Norm：残差连接与层归一化
- FFN & SwiGLU：前馈网络与门控激活函数

**Mamba 状态空间模型**
- Mamba 选择性机制：动态参数生成与状态更新
- 离散化过程：连续系统到离散递归的转换
- Transformer vs Mamba：架构对比与复杂度分析

**训练与优化技术**
- AdamW 优化器：解耦权重衰减机制
- BPE 分词：字节对编码算法原理
- 混合精度训练：FP16/BF16 训练策略
- RoPE 旋转位置编码：相对位置编码机制
- 训练损失函数：交叉熵与梯度优化

### 2. 交互式实验平台
基于 Streamlit 的 Web 应用，提供以下交互功能：

**参数调节实验室**
- 文本输入与 Token 化处理
- Transformer 参数配置：嵌入维度、多头数量
- Mamba 参数设置：状态维度、选择性参数
- 训练超参数：学习率、温度采样

**数学计算可视化**
- Attention 完整计算过程展示
- Softmax 温度调节效果演示
- 位置编码热力图可视化
- 多头注意力权重分布
- FFN 维度变换过程

**模型行为分析**
- 注意力模式分析（局部/全局/因果）
- 层级特征演化追踪
- 参数量对比分析
- 计算复杂度对比（Transformer vs Mamba）

### 3. 训练优化实验
- 学习率调度策略对比（Warmup + Cosine Decay、Step Decay、Exponential Decay）
- 优化器性能对比（SGD、Adam、AdamW、Lion）
- 混合精度训练效果分析
- 损失函数行为分析与过拟合检测

## 安装与使用

### 环境要求
- Python 3.8+
- FFmpeg（用于 Manim 视频渲染）

### 安装步骤

1. **安装依赖包**
```bash
pip install -r requirement.txt
```

2. **生成动画视频**（可选）
```bash
chmod +x generate_all_videos.sh

./generate_all_videos.sh
```


3. **启动交互应用**
```bash
streamlit run app.py
```

### 项目结构
```
Transformer_Explorer/
├── app.py                 # Streamlit 主应用
├── scene/                 # Manim 动画脚本
│   ├── scene_struct.py    # Transformer 架构
│   ├── scene_mamba_core.py # Mamba 机制
│   ├── scene_*.py         # 其他主题动画
├── assets/                # 视频资源文件
├── media/                 # Manim 输出目录
└── requirement.txt        # 项目依赖
```

## 技术实现

- **动画引擎**：Manim - 数学动画生成工具
- **交互框架**：Streamlit - Web 应用框架
- **数值计算**：NumPy, PyTorch
- **数据可视化**：Plotly, Pandas

## 参考文献

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- "Rotary Position Embedding" (Su et al., 2021)
- "AdamW: Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)

