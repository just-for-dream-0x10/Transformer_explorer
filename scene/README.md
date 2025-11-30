# Scene 文件说明

本目录包含Transformer & Mamba各模块的底层数学逻辑可视化脚本。

## 📁 文件结构

### 🤖 Transformer 核心模块

#### 1. 位置编码
- **scene_positional_encoding.py** - 传统位置编码
  - 对应笔记: `1.math.md`
  - 生成: `manim scene_positional_encoding.py PositionalEncoding`
  - 内容: 正弦位置编码公式、矩阵可视化、与词嵌入相加

#### 2. RoPE旋转位置编码  
- **scene_rope_math.py** - 现代位置编码
  - 对应笔记: `7.Advanced.md`
  - 生成: `manim scene_rope_math.py RoPEMath`
  - 内容: 复数旋转、相对位置推导、高维旋转机制

#### 3. 多头注意力
- **scene_multi_head_detailed.py** - 多头注意力机制
  - 对应笔记: `2.multi-headed.md`
  - 生成: `manim scene_multi_head_detailed.py MultiHeadDetailed`
  - 内容: 输入拆分、QKV计算、注意力计算、拼接融合

#### 4. 残差连接与层归一化
- **scene_residual_norm.py** - 深层网络稳定性
  - 对应笔记: `3.ResidualConnection.md`
  - 生成: `manim scene_residual_norm.py ResidualNorm`
  - 内容: 残差连接数学、梯度流、LayerNorm、Pre-LN vs Post-LN

#### 5. FFN/SwiGLU网络
- **scene_ffn_swiglu.py** - 前馈网络进化
  - 对应笔记: `4.encoder.md`
  - 生成: `manim scene_ffn_swiglu.py FFNSwiGLU`
  - 内容: 传统FFN、SwiGLU机制、激活函数对比

#### 6. Cross Attention
- **scene_cross_attn.py** - 编解码协作
  - 对应笔记: `5.decoder.md`
  - 生成: `manim scene_cross_attn.py CrossAttentionFlow`
  - 内容: Q来自Decoder、K/V来自Encoder、协作机制

#### 7. Encoder/Decoder架构
- **scene_struct.py** - 整体架构流程
  - 对应笔记: `4.encoder.md`, `5.decoder.md`
  - 生成: `manim scene_struct.py EncoderFlow`, `manim scene_struct.py DecoderMasking`
  - 内容: Encoder流程、Decoder因果掩码

### 🐍 Mamba 核心模块

#### 8. Mamba核心机制
- **scene_mamba_core.py** - 选择性状态空间
  - 对应笔记: `Appendix_E_Mamba_vs_Transformer.md`
  - 生成: `manim scene_mamba_core.py MambaMechanism`
  - 内容: 选择性扫描、动态阀门、状态压缩

#### 9. Mamba数学原理
- **scene_mamba_math.py** - 数学推导
  - 对应笔记: `Appendix_E_Mamba_vs_Transformer.md`
  - 生成: `manim scene_mamba_math.py MambaMath`
  - 内容: 连续系统、离散化、递归扫描

### 🎯 训练与推理

#### 10. 训练过程
- **scene_training_loss.py** - Next Token Prediction
  - 对应笔记: `10.Training_Essentials.md`
  - 生成: `manim scene_training_loss.py TrainingLoss`
  - 内容: 交叉熵损失、Teacher Forcing、并行训练

#### 11. 解码策略
- **scene_sampling_temperature.py** - 采样方法
  - 对应笔记: `9.Inference_Sampling.md`
  - 生成: `manim scene_sampling_temperature.py SamplingTemperature`
  - 内容: Temperature调节、Top-k、Top-p采样

#### 12. 优化器
- **scene_adamw_optimizer.py** - 权重衰减
  - 对应笔记: `10.Training_Essentials.md`
  - 生成: `manim scene_adamw_optimizer.py AdamWOptimizer`
  - 内容: Adam vs AdamW、解耦权重衰减、超参数

#### 13. 混合精度训练
- **scene_mixed_precision.py** - 数值精度
  - 对应笔记: `10.Training_Essentials.md`
  - 生成: `manim scene_mixed_precision.py MixedPrecision`
  - 内容: FP16 vs BF16、数值范围、硬件支持

### 📊 基础与对比

#### 14. 分词算法
- **scene_bpe_detailed.py** - BPE分词
  - 对应笔记: `8.Tokinization.md`
  - 生成: `manim scene_bpe_detailed.py BPEDetailed`
  - 内容: BPE迭代、词表构建、算法对比

#### 15. 架构对比
- **scene_compare.py** - Transformer vs Mamba
  - 对应笔记: `Appendix_E_Mamba_vs_Transformer.md`
  - 生成: `manim scene_compare.py scene_compare`
  - 内容: 复杂度对比、内存占用、推理速度

## 🚀 批量生成

使用 `generate_all_videos.sh` 脚本批量生成所有视频：

```bash
./generate_all_videos.sh
```

## 📱 Streamlit展示

所有生成的视频会在 `app.py` 中展示，按模块分类：
- Transformer家族
- Mamba家族  
- 架构对比
- 训练推理

## 🎯 设计原则

1. **数学严谨性** - 每个视频都基于笔记中的数学公式
2. **视觉直观** - 通过动画展示抽象概念
3. **对比清晰** - 突出Transformer与Mamba的差异
4. **模块独立** - 每个文件专注一个核心概念
5. **渐进深入** - 从基础到高级的逻辑顺序