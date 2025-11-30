#!/bin/bash
# 统一生成所有Transformer & Mamba可视化视频脚本

echo "=========================================="
echo "Transformer & Mamba 可视化视频生成器"
echo "=========================================="

# 确保assets目录存在
mkdir -p ./assets

echo ""
echo "🎬 开始生成视频..."

# Transformer核心模块视频
echo "📝 生成Transformer核心模块视频..."
echo "   - 位置编码..."
manim scene/scene_positional_encoding.py PositionalEncoding -qh
echo "  - RoPE旋转位置编码..."
manim scene/scene_rope_math.py RoPEMath -qh
echo "  - 多头注意力详细机制..."
manim scene/scene_multi_head_detailed.py MultiHeadDetailed -qh
echo "  - 残差连接与层归一化..."
manim scene/scene_residual_norm.py ResidualNorm -qh
echo "  - FFN/SwiGLU网络..."
manim scene/scene_ffn_swiglu.py FFNSwiGLU -qh
echo "  - Cross Attention机制..."
manim scene/scene_cross_attn.py CrossAttentionFlow -qh
echo "  - Encoder/Decoder架构..."
manim scene/scene_struct.py EncoderFlow -qh
manim scene/scene_struct.py DecoderMasking -qh

# Mamba核心模块视频
echo "🐍 生成Mamba核心模块视频..."
echo "  - Mamba核心机制..."
manim scene/scene_mamba_core.py MambaMechanism -qh
echo "  - Mamba数学原理..."
manim scene/scene_mamba_math.py DiscretizationVisual -qh

# 训练与推理视频
echo "🎯 生成训练与推理视频..."
echo "  - 训练过程与损失..."
manim scene/scene_training_loss.py TrainingLoss -qh
echo "  - 解码策略与采样..."
manim/scene/scene_sampling_temperature.py SamplingTemperature -qh
echo "  - AdamW优化器..."
manim scene/scene_adamw_optimizer.py AdamWOptimizer -qh
echo "  - 混合精度训练..."
manim scene/scene_mixed_precision.py MixedPrecision -qh

# 基础与对比视频
echo "📊 生成基础与对比视频..."
echo "  - BPE分词算法..."
manim scene/scene_bpe_detailed.py BPEDetailed -qh
echo "  - Transformer vs Mamba对比..."
manim scene/scene_compare.py TransformerVsMamba -qh

echo ""
echo "🔄 移动视频文件到assets目录..."

# 移动视频文件
success_count=0

# 定义视频文件映射并移动
move_video() {
    local scene_name=$1
    local output_file=$2
    
    # 检查生成的视频文件
    if [ -f "media/videos/${scene_name}/${output_file}" ]; then
        echo "  ✅ ${output_file} - 移动中..."
        mv "media/videos/${scene_name}/${output_file}" "./assets/${output_file}"
        ((success_count++))
        return 0
    else
        echo "  ❌ ${output_file} - 生成失败"
        return 1
    fi
}

# 移动各个视频文件
move_video "PositionalEncoding" "PositionalEncoding.mp4"
move_video "RoPEMath" "RoPEMath.mp4" 
move_video "MultiHeadDetailed" "MultiHeadDetailed.mp4"
move_video "ResidualNorm" "ResidualNorm.mp4"
move_video "FFNSwiGLU" "FFNSwiGLU.mp4"
move_video "CrossAttentionFlow" "CrossAttentionFlow.mp4"
move_video "EncoderFlow" "EncoderFlow.mp4"
move_video "DecoderMasking" "DecoderMasking.mp4"
move_video "MambaMechanism" "MambaMechanism.mp4"
move_video "DiscretizationVisual" "DiscretizationVisual.mp4"
move_video "TrainingLoss" "TrainingLoss.mp4"
move_video "SamplingTemperature" "SamplingTemperature.mp4"
move_video "AdamWOptimizer" "AdamWOptimizer.mp4"
move_video "MixedPrecision" "MixedPrecision.mp4"
move_video "BPEDetailed" "BPEDetailed.mp4"
move_video "TransformerVsMamba" "TransformerVsMamba.mp4"

echo ""
echo "=========================================="
echo "✅ 视频生成完成！"
echo "=========================================="
echo ""
echo "📊 生成统计:"
echo "  - 总计: ${#video_map[@]} 个视频"
echo "  - 成功: ${success_count} 个"
echo "  - 失败: $((${#video_map[@]} - success_count)) 个"
echo ""
echo "📁 视频文件位置: ./assets/"
echo ""
echo "🎬 查看生成的视频:"
ls -la ./assets/*.mp4 2>/dev/null | while read line; do
    if [[ -n "$line" ]] && [[ "$line" =~ \.mp4$ ]]; then
        echo "  $line"
    fi
done

echo ""
echo "🚀 下一步:"
echo "   1. 运行 streamlit应用: streamlit run app.py"
echo "  2. 在浏览器中查看视频展示"
echo "  3. 享受Transformer & Mamba的可视化之旅！"