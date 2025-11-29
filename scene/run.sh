#!/bin/bash

# 生成基础注意力
manim -ql scene.py AttentionMechanism

# 生成架构流
manim -ql scene_struct.py EncoderFlow
manim -ql scene_struct.py DecoderMasking

# 生成协作机制
manim -ql scene_cross_attn.py CrossAttentionFlow

# 生成对比与 Mamba
manim -ql scene_compare.py TransformerVsMamba
manim -ql scene_mamba_core.py MambaMechanism