"""
Cross Attention 机制详解
======================

对应笔记: 5.decoder.md (Cross Attention部分)
生成命令: manim scene_cross_attn.py CrossAttentionFlow -qh
输出视频: assets/CrossAttentionFlow.mp4

内容要点:
- Q来自Decoder，K,V来自Encoder
- 完整的Q×K^T → Softmax → ×V计算过程
- 编解码协作机制
- 数学公式展示
"""

from manim import *
import numpy as np

config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class CrossAttentionFlow(Scene):
    def construct(self):
        # 1. 标题和公式
        title = Text("Cross Attention: Encoder-Decoder协作", font_size=36).to_edge(UP)
        self.play(Write(title))

        # Cross Attention公式
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V",
            font_size=28,
            color=YELLOW,
        ).move_to(UP * 1)
        self.play(Write(formula))

        # 公式说明
        formula_explanation = VGroup(
            Text("Q: Query (来自Decoder)", font_size=16, color=PURPLE).next_to(
                formula, DOWN, buff=0.2
            ),
            Text("K, V: Key, Value (来自Encoder)", font_size=16, color=BLUE).next_to(
                formula, DOWN, buff=0.5
            ),
        ).move_to(UP * 0.3)
        self.play(Write(formula_explanation))

        self.wait(1)
        self.play(FadeOut(formula), FadeOut(formula_explanation))

        # ==========================================
        # 2. Encoder 侧 (Source Memory)
        # ==========================================
        enc_title = Text("Encoder侧: K, V存储", font_size=24, color=BLUE).move_to(
            LEFT * 4 + UP * 2
        )
        self.play(Write(enc_title))

        # Encoder输出矩阵
        enc_data = np.array([[0.8, 0.2, 0.6], [0.4, 0.9, 0.1], [0.3, 0.7, 0.5]])

        enc_matrix = self.create_matrix(enc_data, "Encoder Output")
        enc_matrix.move_to(LEFT * 4 + UP * 0.5)
        self.play(FadeIn(enc_matrix))

        # 分离K和V
        k_data = enc_data[:, :2]  # 前两列作为K
        v_data = enc_data[:, 2:]  # 最后一列作为V

        k_matrix = self.create_matrix(k_data, "K", GREEN)
        k_matrix.scale(0.8).move_to(LEFT * 5.5 + DOWN * 0.5)

        v_matrix = self.create_matrix(v_data, "V", YELLOW)
        v_matrix.scale(0.8).move_to(LEFT * 2.5 + DOWN * 0.5)

        # 分离箭头
        split_arrow = Arrow(enc_matrix.get_right(), k_matrix.get_left(), color=YELLOW)
        split_arrow2 = Arrow(enc_matrix.get_right(), v_matrix.get_left(), color=YELLOW)

        self.play(
            Create(split_arrow),
            Create(split_arrow2),
            FadeIn(k_matrix),
            FadeIn(v_matrix),
        )

        self.wait(1)
        self.play(FadeOut(enc_matrix))

        # ==========================================
        # 3. Decoder 侧 (Target Query)
        # ==========================================
        dec_title = Text("Decoder侧: Q查询", font_size=24, color=PURPLE).move_to(
            RIGHT * 3 + UP * 2
        )
        self.play(Write(dec_title))

        # Decoder Query矩阵
        q_data = np.array([[0.5, 0.8], [0.6, 0.3]])

        q_matrix = self.create_matrix(q_data, "Q", PURPLE)
        q_matrix.scale(0.8).move_to(RIGHT * 4 + DOWN * 0.5)
        self.play(FadeIn(q_matrix))

        self.wait(1)

        # ==========================================
        # 4. 注意力计算：Q × K^T
        # ==========================================
        attn_title = Text(
            "步骤1: Q × K^T (相似度计算)", font_size=22, color=YELLOW
        ).move_to(UP * 1)
        self.play(Write(attn_title))

        # 计算Q×K^T
        scores = q_data @ k_data.T
        scores_matrix = self.create_matrix(scores, "Scores")
        scores_matrix.scale(0.8).move_to(ORIGIN + DOWN * 1.5)

        # 计算箭头
        qk_arrow = CurvedArrow(
            q_matrix.get_right(), scores_matrix.get_left(), color=YELLOW
        )
        k_arrow = CurvedArrow(
            k_matrix.get_bottom(), scores_matrix.get_top(), color=YELLOW
        )

        self.play(Create(qk_arrow), Create(k_arrow), FadeIn(scores_matrix))

        # 展示数值
        scores_detail = VGroup(
            Text("Q(2,2) × K^T(2,2) = Scores(2,2)", font_size=16, color=GRAY).move_to(
                DOWN * 2.5
            )
        )
        self.play(Write(scores_detail))

        self.wait(1.5)
        self.play(FadeOut(attn_title), FadeOut(scores_detail))

        # ==========================================
        # 5. 缩放和Softmax
        # ==========================================
        scale_title = Text("步骤2: 缩放 & Softmax", font_size=22, color=YELLOW).move_to(
            UP * 1
        )
        self.play(Write(scale_title))

        # 缩放
        d_k = 2  # Key维度
        scaled_scores = scores / np.sqrt(d_k)
        scaled_matrix = self.create_matrix(scaled_scores, "Scaled")
        scaled_matrix.scale(0.8).move_to(ORIGIN + DOWN * 1.5)

        self.play(ReplacementTransform(scores_matrix, scaled_matrix))

        # 缩放公式
        scale_formula = MathTex(
            r"\frac{QK^T}{\sqrt{d_k}}", font_size=20, color=ORANGE
        ).next_to(scaled_matrix, RIGHT)
        self.play(Write(scale_formula))

        # Softmax
        attn_weights = np.exp(scaled_scores) / np.sum(
            np.exp(scaled_scores), axis=1, keepdims=True
        )
        attn_matrix = self.create_matrix(attn_weights, "Attention")
        attn_matrix.scale(0.8).move_to(ORIGIN + DOWN * 1.5)

        self.play(
            ReplacementTransform(scaled_matrix, attn_matrix), FadeOut(scale_formula)
        )

        self.wait(1.5)
        self.play(FadeOut(scale_title))

        # ==========================================
        # 6. 输出计算：Attention × V
        # ==========================================
        output_title = Text(
            "步骤3: Attention × V (加权求和)", font_size=22, color=YELLOW
        ).move_to(UP * 1)
        self.play(Write(output_title))

        # 计算输出
        output = attn_weights @ v_data
        output_matrix = self.create_matrix(output, "Output")
        output_matrix.scale(0.8).move_to(RIGHT * 2 + DOWN * 1.5)

        # 计算箭头
        attn_arrow = CurvedArrow(
            attn_matrix.get_right(), output_matrix.get_left(), color=YELLOW
        )
        v_arrow = CurvedArrow(
            v_matrix.get_bottom(), output_matrix.get_top(), color=YELLOW
        )

        self.play(Create(attn_arrow), Create(v_arrow), FadeIn(output_matrix))

        # 输出公式
        output_formula = MathTex(
            r"\text{Output} = \sum_i \text{Attention}_i \cdot V_i",
            font_size=16,
            color=ORANGE,
        ).next_to(output_matrix, RIGHT)
        self.play(Write(output_formula))

        self.wait(2)

        # ==========================================
        # 7. 总结
        # ==========================================
        summary_text = Text(
            "Cross Attention完成：Decoder查询Encoder记忆", font_size=20, color=GREEN
        ).move_to(DOWN * 3)
        self.play(Write(summary_text))

        self.wait(2)

    def create_matrix(self, data, label="", color=WHITE):
        """创建矩阵可视化"""
        rows, cols = data.shape
        group = VGroup()

        # 创建矩阵格子
        for i in range(rows):
            for j in range(cols):
                sq = (
                    Square(side_length=0.8)
                    .set_stroke(color, 1)
                    .set_fill(color, opacity=0.1)
                )
                sq.move_to(RIGHT * j * 0.8 + DOWN * i * 0.8)

                val = data[i][j]
                num = Text(f"{val:.1f}", font_size=18, color=color).move_to(sq)

                bar_group = VGroup(sq, num)
                group.add(bar_group)

        # 添加标签
        if label:
            label_text = Text(label, font_size=20, color=color).next_to(
                group, UP, buff=0.2
            )
            group.add(label_text)

        return group

        # Decoder 位置
        dec_box = Rectangle(height=2, width=1.5, color=BLUE, fill_opacity=0.3).move_to(
            LEFT * 3 + UP * 0.5
        )

        # Decoder 里的 Query
        q_box = Rectangle(height=0.6, width=0.8, color=RED, fill_opacity=0.5).move_to(
            dec_box.get_center()
        )
        q_text = Text("Q", font_size=24).move_to(q_box)

        dec_group = VGroup(dec_box, dec_label, q_box, q_text)

        self.play(FadeIn(dec_group))

        # ==========================================
        # 4. 协作过程动画
        # ==========================================

        # Step 1: Query 发出探测波 (Attention Score 计算)
        attn_arrow = Arrow(q_box.get_left(), k_box.get_right(), color=RED, buff=0.1)
        step1_text = Text("1. Query scans Keys", font_size=24, color=RED).next_to(
            attn_arrow, UP
        )

        self.play(GrowArrow(attn_arrow), Write(step1_text))

        # 模拟扫描效果 (高亮 K 的不同区域)
        scan_rect = SurroundingRectangle(k_box, color=WHITE, buff=0.1)
        self.play(Create(scan_rect), run_time=0.5)
        self.play(FadeOut(scan_rect), FadeOut(attn_arrow), FadeOut(step1_text))

        # Step 2: 生成 Attention Weights (热力图条)
        weights_bar = Rectangle(height=3.5, width=0.3, fill_opacity=1).next_to(
            k_box, LEFT
        )
        # 渐变色模拟权重
        weights_bar.set_fill(color=[BLACK, RED, BLACK], opacity=1)
        weights_text = Text("Weights", font_size=20).next_to(weights_bar, LEFT)

        self.play(FadeIn(weights_bar), Write(weights_text))

        # Step 3: 提取 Value (加权求和)
        extract_arrow = Arrow(
            v_box.get_right(), q_box.get_top() + UP * 1.5, color=YELLOW, path_arc=-1.5
        )
        step3_text = Text("2. Extract Context (V)", font_size=24, color=YELLOW).move_to(
            UP * 1
        )

        self.play(GrowArrow(extract_arrow), Write(step3_text))

        # 模拟信息流块
        context_chunk = Rectangle(height=0.6, width=0.8, color=YELLOW, fill_opacity=0.8)
        context_chunk.move_to(v_box.get_center())

        self.play(
            context_chunk.animate.move_to(q_box.get_top() + UP * 0.5), run_time=1.5
        )

        # Step 4: 融合 (Add)
        fusion_text = Text("3. Fusion", font_size=24, color=PURPLE).next_to(
            q_box, UP, buff=0.8
        )
        self.play(ReplacementTransform(step3_text, fusion_text), FadeOut(extract_arrow))

        self.play(
            context_chunk.animate.move_to(q_box.get_center()),
            q_box.animate.set_fill(ORANGE, opacity=0.8),  # 融合变色
            run_time=1,
        )

        self.play(FadeOut(context_chunk))  # 融进去了

        # 结束语
        final_text = Text(
            "Decoder now has Source Context!", font_size=28, color=GREEN
        ).to_edge(DOWN, buff=1)
        self.play(Write(final_text))

        self.wait(3)
