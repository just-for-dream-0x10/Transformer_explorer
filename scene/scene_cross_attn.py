from manim import *
import numpy as np

config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class CrossAttentionFlow(Scene):
    def construct(self):
        # 1. 标题
        title = Text(
            "Encoder-Decoder Collaboration (Cross-Attention)", font_size=36
        ).to_edge(UP)
        self.play(Write(title))

        # ==========================================
        # 2. Encoder 侧 (Source Memory)
        # ==========================================
        enc_box = Rectangle(height=4, width=2, color=BLUE, fill_opacity=0.2).to_edge(
            LEFT, buff=1.5
        )
        enc_label = Text("Encoder\nOutput\n(Source)", font_size=24, color=BLUE).next_to(
            enc_box, UP
        )

        # Encoder 里的 Key 和 Value
        k_box = Rectangle(height=3.5, width=0.6, color=GREEN, fill_opacity=0.5).move_to(
            enc_box.get_center() + LEFT * 0.4
        )
        v_box = Rectangle(
            height=3.5, width=0.6, color=YELLOW, fill_opacity=0.5
        ).move_to(enc_box.get_center() + RIGHT * 0.4)

        k_text = Text("K", font_size=24).move_to(k_box.get_top() + DOWN * 0.3)
        v_text = Text("V", font_size=24).move_to(v_box.get_top() + DOWN * 0.3)

        enc_group = VGroup(enc_box, enc_label, k_box, v_box, k_text, v_text)

        self.play(FadeIn(enc_group))

        # ==========================================
        # 3. Decoder 侧 (Target Query)
        # ==========================================
        dec_box = (
            Rectangle(height=1, width=3, color=PURPLE, fill_opacity=0.2)
            .to_edge(RIGHT, buff=1.5)
            .shift(DOWN)
        )
        dec_label = Text("Decoder\n(Current Step)", font_size=24, color=PURPLE).next_to(
            dec_box, DOWN
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
