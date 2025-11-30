"""
多头注意力详细机制可视化场景
=============================

对应笔记: 2.multi-headed.md
生成命令: manim scene_multi_head_detailed.py MultiHeadDetailed -qh
输出视频: assets/MultiHeadDetailed.mp4

内容要点:
- 输入序列拆分成多个头
- 每个头独立计算Q,K,V
- 注意力计算过程详解
- 多头输出拼接与融合
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class MultiHeadDetailed(Scene):
    def construct(self):
        # 标题
        title = Text("多头注意力详细机制", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # 1. 输入展示
        input_text = Text('输入序列: "I love AI"', font_size=28, color=BLUE).move_to(
            UP * 1.5
        )
        self.play(Write(input_text))

        # 输入矩阵表示
        input_matrix = self.create_matrix(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]], [3, 3], "输入 X"
        )
        input_matrix.move_to(UP * 0.5)
        self.play(FadeIn(input_matrix))

        # 2. 拆分成多个头
        self.play(FadeOut(input_text))

        split_title = Text(
            "步骤1: 输入拆分成多个头", font_size=28, color=YELLOW
        ).move_to(UP * 2)
        self.play(Write(split_title))

        # 展示多头拆分
        heads = VGroup()
        head_labels = VGroup()
        num_heads = 4
        head_dim = 2

        # 原始矩阵分裂动画
        original_rect = SurroundingRectangle(input_matrix, color=RED, buff=0.1)
        self.play(Create(original_rect))

        # 创建多个头
        for i in range(num_heads):
            # 每个头的矩阵
            head_data = [[i, i + 1], [i + 2, i + 3]]  # 简化的数据
            head_matrix = self.create_matrix(head_data, [2, 2], f"Head {i+1}")
            head_matrix.scale(0.7)

            # 位置布局
            if i < 2:
                head_matrix.move_to(LEFT * 2.5 + RIGHT * i * 2.5 + DOWN * 1)
            else:
                head_matrix.move_to(LEFT * 2.5 + RIGHT * (i - 2) * 2.5 + DOWN * 2.5)

            heads.add(head_matrix)

        # 分裂动画
        self.play(
            FadeOut(input_matrix),
            FadeOut(original_rect),
            *[FadeIn(head) for head in heads],
            run_time=1.5,
        )

        self.wait(1)

        # 3. 每个头独立计算Q,K,V
        self.play(FadeOut(split_title))

        # 先清理拆分的头，避免遮挡后续内容
        self.play(FadeOut(heads))

        qkv_title = Text(
            "步骤2: 每个头独立计算Q,K,V", font_size=28, color=YELLOW
        ).move_to(UP * 2)
        self.play(Write(qkv_title))

        # 为第一个头展示Q,K,V计算（重新创建一个头用于展示）
        first_head = Rectangle(
            height=0.8, width=0.8, color=GREEN, fill_opacity=0.3
        ).move_to(LEFT * 2 + DOWN * 0.5)
        first_rect = SurroundingRectangle(first_head, color=GREEN, buff=0.1)
        self.play(Create(first_head), Create(first_rect))

        # Q,K,V矩阵
        q_matrix = self.create_matrix([[1, 0], [0, 1]], [2, 2], "Q", RED)
        k_matrix = self.create_matrix([[1, 1], [0, 0]], [2, 2], "K", GREEN)
        v_matrix = self.create_matrix([[0, 1], [1, 0]], [2, 2], "V", BLUE)

        q_matrix.scale(0.6).move_to(first_head.get_center() + LEFT * 2 + UP * 0.5)
        k_matrix.scale(0.6).move_to(first_head.get_center() + UP * 0.5)
        v_matrix.scale(0.6).move_to(first_head.get_center() + RIGHT * 2 + UP * 0.5)

        self.play(FadeIn(q_matrix), FadeIn(k_matrix), FadeIn(v_matrix))

        # 添加投影权重矩阵说明
        weight_text = Text("W_Q, W_K, W_V", font_size=20, color=GRAY).move_to(DOWN * 3)
        self.play(Write(weight_text))

        self.wait(1.5)

        # 4. 注意力计算展示
        self.play(FadeOut(qkv_title), FadeOut(weight_text), FadeOut(first_rect))

        attn_title = Text(
            "步骤3: 注意力计算 QK^T/√d_k", font_size=28, color=YELLOW
        ).move_to(UP * 2)
        self.play(Write(attn_title))

        # 计算注意力分数
        scores = np.array([[1, 0.5], [0.5, 1]])  # 简化的分数
        scores_matrix = self.create_matrix(scores, [2, 2], "Scores")
        scores_matrix.scale(0.6).move_to(first_head.get_center() + DOWN * 1.5)

        # 添加计算箭头
        arrow_qk = CurvedArrow(
            q_matrix.get_right(), scores_matrix.get_left(), color=YELLOW
        )
        self.play(Create(arrow_qk), FadeIn(scores_matrix))

        # Softmax
        softmax_text = Text("Softmax", font_size=20, color=ORANGE).next_to(
            scores_matrix, RIGHT
        )
        self.play(Write(softmax_text))

        # Softmax结果
        attn_weights = np.array([[0.73, 0.27], [0.27, 0.73]])
        attn_matrix = self.create_matrix(attn_weights, [2, 2], "Attention")
        attn_matrix.scale(0.6).move_to(scores_matrix.get_center() + RIGHT * 2.5)

        arrow_softmax = Arrow(
            softmax_text.get_right(), attn_matrix.get_left(), color=ORANGE
        )
        self.play(Create(arrow_softmax), FadeIn(attn_matrix))

        self.wait(1)

        # 5. 输出计算
        self.play(FadeOut(attn_title))

        output_title = Text(
            "步骤4: 输出计算 Attention × V", font_size=28, color=YELLOW
        ).move_to(UP * 2)
        self.play(Write(output_title))

        # 输出矩阵
        output_data = np.array([[0.5, 0.5], [0.5, 0.5]])  # 简化的输出
        output_matrix = self.create_matrix(output_data, [2, 2], "Output", PURPLE)
        output_matrix.scale(0.6).move_to(attn_matrix.get_center() + DOWN * 1.5)

        # 计算箭头
        arrow_output = CurvedArrow(
            attn_matrix.get_bottom(), output_matrix.get_top(), color=PURPLE
        )
        arrow_v = CurvedArrow(
            v_matrix.get_left(), output_matrix.get_right(), color=BLUE
        )

        self.play(Create(arrow_output), Create(arrow_v), FadeIn(output_matrix))
        self.wait(1.5)

        # 6. 拼接所有头
        self.play(FadeOut(output_title))

        concat_title = Text(
            "步骤5: 拼接所有头的输出", font_size=28, color=YELLOW
        ).move_to(UP * 2)
        self.play(Write(concat_title))

        # 清理当前展示的第一个头的细节
        self.play(
            FadeOut(
                VGroup(
                    first_head,
                    first_rect,
                    q_matrix,
                    k_matrix,
                    v_matrix,
                    scores_matrix,
                    softmax_text,
                    attn_matrix,
                    output_matrix,
                    arrow_qk,
                    arrow_softmax,
                    arrow_output,
                    arrow_v,
                )
            )
        )

        # 重新创建所有头用于展示输出
        head_positions = [LEFT * 2.5 + RIGHT * i * 2.5 + DOWN * 0.5 for i in range(4)]

        # 展示所有头的输出
        head_outputs = VGroup()
        for i, pos in enumerate(head_positions):
            output = self.create_matrix(
                [[i, i + 1], [i + 2, i + 3]], [2, 2], f"Out{i+1}", PURPLE
            )
            output.scale(0.5)
            output.move_to(pos)
            head_outputs.add(output)

        self.play(FadeIn(head_outputs))

        # 拼接箭头
        concat_arrows = VGroup()
        for i in range(len(head_outputs) - 1):
            arrow = Arrow(
                head_outputs[i].get_right(),
                head_outputs[i + 1].get_left(),
                color=YELLOW,
            )
            concat_arrows.add(arrow)

        self.play(Create(concat_arrows))

        # 最终拼接结果
        final_output = self.create_matrix(
            [[1, 2, 3, 4], [5, 6, 7, 8]], [2, 4], "Concatenated", GREEN
        )
        final_output.move_to(DOWN * 3)

        concat_arrow = Arrow(
            head_outputs[1].get_bottom(), final_output.get_top(), color=GREEN
        )
        self.play(Create(concat_arrow), FadeIn(final_output))

        self.wait(1)

        # 7. 最终线性变换
        self.play(FadeOut(concat_title))

        linear_title = Text(
            "步骤6: 最终线性变换 W_O", font_size=28, color=YELLOW
        ).move_to(UP * 2)
        self.play(Write(linear_title))

        # W_O矩阵
        wo_matrix = self.create_matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [4, 4],
            "W_O",
            ORANGE,
        )
        wo_matrix.scale(0.6).move_to(final_output.get_left() + LEFT * 2)

        self.play(FadeIn(wo_matrix))

        # 最终结果
        final_result = self.create_matrix([[1, 2], [3, 4]], [2, 2], "Final Output", RED)
        final_result.move_to(final_output.get_right() + RIGHT * 2)

        # 计算箭头
        final_arrow = Arrow(
            final_output.get_right(), final_result.get_left(), color=RED
        )

        self.play(Create(final_arrow), FadeIn(final_result))

        # 总结
        self.play(FadeOut(linear_title))

        summary = Text(
            "多头注意力：从不同子空间视角理解序列，最后融合信息",
            font_size=24,
            color=YELLOW,
        ).move_to(DOWN * 4.5)

        self.play(Write(summary))
        self.wait(3)

    def create_matrix(self, data, shape, label="", color=WHITE):
        """创建矩阵可视化"""
        rows, cols = shape
        group = VGroup()

        # 创建矩阵格子
        for i in range(rows):
            for j in range(cols):
                sq = (
                    Square(side_length=0.8)
                    .set_stroke(color, 1)
                    .set_fill(color, opacity=0.1)
                )
                sq.move_to(
                    RIGHT * (j - cols / 2 + 0.5) * 0.8
                    + DOWN * (i - rows / 2 + 0.5) * 0.8
                )

                if i < len(data) and j < len(data[0]):
                    val = Text(str(data[i][j]), font_size=20, color=color).move_to(sq)
                    group.add(VGroup(sq, val))
                else:
                    group.add(sq)

        # 添加标签
        if label:
            label_text = Text(label, font_size=24, color=color).next_to(
                group, UP, buff=0.2
            )
            group.add(label_text)

        return group


if __name__ == "__main__":
    scene = MultiHeadDetailed()
    scene.render()
