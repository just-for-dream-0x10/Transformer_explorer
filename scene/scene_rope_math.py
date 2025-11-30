"""
RoPE旋转位置编码数学原理可视化场景
=====================================

对应笔记: 7.Advanced.md (RoPE部分)
生成命令: manim scene_rope_math.py RoPEMath -qh
输出视频: assets/RoPEMath.mp4

内容要点:
- 复数旋转的几何直观
- RoPE的数学推导过程
- 相对位置信息如何自然包含在点积中
- 高维空间的旋转机制
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class RoPEMath(Scene):
    def construct(self):
        # 标题
        title = Text("RoPE: 旋转位置编码数学原理", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # 1. 核心思想：复数旋转
        subtitle = Text(
            "核心思想：通过复数旋转注入位置信息", font_size=28, color=YELLOW
        ).move_to(UP * 1.5)
        self.play(Write(subtitle))
        self.wait(1)

        # 2. 2D旋转矩阵展示
        self.play(FadeOut(subtitle))

        # 创建坐标系
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": WHITE},
        ).move_to(LEFT * 3)

        self.play(Create(axes))

        # 原始向量
        original_vector = Arrow(ORIGIN, RIGHT + UP, color=BLUE, stroke_width=4)
        vector_label = Text("v", font_size=24, color=BLUE).next_to(
            original_vector.get_end(), RIGHT
        )

        self.play(Create(original_vector), Write(vector_label))

        # 旋转矩阵公式
        rotation_formula = MathTex(
            r"R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}",
            font_size=32,
            color=YELLOW,
        ).move_to(RIGHT * 2.5 + UP * 2)

        self.play(Write(rotation_formula))

        # 旋转动画
        rotated_vectors = VGroup()
        angles = [PI / 6, PI / 3, PI / 2, 2 * PI / 3]

        for i, angle in enumerate(angles):
            rotated = Arrow(
                ORIGIN, rotate_vector(RIGHT + UP, angle), color=GREEN, stroke_width=3
            )
            angle_label = Text(f"θ={angle*180/PI:.0f}°", font_size=16, color=GREEN)
            angle_label.next_to(rotated.get_end(), RIGHT)

            # 创建副本进行动画
            arrow_copy = original_vector.copy()
            self.play(
                Transform(arrow_copy, rotated, path_arc=angle),
                Write(angle_label),
                run_time=1,
            )
            # 添加实际显示的物体到组中以便清理
            rotated_vectors.add(VGroup(arrow_copy, angle_label))

        self.wait(1.5)

        # 3. RoPE的数学推导
        self.play(
            FadeOut(axes),
            FadeOut(original_vector),
            FadeOut(vector_label),
            FadeOut(rotated_vectors),
            FadeOut(rotation_formula),
        )

        deriv_title = Text("RoPE相对位置推导", font_size=32).move_to(UP * 2.5)
        self.play(Write(deriv_title))

        # Query和Key的位置
        q_pos = Text("Query位置: m", font_size=24, color=BLUE).move_to(
            UP * 0.5 + LEFT * 3
        )
        k_pos = Text("Key位置: n", font_size=24, color=GREEN).move_to(
            UP * 0.5 + RIGHT * 3
        )

        self.play(Write(q_pos), Write(k_pos))

        # 旋转公式
        q_rotate = MathTex(
            r"q_m = R(m\theta) \cdot q", font_size=28, color=BLUE
        ).move_to(LEFT * 3 + DOWN * 0.5)
        k_rotate = MathTex(
            r"k_n = R(n\theta) \cdot k", font_size=28, color=GREEN
        ).move_to(RIGHT * 3 + DOWN * 0.5)

        self.play(Write(q_rotate), Write(k_rotate))

        # 点积计算
        dot_product = MathTex(
            r"q_m^T k_n = q^T R(m\theta)^T R(n\theta) k", font_size=28, color=YELLOW
        ).move_to(DOWN * 1.5)

        self.play(Write(dot_product))
        self.wait(0.5)

        # 关键性质
        key_property = MathTex(
            r"R(a)^T R(b) = R(b-a)", font_size=28, color=RED
        ).next_to(dot_product, DOWN, buff=0.5)

        self.play(Write(key_property))
        self.wait(0.5)

        # 最终结果
        final_result = MathTex(
            r"q_m^T k_n = q^T R((n-m)\theta) k", font_size=32, color=GREEN
        ).next_to(key_property, DOWN, buff=0.5)

        self.play(Write(final_result))

        # 突出相对距离
        highlight = SurroundingRectangle(final_result[0][11:17], color=YELLOW, buff=0.1)
        relative_text = Text(
            "只依赖相对距离(n-m)！", font_size=24, color=YELLOW
        ).next_to(final_result, DOWN, buff=0.5)

        self.play(Create(highlight), Write(relative_text))
        self.wait(2)

        # 4. 高维空间的旋转
        self.play(
            FadeOut(
                VGroup(
                    deriv_title,
                    q_pos,
                    k_pos,
                    q_rotate,
                    k_rotate,
                    dot_product,
                    key_property,
                    final_result,
                    highlight,
                    relative_text,
                )
            )
        )

        high_dim_title = Text("高维空间：每2维一组独立旋转", font_size=32).move_to(
            UP * 2.5
        )
        self.play(Write(high_dim_title))

        # 展示多维旋转
        dim_groups = VGroup()
        for i in range(4):  # 展示4组2D旋转
            # 创建小坐标系
            mini_axes = Axes(
                x_range=[-1, 1, 0.5],
                y_range=[-1, 1, 0.5],
                x_length=1.5,
                y_length=1.5,
                axis_config={"color": GRAY, "stroke_width": 1},
            ).move_to(RIGHT * (i - 1.5) * 2 + DOWN * 0.5)

            # 原始向量
            vector = Arrow(ORIGIN, RIGHT * 0.7 + UP * 0.3, color=BLUE, stroke_width=2)

            # 维度标签
            dim_label = Text(f"dim {2*i},{2*i+1}", font_size=16, color=GRAY).next_to(
                mini_axes, UP
            )

            group = VGroup(mini_axes, vector, dim_label)
            dim_groups.add(group)

        self.play(FadeIn(dim_groups))

        # 同时旋转所有维度
        rotated_groups = VGroup()
        for i, group in enumerate(dim_groups):
            mini_axes, vector, dim_label = group
            angle = (i + 1) * PI / 6  # 不同维度不同角度

            rotated_vector = Arrow(
                ORIGIN,
                rotate_vector(RIGHT * 0.7 + UP * 0.3, angle),
                color=GREEN,
                stroke_width=2,
            )

            self.play(
                Transform(vector, rotated_vector, path_arc=angle),
                run_time=1,
                lag_ratio=0.2,
            )

        # 说明文字
        explanation = Text(
            "每个维度对独立旋转，频率不同，捕获不同尺度的位置信息",
            font_size=20,
            color=GRAY,
        ).move_to(DOWN * 2.5)

        self.play(Write(explanation))
        self.wait(2)

        # 5. RoPE vs 传统位置编码对比
        self.play(FadeOut(VGroup(dim_groups, explanation, high_dim_title)))

        compare_title = Text("RoPE vs 传统位置编码", font_size=32).move_to(UP * 2.5)
        self.play(Write(compare_title))

        # 创建对比表格
        traditional = (
            VGroup(
                Text("传统位置编码", font_size=24, color=BLUE),
                Text("• 绝对位置", font_size=20),
                Text("• 直接相加", font_size=20),
                Text("• 外推性差", font_size=20, color=RED),
            )
            .arrange(DOWN, buff=0.3)
            .move_to(LEFT * 3 + DOWN * 0.5)
        )

        rope = (
            VGroup(
                Text("RoPE", font_size=24, color=GREEN),
                Text("• 相对位置", font_size=20),
                Text("• 旋转变换", font_size=20),
                Text("• 外推性好", font_size=20, color=GREEN),
            )
            .arrange(DOWN, buff=0.3)
            .move_to(RIGHT * 3 + DOWN * 0.5)
        )

        self.play(FadeIn(traditional), FadeIn(rope))

        vs_text = Text("VS", font_size=32, color=YELLOW).move_to(ORIGIN)
        self.play(Write(vs_text))

        # 优势总结
        advantages = Text(
            "RoPE优势：通过旋转矩阵的数学性质，自然地让点积包含相对位置信息",
            font_size=22,
            color=YELLOW,
        ).move_to(DOWN * 2.5)

        self.play(Write(advantages))
        self.wait(3)


if __name__ == "__main__":
    scene = RoPEMath()
    scene.render()
