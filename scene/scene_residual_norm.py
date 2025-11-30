"""
残差连接与层归一化可视化场景
============================

对应笔记: 3.ResidualConnection.md
生成命令: manim scene_residual_norm.py ResidualNorm -qh
输出视频: assets/ResidualNorm.mp4

内容要点:
- 残差连接数学原理
- 梯度流对比（传统vs残差）
- 层归一化计算过程
- Post-LN vs Pre-LN架构对比
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class ResidualNorm(Scene):
    def construct(self):
        # 标题
        title = Text("残差连接与层归一化", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # 1. 核心概念介绍
        concept_text = Text(
            "Add & Norm: 解决深层网络的梯度消失问题", font_size=28, color=YELLOW
        ).move_to(UP * 1.5)
        self.play(Write(concept_text))
        self.wait(1)

        # 2. 残差连接的数学原理
        self.play(FadeOut(concept_text))

        math_title = Text("残差连接数学原理", font_size=32).move_to(UP * 2)
        self.play(Write(math_title))

        # 公式
        formula = MathTex(r"H(x) = x + F(x)", font_size=36, color=GREEN).move_to(
            UP * 0.5
        )
        self.play(Write(formula))

        # 公式解释
        x_text = Text(
            "x: Identity Mapping (恒等映射)", font_size=20, color=BLUE
        ).move_to(LEFT * 3 + DOWN * 0.5)
        fx_text = Text(
            "F(x): Residual Mapping (残差映射)", font_size=20, color=RED
        ).move_to(RIGHT * 3 + DOWN * 0.5)

        self.play(Write(x_text), Write(fx_text))
        self.wait(1.5)

        # 3. 梯度流可视化
        self.play(FadeOut(VGroup(math_title, formula, x_text, fx_text)))

        gradient_title = Text("梯度流对比", font_size=32).move_to(UP * 2)
        self.play(Write(gradient_title))

        # 传统网络梯度流
        traditional_title = Text("传统网络", font_size=24, color=RED).move_to(
            LEFT * 4 + UP * 1
        )
        self.play(Write(traditional_title))

        # 传统网络层
        traditional_layers = VGroup()
        for i in range(5):
            layer = Rectangle(height=0.8, width=2, color=RED, fill_opacity=0.3).move_to(
                LEFT * 4 + DOWN * i * 0.9
            )
            layer_text = Text(f"Layer {i+1}", font_size=16).move_to(layer)
            traditional_layers.add(VGroup(layer, layer_text))

        self.play(FadeIn(traditional_layers))

        # 传统梯度流（越来越小）
        traditional_gradients = VGroup()
        gradient_values = [1.0, 0.5, 0.25, 0.125, 0.0625]
        for i, val in enumerate(gradient_values):
            arrow = Arrow(
                traditional_layers[i].get_bottom(),
                (
                    traditional_layers[i + 1].get_top()
                    if i < 4
                    else traditional_layers[i].get_bottom() + DOWN * 0.5
                ),
                color=RED,
                stroke_width=val * 5,
                max_stroke_width_to_length_ratio=10,
            )
            gradient_text = Text(f"∇={val:.3f}", font_size=12, color=RED).next_to(
                arrow, RIGHT
            )
            traditional_gradients.add(VGroup(arrow, gradient_text))

        self.play(*[Create(grad) for grad in traditional_gradients[:-1]])

        # 残差网络梯度流
        residual_title = Text("残差网络", font_size=24, color=GREEN).move_to(
            RIGHT * 4 + UP * 1
        )
        self.play(Write(residual_title))

        # 残差网络层
        residual_layers = VGroup()
        for i in range(5):
            layer = Rectangle(
                height=0.8, width=2, color=GREEN, fill_opacity=0.3
            ).move_to(RIGHT * 4 + DOWN * i * 0.9)
            layer_text = Text(f"Layer {i+1}", font_size=16).move_to(layer)
            residual_layers.add(VGroup(layer, layer_text))

        self.play(FadeIn(residual_layers))

        # 残差梯度流（保持恒定）
        residual_gradients = VGroup()
        for i in range(4):
            arrow = Arrow(
                residual_layers[i].get_bottom(),
                residual_layers[i + 1].get_top(),
                color=GREEN,
                stroke_width=5,
            )
            gradient_text = Text("∇=1.0", font_size=12, color=GREEN).next_to(
                arrow, RIGHT
            )
            residual_gradients.add(VGroup(arrow, gradient_text))

        # 添加残差连接的"高速公路"
        highway = Line(
            residual_layers[0].get_right() + RIGHT * 0.5,
            residual_layers[4].get_right() + RIGHT * 0.5,
            color=YELLOW,
            stroke_width=3,
        )
        highway_label = Text("梯度高速公路", font_size=14, color=YELLOW).next_to(
            highway, RIGHT
        )

        self.play(
            *[Create(grad) for grad in residual_gradients],
            Create(highway),
            Write(highway_label),
        )

        self.wait(2)

        # 4. 层归一化详解
        self.play(
            FadeOut(
                VGroup(
                    gradient_title,
                    traditional_title,
                    traditional_layers,
                    traditional_gradients,
                    residual_title,
                    residual_layers,
                    residual_gradients,
                    highway,
                    highway_label,
                )
            )
        )

        norm_title = Text("层归一化 (Layer Normalization)", font_size=32).move_to(
            UP * 2
        )
        self.play(Write(norm_title))

        # 层归一化公式
        norm_formula = MathTex(
            r"\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta",
            font_size=28,
            color=YELLOW,
        ).move_to(UP * 0.5)
        self.play(Write(norm_formula))

        # 公式解释
        mu_text = Text("μ: 均值", font_size=18, color=BLUE).move_to(LEFT * 4 + DOWN * 1)
        sigma_text = Text("σ²: 方差", font_size=18, color=GREEN).move_to(
            LEFT * 4 + DOWN * 1.5
        )
        gamma_text = Text("γ: 缩放参数", font_size=18, color=RED).move_to(
            RIGHT * 4 + DOWN * 1
        )
        beta_text = Text("β: 偏移参数", font_size=18, color=ORANGE).move_to(
            RIGHT * 4 + DOWN * 1.5
        )

        self.play(
            Write(mu_text), Write(sigma_text), Write(gamma_text), Write(beta_text)
        )
        self.wait(1.5)

        # 5. 层归一化过程演示
        self.play(
            FadeOut(
                VGroup(
                    norm_title, norm_formula, mu_text, sigma_text, gamma_text, beta_text
                )
            )
        )

        process_title = Text("层归一化过程演示", font_size=32).move_to(UP * 2)
        self.play(Write(process_title))

        # 原始向量
        original_vector = np.array([2.0, 4.0, 6.0, 8.0])
        vector_group = self.create_vector_visualization(original_vector, "原始向量")
        vector_group.move_to(LEFT * 3)
        self.play(FadeIn(vector_group))

        # 计算均值
        mean_val = np.mean(original_vector)
        mean_text = Text(f"μ = {mean_val:.1f}", font_size=24, color=BLUE).move_to(
            LEFT * 3 + DOWN * 2
        )
        self.play(Write(mean_text))

        # 减去均值
        centered_vector = original_vector - mean_val
        centered_group = self.create_vector_visualization(centered_vector, "减去均值")
        centered_group.move_to(ORIGIN)

        arrow1 = Arrow(
            vector_group.get_right(), centered_group.get_left(), color=YELLOW
        )
        self.play(Create(arrow1), FadeIn(centered_group))

        # 计算方差并归一化
        variance = np.var(original_vector)
        std = np.sqrt(variance)
        normalized_vector = centered_vector / std
        normalized_group = self.create_vector_visualization(normalized_vector, "归一化")
        normalized_group.move_to(RIGHT * 3)

        norm_text = Text(f"σ = {std:.2f}", font_size=24, color=GREEN).move_to(DOWN * 2)
        arrow2 = Arrow(
            centered_group.get_right(), normalized_group.get_left(), color=YELLOW
        )
        self.play(Write(norm_text), Create(arrow2), FadeIn(normalized_group))

        self.wait(2)

        # 6. Post-LN vs Pre-LN对比
        self.play(
            FadeOut(
                VGroup(
                    process_title,
                    vector_group,
                    mean_text,
                    centered_group,
                    arrow1,
                    norm_text,
                    arrow2,
                    normalized_group,
                )
            )
        )

        compare_title = Text("Post-LN vs Pre-LN", font_size=32).move_to(UP * 2)
        self.play(Write(compare_title))

        # Post-LN架构
        post_title = Text("Post-LN (原始Transformer)", font_size=24, color=RED).move_to(
            LEFT * 4 + UP * 1
        )
        self.play(Write(post_title))

        # Post-LN流程
        post_input = Rectangle(
            height=0.6, width=1.5, color=BLUE, fill_opacity=0.5
        ).move_to(LEFT * 4 + UP * 0)
        post_input_text = Text("x", font_size=20).move_to(post_input)

        post_sublayer = Rectangle(
            height=0.6, width=1.5, color=GREEN, fill_opacity=0.5
        ).move_to(LEFT * 4 + DOWN * 1)
        post_sublayer_text = Text("F(x)", font_size=20).move_to(post_sublayer)

        post_add = Circle(radius=0.3, color=YELLOW, fill_opacity=0.5).move_to(
            LEFT * 4 + DOWN * 2
        )
        post_add_text = Text("+", font_size=20, color=BLACK).move_to(post_add)

        post_norm = Rectangle(
            height=0.6, width=1.5, color=ORANGE, fill_opacity=0.5
        ).move_to(LEFT * 4 + DOWN * 3)
        post_norm_text = Text("Norm", font_size=20, color=BLACK).move_to(post_norm)

        # Post-LN连接
        post_arrows = VGroup(
            Arrow(post_input.get_bottom(), post_sublayer.get_top(), color=WHITE),
            Arrow(post_input.get_right(), post_add.get_right(), color=YELLOW),
            Arrow(post_sublayer.get_bottom(), post_add.get_left(), color=WHITE),
            Arrow(post_add.get_bottom(), post_norm.get_top(), color=WHITE),
        )

        post_formula = Text("y = Norm(x + F(x))", font_size=18, color=RED).move_to(
            LEFT * 4 + DOWN * 4
        )

        self.play(
            FadeIn(
                VGroup(
                    post_input,
                    post_input_text,
                    post_sublayer,
                    post_sublayer_text,
                    post_add,
                    post_add_text,
                    post_norm,
                    post_norm_text,
                    post_arrows,
                    post_formula,
                )
            )
        )

        # Pre-LN架构
        pre_title = Text("Pre-LN (现代LLM)", font_size=24, color=GREEN).move_to(
            RIGHT * 4 + UP * 1
        )
        self.play(Write(pre_title))

        # Pre-LN流程
        pre_input = Rectangle(
            height=0.6, width=1.5, color=BLUE, fill_opacity=0.5
        ).move_to(RIGHT * 4 + UP * 0)
        pre_input_text = Text("x", font_size=20).move_to(pre_input)

        pre_norm = Rectangle(
            height=0.6, width=1.5, color=ORANGE, fill_opacity=0.5
        ).move_to(RIGHT * 4 + DOWN * 1)
        pre_norm_text = Text("Norm", font_size=20, color=BLACK).move_to(pre_norm)

        pre_sublayer = Rectangle(
            height=0.6, width=1.5, color=GREEN, fill_opacity=0.5
        ).move_to(RIGHT * 4 + DOWN * 2)
        pre_sublayer_text = Text("F(Norm(x))", font_size=18).move_to(pre_sublayer)

        pre_add = Circle(radius=0.3, color=YELLOW, fill_opacity=0.5).move_to(
            RIGHT * 4 + DOWN * 3
        )
        pre_add_text = Text("+", font_size=20, color=BLACK).move_to(pre_add)

        # Pre-LN连接
        pre_arrows = VGroup(
            Arrow(pre_input.get_bottom(), pre_norm.get_top(), color=WHITE),
            Arrow(pre_norm.get_bottom(), pre_sublayer.get_top(), color=WHITE),
            Arrow(pre_input.get_right(), pre_add.get_right(), color=YELLOW),
            Arrow(pre_sublayer.get_bottom(), pre_add.get_left(), color=WHITE),
        )

        pre_formula = Text("y = x + F(Norm(x))", font_size=18, color=GREEN).move_to(
            RIGHT * 4 + DOWN * 4
        )

        self.play(
            FadeIn(
                VGroup(
                    pre_input,
                    pre_input_text,
                    pre_norm,
                    pre_norm_text,
                    pre_sublayer,
                    pre_sublayer_text,
                    pre_add,
                    pre_add_text,
                    pre_arrows,
                    pre_formula,
                )
            )
        )

        # 优势对比
        advantage_text = Text(
            "Pre-LN优势: 训练更稳定，无需warmup", font_size=22, color=YELLOW
        ).move_to(DOWN * 5)
        self.play(Write(advantage_text))

        self.wait(3)

    def create_vector_visualization(self, vector, title=""):
        """创建向量可视化"""
        group = VGroup()

        # 标题
        if title:
            title_text = Text(title, font_size=18, color=WHITE).move_to(UP * 1.5)
            group.add(title_text)

        # 向量条形图
        bars = VGroup()
        for i, val in enumerate(vector):
            height = abs(val) * 0.3
            bar = Rectangle(
                height=height,
                width=0.5,
                color=BLUE if val >= 0 else RED,
                fill_opacity=0.7,
            )
            bar.move_to(DOWN * i * 0.6 + DOWN * 0.5)
            if val < 0:
                bar.shift(DOWN * height)

            # 数值标签
            val_text = Text(f"{val:.1f}", font_size=14, color=WHITE).next_to(bar, RIGHT)
            bars.add(VGroup(bar, val_text))

        group.add(bars)
        return group


if __name__ == "__main__":
    scene = ResidualNorm()
    scene.render()
