"""
FFN/SwiGLU网络可视化场景
=========================

对应笔记: 4.encoder.md (FFN部分)
生成命令: manim scene_ffn_swiglu.py FFNSwiGLU -qh
输出视频: assets/FFNSwiGLU.mp4

内容要点:
- 传统FFN架构与公式
- SwiGLU机制详解
- 激活函数对比（ReLU/GELU/Swish）
- 参数量对比分析
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class FFNSwiGLU(Scene):
    def construct(self):
        # 标题
        title = Text("前馈网络：FFN vs SwiGLU", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # 1. FFN的作用
        purpose_text = Text(
            "FFN: 注意力机制负责聚合信息，FFN负责深度思考", font_size=28, color=YELLOW
        ).move_to(UP * 1.5)
        self.play(Write(purpose_text))
        self.wait(1)

        # 2. 传统FFN (BERT)
        self.play(FadeOut(purpose_text))

        ffn_title = Text("传统FFN (BERT/GPT风格)", font_size=32).move_to(UP * 2)
        self.play(Write(ffn_title))

        # FFN公式
        ffn_formula = MathTex(
            r"\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2",
            font_size=28,
            color=GREEN,
        ).move_to(UP * 1)
        self.play(Write(ffn_formula))

        # 维度变化说明
        dim_text = Text("维度变换: d → 4d → d", font_size=24, color=BLUE).move_to(
            UP * 0.3
        )
        self.play(Write(dim_text))

        # 创建FFN架构图
        self.create_ffn_architecture()
        self.wait(1.5)

        # 3. SwiGLU介绍
        self.play(FadeOut(VGroup(ffn_title, ffn_formula, dim_text)))

        # 清理FFN架构图
        if hasattr(self, "ffn_arch"):
            self.play(FadeOut(self.ffn_arch))

        swiglu_title = Text("SwiGLU (现代LLM: LLaMA, PaLM)", font_size=32).move_to(
            UP * 2
        )
        self.play(Write(swiglu_title))

        # SwiGLU公式
        swiglu_formula = MathTex(
            r"\text{SwiGLU}(x) = (\text{Swish}(xW_g) \otimes (xW_{in}))W_{out}",
            font_size=26,
            color=GREEN,
        ).move_to(UP * 1)
        self.play(Write(swiglu_formula))

        # Swish函数解释
        swish_formula = MathTex(
            r"\text{Swish}(x) = x \cdot \sigma(\beta x)", font_size=24, color=YELLOW
        ).move_to(UP * 0.2)
        self.play(Write(swish_formula))

        self.wait(1)

        # 4. SwiGLU架构图
        self.create_swiglu_architecture()
        self.wait(1.5)

        # 5. Swish vs ReLU vs GELU对比
        self.play(FadeOut(VGroup(swiglu_title, swiglu_formula, swish_formula)))

        # 清理SwiGLU架构图
        if hasattr(self, "swiglu_arch"):
            self.play(FadeOut(self.swiglu_arch))

        activation_title = Text("激活函数对比", font_size=32).move_to(UP * 2)
        self.play(Write(activation_title))

        self.compare_activation_functions()

        # 6. 参数量对比
        self.play(FadeOut(activation_title))

        param_title = Text("参数量对比", font_size=32).move_to(UP * 2)
        self.play(Write(param_title))

        self.compare_parameters()

        self.wait(3)

    def create_ffn_architecture(self):
        """创建传统FFN架构图"""
        # 输入层
        input_layer = Rectangle(
            height=1.5, width=0.8, color=BLUE, fill_opacity=0.6
        ).move_to(LEFT * 5 + DOWN * 0.5)
        input_text = Text("Input\n[d]", font_size=16, color=WHITE).move_to(input_layer)

        # 第一个线性层 (扩展)
        w1_layer = Rectangle(
            height=3, width=0.8, color=GREEN, fill_opacity=0.6
        ).move_to(LEFT * 2.5 + DOWN * 0.5)
        w1_text = Text("W₁\n[d×4d]", font_size=16, color=WHITE).move_to(w1_layer)

        # 激活函数
        activation = Circle(radius=0.6, color=ORANGE, fill_opacity=0.6).move_to(
            ORIGIN + DOWN * 0.5
        )
        activation_text = Text("GELU", font_size=16, color=BLACK).move_to(activation)

        # 第二个线性层 (压缩)
        w2_layer = Rectangle(
            height=1.5, width=0.8, color=GREEN, fill_opacity=0.6
        ).move_to(RIGHT * 2.5 + DOWN * 0.5)
        w2_text = Text("W₂\n[4d×d]", font_size=16, color=WHITE).move_to(w2_layer)

        # 输出层
        output_layer = Rectangle(
            height=1.5, width=0.8, color=PURPLE, fill_opacity=0.6
        ).move_to(RIGHT * 5 + DOWN * 0.5)
        output_text = Text("Output\n[d]", font_size=16, color=WHITE).move_to(
            output_layer
        )

        # 连接箭头
        arrows = VGroup(
            Arrow(input_layer.get_right(), w1_layer.get_left(), color=YELLOW),
            Arrow(w1_layer.get_right(), activation.get_left(), color=YELLOW),
            Arrow(activation.get_right(), w2_layer.get_left(), color=YELLOW),
            Arrow(w2_layer.get_right(), output_layer.get_left(), color=YELLOW),
        )

        # 维度标注
        dim_labels = VGroup(
            Text("d", font_size=14, color=YELLOW).move_to(
                (input_layer.get_right() + w1_layer.get_left()) / 2 + UP * 0.3
            ),
            Text("4d", font_size=14, color=YELLOW).move_to(
                (w1_layer.get_right() + activation.get_left()) / 2 + UP * 0.3
            ),
            Text("4d", font_size=14, color=YELLOW).move_to(
                (activation.get_right() + w2_layer.get_left()) / 2 + UP * 0.3
            ),
            Text("d", font_size=14, color=YELLOW).move_to(
                (w2_layer.get_right() + output_layer.get_left()) / 2 + UP * 0.3
            ),
        )

        self.play(
            FadeIn(
                VGroup(
                    input_layer,
                    input_text,
                    w1_layer,
                    w1_text,
                    activation,
                    activation_text,
                    w2_layer,
                    w2_text,
                    output_layer,
                    output_text,
                    arrows,
                    dim_labels,
                )
            )
        )

        # 保存引用以便后续清理
        self.ffn_arch = VGroup(
            input_layer,
            input_text,
            w1_layer,
            w1_text,
            activation,
            activation_text,
            w2_layer,
            w2_text,
            output_layer,
            output_text,
            arrows,
            dim_labels,
        )

    def create_swiglu_architecture(self):
        """创建SwiGLU架构图"""
        # 清理之前的架构
        if hasattr(self, "ffn_arch"):
            self.play(FadeOut(self.ffn_arch))

        # 输入层
        input_layer = Rectangle(
            height=1.5, width=0.8, color=BLUE, fill_opacity=0.6
        ).move_to(LEFT * 5 + DOWN * 0.5)
        input_text = Text("Input\n[d]", font_size=16, color=WHITE).move_to(input_layer)

        # 分叉点
        fork_point = Dot(color=YELLOW).move_to(LEFT * 3.5 + DOWN * 0.5)

        # 上分支 (门控)
        w_g_layer = Rectangle(
            height=2, width=0.8, color=GREEN, fill_opacity=0.6
        ).move_to(LEFT * 2 + UP * 0.5)
        w_g_text = Text("W_g\n[d×2d]", font_size=16, color=WHITE).move_to(w_g_layer)

        swish_activation = Circle(radius=0.5, color=ORANGE, fill_opacity=0.6).move_to(
            LEFT * 0.5 + UP * 0.5
        )
        swish_text = Text("Swish", font_size=14, color=BLACK).move_to(swish_activation)

        # 下分支 (输入)
        w_in_layer = Rectangle(
            height=2, width=0.8, color=GREEN, fill_opacity=0.6
        ).move_to(LEFT * 2 + DOWN * 1.5)
        w_in_text = Text("W_in\n[d×2d]", font_size=16, color=WHITE).move_to(w_in_layer)

        # 逐元素乘法
        multiply = Circle(radius=0.5, color=RED, fill_opacity=0.6).move_to(
            RIGHT * 1.5 + DOWN * 0.5
        )
        multiply_text = Text("⊗", font_size=20, color=WHITE).move_to(multiply)

        # 输出层
        w_out_layer = Rectangle(
            height=1.5, width=0.8, color=GREEN, fill_opacity=0.6
        ).move_to(RIGHT * 3.5 + DOWN * 0.5)
        w_out_text = Text("W_out\n[2d×d]", font_size=16, color=WHITE).move_to(
            w_out_layer
        )

        # 最终输出
        output_layer = Rectangle(
            height=1.5, width=0.8, color=PURPLE, fill_opacity=0.6
        ).move_to(RIGHT * 5.5 + DOWN * 0.5)
        output_text = Text("Output\n[d]", font_size=16, color=WHITE).move_to(
            output_layer
        )

        # 连接箭头
        arrows = VGroup(
            Arrow(input_layer.get_right(), fork_point, color=YELLOW),
            Arrow(fork_point, w_g_layer.get_left(), color=YELLOW),
            Arrow(fork_point, w_in_layer.get_left(), color=YELLOW),
            Arrow(w_g_layer.get_right(), swish_activation.get_left(), color=YELLOW),
            Arrow(w_in_layer.get_right(), multiply.get_left(), color=YELLOW),
            Arrow(swish_activation.get_right(), multiply.get_left(), color=YELLOW),
            Arrow(multiply.get_right(), w_out_layer.get_left(), color=YELLOW),
            Arrow(w_out_layer.get_right(), output_layer.get_left(), color=YELLOW),
        )

        self.play(
            FadeIn(
                VGroup(
                    input_layer,
                    input_text,
                    fork_point,
                    w_g_layer,
                    w_g_text,
                    swish_activation,
                    swish_text,
                    w_in_layer,
                    w_in_text,
                    multiply,
                    multiply_text,
                    w_out_layer,
                    w_out_text,
                    output_layer,
                    output_text,
                    arrows,
                )
            )
        )

        # 保存引用
        self.swiglu_arch = VGroup(
            input_layer,
            input_text,
            fork_point,
            w_g_layer,
            w_g_text,
            swish_activation,
            swish_text,
            w_in_layer,
            w_in_text,
            multiply,
            multiply_text,
            w_out_layer,
            w_out_text,
            output_layer,
            output_text,
            arrows,
        )

    def compare_activation_functions(self):
        """比较不同激活函数"""
        # 创建坐标系
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 3, 1],
            x_length=6,
            y_length=3,
            axis_config={"color": WHITE},
        ).move_to(DOWN * 0.5)

        x_label = Text("x", font_size=16).next_to(axes.x_axis, RIGHT)
        y_label = Text("f(x)", font_size=16).next_to(axes.y_axis, UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # ReLU函数
        relu_func = axes.plot(lambda x: max(0, x), color=RED, stroke_width=3)
        relu_label = (
            Text("ReLU", font_size=18, color=RED)
            .next_to(axes, UP, buff=0.5)
            .shift(LEFT * 3)
        )

        # GELU函数 (近似)
        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

        gelu_func = axes.plot(gelu, color=GREEN, stroke_width=3)
        gelu_label = Text("GELU", font_size=18, color=GREEN).next_to(axes, UP, buff=0.5)

        # Swish函数
        def swish(x):
            return x / (1 + np.exp(-x))

        swish_func = axes.plot(swish, color=BLUE, stroke_width=3)
        swish_label = (
            Text("Swish", font_size=18, color=BLUE)
            .next_to(axes, UP, buff=0.5)
            .shift(RIGHT * 3)
        )

        self.play(Create(relu_func), Write(relu_label))
        self.wait(0.5)
        self.play(Create(gelu_func), Write(gelu_label))
        self.wait(0.5)
        self.play(Create(swish_func), Write(swish_label))

        # 特性说明
        features = VGroup(
            Text("• ReLU: 简单但死神经元问题", font_size=16, color=RED).move_to(
                DOWN * 2.5 + LEFT * 3
            ),
            Text("• GELU: 平滑版本，性能更好", font_size=16, color=GREEN).move_to(
                DOWN * 2.5
            ),
            Text("• Swish: 自门控，适合SwiGLU", font_size=16, color=BLUE).move_to(
                DOWN * 2.5 + RIGHT * 3
            ),
        )

        self.play(Write(features))
        self.wait(2)

        # 清理
        self.play(
            FadeOut(
                VGroup(
                    axes,
                    x_label,
                    y_label,
                    relu_func,
                    relu_label,
                    gelu_func,
                    gelu_label,
                    swish_func,
                    swish_label,
                    features,
                )
            )
        )

    def compare_parameters(self):
        """比较参数量"""
        # 创建对比表格
        table = VGroup()

        # 表头
        headers = VGroup(
            Text("架构", font_size=20, color=YELLOW),
            Text("中间维度", font_size=20, color=YELLOW),
            Text("参数量", font_size=20, color=YELLOW),
            Text("特点", font_size=20, color=YELLOW),
        ).arrange(RIGHT, buff=1.5)

        # FFN行
        ffn_row = VGroup(
            Text("传统FFN", font_size=18, color=BLUE),
            Text("4d", font_size=18, color=WHITE),
            Text("8d²", font_size=18, color=WHITE),
            Text("简单直接", font_size=18, color=GRAY),
        ).arrange(RIGHT, buff=1.5)
        ffn_row.next_to(headers, DOWN, buff=0.5)

        # SwiGLU行
        swiglu_row = VGroup(
            Text("SwiGLU", font_size=18, color=GREEN),
            Text("2.67d", font_size=18, color=WHITE),
            Text("8d²", font_size=18, color=WHITE),
            Text("性能更强", font_size=18, color=GREEN),
        ).arrange(RIGHT, buff=1.5)
        swiglu_row.next_to(ffn_row, DOWN, buff=0.5)

        # 添加表格线
        h_line1 = Line(LEFT * 4, RIGHT * 4, color=GRAY).next_to(headers, DOWN, buff=0.2)
        h_line2 = Line(LEFT * 4, RIGHT * 4, color=GRAY).next_to(ffn_row, DOWN, buff=0.2)

        v_lines = VGroup()
        for i in range(5):
            x_pos = LEFT * 4 + RIGHT * i * 2
            v_line = Line(x_pos + UP * 0.5, x_pos + DOWN * 2, color=GRAY)
            v_lines.add(v_line)

        table.add(headers, ffn_row, swiglu_row, h_line1, h_line2, v_lines)
        table.center()

        self.play(FadeIn(table))

        # 说明文字
        explanation = Text(
            "SwiGLU通过3个矩阵实现与传统FFN相同的参数量，但性能更强",
            font_size=20,
            color=YELLOW,
        ).move_to(DOWN * 2.5)

        self.play(Write(explanation))

        # 保存引用以便清理
        self.param_table = VGroup(table, explanation)


if __name__ == "__main__":
    scene = FFNSwiGLU()
    scene.render()
