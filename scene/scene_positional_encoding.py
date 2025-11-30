"""
位置编码可视化场景
===================

对应笔记: 1.math.md (位置编码部分)
生成命令: manim scene_positional_encoding.py PositionalEncoding -qh
输出视频: assets/PositionalEncoding.mp4

内容要点:
- Transformer排列不变性问题
- 正弦位置编码公式推导
- 位置编码矩阵可视化
- 不同频率正弦波展示
- 位置编码与词嵌入相加
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class PositionalEncoding(Scene):
    def construct(self):
        # 标题
        title = Text("位置编码 (Positional Encoding)", font_size=40).to_edge(
            UP, buff=0.5
        )
        self.play(Write(title))
        self.wait(0.5)

        # 1. 问题：Transformer的排列不变性
        problem_text = Text(
            "问题：Transformer是排列不变的！", font_size=28, color=RED
        ).to_edge(DOWN, buff=1.0)
        self.play(Write(problem_text))
        self.wait(0.5)

        # 展示词语重排的例子
        words = ["我", "爱", "AI"]
        word_objs = VGroup()
        for i, word in enumerate(words):
            txt = Text(word, font_size=36, color=BLUE)
            txt.move_to(RIGHT * (i - 1) * 2)
            word_objs.add(txt)

        word_objs.move_to(UP * 0.5)
        self.play(FadeIn(word_objs))
        self.wait(0.5)

        # 重排词语
        arrow = Arrow(LEFT, RIGHT, color=YELLOW).next_to(word_objs, DOWN, buff=0.5)
        self.play(Create(arrow))

        words_reordered = ["AI", "爱", "我"]
        word_objs_reordered = VGroup()
        for i, word in enumerate(words_reordered):
            txt = Text(word, font_size=36, color=GREEN)
            txt.move_to(RIGHT * (i - 1) * 2)
            word_objs_reordered.add(txt)

        word_objs_reordered.move_to(DOWN * 1.0)
        self.play(FadeIn(word_objs_reordered))
        self.wait(1)

        # 清理
        self.play(
            FadeOut(problem_text),
            FadeOut(word_objs),
            FadeOut(arrow),
            FadeOut(word_objs_reordered),
        )

        # 2. 数学公式展示
        formula_title = Text("正弦位置编码公式", font_size=32).move_to(UP * 2.5)
        self.play(Write(formula_title))

        # 公式
        formula1 = MathTex(
            r"PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)",
            font_size=36,
            color=YELLOW,
        ).move_to(UP * 0.5)
        formula2 = MathTex(
            r"PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)",
            font_size=36,
            color=YELLOW,
        ).next_to(formula1, DOWN, buff=0.5)

        self.play(Write(formula1), Write(formula2))
        self.wait(1.5)

        # 3. 可视化位置编码矩阵
        self.play(FadeOut(formula1), FadeOut(formula2), FadeOut(formula_title))

        matrix_title = Text("位置编码矩阵可视化", font_size=32).move_to(UP * 2.5)
        self.play(Write(matrix_title))

        # 创建位置编码矩阵
        seq_len = 6
        d_model = 8

        # 计算位置编码
        pe = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                div_term = 10000 ** (i / d_model)
                pe[pos, i] = np.sin(pos / div_term)
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / div_term)

        # 创建热力图
        cell_size = 0.6
        matrix = VGroup()
        for i in range(seq_len):
            for j in range(d_model):
                value = pe[i, j]
                # 根据值映射颜色
                if value > 0:
                    color = interpolate_color(WHITE, RED, abs(value))
                else:
                    color = interpolate_color(WHITE, BLUE, abs(value))

                sq = (
                    Square(side_length=cell_size)
                    .set_fill(color, opacity=0.8)
                    .set_stroke(GRAY, 1)
                )
                sq.move_to(
                    RIGHT * (j - d_model / 2 + 0.5) * cell_size
                    + DOWN * (i - seq_len / 2 + 0.5) * cell_size
                )

                # 显示数值（保留2位小数）
                num_text = Text(
                    f"{value:.1f}",
                    font_size=14,
                    color=BLACK if abs(value) > 0.5 else WHITE,
                ).move_to(sq)
                matrix.add(VGroup(sq, num_text))

        matrix.center()
        self.play(FadeIn(matrix))

        # 添加行列标签
        row_labels = VGroup()
        col_labels = VGroup()
        for i in range(seq_len):
            label = (
                Text(f"pos={i}", font_size=16, color=GREEN)
                .next_to(matrix, LEFT, buff=0.3)
                .align_to(matrix[i * d_model], UP)
            )
            row_labels.add(label)

        for i in range(d_model):
            label = (
                Text(f"dim={i}", font_size=16, color=YELLOW)
                .next_to(matrix, UP, buff=0.3)
                .align_to(matrix[i], LEFT)
            )
            col_labels.add(label)

        self.play(Write(row_labels), Write(col_labels))
        self.wait(1.5)

        # 4. 展示不同频率的正弦波
        self.play(
            FadeOut(matrix),
            FadeOut(row_labels),
            FadeOut(col_labels),
            FadeOut(matrix_title),
        )

        wave_title = Text("不同维度的正弦波", font_size=32).move_to(UP * 2.5)
        self.play(Write(wave_title))

        # 创建坐标系
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=8,
            y_length=3,
            axis_config={"color": WHITE},
            x_axis_config={"numbers_to_include": np.arange(0, 11, 2)},
            y_axis_config={"numbers_to_include": [-1, 0, 1]},
        ).move_to(DOWN * 0.5)

        x_label = Text("位置 (pos)", font_size=20).next_to(axes.x_axis, RIGHT)
        y_label = (
            Text("编码值", font_size=20).next_to(axes.y_axis, UP).rotate(90 * DEGREES)
        )

        self.play(Create(axes), Write(x_label), Write(y_label))

        # 绘制不同频率的正弦波
        colors = [RED, GREEN, BLUE, YELLOW]
        frequencies = [1, 2, 4, 8]
        waves = VGroup()

        for freq, color in zip(frequencies, colors):
            wave = axes.plot(
                lambda x: np.sin(freq * x / 2), color=color, stroke_width=3
            )
            waves.add(wave)
            self.play(Create(wave), run_time=0.8)

        # 添加图例
        legends = VGroup()
        for freq, color in zip(frequencies, colors):
            legend_line = Line(LEFT, RIGHT, color=color, stroke_width=3).scale(0.5)
            legend_text = Text(f"freq={freq}", font_size=16, color=color).next_to(
                legend_line, RIGHT
            )
            legend = VGroup(legend_line, legend_text)
            legends.add(legend)

        legends.arrange(DOWN, buff=0.2).to_edge(RIGHT, buff=1.0)
        self.play(FadeIn(legends))
        self.wait(2)

        # 5. 展示位置编码如何与词嵌入相加
        self.play(
            FadeOut(waves),
            FadeOut(legends),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(wave_title),
        )

        add_title = Text("位置编码 + 词嵌入", font_size=32).move_to(UP * 2.5)
        self.play(Write(add_title))

        # 词嵌入向量
        embedding = Rectangle(
            height=2, width=0.8, color=BLUE, fill_opacity=0.6
        ).move_to(LEFT * 3)
        embedding_text = Text("词嵌入\n[512维]", font_size=20, color=WHITE).move_to(
            embedding
        )

        # 位置编码向量
        pos_enc = Rectangle(height=2, width=0.8, color=GREEN, fill_opacity=0.6).move_to(
            ORIGIN
        )
        pos_text = Text("位置编码\n[512维]", font_size=20, color=WHITE).move_to(pos_enc)

        # 加号
        plus = Text("+", font_size=48, color=YELLOW).move_to(RIGHT * 1.5)

        # 结果
        result = Rectangle(height=2, width=0.8, color=PURPLE, fill_opacity=0.6).move_to(
            RIGHT * 4
        )
        result_text = Text("最终输入\n[512维]", font_size=20, color=WHITE).move_to(
            result
        )

        self.play(FadeIn(embedding), Write(embedding_text))
        self.play(FadeIn(pos_enc), Write(pos_text))
        self.play(Write(plus))

        # 箭头
        arrow1 = Arrow(
            embedding.get_right(), embedding.get_right() + RIGHT * 0.5, color=YELLOW
        )
        arrow2 = Arrow(
            pos_enc.get_right(), pos_enc.get_right() + RIGHT * 0.5, color=YELLOW
        )
        self.play(Create(arrow1), Create(arrow2))

        self.play(FadeIn(result), Write(result_text))

        # 添加说明
        explanation = Text(
            "通过加法，位置信息被注入到词向量中", font_size=24, color=GRAY
        ).move_to(DOWN * 2.5)
        self.play(Write(explanation))

        self.wait(3)


if __name__ == "__main__":
    scene = PositionalEncoding()
    scene.render()
