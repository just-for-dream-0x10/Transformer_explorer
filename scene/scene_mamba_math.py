from manim import *
import numpy as np

config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class DiscretizationVisual(Scene):
    def construct(self):
        # ==========================================
        # 1. 引入连续系统 (物理世界)
        # ==========================================
        title = Text(
            "From Continuous Physics to Discrete Digital", font_size=40
        ).to_edge(UP, buff=0.5)
        self.play(Write(title))

        # 调整 Axes 位置：稍微下移，给顶部留出更多空间
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=4,
            axis_config={"include_tip": False, "color": GRAY},
        ).move_to(
            DOWN * 0.8
        )  # 下移一点

        # 绘制曲线
        curve = axes.plot(
            lambda x: np.sin(x) + 0.5 * np.cos(2 * x), color=BLUE, stroke_width=3
        )

        # --- 修复 1: 标签位置 ---
        # 原来是 next_to(curve)，容易重叠。现在改为放在坐标轴右上角的空白处
        label_cont = Text("Continuous Signal x(t)", font_size=24, color=BLUE)
        label_cont.next_to(axes, UP, aligned_edge=RIGHT, buff=0.2)

        self.play(Create(axes), Create(curve), Write(label_cont))
        self.wait(1)

        # ==========================================
        # 2. 引入采样 (Delta)
        # ==========================================

        # 放在标题下方
        delta_text = MathTex(r"\Delta \text{ (Step Size)}", color=YELLOW).next_to(
            title, DOWN, buff=0.5
        )
        self.play(Write(delta_text))

        # 演示大步长 (粗糙采样)
        lines_coarse = VGroup()
        rects_coarse = VGroup()
        step_coarse = 1.0

        for x in np.arange(0, 10, step_coarse):
            y = np.sin(x) + 0.5 * np.cos(2 * x)
            # 虚线
            line = DashedLine(
                axes.c2p(x, 0), axes.c2p(x, y), color=YELLOW, stroke_width=2
            )
            dot = Dot(axes.c2p(x, y), color=YELLOW, radius=0.06)
            lines_coarse.add(line, dot)

            # 矩形 (ZOH)
            if x < 10 - step_coarse:  # 防止画出界
                rect = Rectangle(
                    width=axes.x_axis.unit_size * step_coarse,
                    height=abs(y) * axes.y_axis.unit_size,
                    stroke_width=1,
                    stroke_color=YELLOW,
                    fill_opacity=0.2,
                    fill_color=YELLOW,
                )
                # 锚点对齐
                if y >= 0:
                    rect.align_to(axes.c2p(x, 0), DOWN + LEFT)
                else:
                    rect.align_to(axes.c2p(x, 0), UP + LEFT)
                rects_coarse.add(rect)

        self.play(Create(lines_coarse), FadeIn(rects_coarse), run_time=1.5)

        # 公式放在底部，留出 buff 防止遮挡坐标轴
        formula_zoh = MathTex(
            r"\bar{A} = \exp(\Delta A)",
            r"\quad \text{Large } \Delta \to \text{Info Loss}",
        ).next_to(
            axes, DOWN, buff=0.8
        )  # 增加间距

        self.play(Write(formula_zoh))
        self.wait(2)

        # ==========================================
        # 3. 动态调整：Mamba 的核心魔法
        # ==========================================

        # 清除旧元素
        self.play(
            FadeOut(lines_coarse),
            FadeOut(rects_coarse),
            FadeOut(formula_zoh),
            FadeOut(delta_text),  # 把上面的 Delta 标题也清掉，避免和 Conclusion 重叠
        )

        # --- 修复 2: 底部文字位置 ---
        magic_text = Text("Mamba: Δ changes at every step!", font_size=32, color=RED)
        magic_text.next_to(axes, DOWN, buff=0.8)  # 保持足够的底部距离

        self.play(Write(magic_text))

        # 演示变步长
        x_points = [0, 1.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.0, 9.0, 10.0]
        lines_adaptive = VGroup()
        rects_adaptive = VGroup()

        for i in range(len(x_points) - 1):
            x = x_points[i]
            next_x = x_points[i + 1]
            width = next_x - x
            y = np.sin(x) + 0.5 * np.cos(2 * x)

            line = Line(axes.c2p(x, 0), axes.c2p(x, y), color=RED, stroke_width=2)
            dot = Dot(axes.c2p(x, y), color=RED, radius=0.06)
            lines_adaptive.add(line, dot)

            rect = Rectangle(
                width=axes.x_axis.unit_size * width,
                height=abs(y) * axes.y_axis.unit_size,
                stroke_width=1,
                stroke_color=RED,
                fill_opacity=0.3,
                fill_color=RED,
            )
            if y >= 0:
                rect.align_to(axes.c2p(x, 0), DOWN + LEFT)
            else:
                rect.align_to(axes.c2p(x, 0), UP + LEFT)
            rects_adaptive.add(rect)

        self.play(Create(lines_adaptive), FadeIn(rects_adaptive), run_time=2)

        # --- 修复 3: 顶部结论文字位置 ---
        # 放在标题正下方，而不是用绝对坐标 move_to
        conclusion = Text(
            "Focus on details (small Δ) or Skip noise (large Δ)",
            font_size=24,
            color=WHITE,
        )
        conclusion.next_to(title, DOWN, buff=0.5)

        self.play(Write(conclusion))

        self.wait(3)
