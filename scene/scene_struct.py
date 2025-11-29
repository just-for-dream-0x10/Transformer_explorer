from manim import *
import numpy as np

# 通用配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


# =========================================================
# 场景 1: Encoder 的多头拆分与残差连接 (布局优化版)
# =========================================================
class EncoderFlow(Scene):
    def construct(self):
        # 1. 标题
        title = Text("Encoder Block: Multi-Head & Residual", font_size=40).to_edge(
            UP, buff=0.5
        )
        self.play(Write(title))

        # --- 布局坐标定义 ---
        POS_INPUT = DOWN * 3.0
        POS_MIDDLE = ORIGIN  # QKV 和 Attention 发生的地方
        POS_OUTPUT = UP * 2.0  # 最终输出的地方

        # 2. 创建输入向量 X
        input_box = Rectangle(height=0.8, width=3, color=BLUE, fill_opacity=0.5)
        input_text = Text("Input X (512)", font_size=24).move_to(input_box)
        input_group = VGroup(input_box, input_text).move_to(POS_INPUT)

        self.play(FadeIn(input_group))

        # 3. 复制出三份 Q, K, V (移动到中间层)
        qkv_group = VGroup()
        labels = ["Q", "K", "V"]
        colors = [RED, GREEN, YELLOW]

        # 初始在 Input 位置生成
        for i in range(3):
            box = Rectangle(height=0.6, width=0.8, color=colors[i], fill_opacity=0.5)
            lbl = Text(labels[i], font_size=24).move_to(box)
            g = VGroup(box, lbl).move_to(POS_INPUT)  # 从底下生出来
            qkv_group.add(g)

        # 动画：从 Input 分裂并上升到中间
        self.play(
            qkv_group[0].animate.move_to(POS_MIDDLE + LEFT * 2.5),
            qkv_group[1].animate.move_to(POS_MIDDLE),
            qkv_group[2].animate.move_to(POS_MIDDLE + RIGHT * 2.5),
            run_time=1,
        )

        # 连线 X -> QKV
        lines_qkv = VGroup()
        for box in qkv_group:
            l = Line(input_box.get_top(), box.get_bottom(), color=GRAY_C)
            lines_qkv.add(l)
        self.play(Create(lines_qkv), run_time=0.5)

        # 4. 多头拆分 (Split Heads) - 在 QKV 上方展示
        heads_group = VGroup()
        head_lines = VGroup()  # 单独存线，方便后面清理

        for i, parent in enumerate(qkv_group):
            # 每个 Q/K/V 上方生成两个小方块
            h1 = Rectangle(height=0.4, width=0.35, color=parent[0].get_color())
            h2 = Rectangle(height=0.4, width=0.35, color=parent[0].get_color())

            # 位置在 QKV 的上方一点
            h1.move_to(parent.get_center() + UP * 1.2 + LEFT * 0.25)
            h2.move_to(parent.get_center() + UP * 1.2 + RIGHT * 0.25)

            heads_group.add(h1, h2)

            # 连线
            l1 = Line(parent.get_top(), h1.get_bottom(), color=GRAY, stroke_width=2)
            l2 = Line(parent.get_top(), h2.get_bottom(), color=GRAY, stroke_width=2)
            head_lines.add(l1, l2)

        self.play(Create(head_lines), FadeIn(heads_group), run_time=0.5)

        head_label = Text("Split Heads", font_size=24, color=GRAY).next_to(
            heads_group, RIGHT, buff=0.5
        )
        self.play(Write(head_label))
        self.wait(0.5)

        # 5. Attention 计算 (变成一个大黑盒，覆盖掉 QKV 和 Heads)
        # 这一步我们要清理掉 QKV, Heads, Lines，让界面变干净

        attn_box = Rectangle(
            height=1.5, width=6, color=WHITE, fill_opacity=0.1
        ).move_to(POS_MIDDLE + UP * 0.5)
        attn_text = Text("Multi-Head Attention", font_size=30).move_to(attn_box)

        self.play(
            FadeOut(qkv_group),
            FadeOut(lines_qkv),
            FadeOut(heads_group),
            FadeOut(head_lines),  # 关键：把那几根V形线也擦掉
            FadeOut(head_label),
            FadeIn(attn_box),
            Write(attn_text),
        )

        # 6. Concat & Output (上升到顶部)
        output_box = Rectangle(height=0.8, width=3, color=PURPLE, fill_opacity=0.5)
        output_text = Text("Output Z", font_size=24).move_to(output_box)
        output_group = VGroup(output_box, output_text).move_to(
            POS_MIDDLE + UP * 0.5
        )  # 初始位置在 Attn 里面

        # 动画：Attention 盒子收缩成 Output 盒子，并向上移动
        self.play(FadeOut(attn_box), FadeOut(attn_text), FadeIn(output_group))
        self.play(output_group.animate.move_to(POS_OUTPUT))  # 移到最上面

        # 7. 残差连接 (Residual Connection) - 巨大的弧线
        # 从 Input (底) 连到 Output (顶)

        p_start = input_box.get_right() + RIGHT * 0.2
        p_end = output_box.get_right() + RIGHT * 0.2

        # 贝塞尔曲线控制点：向右拉得更远，形成 "D" 字形的大弧线
        control_p1 = p_start + RIGHT * 4.0
        control_p2 = p_end + RIGHT * 4.0

        residual_line = CubicBezier(
            p_start, control_p1, control_p2, p_end, color=YELLOW, stroke_width=5
        )
        residual_text = Text("Residual (+)", font_size=24, color=YELLOW).next_to(
            residual_line, RIGHT, buff=0.2
        )

        self.play(Create(residual_line), Write(residual_text), run_time=1.5)

        # Add & Norm 变换
        norm_box = Rectangle(
            height=0.8, width=3.2, color=ORANGE, fill_opacity=0.8
        )  # 不透明度高一点，盖住线
        norm_text = Text("Add & Norm", font_size=24, color=BLACK).move_to(
            norm_box
        )  # 文字用黑色对比

        self.play(
            ReplacementTransform(output_box, norm_box),
            ReplacementTransform(output_text, norm_text),
        )

        self.wait(2)


# =========================================================
# 场景 2: Decoder 的因果掩码 (微调版)
# =========================================================
class DecoderMasking(Scene):
    def construct(self):
        title = Text("Decoder: Causal Masking (-inf)", font_size=36).to_edge(UP)
        self.play(Write(title))

        size = 4
        np.random.seed(42)
        scores_data = np.random.randint(1, 10, (size, size))

        cell_size = 1.0  # 稍微大一点
        grid = VGroup()
        nums = VGroup()

        # 创建网格
        for i in range(size):
            for j in range(size):
                sq = Square(side_length=cell_size).set_stroke(GRAY, 1)
                # 居中布局
                sq.move_to(
                    RIGHT * (j - size / 2 + 0.5) * cell_size
                    + DOWN * (i - size / 2 + 0.5) * cell_size
                )

                val = scores_data[i][j]
                num = Text(str(val), font_size=24).move_to(sq)

                grid.add(sq)
                nums.add(num)

        matrix_group = VGroup(grid, nums).center()

        # 行列标签
        row_labels = VGroup()
        col_labels = VGroup()
        for i in range(size):
            # 获取对应格子的位置
            left_cell = grid[i * size]
            top_cell = grid[i]

            rl = Text(f"t{i+1}", font_size=24, color=BLUE).next_to(
                left_cell, LEFT, buff=0.3
            )
            cl = Text(f"t{i+1}", font_size=24, color=BLUE).next_to(
                top_cell, UP, buff=0.3
            )
            row_labels.add(rl)
            col_labels.add(cl)

        self.play(Create(grid), Write(nums), Write(row_labels), Write(col_labels))
        self.wait(0.5)

        # Mask 动画
        mask_text = Text(
            "Mask Future (Upper Triangle)", font_size=30, color=RED
        ).to_edge(RIGHT, buff=1.0)
        self.play(Write(mask_text))

        anims = []
        for i in range(size):
            for j in range(size):
                if j > i:  # 上三角
                    idx = i * size + j
                    sq = grid[idx]
                    nm = nums[idx]
                    anims.append(sq.animate.set_fill(RED_E, opacity=0.6))
                    anims.append(Transform(nm, Text("-inf", font_size=20).move_to(sq)))

        self.play(*anims, run_time=1.5)
        self.wait(0.5)

        # Softmax 动画
        softmax_text = Text("Softmax -> 0", font_size=30, color=GREEN).next_to(
            mask_text, DOWN, buff=0.5
        )
        self.play(Write(softmax_text))

        anims2 = []
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                sq = grid[idx]
                nm = nums[idx]

                if j > i:
                    anims2.append(sq.animate.set_fill(BLACK, opacity=0.8))  # 变黑
                    anims2.append(
                        Transform(nm, Text("0", font_size=24, color=GRAY).move_to(sq))
                    )
                else:
                    anims2.append(sq.animate.set_stroke(GREEN, 3))  # 高亮有效区域

        self.play(*anims2, run_time=1.5)

        self.wait(2)
