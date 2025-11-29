from manim import *
import numpy as np

config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class TransformerVsMamba(Scene):
    def construct(self):
        # 1. 标题
        title = Text("Compute Cost: Transformer vs Mamba", font_size=40).to_edge(UP)
        self.play(Write(title))

        # ==========================================
        # 左侧：Transformer (O(L^2))
        # ==========================================
        tf_label = Text("Transformer (Attention)", font_size=28, color=RED).move_to(
            LEFT * 3.5 + UP * 2.5
        )
        self.play(Write(tf_label))

        # 初始矩阵 (2x2)
        grid_size = 2
        tf_group = VGroup()

        # 创建初始网格
        for i in range(grid_size):
            for j in range(grid_size):
                sq = Square(side_length=0.4).set_stroke(RED_E, 1).set_fill(RED, 0.2)
                sq.move_to(LEFT * 4.5 + RIGHT * j * 0.4 + DOWN * i * 0.4)
                tf_group.add(sq)

        tf_group.center().shift(LEFT * 3.5)
        self.play(FadeIn(tf_group))

        # ==========================================
        # 右侧：Mamba (O(L))
        # ==========================================
        mamba_label = Text("Mamba (SSM Scan)", font_size=28, color=GREEN).move_to(
            RIGHT * 3.5 + UP * 2.5
        )
        self.play(Write(mamba_label))

        # 初始状态 (固定高度，长度随时间增加)
        state_dim = 4  # 隐状态高度
        mamba_group = VGroup()

        # 创建初始状态条 (2列)
        for i in range(state_dim):
            for j in range(grid_size):
                sq = Square(side_length=0.4).set_stroke(GREEN_E, 1).set_fill(GREEN, 0.2)
                sq.move_to(RIGHT * 2.5 + RIGHT * j * 0.4 + DOWN * i * 0.4)
                mamba_group.add(sq)

        mamba_group.center().shift(RIGHT * 3.5)
        self.play(FadeIn(mamba_group))

        # ==========================================
        # 动态演示：序列长度增长 (L: 2 -> 8)
        # ==========================================

        l_text = Variable(2, Text("Sequence Length (L)"), var_type=Integer).to_edge(
            DOWN
        )
        self.play(Write(l_text))

        for l in range(3, 9):  # 增长到 8
            # 更新 L 显示
            self.play(l_text.tracker.animate.set_value(l), run_time=0.1)

            # --- 更新 Transformer (N*N) ---
            # 需要增加一行和一列，填满之前的空缺
            new_tf_cells = VGroup()

            # 1. 增加新的一行 (第 l 行，列 0 到 l-1)
            for j in range(l):
                sq = Square(side_length=0.4).set_stroke(RED_E, 1).set_fill(RED, 0.2)
                # 计算位置：基于左上角参考点
                ref = tf_group[0].get_center()
                sq.move_to(ref + RIGHT * j * 0.4 + DOWN * (l - 1) * 0.4)
                new_tf_cells.add(sq)

            # 2. 增加新的一列 (第 l 列，行 0 到 l-2) -> 因为 (l-1, l-1) 已经在上面加过了
            for i in range(l - 1):
                sq = Square(side_length=0.4).set_stroke(RED_E, 1).set_fill(RED, 0.2)
                ref = tf_group[0].get_center()
                sq.move_to(ref + RIGHT * (l - 1) * 0.4 + DOWN * i * 0.4)
                new_tf_cells.add(sq)

            tf_group.add(new_tf_cells)

            # --- 更新 Mamba (d * L) ---
            # 只需要在右边增加一列
            new_mamba_cells = VGroup()
            for i in range(state_dim):
                sq = Square(side_length=0.4).set_stroke(GREEN_E, 1).set_fill(GREEN, 0.2)
                # 找到最右边的参考
                # mamba_group 里的元素顺序比较乱，重新计算位置最稳
                # 假设初始参考点
                ref_m = (
                    RIGHT * 3.5
                    + LEFT * (grid_size * 0.4 / 2)
                    + UP * (state_dim * 0.4 / 2)
                )
                # 微调位置逻辑：直接往右堆
                # 简单起见，我们重新画一个列
                last_col_x = mamba_group[-1].get_center()[0]
                # 这是一个hack，并不是特别准，但视觉上够用
                # 更准的方法是记录 current_L

                # 重新计算坐标
                # Mamba Center: RIGHT * 3.5
                # Start X: RIGHT * 3.5 - (Current_Width / 2)

                pass

            # Mamba 更新逻辑重写：
            # 每次只加一竖列
            col_group = VGroup()
            for i in range(state_dim):
                sq = Square(side_length=0.4).set_stroke(GREEN_E, 1).set_fill(GREEN, 0.2)
                # 放到 Mamba 组的右侧
                sq.move_to(
                    mamba_group.get_right()
                    + RIGHT * 0.2
                    + DOWN * (i - state_dim / 2 + 0.5) * 0.4
                    + LEFT * 0.2
                )
                # 上面坐标算得有点晕，用 next_to 最稳
                if i == 0:
                    # 找到当前最右上角的格子
                    top_right = mamba_group[state_dim - 1]  # 并不是...
                    # 简单粗暴：获取 Mamba Group 的 Right 边界
                    right_edge = mamba_group.get_right()
                    top_edge = mamba_group.get_top()
                    sq.move_to([right_edge[0] + 0.2, top_edge[1] - 0.2, 0])
                else:
                    sq.next_to(col_group[-1], DOWN, buff=0)

                col_group.add(sq)

            mamba_group.add(col_group)

            # 播放动画
            self.play(FadeIn(new_tf_cells), FadeIn(col_group), run_time=0.3)

            # 重新居中，防止跑出屏幕
            if l % 3 == 0:
                self.play(
                    tf_group.animate.center().shift(LEFT * 3.5),
                    mamba_group.animate.center().shift(RIGHT * 3.5),
                    run_time=0.2,
                )

        # ==========================================
        # 总结文本
        # ==========================================

        t_cost = Text("Cost: Quadratic O(L^2)", font_size=24, color=RED).next_to(
            tf_group, DOWN
        )
        m_cost = Text("Cost: Linear O(L)", font_size=24, color=GREEN).next_to(
            mamba_group, DOWN
        )

        self.play(Write(t_cost), Write(m_cost))

        self.wait(2)
