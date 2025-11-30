"""
Transformer vs Mamba 计算复杂度对比
===================================

对应笔记: Appendix_E_Mamba_vs_Transformer.md
生成命令: manim scene_compare.py TransformerVsMamba -qh
输出视频: assets/TransformerVsMamba.mp4

内容要点:
- Attention矩阵计算 O(L²) vs SSM扫描 O(L)
- KV Cache显存占用 vs 固定状态
- 训练并行性对比
- 推理效率对比
"""

from manim import *
import numpy as np

config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class TransformerVsMamba(Scene):
    def construct(self):
        # 1. 标题
        title = Text("Transformer vs Mamba: 计算复杂度深度解析", font_size=40).to_edge(
            UP
        )
        self.play(Write(title))

        # 2. 复杂度公式展示
        complexity_title = Text("计算复杂度对比", font_size=32).move_to(UP * 1.5)
        self.play(Write(complexity_title))

        # Transformer复杂度
        tf_complexity = MathTex(
            r"\text{Attention: } O(L^2 \cdot d)", font_size=28, color=RED
        ).move_to(LEFT * 4 + UP * 0.5)
        self.play(Write(tf_complexity))

        # Mamba复杂度
        mamba_complexity = MathTex(
            r"\text{SSM Scan: } O(L \cdot d)", font_size=28, color=GREEN
        ).move_to(RIGHT * 4 + UP * 0.5)
        self.play(Write(mamba_complexity))

        self.wait(1.5)
        self.play(FadeOut(complexity_title), FadeOut(tf_complexity), FadeOut(mamba_complexity))

        # 3. 训练阶段对比
        training_title = Text("训练阶段：并行性对比", font_size=32).move_to(UP * 2.5)
        self.play(Write(training_title))

        # Transformer训练
        tf_train_title = Text("Transformer训练", font_size=24, color=RED).move_to(
            LEFT * 4 + UP * 1.5
        )
        self.play(Write(tf_train_title))

        # 创建GEMM可视化
        gemm_box = Rectangle(height=2.5, width=3.5, color=RED, fill_opacity=0.3).move_to(
            LEFT * 4
        )
        gemm_text = Text("GEMM\n矩阵乘法", font_size=20, color=WHITE).move_to(gemm_box).shift(UP * 0.5)

        # 展示并行计算
        parallel_arrows = VGroup()
        for i in range(3):
            arrow = Arrow(UP * 0.3, DOWN * 0.3, color=YELLOW).move_to(
                LEFT * 4 + LEFT * 0.8 + RIGHT * i * 0.8 + DOWN * 0.5
            )
            parallel_arrows.add(arrow)

        self.play(FadeIn(gemm_box), Write(gemm_text), Create(parallel_arrows))

        parallel_text = Text("完全并行", font_size=20, color=YELLOW).next_to(gemm_box, DOWN, buff=0.3)
        self.play(Write(parallel_text))

        # Mamba训练
        mamba_train_title = Text("Mamba训练", font_size=24, color=GREEN).move_to(
            RIGHT * 4 + UP * 1.5
        )
        self.play(Write(mamba_train_title))

        # 创建并行扫描可视化
        scan_box = Rectangle(height=2.5, width=3.5, color=GREEN, fill_opacity=0.3).move_to(
            RIGHT * 4
        )
        scan_text = Text("Parallel Scan\n并行扫描", font_size=20, color=WHITE).move_to(
            scan_box
        ).shift(UP * 0.5)

        # 展示扫描过程
        scan_steps = VGroup()
        for i in range(4):
            step = Circle(radius=0.2, color=YELLOW).move_to(
                RIGHT * 4 + LEFT * 0.9 + RIGHT * i * 0.6 + DOWN * 0.5
            )
            scan_steps.add(step)

        self.play(FadeIn(scan_box), Write(scan_text), FadeIn(scan_steps))

        scan_text2 = Text("O(log L)并行", font_size=20, color=YELLOW).next_to(scan_box, DOWN, buff=0.3)
        self.play(Write(scan_text2))

        self.wait(2)
        self.play(
            FadeOut(
                VGroup(
                    training_title,
                    tf_train_title,
                    gemm_box,
                    gemm_text,
                    parallel_arrows,
                    parallel_text,
                    mamba_train_title,
                    scan_box,
                    scan_text,
                    scan_steps,
                    scan_text2,
                )
            )
        )

        # 4. 推理阶段对比
        inference_title = Text("推理阶段：内存占用对比", font_size=32).move_to(UP * 2.5)
        self.play(Write(inference_title))

        # Transformer推理
        tf_inf_title = Text("Transformer推理", font_size=24, color=RED).move_to(
            LEFT * 4 + UP * 1.5
        )
        self.play(Write(tf_inf_title))

        # KV Cache可视化
        kv_cache_box = Rectangle(
            height=3.5, width=2.5, color=RED, fill_opacity=0.3
        ).move_to(LEFT * 4)
        kv_text = Text("KV Cache", font_size=20, color=WHITE).move_to(kv_cache_box).to_edge(UP, buff=0.2).shift(DOWN*2.5) # Relative to box center is tricky, let's use move_to and shift
        kv_text.move_to(kv_cache_box.get_top() + DOWN * 0.5)

        # 展示KV Cache增长
        cache_levels = VGroup()
        for i in range(4):
            level = Rectangle(
                height=0.4, width=2.0, color=ORANGE, fill_opacity=0.5
            ).move_to(LEFT * 4 + UP * 0.5 - i * 0.6)
            cache_levels.add(level)

        self.play(FadeIn(kv_cache_box), Write(kv_text), FadeIn(cache_levels))

        cache_formula = MathTex(
            r"O(L \cdot d_{model})", font_size=24, color=ORANGE
        ).next_to(kv_cache_box, DOWN, buff=0.3)
        self.play(Write(cache_formula))

        # Mamba推理
        mamba_inf_title = Text("Mamba推理", font_size=24, color=GREEN).move_to(
            RIGHT * 4 + UP * 1.5
        )
        self.play(Write(mamba_inf_title))

        # 固定状态可视化
        state_box = Rectangle(height=1.5, width=2.5, color=GREEN, fill_opacity=0.3).move_to(
            RIGHT * 4
        )
        state_text = Text("Fixed State", font_size=20, color=WHITE).move_to(state_box)

        self.play(FadeIn(state_box), Write(state_text))

        state_formula = MathTex(
            r"O(d_{state} \cdot d_{model})", font_size=24, color=YELLOW
        ).next_to(state_box, DOWN, buff=0.3)
        self.play(Write(state_formula))

        # 关键差异说明
        diff_text = Text(
            "关键差异：线性 vs 平方增长", font_size=24, color=YELLOW
        ).move_to(DOWN * 3)
        self.play(Write(diff_text))

        self.wait(2)

        # 5. 总结对比
        self.play(
            FadeOut(
                VGroup(
                    inference_title,
                    tf_inf_title,
                    kv_cache_box,
                    kv_text,
                    cache_levels,
                    cache_formula,
                    mamba_inf_title,
                    state_box,
                    state_text,
                    state_formula,
                    diff_text,
                )
            )
        )

        summary_title = Text("总结对比", font_size=32).move_to(UP * 2.5)
        self.play(Write(summary_title))

        # 创建对比表格
        comparison_table = self.create_comparison_table()
        self.play(FadeIn(comparison_table))

        self.wait(3)

    def create_comparison_table(self):
        """创建对比表格 - 卡片式布局"""
        # 卡片背景
        card_bg = RoundedRectangle(
            corner_radius=0.5,
            height=6,
            width=12,
            fill_color="#1E1E1E",
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=2
        )

        # 卡片标题
        card_title = Text("Transformer vs Mamba 核心对比", font_size=32, color=BLUE).next_to(card_bg, UP, buff=-0.8)
        
        # 分隔线
        h_line = Line(LEFT * 5.5, RIGHT * 5.5, color=GRAY).next_to(card_title, DOWN, buff=0.3)

        # 表头
        headers = VGroup(
            Text("特性", font_size=24, color=YELLOW).move_to(LEFT * 4),
            Text("Transformer", font_size=24, color=RED).move_to(ORIGIN),
            Text("Mamba", font_size=24, color=GREEN).move_to(RIGHT * 4),
        ).next_to(h_line, DOWN, buff=0.3)

        # 数据行
        rows_data = [
            ("计算复杂度", "O(L²·d) - 二次", "O(L·d) - 线性"),
            ("训练并行性", "完全并行 (高效)", "O(log L) (较好)"),
            ("推理内存", "O(L·d) (KV Cache)", "O(d_state·d) (固定)"),
            ("适用场景", "短序列 / 通用", "长序列 / 线性关注"),
        ]

        rows_group = VGroup()
        start_point = headers.get_bottom() + DOWN * 0.4
        
        for i, (feature, tf_val, mamba_val) in enumerate(rows_data):
            # 行背景（交替颜色）
            row_bg = Rectangle(
                height=0.8, 
                width=11, 
                fill_color=GRAY if i % 2 == 0 else BLACK, 
                fill_opacity=0.2,
                stroke_width=0
            ).move_to(DOWN * (start_point[1] - 0.4 - i * 0.9)) # 手动计算位置有点麻烦，用 next_to 更好

            # 重新定位逻辑
            if i == 0:
                y_pos = start_point[1] - 0.4
            else:
                y_pos = rows_group[-1].get_center()[1] - 0.9

            row_bg.move_to([0, y_pos, 0])

            row_content = VGroup(
                Text(feature, font_size=20, color=WHITE).move_to([headers[0].get_center()[0], y_pos, 0]),
                Text(tf_val, font_size=20, color=RED_B).move_to([headers[1].get_center()[0], y_pos, 0]),
                Text(mamba_val, font_size=20, color=GREEN_B).move_to([headers[2].get_center()[0], y_pos, 0]),
            )
            
            rows_group.add(VGroup(row_bg, row_content))

        return VGroup(card_bg, card_title, h_line, headers, rows_group)

    def construct_complexity_visualization(self):
        """创建O(L²) vs O(L)复杂度可视化"""
        grid_size = 5
        transformer_group = VGroup()
        mamba_group = VGroup()

        # Transformer O(L²) 网格
        for i in range(grid_size):
            for j in range(grid_size):
                sq = Square(side_length=0.4).set_stroke(RED_E, 1).set_fill(RED, 0.2)
                sq.move_to(LEFT * 2.5 + RIGHT * j * 0.4 + DOWN * i * 0.4)
                transformer_group.add(sq)

        # Mamba O(L) 线性
        for i in range(grid_size):
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
