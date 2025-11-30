"""
Mamba 核心机制：选择性状态空间
==============================

对应笔记: Appendix_E_Mamba_vs_Transformer.md
生成命令: manim scene_mamba_core.py MambaMechanism -qh
输出视频: assets/MambaMechanism.mp4

内容要点:
- 选择性机制：Δ, B, C根据输入动态生成
- 离散化过程：连续系统到数字信号
- 递归扫描：h_t = ĀA h_{t-1} + ĀB x_t
- 与Attention的对比：O(L)复杂度，固定内存
"""

from manim import *
import numpy as np

config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class MambaMechanism(Scene):
    def construct(self):
        # 1. 标题和核心公式
        title = Text("Mamba: 选择性状态空间机制", font_size=36).to_edge(UP)
        self.play(Write(title))

        # 核心递归公式
        core_formula = MathTex(
            r"h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t", font_size=28, color=YELLOW
        ).move_to(UP * 1.8)
        self.play(Write(core_formula))

        # 公式说明（垂直排列，数学符号+中文说明）
        h_symbol = MathTex(r"h_t:", font_size=16, color=YELLOW)
        h_desc = Text("当前隐状态", font_size=16, color=YELLOW)
        h_row = VGroup(h_symbol, h_desc).arrange(RIGHT, buff=0.2)
        
        x_symbol = MathTex(r"x_t:", font_size=16, color=BLUE)
        x_desc = Text("当前输入", font_size=16, color=BLUE)
        x_row = VGroup(x_symbol, x_desc).arrange(RIGHT, buff=0.2)
        
        ab_symbol = MathTex(r"\bar{A}_t, \bar{B}_t:", font_size=16, color=GREEN)
        ab_desc = Text("离散化参数", font_size=16, color=GREEN)
        ab_row = VGroup(ab_symbol, ab_desc).arrange(RIGHT, buff=0.2)
        
        formula_explanation = VGroup(h_row, x_row, ab_row).arrange(
            DOWN, buff=0.3, aligned_edge=LEFT
        ).next_to(core_formula, DOWN, buff=0.5)
        
        self.play(Write(formula_explanation))

        self.wait(1.5)
        self.play(FadeOut(core_formula), FadeOut(formula_explanation))

        # 2. 选择性机制介绍
        selection_title = Text("核心创新：输入依赖的选择性", font_size=32).move_to(
            UP * 2
        )
        self.play(Write(selection_title))

        # 选择性公式
        selection_formula = MathTex(
            r"\mathbf{B}_t, \mathbf{C}_t, \Delta_t = \text{Linear}(x_t)",
            font_size=24,
            color=ORANGE,
        ).move_to(UP * 1.2)
        self.play(Write(selection_formula))

        # 展示动态参数生成
        self.show_dynamic_parameters()

        self.wait(1.5)
        self.play(FadeOut(selection_title), FadeOut(selection_formula))

        # 3. 离散化过程
        discretization_title = Text("离散化：连续 → 数字", font_size=32).move_to(UP * 2)
        self.play(Write(discretization_title))

        # 连续系统公式
        continuous_eq1 = MathTex(
            r"h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)",
            font_size=20,
            color=GREEN,
        ).move_to(UP * 0.5)
        
        continuous_eq2 = MathTex(
            r"y(t) = \mathbf{C}h(t)",
            font_size=20,
            color=GREEN,
        ).next_to(continuous_eq1, DOWN)
        
        continuous_formula = VGroup(continuous_eq1, continuous_eq2)
        self.play(Write(continuous_formula))

        # 离散化步骤
        self.show_discretization_process()

        self.wait(1.5)
        self.play(FadeOut(discretization_title), FadeOut(continuous_formula))

        # 4. 递归扫描过程
        scan_title = Text("递归扫描：状态更新", font_size=32).move_to(UP * 2)
        self.play(Write(scan_title))

        self.show_recursive_scan()

        self.wait(1.5)
        self.play(FadeOut(scan_title))

        # 5. 与Attention对比
        comparison_title = Text("与Attention的关键差异", font_size=32).move_to(UP * 2)
        self.play(Write(comparison_title))

        self.show_attention_comparison()

        self.wait(3)

    def show_dynamic_parameters(self):
        """展示动态参数生成"""
        # 输入序列
        inputs = ["I", "love", "AI", "models", "!"]
        input_group = VGroup()

        for i, token in enumerate(inputs):
            token_box = Rectangle(height=0.6, width=1.2, color=BLUE, fill_opacity=0.5)
            token_text = Text(token, font_size=20, color=WHITE).move_to(token_box)
            input_group.add(VGroup(token_box, token_text))

        input_group.arrange(RIGHT, buff=0.3).move_to(UP * 0.5)
        self.play(FadeIn(input_group))

        # 参数生成器
        param_generator = Rectangle(
            height=1, width=4, color=PURPLE, fill_opacity=0.3
        ).move_to(DOWN * 0.5)
        param_text = Text("Linear Layer", font_size=18, color=WHITE).move_to(
            param_generator
        )

        self.play(FadeIn(param_generator), Write(param_text))

        # 动态生成Δ, B, C
        for i, token in enumerate(inputs):
            # 高亮当前token
            highlight = SurroundingRectangle(input_group[i][0], color=YELLOW, buff=0.1)
            self.play(Create(highlight))

            # 生成参数
            delta = np.random.uniform(0.1, 1.0)
            b_param = np.random.randn(2, 4)
            c_param = np.random.randn(2, 4)

            # 展示参数值
            delta_text = Text(f"Δ={delta:.2f}", font_size=16, color=ORANGE).move_to(
                DOWN * 1.5
            )
            b_text = Text(f"B={b_param[0][0]:.1f}", font_size=16, color=GREEN).next_to(
                delta_text, RIGHT
            )
            c_text = Text(f"C={c_param[0][0]:.1f}", font_size=16, color=YELLOW).next_to(
                b_text, RIGHT
            )

            self.play(Write(delta_text), Write(b_text), Write(c_text))

            # 清理
            self.play(FadeOut(VGroup(delta_text, b_text, c_text)), FadeOut(highlight))

            if i < len(inputs) - 1:
                self.wait(0.3)

        self.wait(1)
        self.play(FadeOut(VGroup(input_group, param_generator, param_text)))

    def show_discretization_process(self):
        """展示离散化过程"""
        # 创建连续系统可视化（左侧）
        continuous_box = Rectangle(
            height=1.8, width=2.5, color=GREEN, fill_opacity=0.2
        ).move_to(LEFT * 4 + DOWN * 0.5)
        continuous_label = Text("连续系统", font_size=16, color=GREEN).next_to(
            continuous_box, UP, buff=0.2
        )

        A_text = Text("A", font_size=18, color=WHITE).move_to(
            continuous_box.get_center() + UP * 0.3
        )
        B_text = Text("B", font_size=18, color=WHITE).move_to(
            continuous_box.get_center() + DOWN * 0.3
        )

        self.play(
            FadeIn(continuous_box), Write(continuous_label), Write(A_text), Write(B_text)
        )

        # 离散化公式（中间，方框外）
        bar_A = MathTex(r"\bar{A} = e^{\Delta A}", font_size=18, color=YELLOW).move_to(UP * 0.8)
        bar_B = MathTex(r"\bar{B} \approx \Delta B", font_size=18, color=YELLOW).move_to(UP * 0.3)
        
        # 步长Δ（放在公式下方）
        delta_text = Text("Δ = 0.1", font_size=18, color=ORANGE).move_to(DOWN * 0.3)
        
        self.play(Write(bar_A), Write(bar_B))
        self.wait(0.5)
        self.play(Write(delta_text))

        # 离散化系统（右侧）
        discrete_box = Rectangle(
            height=1.8, width=2.5, color=ORANGE, fill_opacity=0.2
        ).move_to(RIGHT * 1.5 + DOWN * 0.5)
        discrete_label = Text("离散系统", font_size=16, color=ORANGE).next_to(
            discrete_box, UP, buff=0.2
        )

        discrete_A = Text("ĀA", font_size=18, color=WHITE).move_to(
            discrete_box.get_center() + UP * 0.3
        )
        discrete_B = Text("ĀB", font_size=18, color=WHITE).move_to(
            discrete_box.get_center() + DOWN * 0.3
        )

        self.play(
            FadeIn(discrete_box),
            Write(discrete_label),
            Write(discrete_A),
            Write(discrete_B),
        )

        # 箭头表示转换
        arrow = Arrow(continuous_box.get_right(), discrete_box.get_left(), color=YELLOW)
        self.play(Create(arrow))

        self.wait(1.5)
        self.play(
            FadeOut(
                VGroup(
                    continuous_box,
                    continuous_label,
                    A_text,
                    B_text,
                    delta_text,
                    bar_A,
                    bar_B,
                    discrete_box,
                    discrete_label,
                    discrete_A,
                    discrete_B,
                    arrow,
                )
            )
        )

    def show_recursive_scan(self):
        """展示递归扫描过程"""
        # 初始化状态
        h_initial = np.array([[0.1, 0.2], [0.3, 0.4]])
        h_current = h_initial.copy()

        # 状态容器
        state_container = Rectangle(
            height=2.5, width=2.5, color=YELLOW, fill_opacity=0.2
        ).move_to(ORIGIN)
        state_label = Text("隐状态 h", font_size=18, color=YELLOW).next_to(
            state_container, UP, buff=0.2
        )

        self.play(FadeIn(state_container), Write(state_label))

        # 展示递归过程
        x_inputs = [np.array([0.5, 0.3]), np.array([0.8, 0.6]), np.array([0.3, 0.9])]
        
        # 初始状态显示
        state_value = Text(
            f"h_0 = [0.39, 0.80]",
            font_size=18,
            color=WHITE,
        ).move_to(state_container)
        self.play(Write(state_value))

        for t, x_input in enumerate(x_inputs):
            # 当前输入（顶部）
            input_text = Text(
                f"x_{t+1} = [{x_input[0]:.1f}, {x_input[1]:.1f}]",
                font_size=18,
                color=BLUE,
            ).to_edge(UP, buff=1.0)
            self.play(Write(input_text))
            self.wait(0.3)

            # 更新状态
            h_new = h_current * 0.9 + x_input.reshape(2, 1) * 0.5
            h_current = h_new

            # 展示状态变化
            new_state_value = Text(
                f"h_{t+1} = [{h_current[0,0]:.2f}, {h_current[1,0]:.2f}]",
                font_size=18,
                color=WHITE,
            ).move_to(state_container)
            self.play(Transform(state_value, new_state_value))

            # 清理输入
            self.play(FadeOut(input_text))
            if t < len(x_inputs) - 1:
                self.wait(0.4)

        self.wait(1)
        self.play(FadeOut(VGroup(state_container, state_label, state_value)))

    def show_attention_comparison(self):
        """展示与Attention的对比"""
        # 创建对比表格（缩小间距）
        comparison_table = VGroup()

        # 表头
        headers = VGroup(
            Text("特性", font_size=18, color=YELLOW),
            Text("Mamba", font_size=18, color=GREEN),
            Text("Attention", font_size=18, color=RED),
        ).arrange(RIGHT, buff=1.2)

        # 计算复杂度
        complexity_row = VGroup(
            Text("计算复杂度", font_size=15, color=WHITE),
            Text("O(L·d)", font_size=15, color=GREEN),
            Text("O(L²·d)", font_size=15, color=RED),
        ).arrange(RIGHT, buff=1.2)
        complexity_row.next_to(headers, DOWN, buff=0.4)

        # 内存占用
        memory_row = VGroup(
            Text("推理内存", font_size=15, color=WHITE),
            Text("O(d_state·d)", font_size=14, color=GREEN),
            Text("O(L·d)", font_size=15, color=RED),
        ).arrange(RIGHT, buff=1.2)
        memory_row.next_to(complexity_row, DOWN, buff=0.4)

        # 选择性
        selectivity_row = VGroup(
            Text("选择性", font_size=15, color=WHITE),
            Text("✓ 输入依赖", font_size=15, color=GREEN),
            Text("✗ 静态权重", font_size=15, color=RED),
        ).arrange(RIGHT, buff=1.2)
        selectivity_row.next_to(memory_row, DOWN, buff=0.4)

        comparison_table.add(headers, complexity_row, memory_row, selectivity_row)
        comparison_table.move_to(UP * 0.5)  # 向上移动避免被遮挡

        self.play(FadeIn(comparison_table))

        # 关键优势说明（移到底部）
        advantages = Text(
            "Mamba优势：长序列友好，固定内存", 
            font_size=16, 
            color=YELLOW
        ).to_edge(DOWN, buff=0.8)
        
        self.play(Write(advantages))

        self.wait(2)
        
        # 清除对比表格
        self.play(FadeOut(comparison_table), FadeOut(advantages))

        # 创建演示场景
        # Input tokens（缩小尺寸）
        input_tokens = ["x1", "x2", "x3", "x4", "x5"]
        input_group = VGroup()
        for token in input_tokens:
            token_box = Rectangle(height=0.5, width=0.7, color=BLUE, fill_opacity=0.3)
            token_text = Text(token, font_size=16, color=WHITE).move_to(token_box.get_center())
            input_group.add(VGroup(token_box, token_text))
        input_group.arrange(RIGHT, buff=0.25).move_to(LEFT * 3.5 + UP * 1.2)
        input_label = Text("Key Info  Input Sequence", font_size=14, color=BLUE).next_to(input_group, UP, buff=0.2)
        
        # Parameter generator
        param_box = Rectangle(height=0.9, width=1.3, color=PURPLE, fill_opacity=0.3).move_to(
            LEFT * 0.5 + DOWN * 0.5
        )
        param_text = Text("Linear Layer", font_size=14, color=WHITE).move_to(param_box.get_center())

        # Tank (状态容器)
        tank = Rectangle(height=2, width=1.4, fill_opacity=0.1, color=BLUE, stroke_width=2).move_to(
            RIGHT * 3.5 + DOWN * 0.3
        )
        tank_label = Text("h (state)", font_size=14, color=BLUE).next_to(tank, UP, buff=0.15)
        
        # Fluid (当前状态)
        fluid = Rectangle(height=0.2, width=1.3, fill_opacity=0.8, color=YELLOW, stroke_width=0)
        fluid.align_to(tank, DOWN)
        
        # Valve (选择性机制)
        valve = Triangle(fill_opacity=1, color=GRAY).scale(0.25).rotate(PI)
        valve.move_to(tank.get_top() + UP * 0.8)

        # 连线
        # Input -> Param
        arrow_1 = Arrow(input_group.get_bottom(), param_box.get_top(), color=GRAY)
        # Param -> Valve (控制信号)
        arrow_2 = Arrow(param_box.get_bottom(), valve.get_top(), color=PURPLE)
        # Valve -> Tank (液体流)
        pipe = Line(valve.get_bottom(), tank.get_top(), color=GRAY, stroke_width=2)

        self.play(
            Create(input_group),
            Write(input_label),
            Create(tank),
            Write(tank_label),
            FadeIn(fluid),
            Create(param_box),
            Write(param_text),
            Create(valve),
            Create(arrow_1),
            Create(arrow_2),
            Create(pipe),
        )
        self.wait(0.5)

        # ==========================================
        # 动画循环
        # ==========================================

        # 定义 Token 类型
        token_infos = [
            {"type": "Noise", "color": RED, "delta": -0.2},
            {"type": "Key Info", "color": GREEN, "delta": 1.0},
            {"type": "Noise", "color": RED, "delta": -0.2},
            {"type": "Noise", "color": RED, "delta": -0.2},
            {"type": "Key Info", "color": GREEN, "delta": 0.8},
        ]

        current_h = 0.2

        # 指示器 (Scanner)
        scanner = SurroundingRectangle(input_group[0], color=WHITE)
        self.play(Create(scanner))

        for i, info in enumerate(token_infos):
            # 1. 扫描移动
            target_token = input_group[i]

            # 更新连线位置
            new_arrow_1 = Arrow(
                target_token.get_bottom(), param_box.get_top(), color=BLUE
            )

            self.play(
                scanner.animate.move_to(target_token),
                ReplacementTransform(arrow_1, new_arrow_1),
                run_time=0.3,
            )
            arrow_1 = new_arrow_1  # update ref

            # 2. 显示 Token 类型
            type_text = Text(info["type"], font_size=16, color=info["color"]).next_to(
                target_token, UP
            )
            self.play(FadeIn(type_text), run_time=0.2)

            # 3. 参数生成器“思考” (闪烁)
            self.play(param_box.animate.set_fill(info["color"], 0.5), run_time=0.2)

            # 4. 阀门动作
            # 绿色=开，红色=关
            valve_scale = 1.5 if info["color"] == GREEN else 0.6
            flow_width = 8 if info["color"] == GREEN else 1

            self.play(
                valve.animate.set_color(info["color"]).scale(valve_scale),
                param_box.animate.set_fill(PURPLE, 0.2),  # 恢复
                run_time=0.2,
            )

            # 5. 液体流入与状态更新
            # 流动的线
            flow = Line(
                valve.get_bottom(),
                fluid.get_top(),
                color=info["color"],
                stroke_width=flow_width,
            )
            self.play(FadeIn(flow), run_time=0.1)

            # 计算新高度 (模拟遗忘和输入)
            # h_new = A*h + B*x
            # 简单模拟: Noise 会让水位下降(遗忘), Key 会让水位上升
            next_h = max(0.1, min(1.9, current_h + info["delta"]))

            new_fluid = Rectangle(
                height=next_h, width=1.5, fill_opacity=0.8, color=YELLOW, stroke_width=0
            )
            new_fluid.align_to(tank, DOWN)

            self.play(Transform(fluid, new_fluid), run_time=0.5)

            # 6. 清理
            self.play(
                FadeOut(flow),
                FadeOut(type_text),
                valve.animate.set_color(GRAY).scale(1 / valve_scale),  # 阀门复位
                run_time=0.2,
            )
            current_h = next_h

        # 总结
        summary = Text(
            "Result: Only useful info is compressed into h", font_size=30, color=YELLOW
        ).next_to(tank, RIGHT, buff=1)
        self.play(Write(summary))
        self.wait(2)
