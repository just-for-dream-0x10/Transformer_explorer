from manim import *
import numpy as np

config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class MambaMechanism(Scene):
    def construct(self):
        # 1. 标题
        title = Text("Mamba Core: Input-Dependent Selection", font_size=36).to_edge(UP)
        self.play(Write(title))

        # ==========================================
        # 布局：垂直流 (Top-Down Flow)
        # ==========================================

        # --- TOP: 输入序列 ---
        # 放到屏幕上方
        input_group = VGroup()
        for i in range(5):
            sq = Square(side_length=0.8).set_stroke(BLUE, 2)
            input_group.add(sq)
        input_group.arrange(RIGHT, buff=0.2).to_edge(UP, buff=1.5)

        input_label = Text("Input Sequence (x)", font_size=24, color=BLUE).next_to(
            input_group, UP
        )

        # --- BOTTOM: 隐状态 (水箱) ---
        # 放到屏幕下方居中
        tank = Rectangle(height=2.0, width=1.5, color=WHITE).to_edge(DOWN, buff=1.0)
        tank_label = Text("Hidden State (h)", font_size=24, color=YELLOW).next_to(
            tank, DOWN
        )

        # 液位 (初始很低)
        fluid = Rectangle(
            height=0.2, width=1.5, fill_opacity=0.8, color=YELLOW, stroke_width=0
        )
        fluid.align_to(tank, DOWN)

        # --- MIDDLE: 控制中心 ---
        # SSM 参数生成器
        param_box = Rectangle(height=0.8, width=2.5, color=PURPLE, fill_opacity=0.2)
        param_box.move_to(UP * 0.5)  # 屏幕中心偏上
        param_text = Text("Linear(x) -> Δ, B, C", font_size=20).move_to(param_box)

        # 阀门 (Gate) - 连接 Param 和 Tank
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
