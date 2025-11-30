"""
AdamW优化器可视化场景
=====================

对应笔记: 10.Training_Essentials.md (优化器部分)
生成命令: manim scene_adamw_optimizer.py AdamWOptimizer -qh
输出视频: assets/AdamWOptimizer.mp4

内容要点:
- Adam权重衰减的问题
- AdamW解耦权重衰减
- 更新步骤详解
- 训练效果对比
- 超参数建议
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class AdamWOptimizer(Scene):
    def construct(self):
        # 标题
        title = Text("AdamW优化器：解耦权重衰减", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # 1. 为什么需要AdamW
        intro_text = Text(
            "标准Adam对权重衰减的处理有数学缺陷", font_size=28, color=YELLOW
        ).move_to(UP * 1.5)
        self.play(Write(intro_text))
        self.wait(1)

        # 2. Adam的问题
        self.play(FadeOut(intro_text))

        problem_title = Text("Adam的问题：权重衰减与梯度耦合", font_size=32).move_to(
            UP * 2
        )
        self.play(Write(problem_title))

        # Adam更新公式（有问题）
        adam_formula = MathTex(
            r"g_t \leftarrow \nabla f(\theta_{t-1}) + \lambda \theta_{t-1}",
            font_size=24,
            color=RED,
        ).move_to(UP * 1)

        problem_text = Text(
            "问题：权重衰减被加入梯度，受动量和方差影响", font_size=20, color=RED
        ).move_to(UP * 0.3)

        self.play(Write(adam_formula), Write(problem_text))
        self.wait(1.5)

        # 3. AdamW解决方案
        self.play(FadeOut(VGroup(problem_title, adam_formula, problem_text)))

        solution_title = Text("AdamW：解耦权重衰减", font_size=32).move_to(UP * 2)
        self.play(Write(solution_title))

        # AdamW更新步骤
        self.show_adamw_steps()

        # 4. 可视化对比
        self.play(FadeOut(solution_title))

        compare_title = Text("Adam vs AdamW 对比", font_size=32).move_to(UP * 2)
        self.play(Write(compare_title))

        self.compare_adam_adamw()

        # 5. 实际效果
        self.play(FadeOut(compare_title))

        effect_title = Text("AdamW的实际效果", font_size=32).move_to(UP * 2)
        self.play(Write(effect_title))

        self.show_adamw_effects()

        # 6. 超参数建议
        self.play(FadeOut(effect_title))

        hyperparam_title = Text("超参数建议", font_size=32).move_to(UP * 2)
        self.play(Write(hyperparam_title))

        self.show_hyperparameters()

        self.wait(3)

    def show_adamw_steps(self):
        """展示AdamW更新步骤"""
        # 步骤1：计算梯度
        step1 = Text("步骤1: 计算梯度", font_size=20, color=BLUE).move_to(UP * 1.5)
        gradient_formula = MathTex(
            r"g_t = \nabla f(\theta_{t-1})", font_size=20, color=BLUE
        ).move_to(UP * 1.0)
        self.play(Write(step1), Write(gradient_formula))
        self.wait(0.5)

        # 步骤2：更新动量
        step2 = Text("步骤2: 更新动量", font_size=20, color=GREEN).move_to(UP * 0.2)
        momentum_formula = MathTex(
            r"m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t", font_size=20, color=GREEN
        ).move_to(DOWN * 0.3)
        self.play(Write(step2), Write(momentum_formula))
        self.wait(0.5)

        # 步骤3：更新方差
        step3 = Text("步骤3: 更新方差", font_size=20, color=YELLOW).move_to(DOWN * 1.1)
        variance_formula = MathTex(
            r"v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2", font_size=20, color=YELLOW
        ).move_to(DOWN * 1.6)
        self.play(Write(step3), Write(variance_formula))
        self.wait(0.5)

        # 步骤4：Adam更新（不包含权重衰减）
        step4 = Text("步骤4: Adam更新", font_size=20, color=ORANGE).move_to(DOWN * 2.4)
        adam_update = MathTex(
            r"\theta'_t = \theta_{t-1} - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}",
            font_size=18,
            color=ORANGE,
        ).move_to(DOWN * 2.9)
        self.play(Write(step4), Write(adam_update))
        self.wait(0.5)

        # 步骤5：权重衰减（独立）
        step5 = Text("步骤5: 权重衰减（独立）", font_size=20, color=RED).move_to(
            DOWN * 3.7
        )
        weight_decay = MathTex(
            r"\theta_t = \theta'_t - \eta \lambda \theta_{t-1}", font_size=18, color=RED
        ).move_to(DOWN * 4.2)
        self.play(Write(step5), Write(weight_decay))
        self.wait(0.5)

        # 关键点
        key_point = Text(
            "关键：权重衰减独立于梯度统计量！", font_size=18, color=YELLOW
        ).to_edge(DOWN, buff=0.3)
        self.play(Write(key_point))

        self.wait(2)

        # 清理
        self.play(
            FadeOut(
                VGroup(
                    step1,
                    gradient_formula,
                    step2,
                    momentum_formula,
                    step3,
                    variance_formula,
                    step4,
                    adam_update,
                    step5,
                    weight_decay,
                    key_point,
                )
            )
        )

    def compare_adam_adamw(self):
        """对比Adam和AdamW"""
        # 创建对比图
        # Adam部分
        adam_title = Text("Adam", font_size=24, color=RED).move_to(LEFT * 3 + UP * 1)
        self.play(Write(adam_title))

        # Adam流程
        adam_flow = VGroup()

        # 参数
        adam_param = Rectangle(
            height=0.6, width=1.2, color=BLUE, fill_opacity=0.5
        ).move_to(LEFT * 3 + UP * 0)
        adam_param_text = Text("θ", font_size=16, color=WHITE).move_to(adam_param)

        # 梯度+权重衰减
        adam_grad = Rectangle(
            height=0.6, width=1.2, color=RED, fill_opacity=0.5
        ).move_to(LEFT * 3 + DOWN * 0.8)
        adam_grad_text = Text("g+λθ", font_size=14, color=WHITE).move_to(adam_grad)

        # 动量方差
        adam_mv = Rectangle(
            height=0.6, width=1.2, color=GREEN, fill_opacity=0.5
        ).move_to(LEFT * 3 + DOWN * 1.6)
        adam_mv_text = Text("m,v", font_size=14, color=WHITE).move_to(adam_mv)

        # 更新
        adam_update = Rectangle(
            height=0.6, width=1.2, color=YELLOW, fill_opacity=0.5
        ).move_to(LEFT * 3 + DOWN * 2.4)
        adam_update_text = Text("θ'", font_size=16, color=BLACK).move_to(adam_update)

        adam_flow.add(
            adam_param,
            adam_param_text,
            adam_grad,
            adam_grad_text,
            adam_mv,
            adam_mv_text,
            adam_update,
            adam_update_text,
        )

        # AdamW部分
        adamw_title = Text("AdamW", font_size=24, color=GREEN).move_to(
            RIGHT * 3 + UP * 1
        )
        self.play(Write(adamw_title))

        # AdamW流程
        adamw_flow = VGroup()

        # 参数
        adamw_param = Rectangle(
            height=0.6, width=1.2, color=BLUE, fill_opacity=0.5
        ).move_to(RIGHT * 3 + UP * 0)
        adamw_param_text = Text("θ", font_size=16, color=WHITE).move_to(adamw_param)

        # 梯度（纯梯度）
        adamw_grad = Rectangle(
            height=0.6, width=1.2, color=RED, fill_opacity=0.5
        ).move_to(RIGHT * 3 + DOWN * 0.8)
        adamw_grad_text = Text("g", font_size=14, color=WHITE).move_to(adamw_grad)

        # 动量方差
        adamw_mv = Rectangle(
            height=0.6, width=1.2, color=GREEN, fill_opacity=0.5
        ).move_to(RIGHT * 3 + DOWN * 1.6)
        adamw_mv_text = Text("m,v", font_size=14, color=WHITE).move_to(adamw_mv)

        # Adam更新
        adamw_adam_update = Rectangle(
            height=0.6, width=1.2, color=YELLOW, fill_opacity=0.5
        ).move_to(RIGHT * 3 + DOWN * 2.4)
        adamw_adam_text = Text("θ'", font_size=16, color=BLACK).move_to(
            adamw_adam_update
        )

        # 权重衰减（独立）
        adamw_decay = Rectangle(
            height=0.6, width=1.2, color=ORANGE, fill_opacity=0.5
        ).move_to(RIGHT * 3 + DOWN * 3.2)
        adamw_decay_text = Text("-ηλθ", font_size=14, color=BLACK).move_to(adamw_decay)

        # 最终参数
        adamw_final = Rectangle(
            height=0.6, width=1.2, color=PURPLE, fill_opacity=0.5
        ).move_to(RIGHT * 3 + DOWN * 4.0)
        adamw_final_text = Text("θ", font_size=16, color=WHITE).move_to(adamw_final)

        adamw_flow.add(
            adamw_param,
            adamw_param_text,
            adamw_grad,
            adamw_grad_text,
            adamw_mv,
            adamw_mv_text,
            adamw_adam_update,
            adamw_adam_text,
            adamw_decay,
            adamw_decay_text,
            adamw_final,
            adamw_final_text,
        )

        # 连接箭头
        adam_arrows = VGroup(
            Arrow(adam_param.get_bottom(), adam_grad.get_top(), color=WHITE),
            Arrow(adam_grad.get_bottom(), adam_mv.get_top(), color=WHITE),
            Arrow(adam_mv.get_bottom(), adam_update.get_top(), color=WHITE),
        )

        adamw_arrows = VGroup(
            Arrow(adamw_param.get_bottom(), adamw_grad.get_top(), color=WHITE),
            Arrow(adamw_grad.get_bottom(), adamw_mv.get_top(), color=WHITE),
            Arrow(adamw_mv.get_bottom(), adamw_adam_update.get_top(), color=WHITE),
            Arrow(adamw_adam_update.get_bottom(), adamw_decay.get_top(), color=WHITE),
            Arrow(adamw_decay.get_bottom(), adamw_final.get_top(), color=WHITE),
        )

        self.play(FadeIn(VGroup(adam_flow, adamw_flow, adam_arrows, adamw_arrows)))

        # 优势说明
        advantage_text = Text(
            "AdamW: 权重衰减独立，正则化效果更好", font_size=18, color=YELLOW
        ).move_to(DOWN * 5)

        self.play(Write(advantage_text))
        self.wait(2)

        # 清理
        self.play(
            FadeOut(
                VGroup(
                    adam_title,
                    adam_flow,
                    adam_arrows,
                    adamw_title,
                    adamw_flow,
                    adamw_arrows,
                    advantage_text,
                )
            )
        )

    def show_adamw_effects(self):
        """展示AdamW的实际效果"""
        # 创建训练曲线对比
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 2, 0.5],
            x_length=6,
            y_length=4,
            axis_config={"color": WHITE},
            x_axis_config={"numbers_to_include": [0, 20, 40, 60, 80, 100]},
            y_axis_config={"numbers_to_include": [0, 0.5, 1.0, 1.5, 2.0]},
        ).move_to(ORIGIN)

        x_label = Text("训练步数", font_size=16).next_to(axes.x_axis, RIGHT)
        y_label = Text("损失值", font_size=16).next_to(axes.y_axis, UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Adam曲线（收敛较慢）
        def adam_loss(x):
            return 1.8 * np.exp(-x / 50) + 0.3

        adam_curve = axes.plot(adam_loss, color=RED, stroke_width=3)
        adam_label = Text("Adam", font_size=18, color=RED).next_to(adam_curve, RIGHT)

        # AdamW曲线（收敛更快）
        def adamw_loss(x):
            return 1.6 * np.exp(-x / 35) + 0.2

        adamw_curve = axes.plot(adamw_loss, color=GREEN, stroke_width=3)
        adamw_label = Text("AdamW", font_size=18, color=GREEN).next_to(
            adamw_curve, RIGHT
        )

        self.play(Create(adam_curve), Write(adam_label))
        self.play(Create(adamw_curve), Write(adamw_label))

        # 效果说明
        effects = (
            VGroup(
                Text("• 收敛速度更快", font_size=16, color=GREEN),
                Text("• 最终损失更低", font_size=16, color=GREEN),
                Text("• 泛化能力更好", font_size=16, color=GREEN),
            )
            .arrange(DOWN, buff=0.3)
            .move_to(RIGHT * 3)
        )

        self.play(Write(effects))

        self.wait(2)

        # 清理
        self.play(
            FadeOut(
                VGroup(
                    axes,
                    x_label,
                    y_label,
                    adam_curve,
                    adam_label,
                    adamw_curve,
                    adamw_label,
                    effects,
                )
            )
        )

    def show_hyperparameters(self):
        """展示超参数建议"""
        # 创建参数表格（缩小尺寸）
        table = VGroup()

        # 表头
        headers = VGroup(
            Text("参数", font_size=18, color=YELLOW),
            Text("推荐值", font_size=18, color=YELLOW),
            Text("作用", font_size=18, color=YELLOW),
        ).arrange(RIGHT, buff=1.2)

        # 学习率
        lr_row = VGroup(
            Text("学习率 η", font_size=16, color=WHITE),
            Text("1e-4 ~ 3e-4", font_size=14, color=WHITE),
            Text("控制更新步长", font_size=14, color=WHITE),
        ).arrange(RIGHT, buff=1.2)
        lr_row.next_to(headers, DOWN, buff=0.4)

        # Beta1
        beta1_row = VGroup(
            Text("β₁", font_size=16, color=WHITE),
            Text("0.9", font_size=14, color=WHITE),
            Text("动量衰减率", font_size=14, color=WHITE),
        ).arrange(RIGHT, buff=1.2)
        beta1_row.next_to(lr_row, DOWN, buff=0.4)

        # Beta2
        beta2_row = VGroup(
            Text("β₂", font_size=16, color=WHITE),
            Text("0.95 ~ 0.999", font_size=14, color=WHITE),
            Text("方差衰减率", font_size=14, color=WHITE),
        ).arrange(RIGHT, buff=1.2)
        beta2_row.next_to(beta1_row, DOWN, buff=0.4)

        # 权重衰减
        wd_row = VGroup(
            Text("权重衰减 λ", font_size=16, color=WHITE),
            Text("0.01 ~ 0.1", font_size=14, color=WHITE),
            Text("正则化强度", font_size=14, color=WHITE),
        ).arrange(RIGHT, buff=1.2)
        wd_row.next_to(beta2_row, DOWN, buff=0.4)

        # Epsilon
        eps_row = VGroup(
            Text("ε", font_size=16, color=WHITE),
            Text("1e-8", font_size=14, color=WHITE),
            Text("数值稳定性", font_size=14, color=WHITE),
        ).arrange(RIGHT, buff=1.2)
        eps_row.next_to(wd_row, DOWN, buff=0.4)

        # 添加表格线
        h_lines = VGroup()
        for i in range(6):
            y_pos = UP * 0.8 - DOWN * i * 0.4
            h_line = Line(LEFT * 2.5, RIGHT * 2.5, color=GRAY).move_to(y_pos)
            h_lines.add(h_line)

        v_lines = VGroup()
        for i in range(4):
            x_pos = LEFT * 2.5 + RIGHT * i * 1.7
            v_line = Line(x_pos + UP * 0.8, x_pos + DOWN * 1.2, color=GRAY)
            v_lines.add(v_line)

        table.add(
            headers, lr_row, beta1_row, beta2_row, wd_row, eps_row, h_lines, v_lines
        )
        table.move_to(UP * 0.5)  # 向上移动避免被遮挡

        self.play(FadeIn(table))

        # 注意事项（缩小并调整位置）
        notes = (
            VGroup(
                Text("• 大模型常用较大的权重衰减 (0.1)", font_size=14, color=YELLOW),
                Text("• 学习率调度器配合使用效果更佳", font_size=14, color=YELLOW),
                Text("• 不同任务可能需要调整超参数", font_size=14, color=YELLOW),
            )
            .arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            .to_edge(DOWN, buff=0.5)
        )

        self.play(Write(notes))

        self.wait(2)


if __name__ == "__main__":
    scene = AdamWOptimizer()
    scene.render()
