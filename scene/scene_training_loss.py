"""
训练过程与损失函数可视化场景
===========================

对应笔记: 10.Training_Essentials.md
生成命令: manim scene_training_loss.py TrainingLoss -qh
输出视频: assets/TrainingLoss.mp4

内容要点:
- Next Token Prediction目标
- 交叉熵损失计算
- Teacher Forcing并行训练
- 损失函数可视化
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class TrainingLoss(Scene):
    def construct(self):
        # 标题
        title = Text("LLM训练：Next Token Prediction", font_size=40).to_edge(
            UP, buff=0.5
        )
        self.play(Write(title))
        self.wait(0.5)

        # 1. 训练目标
        objective_text = Text(
            "目标：给定前文，预测下一个词", font_size=28, color=YELLOW
        ).move_to(UP * 1.5)
        self.play(Write(objective_text))
        self.wait(1)

        # 2. 训练数据示例
        self.play(FadeOut(objective_text))

        data_title = Text("训练数据：Teacher Forcing", font_size=32).move_to(UP * 2)
        self.play(Write(data_title))

        # 输入序列
        input_text = Text("输入: [I, love, AI]", font_size=24, color=BLUE).move_to(
            UP * 0.5
        )
        self.play(Write(input_text))

        # 目标序列
        target_text = Text(
            "目标: [love, AI, models]", font_size=24, color=GREEN
        ).move_to(ORIGIN)
        self.play(Write(target_text))

        # 说明
        explanation = Text(
            "模型需要同时预测3个位置的下一个词", font_size=20, color=GRAY
        ).move_to(DOWN * 0.5)
        self.play(Write(explanation))

        self.wait(1.5)

        # 3. 交叉熵损失详解
        self.play(FadeOut(VGroup(data_title, input_text, target_text, explanation)))

        loss_title = Text("交叉熵损失 (Cross-Entropy Loss)", font_size=32).move_to(
            UP * 2
        )
        self.play(Write(loss_title))

        # 交叉熵公式
        ce_formula = MathTex(
            r"\mathcal{L} = -\sum_{v \in V} y_v \log(p_v)", font_size=28, color=YELLOW
        ).move_to(UP * 1)
        self.play(Write(ce_formula))

        # 简化公式
        simple_formula = MathTex(
            r"\mathcal{L} = -\log(p_{\text{target}})", font_size=24, color=GREEN
        ).move_to(UP * 0.3)
        self.play(Write(simple_formula))

        self.wait(1)

        # 4. 损失计算示例
        self.play(FadeOut(VGroup(loss_title, ce_formula, simple_formula)))

        example_title = Text("损失计算示例", font_size=32).move_to(UP * 2)
        self.play(Write(example_title))

        # 展示一个具体的预测例子
        self.show_loss_calculation_example()

        # 5. 并行计算优势
        self.play(FadeOut(example_title))

        parallel_title = Text("Transformer的并行训练优势", font_size=32).move_to(UP * 2)
        self.play(Write(parallel_title))

        self.show_parallel_training()

        # 6. 损失函数可视化
        self.play(FadeOut(parallel_title))

        viz_title = Text("损失函数可视化", font_size=32).move_to(UP * 2)
        self.play(Write(viz_title))

        self.visualize_loss_function()

        # 7. 训练过程总结
        self.play(FadeOut(viz_title))

        summary_title = Text("训练过程总结", font_size=32).move_to(UP * 2)
        self.play(Write(summary_title))

        self.summarize_training_process()

        self.wait(3)

    def show_loss_calculation_example(self):
        """展示损失计算示例"""
        # 上下文
        context = Text("上下文: 'I love'", font_size=20, color=BLUE).move_to(UP * 1)
        self.play(Write(context))

        # 目标词
        target = Text("目标词: 'AI'", font_size=20, color=GREEN).move_to(UP * 0.5)
        self.play(Write(target))

        # 模型输出的概率分布
        vocab = ["the", "AI", "model", "computer", "data", "science"]
        probabilities = [0.1, 0.6, 0.15, 0.05, 0.07, 0.03]

        # 创建概率分布可视化
        prob_visual = self.create_probability_distribution(vocab, probabilities)
        prob_visual.move_to(DOWN * 0.5)

        self.play(FadeIn(prob_visual))

        # 高亮目标词 ("AI" 是第二个词，索引为1)
        target_highlight = SurroundingRectangle(
            prob_visual[0][1][0], color=YELLOW, buff=0.1
        )
        self.play(Create(target_highlight))

        # 计算损失
        loss_value = -np.log(0.6)
        loss_calculation = Text(
            f"Loss = -log(0.6) = {loss_value:.3f}", font_size=22, color=RED
        ).move_to(DOWN * 2)
        self.play(Write(loss_calculation))

        # 说明
        explanation = Text("概率越高，损失越低", font_size=18, color=GRAY).move_to(
            DOWN * 2.5
        )
        self.play(Write(explanation))

        self.wait(2)

        # 清理
        self.play(
            FadeOut(
                VGroup(
                    context,
                    target,
                    prob_visual,
                    target_highlight,
                    loss_calculation,
                    explanation,
                )
            )
        )

    def create_probability_distribution(self, tokens, probs):
        """创建概率分布可视化"""
        group = VGroup()

        bars = VGroup()
        max_prob = max(probs)

        for i, (token, prob) in enumerate(zip(tokens, probs)):
            height = (prob / max_prob) * 1.5
            color = RED if token == "AI" else BLUE
            bar = Rectangle(height=height, width=0.6, color=color, fill_opacity=0.7)
            bar.move_to(RIGHT * i * 0.8 + DOWN * 0.5)

            # 概率标签
            prob_text = Text(f"{prob:.2f}", font_size=14, color=WHITE).next_to(bar, UP)

            # token标签
            token_text = Text(token, font_size=12, color=WHITE).move_to(bar)

            bar_group = VGroup(bar, prob_text, token_text)
            bars.add(bar_group)

        group.add(bars)
        return group

    def show_parallel_training(self):
        """展示并行训练优势"""
        # RNN vs Transformer对比

        # RNN部分
        rnn_title = Text("RNN (串行)", font_size=24, color=RED).move_to(
            LEFT * 3 + UP * 1
        )
        self.play(Write(rnn_title))

        # RNN时间步
        rnn_steps = VGroup()
        for i in range(3):
            step = Rectangle(
                height=0.6, width=1.5, color=RED, fill_opacity=0.5
            ).move_to(LEFT * 3 + DOWN * i * 0.8)
            step_text = Text(f"Step {i+1}", font_size=16, color=WHITE).move_to(step)
            rnn_steps.add(VGroup(step, step_text))

        # RNN连接箭头
        rnn_arrows = VGroup()
        for i in range(2):
            arrow = Arrow(
                rnn_steps[i].get_bottom(), rnn_steps[i + 1].get_top(), color=YELLOW
            )
            rnn_arrows.add(arrow)

        self.play(FadeIn(rnn_steps), Create(rnn_arrows))

        # Transformer部分
        transformer_title = Text(
            "Transformer (并行)", font_size=24, color=GREEN
        ).move_to(RIGHT * 3 + UP * 1)
        self.play(Write(transformer_title))

        # Transformer并行层
        transformer_layers = VGroup()
        for i in range(3):
            layer = Rectangle(
                height=0.6, width=1.5, color=GREEN, fill_opacity=0.5
            ).move_to(RIGHT * 3 + DOWN * 0.5)
            layer_text = Text(f"Pos {i+1}", font_size=16, color=WHITE).move_to(layer)
            transformer_layers.add(VGroup(layer, layer_text))

        # 并行处理示意
        parallel_arrows = VGroup()
        for layer in transformer_layers:
            arrow = Arrow(UP * 1.5, layer.get_top(), color=YELLOW)
            parallel_arrows.add(arrow)

        self.play(FadeIn(transformer_layers), Create(parallel_arrows))

        # 优势说明
        advantage_text = Text(
            "Transformer可以同时处理所有位置，训练速度大幅提升",
            font_size=18,
            color=YELLOW,
        ).move_to(DOWN * 2.5)

        self.play(Write(advantage_text))
        self.wait(2)

        # 清理
        self.play(
            FadeOut(
                VGroup(
                    rnn_title,
                    rnn_steps,
                    rnn_arrows,
                    transformer_title,
                    transformer_layers,
                    parallel_arrows,
                    advantage_text,
                )
            )
        )

    def visualize_loss_function(self):
        """可视化损失函数"""
        # 创建坐标系，调整y轴范围以适应损失函数
        axes = Axes(
            x_range=[0.01, 1, 0.2],
            y_range=[0, 5, 1],
            x_length=6,
            y_length=4,
            axis_config={"color": WHITE},
            x_axis_config={"numbers_to_include": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
            y_axis_config={"numbers_to_include": [0, 1, 2, 3, 4, 5]},
        ).move_to(ORIGIN)

        x_label = Text("预测概率", font_size=16).next_to(axes.x_axis, RIGHT)
        y_label = Text("损失值", font_size=16).next_to(axes.y_axis, UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # 交叉熵损失曲线，限制最小概率避免无穷大
        def cross_entropy_loss(p):
            p = np.maximum(p, 0.01)  # 限制最小概率为0.01
            return -np.log(p)

        loss_curve = axes.plot(cross_entropy_loss, color=RED, stroke_width=3)
        self.play(Create(loss_curve))

        # 标记特殊点
        good_point = Dot(color=GREEN).move_to(axes.c2p(0.9, cross_entropy_loss(0.9)))
        good_label = Text("预测准确\n损失小", font_size=14, color=GREEN).next_to(
            good_point, RIGHT
        )

        bad_point = Dot(color=RED).move_to(axes.c2p(0.1, cross_entropy_loss(0.1)))
        bad_label = Text("预测错误\n损失大", font_size=14, color=RED).next_to(
            bad_point, LEFT
        )

        self.play(Create(good_point), Write(good_label))
        self.play(Create(bad_point), Write(bad_label))

        self.wait(2)

        # 清理
        self.play(
            FadeOut(
                VGroup(
                    axes,
                    x_label,
                    y_label,
                    loss_curve,
                    good_point,
                    good_label,
                    bad_point,
                    bad_label,
                )
            )
        )

    def summarize_training_process(self):
        """总结训练过程"""
        # 训练流程图
        flow_chart = VGroup()

        # 输入
        input_box = Rectangle(
            height=0.8, width=2, color=BLUE, fill_opacity=0.5
        ).move_to(UP * 1.5)
        input_text = Text("输入序列", font_size=16, color=WHITE).move_to(input_box)

        # 模型
        model_box = Rectangle(
            height=0.8, width=2, color=GREEN, fill_opacity=0.5
        ).move_to(UP * 0.5)
        model_text = Text("Transformer", font_size=16, color=WHITE).move_to(model_box)

        # 输出
        output_box = Rectangle(
            height=0.8, width=2, color=YELLOW, fill_opacity=0.5
        ).move_to(DOWN * 0.5)
        output_text = Text("Logits", font_size=16, color=BLACK).move_to(output_box)

        # 损失计算
        loss_box = Rectangle(height=0.8, width=2, color=RED, fill_opacity=0.5).move_to(
            DOWN * 1.5
        )
        loss_text = Text("交叉熵损失", font_size=16, color=WHITE).move_to(loss_box)

        # 连接箭头
        arrows = VGroup(
            Arrow(input_box.get_bottom(), model_box.get_top(), color=WHITE),
            Arrow(model_box.get_bottom(), output_box.get_top(), color=WHITE),
            Arrow(output_box.get_bottom(), loss_box.get_top(), color=WHITE),
        )

        flow_chart.add(
            input_box,
            input_text,
            model_box,
            model_text,
            output_box,
            output_text,
            loss_box,
            loss_text,
            arrows,
        )

        self.play(FadeIn(flow_chart))

        # 关键点说明
        key_points = (
            VGroup(
                Text(
                    "• Teacher Forcing: 并行计算所有位置的损失",
                    font_size=16,
                    color=YELLOW,
                ),
                Text(
                    "• 因果掩码: 确保每个位置只能看到前面的词",
                    font_size=16,
                    color=YELLOW,
                ),
                Text(
                    "• 交叉熵: 衡量预测分布与真实分布的差异", font_size=16, color=YELLOW
                ),
                Text("• 反向传播: 根据损失更新模型参数", font_size=16, color=YELLOW),
            )
            .arrange(DOWN, buff=0.3)
            .move_to(RIGHT * 3)
        )

        self.play(Write(key_points))

        self.wait(2)

        # 清理
        self.play(FadeOut(VGroup(flow_chart, key_points)))


if __name__ == "__main__":
    scene = TrainingLoss()
    scene.render()
