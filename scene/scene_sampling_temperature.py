"""
解码策略与采样可视化场景
=======================

对应笔记: 9.Inference_Sampling.md
生成命令: manim scene_sampling_temperature.py SamplingTemperature -qh
输出视频: assets/SamplingTemperature.mp4

内容要点:
- Temperature调节概率分布
- Top-k采样硬截断
- Top-p采样动态截断
- 不同策略对比总结
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class SamplingTemperature(Scene):
    def construct(self):
        # 标题
        title = Text("解码策略与采样", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # 1. 问题引入
        intro_text = Text(
            "模型输出logits，如何选择下一个词？", font_size=28, color=YELLOW
        ).move_to(UP * 1.5)
        self.play(Write(intro_text))
        self.wait(1)

        # 2. Logits示例
        self.play(FadeOut(intro_text))

        logits_title = Text("模型输出的Logits示例", font_size=32).move_to(UP * 2)
        self.play(Write(logits_title))

        # 创建logits可视化
        tokens = ["the", "cat", "sat", "on", "mat", "dog", "ran"]
        logits_values = [2.1, 1.8, 1.2, 0.8, 0.5, 0.3, 0.1]

        logits_visual = self.create_logits_visualization(
            tokens, logits_values, "Logits"
        )
        logits_visual.move_to(ORIGIN)

        self.play(FadeIn(logits_visual))
        self.wait(1.5)

        # 3. Temperature调节
        self.play(FadeOut(logits_title))

        temp_title = Text("Temperature: 熵的调节器", font_size=32).move_to(UP * 2)
        self.play(Write(temp_title))

        # Temperature公式
        temp_formula = MathTex(
            r"P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}",
            font_size=28,
            color=YELLOW,
        ).move_to(UP * 1)
        self.play(Write(temp_formula))

        self.wait(1)

        # 4. 不同温度下的分布变化
        self.play(FadeOut(temp_title), FadeOut(temp_formula), FadeOut(logits_visual))

        dist_title = Text("不同Temperature下的概率分布", font_size=32).move_to(UP * 2)
        self.play(Write(dist_title))

        # 展示不同温度
        self.show_temperature_effects(tokens, logits_values)

        # 5. Top-k采样
        self.play(FadeOut(dist_title))

        topk_title = Text("Top-k采样: 硬截断", font_size=32).move_to(UP * 2)
        self.play(Write(topk_title))

        self.show_topk_sampling(tokens, logits_values)

        # 6. Top-p采样
        self.play(FadeOut(topk_title))

        topp_title = Text("Top-p采样: 动态截断", font_size=32).move_to(UP * 2)
        self.play(Write(topp_title))

        self.show_topp_sampling(tokens, logits_values)

        # 7. 策略对比总结
        self.play(FadeOut(topp_title))

        summary_title = Text("解码策略对比", font_size=32).move_to(UP * 2)
        self.play(Write(summary_title))

        self.compare_strategies()

        self.wait(3)

    def create_logits_visualization(self, tokens, values, label=""):
        """创建logits可视化"""
        group = VGroup()

        # 标题
        if label:
            title_text = Text(label, font_size=20, color=YELLOW).move_to(UP * 1.5)
            group.add(title_text)

        # 创建条形图
        bars = VGroup()
        max_val = max(values)

        for i, (token, val) in enumerate(zip(tokens, values)):
            height = (val / max_val) * 2
            bar = Rectangle(height=height, width=0.8, color=BLUE, fill_opacity=0.7)
            bar.move_to(DOWN * i * 0.5 + DOWN * 1.5)

            # 数值标签
            val_text = Text(f"{val:.1f}", font_size=14, color=WHITE).next_to(bar, RIGHT)

            # token标签
            token_text = Text(token, font_size=16, color=WHITE).next_to(
                bar, LEFT, buff=0.3
            )

            bars.add(VGroup(bar, val_text, token_text))

        group.add(bars)
        return group

    def create_probability_visualization(self, tokens, probabilities, label=""):
        """创建概率分布可视化"""
        group = VGroup()

        # 标题
        if label:
            title_text = Text(label, font_size=20, color=YELLOW).move_to(UP * 1.5)
            group.add(title_text)

        # 创建条形图
        bars = VGroup()
        max_prob = max(probabilities)

        for i, (token, prob) in enumerate(zip(tokens, probabilities)):
            height = (prob / max_prob) * 2 if max_prob > 0 else 0
            color = self.get_probability_color(prob)
            bar = Rectangle(height=height, width=0.8, color=color, fill_opacity=0.7)
            bar.move_to(DOWN * i * 0.5 + DOWN * 1.5)

            # 概率标签
            prob_text = Text(f"{prob:.3f}", font_size=14, color=WHITE).next_to(
                bar, RIGHT
            )

            # token标签
            token_text = Text(token, font_size=16, color=WHITE).next_to(
                bar, LEFT, buff=0.3
            )

            bars.add(VGroup(bar, prob_text, token_text))

        group.add(bars)
        return group

    def get_probability_color(self, prob):
        """根据概率获取颜色"""
        if prob > 0.5:
            return RED
        elif prob > 0.2:
            return ORANGE
        elif prob > 0.1:
            return YELLOW
        else:
            return GREEN

    def show_temperature_effects(self, tokens, logits_values):
        """展示不同温度的效果"""
        temperatures = [0.1, 1.0, 2.0]
        temp_labels = ["低温 (T=0.1)", "标准 (T=1.0)", "高温 (T=2.0)"]

        for i, (temp, label) in enumerate(zip(temperatures, temp_labels)):
            # 计算softmax概率
            probs = self.softmax(np.array(logits_values) / temp)

            # 创建可视化
            prob_visual = self.create_probability_visualization(tokens, probs, label)
            prob_visual.move_to(LEFT * 4 + RIGHT * i * 4)

            if i == 0:
                self.play(FadeIn(prob_visual))
            else:
                self.play(FadeIn(prob_visual))

        self.wait(2)

        # 温度效果说明
        explanation = VGroup(
            Text("低温: 分布尖锐，选择确定性高", font_size=16, color=BLUE).move_to(
                DOWN * 1
            ),
            Text("高温: 分布平坦，增加随机性", font_size=16, color=RED).move_to(
                DOWN * 1.5
            ),
        )

        self.play(Write(explanation))
        self.wait(1.5)

        # 清理
        self.play(FadeOut(VGroup(prob_visual, explanation)))

    def show_topk_sampling(self, tokens, logits_values):
        """展示Top-k采样"""
        # 计算概率
        probs = self.softmax(np.array(logits_values))

        # 展示原始分布
        original_dist = self.create_probability_visualization(tokens, probs, "原始分布")
        original_dist.move_to(LEFT * 2)

        self.play(FadeIn(original_dist))

        # Top-k=3的截断
        k = 3
        topk_indices = np.argsort(probs)[-k:]

        # 创建Top-k分布
        topk_probs = np.zeros_like(probs)
        topk_probs[topk_indices] = probs[topk_indices]
        topk_probs = topk_probs / np.sum(topk_probs)  # 重新归一化

        topk_dist = self.create_probability_visualization(tokens, topk_probs, "Top-k=3")
        topk_dist.move_to(RIGHT * 2)

        # 高亮被选中的tokens
        highlights = VGroup()
        for i, idx in enumerate(topk_indices):
            bar = original_dist[1][i][0]  # 获取条形图
            highlight = SurroundingRectangle(bar, color=YELLOW, buff=0.05)
            highlights.add(highlight)

        self.play(Create(highlights), FadeIn(topk_dist))

        # 说明文字
        explanation = Text(
            "只保留概率最高的k个token，重新归一化", font_size=18, color=YELLOW
        ).move_to(DOWN * 2)
        self.play(Write(explanation))

        self.wait(2)

        # 清理
        self.play(FadeOut(VGroup(original_dist, highlights, topk_dist, explanation)))

    def show_topp_sampling(self, tokens, logits_values):
        """展示Top-p采样"""
        # 计算概率
        probs = self.softmax(np.array(logits_values))

        # 展示原始分布
        original_dist = self.create_probability_visualization(tokens, probs, "原始分布")
        original_dist.move_to(LEFT * 2)

        self.play(FadeIn(original_dist))

        # Top-p=0.9的截断
        p_threshold = 0.9
        sorted_indices = np.argsort(probs)[::-1]  # 降序
        cumulative_probs = np.cumsum(probs[sorted_indices])

        # 找到最小的满足累积概率>=p的集合
        topp_indices = sorted_indices[cumulative_probs <= p_threshold]
        if len(topp_indices) < len(sorted_indices):
            topp_indices = np.append(topp_indices, sorted_indices[len(topp_indices)])

        # 创建Top-p分布
        topp_probs = np.zeros_like(probs)
        topp_probs[topp_indices] = probs[topp_indices]
        topp_probs = topp_probs / np.sum(topp_probs)  # 重新归一化

        topp_dist = self.create_probability_visualization(
            tokens, topp_probs, "Top-p=0.9"
        )
        topp_dist.move_to(RIGHT * 2)

        # 高亮被选中的tokens
        highlights = VGroup()
        for i, idx in enumerate(topp_indices):
            bar = original_dist[1][i][0]  # 获取条形图
            highlight = SurroundingRectangle(bar, color=YELLOW, buff=0.05)
            highlights.add(highlight)

        self.play(Create(highlights), FadeIn(topp_dist))

        # 说明文字
        explanation = Text(
            "动态选择token数量，使累积概率达到p", font_size=18, color=YELLOW
        ).move_to(DOWN * 2)
        self.play(Write(explanation))

        self.wait(2)

        # 清理
        self.play(FadeOut(VGroup(original_dist, highlights, topp_dist, explanation)))

    def compare_strategies(self):
        """比较不同策略"""
        # 创建对比表格
        table = VGroup()

        # 表头
        headers = VGroup(
            Text("策略", font_size=20, color=YELLOW),
            Text("Temperature", font_size=20, color=YELLOW),
            Text("特点", font_size=20, color=YELLOW),
            Text("适用场景", font_size=20, color=YELLOW),
        ).arrange(RIGHT, buff=1.2)

        # 贪婪搜索
        greedy_row = VGroup(
            Text("贪婪搜索", font_size=18, color=RED),
            Text("T≈0", font_size=16, color=WHITE),
            Text("确定性", font_size=16, color=WHITE),
            Text("数学解题", font_size=16, color=WHITE),
        ).arrange(RIGHT, buff=1.2)
        greedy_row.next_to(headers, DOWN, buff=0.5)

        # Top-k采样
        topk_row = VGroup(
            Text("Top-k", font_size=18, color=GREEN),
            Text("T=0.7", font_size=16, color=WHITE),
            Text("固定候选集", font_size=16, color=WHITE),
            Text("平衡场景", font_size=16, color=WHITE),
        ).arrange(RIGHT, buff=1.2)
        topk_row.next_to(greedy_row, DOWN, buff=0.5)

        # Top-p采样
        topp_row = VGroup(
            Text("Top-p", font_size=18, color=BLUE),
            Text("T=0.9", font_size=16, color=WHITE),
            Text("动态候选集", font_size=16, color=WHITE),
            Text("创意写作", font_size=16, color=WHITE),
        ).arrange(RIGHT, buff=1.2)
        topp_row.next_to(topk_row, DOWN, buff=0.5)

        # 添加表格线
        h_lines = VGroup()
        for i in range(4):
            y_pos = UP * 0.5 - DOWN * i * 0.6
            h_line = Line(LEFT * 3.5, RIGHT * 3.5, color=GRAY).move_to(y_pos)
            h_lines.add(h_line)

        v_lines = VGroup()
        for i in range(5):
            x_pos = LEFT * 3.5 + RIGHT * i * 1.7
            v_line = Line(x_pos + UP * 0.5, x_pos + DOWN * 1.8, color=GRAY)
            v_lines.add(v_line)

        table.add(headers, greedy_row, topk_row, topp_row, h_lines, v_lines)
        table.center()

        self.play(FadeIn(table))

        # 总结
        summary = Text(
            "Temperature控制创造性，Top-k/p控制质量", font_size=20, color=YELLOW
        ).move_to(DOWN * 2.5)

        self.play(Write(summary))
        self.wait(2)

    def softmax(self, x):
        """计算softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


if __name__ == "__main__":
    scene = SamplingTemperature()
    scene.render()
