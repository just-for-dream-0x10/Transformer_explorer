"""
混合精度训练可视化场景
=====================

对应笔记: 10.Training_Essentials.md (精度部分)
生成命令: manim scene_mixed_precision.py MixedPrecision -qh
输出视频: assets/MixedPrecision.mp4

内容要点:
- 数值精度格式对比
- FP16的问题与限制
- BF16的优势
- 混合精度训练流程
- 硬件支持与性能
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class MixedPrecision(Scene):
    def construct(self):
        # 标题
        title = Text("混合精度训练：FP16 vs BF16", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # 1. 为什么需要混合精度
        intro_text = Text(
            "混合精度：在有限显存中训练更大模型", font_size=28, color=YELLOW
        ).move_to(UP * 1.5)
        self.play(Write(intro_text))
        self.wait(1)

        # 2. 精度格式对比
        self.play(FadeOut(intro_text))

        precision_title = Text("数值精度格式对比", font_size=32).move_to(UP * 2)
        self.play(Write(precision_title))

        self.compare_precision_formats(precision_title)

        # 3. FP16的问题
        # precision_title 已在 compare_precision_formats 中清理

        fp16_title = Text("FP16的问题：数值范围有限", font_size=32).move_to(UP * 2)
        self.play(Write(fp16_title))

        self.show_fp16_problems()

        # 4. BF16的优势
        self.play(FadeOut(fp16_title))

        bf16_title = Text("BF16：现代训练的标准选择", font_size=32).move_to(UP * 2)
        self.play(Write(bf16_title))

        self.show_bf16_advantages()

        # 5. 混合精度训练流程
        self.play(FadeOut(bf16_title))

        workflow_title = Text("混合精度训练流程", font_size=32).move_to(UP * 2)
        self.play(Write(workflow_title))

        self.show_mixed_precision_workflow()

        # 6. 硬件支持
        self.play(FadeOut(workflow_title))

        hardware_title = Text("硬件支持与性能", font_size=32).move_to(UP * 2)
        self.play(Write(hardware_title))

        self.show_hardware_support()

        # 7. 实践建议
        self.play(FadeOut(hardware_title))

        practice_title = Text("实践建议", font_size=32).move_to(UP * 2)
        self.play(Write(practice_title))

        self.show_practical_tips()

        self.wait(3)

    def compare_precision_formats(self, precision_title):
        """对比不同精度格式 - 使用卡片式布局"""
        
        # FP32 卡片
        fp32_card = VGroup(
            Rectangle(height=2, width=2.5, color=BLUE, fill_opacity=0.2),
            Text("FP32", font_size=24, color=BLUE).shift(UP * 0.7),
            Text("符号位: 1", font_size=16, color=WHITE).shift(UP * 0.3),
            Text("指数位: 8", font_size=16, color=WHITE).shift(ORIGIN),
            Text("尾数位: 23", font_size=16, color=WHITE).shift(DOWN * 0.3),
            Text("±3.4×10³⁸", font_size=14, color=YELLOW).shift(DOWN * 0.7),
        ).move_to(LEFT * 3.5)
        
        # FP16 卡片
        fp16_card = VGroup(
            Rectangle(height=2, width=2.5, color=RED, fill_opacity=0.2),
            Text("FP16", font_size=24, color=RED).shift(UP * 0.7),
            Text("符号位: 1", font_size=16, color=WHITE).shift(UP * 0.3),
            Text("指数位: 5", font_size=16, color=WHITE).shift(ORIGIN),
            Text("尾数位: 10", font_size=16, color=WHITE).shift(DOWN * 0.3),
            Text("±6.5×10⁴", font_size=14, color=YELLOW).shift(DOWN * 0.7),
        ).move_to(ORIGIN)
        
        # BF16 卡片
        bf16_card = VGroup(
            Rectangle(height=2, width=2.5, color=GREEN, fill_opacity=0.2),
            Text("BF16", font_size=24, color=GREEN).shift(UP * 0.7),
            Text("符号位: 1", font_size=16, color=WHITE).shift(UP * 0.3),
            Text("指数位: 8", font_size=16, color=WHITE).shift(ORIGIN),
            Text("尾数位: 7", font_size=16, color=WHITE).shift(DOWN * 0.3),
            Text("±3.4×10³⁸", font_size=14, color=YELLOW).shift(DOWN * 0.7),
        ).move_to(RIGHT * 3.5)
        
        cards = VGroup(fp32_card, fp16_card, bf16_card).move_to(UP * 0.2)
        
        self.play(FadeIn(cards))
        
        # 关键对比说明
        comparison_note = Text(
            "BF16 = FP32的数值范围 + FP16的内存占用",
            font_size=18,
            color=YELLOW
        ).to_edge(DOWN, buff=0.8)
        
        self.play(Write(comparison_note))
        self.wait(2)

        # 清理卡片和标题
        self.play(FadeOut(VGroup(cards, comparison_note, precision_title)))
        
        # 可视化位数分配
        self.visualize_bit_allocation()

        self.wait(2)

    def visualize_bit_allocation(self):
        """可视化位数分配 - 垂直堆叠布局"""
        
        # FP32 可视化（顶部）
        fp32_title = Text("FP32 (32位)", font_size=18, color=BLUE).move_to(UP * 2.5 + LEFT * 4)
        
        fp32_bits = VGroup()
        start_x = LEFT * 5
        # 符号位
        sign_bit = Rectangle(height=0.4, width=0.3, color=RED, fill_opacity=0.8).move_to(start_x + UP * 2)
        # 指数位 (8位)
        exp_bits = VGroup()
        for i in range(8):
            bit = Rectangle(height=0.4, width=0.3, color=GREEN, fill_opacity=0.8).move_to(
                start_x + RIGHT * (i + 1) * 0.35 + UP * 2
            )
            exp_bits.add(bit)
        # 尾数位 (23位)
        mant_bits = VGroup()
        for i in range(23):
            bit = Rectangle(height=0.4, width=0.3, color=YELLOW, fill_opacity=0.8).move_to(
                start_x + RIGHT * (i + 9) * 0.35 + UP * 2
            )
            mant_bits.add(bit)
        
        fp32_bits.add(sign_bit, exp_bits, mant_bits)
        
        # FP32 标签
        fp32_labels = VGroup(
            Text("符号", font_size=12, color=RED).next_to(sign_bit, DOWN, buff=0.1),
            Text("指数位", font_size=12, color=GREEN).next_to(exp_bits, DOWN, buff=0.1),
            Text("尾数位", font_size=12, color=YELLOW).next_to(mant_bits, DOWN, buff=0.1),
        )
        
        self.play(Write(fp32_title), FadeIn(fp32_bits), Write(fp32_labels))
        self.wait(1)
        
        # FP16 可视化（中间）
        fp16_title = Text("FP16 (16位)", font_size=18, color=RED).move_to(UP * 0.3 + LEFT * 4)
        
        fp16_bits = VGroup()
        # 符号位
        sign_bit16 = Rectangle(height=0.4, width=0.3, color=RED, fill_opacity=0.8).move_to(start_x + DOWN * 0.2)
        # 指数位 (5位)
        exp_bits16 = VGroup()
        for i in range(5):
            bit = Rectangle(height=0.4, width=0.3, color=GREEN, fill_opacity=0.8).move_to(
                start_x + RIGHT * (i + 1) * 0.35 + DOWN * 0.2
            )
            exp_bits16.add(bit)
        # 尾数位 (10位)
        mant_bits16 = VGroup()
        for i in range(10):
            bit = Rectangle(height=0.4, width=0.3, color=YELLOW, fill_opacity=0.8).move_to(
                start_x + RIGHT * (i + 6) * 0.35 + DOWN * 0.2
            )
            mant_bits16.add(bit)
        
        fp16_bits.add(sign_bit16, exp_bits16, mant_bits16)
        
        # FP16 标签
        fp16_labels = VGroup(
            Text("符号", font_size=12, color=RED).next_to(sign_bit16, DOWN, buff=0.1),
            Text("指数位", font_size=12, color=GREEN).next_to(exp_bits16, DOWN, buff=0.1),
            Text("尾数位", font_size=12, color=YELLOW).next_to(mant_bits16, DOWN, buff=0.1),
        )
        
        self.play(Write(fp16_title), FadeIn(fp16_bits), Write(fp16_labels))
        self.wait(1)
        
        # BF16 可视化（底部）
        bf16_title = Text("BF16 (16位)", font_size=18, color=GREEN).move_to(DOWN * 1.9 + LEFT * 4)
        
        bf16_bits = VGroup()
        # 符号位
        sign_bitbf = Rectangle(height=0.4, width=0.3, color=RED, fill_opacity=0.8).move_to(start_x + DOWN * 2.4)
        # 指数位 (8位)
        exp_bitsbf = VGroup()
        for i in range(8):
            bit = Rectangle(height=0.4, width=0.3, color=GREEN, fill_opacity=0.8).move_to(
                start_x + RIGHT * (i + 1) * 0.35 + DOWN * 2.4
            )
            exp_bitsbf.add(bit)
        # 尾数位 (7位)
        mant_bitsbf = VGroup()
        for i in range(7):
            bit = Rectangle(height=0.4, width=0.3, color=YELLOW, fill_opacity=0.8).move_to(
                start_x + RIGHT * (i + 9) * 0.35 + DOWN * 2.4
            )
            mant_bitsbf.add(bit)
        
        bf16_bits.add(sign_bitbf, exp_bitsbf, mant_bitsbf)
        
        # BF16 标签
        bf16_labels = VGroup(
            Text("符号", font_size=12, color=RED).next_to(sign_bitbf, DOWN, buff=0.1),
            Text("指数位", font_size=12, color=GREEN).next_to(exp_bitsbf, DOWN, buff=0.1),
            Text("尾数位", font_size=12, color=YELLOW).next_to(mant_bitsbf, DOWN, buff=0.1),
        )
        
        self.play(Write(bf16_title), FadeIn(bf16_bits), Write(bf16_labels))
        self.wait(2)
        
        # 清理
        self.play(
            FadeOut(
                VGroup(
                    fp32_title, fp32_bits, fp32_labels,
                    fp16_title, fp16_bits, fp16_labels,
                    bf16_title, bf16_bits, bf16_labels,
                )
            )
        )

    def show_fp16_problems(self):
        """展示FP16的问题"""
        # 溢出问题（左侧）
        overflow_title = Text("溢出问题", font_size=22, color=RED).move_to(
            LEFT * 3 + UP * 1.5
        )
        self.play(Write(overflow_title))

        # 数值范围可视化
        range_line = NumberLine(
            x_range=[-70000, 70000, 10000],
            length=5,
            color=WHITE,
            include_numbers=True,
            numbers_to_include=[-60000, -30000, 0, 30000, 60000],
            font_size=12,
        ).move_to(LEFT * 3 + UP * 0.3)

        safe_range = Rectangle(
            height=0.3, width=4.5, color=GREEN, fill_opacity=0.3
        ).move_to(range_line)
        overflow_left = Rectangle(
            height=0.3, width=0.4, color=RED, fill_opacity=0.5
        ).move_to(range_line.get_left() + RIGHT * 0.2)
        overflow_right = Rectangle(
            height=0.3, width=0.4, color=RED, fill_opacity=0.5
        ).move_to(range_line.get_right() + LEFT * 0.2)

        self.play(
            Create(range_line),
            FadeIn(safe_range),
            FadeIn(overflow_left),
            FadeIn(overflow_right),
        )

        overflow_text = Text("超出范围 = ±inf", font_size=14, color=RED).move_to(
            LEFT * 3 + DOWN * 0.5
        )
        self.play(Write(overflow_text))

        # 下溢问题（右侧）
        underflow_title = Text("下溢问题", font_size=22, color=ORANGE).move_to(
            RIGHT * 3 + UP * 1.5
        )
        self.play(Write(underflow_title))

        # 小数值可视化
        small_values = (
            VGroup(
                Text("1e-8", font_size=16, color=RED),
                Text("→", font_size=18, color=WHITE),
                Text("0", font_size=16, color=RED),
            )
            .arrange(RIGHT, buff=0.2)
            .move_to(RIGHT * 3 + UP * 0.3)
        )

        underflow_text = Text(
            "小数值被置零，梯度消失", font_size=14, color=ORANGE
        ).move_to(RIGHT * 3 + DOWN * 0.5)

        self.play(FadeIn(small_values), Write(underflow_text))

        # Loss Scaling说明（底部）
        loss_scaling = Text(
            "FP16需要Loss Scaling技术", font_size=16, color=YELLOW
        ).to_edge(DOWN, buff=0.8)
        self.play(Write(loss_scaling))

        self.wait(2)

        # 清理
        self.play(
            FadeOut(
                VGroup(
                    overflow_title,
                    range_line,
                    safe_range,
                    overflow_left,
                    overflow_right,
                    overflow_text,
                    underflow_title,
                    small_values,
                    underflow_text,
                    loss_scaling,
                )
            )
        )

    def show_bf16_advantages(self):
        """展示BF16的优势"""
        # 优势列表
        advantages = (
            VGroup(
                Text("✓ 与FP32相同的数值范围", font_size=20, color=GREEN),
                Text("✓ 无需Loss Scaling", font_size=20, color=GREEN),
                Text("✓ 训练更稳定", font_size=20, color=GREEN),
                Text("✓ 精度损失可接受", font_size=20, color=GREEN),
            )
            .arrange(DOWN, buff=0.5)
            .move_to(UP * 0.5)
        )

        self.play(FadeIn(advantages))

        # 对比图
        comparison = VGroup(
            Text("FP16: 需要复杂处理", font_size=18, color=RED).move_to(
                LEFT * 3 + DOWN * 1.5
            ),
            Text("BF16: 开箱即用", font_size=18, color=GREEN).move_to(
                RIGHT * 3 + DOWN * 1.5
            ),
        )

        self.play(Write(comparison))

        # 硬件支持
        hardware_support = Text(
            "A100/H100/RTX 5090 原生支持BF16", font_size=18, color=YELLOW
        ).move_to(DOWN * 2.5)

        self.play(Write(hardware_support))

        self.wait(2)

        # 清理
        self.play(FadeOut(VGroup(advantages, comparison, hardware_support)))

    def show_mixed_precision_workflow(self):
        """展示混合精度训练流程"""
        # 流程图（左侧，缩小尺寸）
        workflow = VGroup()

        # 前向传播（FP16/BF16）
        forward_box = Rectangle(
            height=0.7, width=2.2, color=BLUE, fill_opacity=0.5
        ).move_to(LEFT * 2 + UP * 1.2)
        forward_text = Text("前向传播 (FP16/BF16)", font_size=14, color=WHITE).move_to(
            forward_box
        )

        # 损失计算（FP32）
        loss_box = Rectangle(
            height=0.7, width=2.2, color=GREEN, fill_opacity=0.5
        ).move_to(LEFT * 2 + UP * 0.3)
        loss_text = Text("损失计算 (FP32)", font_size=14, color=WHITE).move_to(loss_box)

        # 反向传播（FP16/BF16）
        backward_box = Rectangle(
            height=0.7, width=2.2, color=YELLOW, fill_opacity=0.5
        ).move_to(LEFT * 2 + DOWN * 0.6)
        backward_text = Text("反向传播 (FP16/BF16)", font_size=14, color=BLACK).move_to(
            backward_box
        )

        # 参数更新（FP32）
        update_box = Rectangle(
            height=0.7, width=2.2, color=RED, fill_opacity=0.5
        ).move_to(LEFT * 2 + DOWN * 1.5)
        update_text = Text("参数更新 (FP32)", font_size=14, color=WHITE).move_to(
            update_box
        )

        # 连接箭头
        arrows = VGroup(
            Arrow(forward_box.get_bottom(), loss_box.get_top(), color=WHITE, buff=0.05),
            Arrow(loss_box.get_bottom(), backward_box.get_top(), color=WHITE, buff=0.05),
            Arrow(backward_box.get_bottom(), update_box.get_top(), color=WHITE, buff=0.05),
        )

        workflow.add(
            forward_box,
            forward_text,
            loss_box,
            loss_text,
            backward_box,
            backward_text,
            update_box,
            update_text,
            arrows,
        )

        self.play(FadeIn(workflow))

        # 说明文字（底部，避免与流程图重叠）
        memory_saving = Text(
            "内存节省：FP16/BF16参数占FP32的50%", font_size=16, color=YELLOW
        ).to_edge(DOWN, buff=1.2)

        speed_boost = Text(
            "速度提升：Tensor Core加速2-4倍", font_size=16, color=GREEN
        ).to_edge(DOWN, buff=0.6)

        self.play(Write(memory_saving))
        self.play(Write(speed_boost))

        self.wait(2)
        # 清理
        self.play(FadeOut(VGroup(workflow, memory_saving, speed_boost)))

    def show_hardware_support(self):
        """展示硬件支持 - 使用卡片式布局"""
        
        # V100 卡片
        v100_card = VGroup(
            Rectangle(height=1.8, width=2.2, color=BLUE, fill_opacity=0.2),
            Text("V100", font_size=20, color=BLUE).shift(UP * 0.6),
            Text("FP16: ✓", font_size=16, color=GREEN).shift(UP * 0.2),
            Text("BF16: ✗", font_size=16, color=RED).shift(DOWN * 0.1),
            Text("Tensor Core: ✓", font_size=14, color=GREEN).shift(DOWN * 0.5),
        ).move_to(LEFT * 4 + UP * 0.3)
        
        # A100 卡片
        a100_card = VGroup(
            Rectangle(height=1.8, width=2.2, color=GREEN, fill_opacity=0.2),
            Text("A100", font_size=20, color=GREEN).shift(UP * 0.6),
            Text("FP16: ✓", font_size=16, color=GREEN).shift(UP * 0.2),
            Text("BF16: ✓", font_size=16, color=GREEN).shift(DOWN * 0.1),
            Text("Tensor Core: ✓", font_size=14, color=GREEN).shift(DOWN * 0.5),
        ).move_to(LEFT * 1.3 + UP * 0.3)
        
        # H100 卡片
        h100_card = VGroup(
            Rectangle(height=1.8, width=2.2, color=GREEN, fill_opacity=0.2),
            Text("H100", font_size=20, color=GREEN).shift(UP * 0.6),
            Text("FP16: ✓", font_size=16, color=GREEN).shift(UP * 0.2),
            Text("BF16: ✓", font_size=16, color=GREEN).shift(DOWN * 0.1),
            Text("Tensor Core: ✓", font_size=14, color=GREEN).shift(DOWN * 0.5),
        ).move_to(RIGHT * 1.3 + UP * 0.3)
        
        # RTX 5090 卡片
        rtx_card = VGroup(
            Rectangle(height=1.8, width=2.2, color=GREEN, fill_opacity=0.2),
            Text("RTX 5090", font_size=18, color=GREEN).shift(UP * 0.6),
            Text("FP16: ✓", font_size=16, color=GREEN).shift(UP * 0.2),
            Text("BF16: ✓", font_size=16, color=GREEN).shift(DOWN * 0.1),
            Text("Tensor Core: ✓", font_size=14, color=GREEN).shift(DOWN * 0.5),
        ).move_to(RIGHT * 4 + UP * 0.3)
        
        cards = VGroup(v100_card, a100_card, h100_card, rtx_card)
        
        self.play(FadeIn(cards))

        # 推荐建议
        recommendation = Text(
            "推荐：A100/H100/RTX 5090 → 直接用BF16",
            font_size=18,
            color=YELLOW
        ).to_edge(DOWN, buff=0.8)

        self.play(Write(recommendation))

        self.wait(2)

        # 清理
        self.play(FadeOut(VGroup(cards, recommendation)))

    def show_practical_tips(self):
        """展示实践建议"""
        # 实践建议列表
        tips = (
            VGroup(
                Text("1. 现代LLM训练首选BF16", font_size=20, color=GREEN),
                Text("2. 只有用老硬件(V100)时才考虑FP16", font_size=20, color=YELLOW),
                Text("3. 梯度裁剪依然重要(防止梯度爆炸)", font_size=20, color=BLUE),
                Text("4. 监控损失和梯度，确保数值稳定", font_size=20, color=BLUE),
                Text("5. 权重保持FP32，避免精度累积误差", font_size=20, color=RED),
            )
            .arrange(DOWN, buff=0.5)
            .move_to(ORIGIN)
        )

        self.play(FadeIn(tips))

        # 总结
        summary = Text(
            "混合精度 = 更大模型 + 更快训练 + 更低成本", font_size=22, color=YELLOW
        ).move_to(DOWN * 3)

        self.play(Write(summary))

        self.wait(2)


if __name__ == "__main__":
    scene = MixedPrecision()
    scene.render()
