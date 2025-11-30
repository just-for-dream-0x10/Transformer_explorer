"""
BPE分词算法详解可视化场景
=========================

对应笔记: 8.Tokinization.md
生成命令: manim scene_bpe_detailed.py BPEDetailed -qh
输出视频: assets/BPEDetailed.mp4

内容要点:
- BPE核心思想与迭代过程
- 字符对频率统计与合并
- 最终词表构建
- 与其他分词算法对比
- GPT分词实际示例
"""

from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200


class BPEDetailed(Scene):
    def construct(self):
        # 标题
        title = Text("BPE分词算法详解", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # 1. BPE核心思想
        concept_text = Text(
            "BPE: 迭代合并最频繁的字符对，最大化压缩率", font_size=28, color=YELLOW
        ).move_to(UP * 1.5)
        self.play(Write(concept_text))
        self.wait(1)

        # 2. 初始语料展示
        self.play(FadeOut(concept_text))

        corpus_title = Text("初始语料库", font_size=32).move_to(UP * 2)
        self.play(Write(corpus_title))

        # 语料示例
        corpus_examples = (
            VGroup(
                Text('"hug" × 10', font_size=24, color=BLUE),
                Text('"pug" × 5', font_size=24, color=GREEN),
                Text('"pun" × 12', font_size=24, color=RED),
                Text('"bun" × 4', font_size=24, color=YELLOW),
            )
            .arrange(DOWN, buff=0.5)
            .move_to(UP * 0.5)
        )

        self.play(FadeIn(corpus_examples))
        self.wait(1)

        # 3. 初始分词状态 - 先清理语料库示例
        self.play(FadeOut(corpus_title), FadeOut(corpus_examples))

        initial_title = Text("初始分词状态 (字符级别)", font_size=32).move_to(UP * 2)
        self.play(Write(initial_title))

        # 展示初始分词
        initial_tokens = self.create_token_visualization(
            ["h u g ×10", "p u g ×5", "p u n ×12", "b u n ×4"]
        )
        initial_tokens.move_to(ORIGIN)

        self.play(FadeIn(initial_tokens))
        self.wait(1)

        # 4. BPE迭代过程 - 清理初始分词
        self.play(FadeOut(initial_title), FadeOut(initial_tokens))

        iteration_title = Text("BPE迭代过程", font_size=32).move_to(UP * 2)
        self.play(Write(iteration_title))

        # 迭代1: 合并 u+n
        self.run_bpe_iteration(
            "迭代1",
            "u n",
            "un",
            ["h u g ×10", "p u g ×5", "p un ×12", "b un ×4"],
            "最高频: 16次",
        )

        # 迭代2: 合并 p+un
        self.run_bpe_iteration(
            "迭代2",
            "p un",
            "pun",
            ["h u g ×10", "p u g ×5", "pun ×12", "b un ×4"],
            "最高频: 12次",
        )

        # 迭代3: 合并 h+ug
        self.run_bpe_iteration(
            "迭代3",
            "h u g",
            "hug",
            ["hug ×10", "p u g ×5", "pun ×12", "b un ×4"],
            "最高频: 10次",
        )

        # 迭代4: 合并 b+un
        self.run_bpe_iteration(
            "迭代4",
            "b un",
            "bun",
            ["hug ×10", "p u g ×5", "pun ×12", "bun ×4"],
            "最高频: 4次",
        )

        self.wait(1)

        # 5. 最终词表
        self.play(FadeOut(iteration_title))

        vocab_title = Text("最终词表示例", font_size=32).move_to(UP * 2)
        self.play(Write(vocab_title))

        # 展示最终词表
        final_vocab = (
            VGroup(
                Text("基础字符: h, u, g, p, n, b", font_size=20, color=BLUE),
                Text("合并tokens: un, pun, hug, bun", font_size=20, color=GREEN),
                Text("总计: 10个tokens", font_size=20, color=YELLOW),
            )
            .arrange(DOWN, buff=0.5)
            .move_to(ORIGIN)
        )

        self.play(FadeIn(final_vocab))
        self.wait(1.5)

        # 6. BPE vs 其他算法对比
        self.play(FadeOut(VGroup(vocab_title, final_vocab)))

        compare_title = Text("分词算法对比", font_size=32).move_to(UP * 2)
        self.play(Write(compare_title))

        self.compare_tokenization_algorithms()

        # 7. 实际应用示例
        self.play(FadeOut(compare_title))

        practical_title = Text("实际应用: GPT如何分词", font_size=32).move_to(UP * 2)
        self.play(Write(practical_title))

        self.show_gpt_tokenization()

        self.wait(3)

    def create_token_visualization(self, token_strings):
        """创建分词可视化"""
        group = VGroup()

        for token_str in token_strings:
            # 解析字符串
            parts = token_str.split(" ")
            tokens = parts[:-1]  # 除了最后的计数
            count = parts[-1]  # 计数部分

            # 创建token组
            token_group = VGroup()
            for i, token in enumerate(tokens):
                if token in ["h", "u", "g", "p", "n", "b"]:
                    # 单字符
                    token_box = Rectangle(
                        height=0.6, width=0.5, color=BLUE, fill_opacity=0.3
                    )
                else:
                    # 合并token
                    token_box = Rectangle(
                        height=0.6, width=0.8, color=GREEN, fill_opacity=0.5
                    )

                token_text = Text(token, font_size=16, color=WHITE).move_to(token_box)
                token_box.move_to(RIGHT * i * 0.6)
                token_group.add(VGroup(token_box, token_text))

            # 添加计数
            count_text = Text(count, font_size=16, color=YELLOW).next_to(
                token_group, RIGHT, buff=0.3
            )
            token_group.add(count_text)

            group.add(token_group)

        group.arrange(DOWN, buff=0.5)
        return group

    def run_bpe_iteration(
        self, iteration_name, merge_pair, new_token, result_tokens, frequency_text
    ):
        """运行BPE迭代"""
        # 迭代标题
        iter_text = Text(iteration_name, font_size=24, color=YELLOW).move_to(UP * 1)
        self.play(Write(iter_text))

        # 合并说明
        merge_text = Text(
            f"合并: '{merge_pair}' → '{new_token}'", font_size=20, color=GREEN
        ).move_to(UP * 0.3)
        self.play(Write(merge_text))

        # 频率说明
        freq_text = Text(frequency_text, font_size=18, color=ORANGE).move_to(DOWN * 0.3)
        self.play(Write(freq_text))

        # 展示结果
        result_visual = self.create_token_visualization(result_tokens)
        result_visual.move_to(DOWN * 1.5)

        self.play(FadeIn(result_visual))
        self.wait(1.5)

        # 清理
        self.play(FadeOut(VGroup(iter_text, merge_text, freq_text, result_visual)))

    def compare_tokenization_algorithms(self):
        """比较不同分词算法"""
        # 创建对比表格
        table = VGroup()

        # 表头
        headers = VGroup(
            Text("算法", font_size=20, color=YELLOW),
            Text("策略", font_size=20, color=YELLOW),
            Text("方向", font_size=20, color=YELLOW),
            Text("代表模型", font_size=20, color=YELLOW),
        ).arrange(RIGHT, buff=1.5)

        # BPE行
        bpe_row = VGroup(
            Text("BPE", font_size=18, color=GREEN),
            Text("频率最高", font_size=16, color=WHITE),
            Text("自底向上", font_size=16, color=WHITE),
            Text("GPT, LLaMA", font_size=16, color=WHITE),
        ).arrange(RIGHT, buff=1.5)
        bpe_row.next_to(headers, DOWN, buff=0.5)

        # WordPiece行
        wp_row = VGroup(
            Text("WordPiece", font_size=18, color=BLUE),
            Text("互信息最高", font_size=16, color=WHITE),
            Text("自底向上", font_size=16, color=WHITE),
            Text("BERT", font_size=16, color=WHITE),
        ).arrange(RIGHT, buff=1.5)
        wp_row.next_to(bpe_row, DOWN, buff=0.5)

        # Unigram行
        unigram_row = VGroup(
            Text("Unigram LM", font_size=18, color=RED),
            Text("损失最小", font_size=16, color=WHITE),
            Text("自顶向下", font_size=16, color=WHITE),
            Text("T5, LLaMA", font_size=16, color=WHITE),
        ).arrange(RIGHT, buff=1.5)
        unigram_row.next_to(wp_row, DOWN, buff=0.5)

        # 添加表格线
        h_line1 = Line(LEFT * 4, RIGHT * 4, color=GRAY).next_to(headers, DOWN, buff=0.2)
        h_line2 = Line(LEFT * 4, RIGHT * 4, color=GRAY).next_to(bpe_row, DOWN, buff=0.2)
        h_line3 = Line(LEFT * 4, RIGHT * 4, color=GRAY).next_to(wp_row, DOWN, buff=0.2)

        v_lines = VGroup()
        for i in range(5):
            x_pos = LEFT * 4 + RIGHT * i * 2
            v_line = Line(x_pos + UP * 0.5, x_pos + DOWN * 2.5, color=GRAY)
            v_lines.add(v_line)

        table.add(
            headers, bpe_row, wp_row, unigram_row, h_line1, h_line2, h_line3, v_lines
        )
        table.center()

        self.play(FadeIn(table))
        self.wait(2)

        # 清理
        self.play(FadeOut(table))

    def show_gpt_tokenization(self):
        """展示GPT分词示例"""
        # 示例文本（顶部）
        example_text = Text('示例: "strawberry"', font_size=28, color=BLUE).to_edge(UP, buff=1.0)
        self.play(Write(example_text))
        self.wait(0.5)

        # 分词结果（上半部分）
        tokenization_title = Text("GPT-4分词结果:", font_size=24, color=YELLOW).move_to(UP * 1.5)
        self.play(Write(tokenization_title))
        
        token_result = Text("str → aw → berry", font_size=24, color=GREEN).move_to(UP * 0.8)
        self.play(Write(token_result))
        
        token_ids = Text("IDs: [496, 675, 5666]", font_size=20, color=WHITE).move_to(UP * 0.2)
        self.play(Write(token_ids))
        self.wait(1)

        # 可视化向量（中间部分，清理上面的详细信息）
        self.play(FadeOut(VGroup(tokenization_title, token_result, token_ids)))
        
        vector_title = Text("向量表示", font_size=22, color=YELLOW).move_to(UP * 0.5)
        self.play(Write(vector_title))
        
        vectors = VGroup()
        for i, token in enumerate(["str", "aw", "berry"]):
            vector = Rectangle(
                height=1.8, width=0.7, color=GREEN, fill_opacity=0.6
            ).move_to(LEFT * 2.5 + RIGHT * i * 2.5 + DOWN * 0.8)
            vector_text = Text(token, font_size=16, color=WHITE).move_to(vector)
            vectors.add(VGroup(vector, vector_text))

        self.play(FadeIn(vectors))
        self.wait(1)

        # 关键说明（底部，分两行显示）
        self.play(FadeOut(vector_title))
        
        key_point1 = Text(
            "模型看到的是 3 个向量，不是 9 个字母!", 
            font_size=20, 
            color=RED
        ).move_to(DOWN * 2.3)
        
        key_point2 = Text(
            "这就是为什么 GPT 数不清 'strawberry' 里的 'r'!", 
            font_size=20, 
            color=YELLOW
        ).move_to(DOWN * 2.9)

        self.play(Write(key_point1))
        self.wait(0.8)
        self.play(Write(key_point2))
        self.wait(2)


if __name__ == "__main__":
    scene = BPEDetailed()
    scene.render()
