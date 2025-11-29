from manim import *
import numpy as np

# 配置
config.background_color = "#0E1117"
config.pixel_height = 800
config.pixel_width = 1200  # 稍微宽一点，防止拥挤


class AttentionMechanism(Scene):
    def construct(self):
        # ==========================================
        # 1. 核心数据定义
        # ==========================================
        q_data = np.array([[1, 0], [0, 1], [1, 1]])  # 3x2
        k_t_data = np.array([[1, 1, 0], [0, 0, 1]])  # 2x3
        scores_data = np.dot(q_data, k_t_data)  # 3x3

        CELL_SIZE = 0.8  # 格子大小
        BUFF_H = 1.0  # 水平间距
        BUFF_V = 1.0  # 垂直间距

        # ==========================================
        # 2. 辅助函数：创建矩阵图形组
        # ==========================================
        def create_matrix_viz(data, color=WHITE):
            rows, cols = data.shape
            group = VGroup()
            cells = VGroup()

            for i in range(rows):
                for j in range(cols):
                    # 创建格子
                    sq = Square(side_length=CELL_SIZE).set_stroke(GRAY, 1)
                    # 移动到正确位置 (相对于组中心)
                    sq.move_to(RIGHT * j * CELL_SIZE + DOWN * i * CELL_SIZE)

                    # 创建数字
                    val = data[i][j]
                    num = Text(str(val), font_size=24, color=color).move_to(sq)

                    cells.add(VGroup(sq, num))

            group.add(cells)

            # 添加简单的矩阵括号装饰 (上下左右四条线)
            h = rows * CELL_SIZE
            w = cols * CELL_SIZE
            # 左括号
            l_bracket = (
                VGroup(
                    Line(UP * h / 2 + LEFT * 0.1, UP * h / 2 + LEFT * 0.3),
                    Line(UP * h / 2 + LEFT * 0.3, DOWN * h / 2 + LEFT * 0.3),
                    Line(DOWN * h / 2 + LEFT * 0.3, DOWN * h / 2 + LEFT * 0.1),
                )
                .set_stroke(color, 2)
                .move_to(cells.get_left() + LEFT * 0.2)
            )

            # 右括号
            r_bracket = (
                VGroup(
                    Line(UP * h / 2 + RIGHT * 0.1, UP * h / 2 + RIGHT * 0.3),
                    Line(UP * h / 2 + RIGHT * 0.3, DOWN * h / 2 + RIGHT * 0.3),
                    Line(DOWN * h / 2 + RIGHT * 0.3, DOWN * h / 2 + RIGHT * 0.1),
                )
                .set_stroke(color, 2)
                .move_to(cells.get_right() + RIGHT * 0.2)
            )

            group.add(l_bracket, r_bracket)
            return group, cells

        # ==========================================
        # 3. 创建并布局所有元素 (核心修复)
        # ==========================================

        # 创建 Q
        q_group, q_cells = create_matrix_viz(q_data, BLUE)
        q_label = Text("Q", font_size=40, color=BLUE).next_to(q_group, LEFT, buff=0.5)

        # 创建 K^T
        k_group, k_cells = create_matrix_viz(k_t_data, GREEN)
        k_label = Text("K_T", font_size=40, color=GREEN).next_to(k_group, UP, buff=0.3)

        # 创建 Score (空网格)
        s_rows, s_cols = scores_data.shape
        score_group = VGroup()
        score_cells = VGroup()  # 用于动画索引

        # 先生成 Score 网格，为了对齐，我们基于 Q 和 K 的位置
        # 这里先占位，后面统一移动
        for i in range(s_rows):
            for j in range(s_cols):
                sq = Square(side_length=CELL_SIZE).set_stroke(DARK_GRAY, 1)
                # 初始都设为0位置，后面移
                num = Text("", font_size=24).move_to(sq)
                cell = VGroup(sq, num)
                score_cells.add(cell)
                score_group.add(cell)

        # --- 布局逻辑 (Layout Logic) ---
        # 1. 把 Q 放在屏幕左边偏下
        q_group.move_to(LEFT * 3 + DOWN * 1)
        q_label.next_to(q_group, LEFT)

        # 2. 把 Score 放在 Q 的右边 (水平对齐)
        # 关键：Score 的第一行 要对齐 Q 的第一行
        # 我们用一种更简单的方法：排列成网格
        score_cells.arrange_in_grid(rows=s_rows, cols=s_cols, buff=0)
        score_group.next_to(q_group, RIGHT, buff=BUFF_H, aligned_edge=UP)

        # 3. 把 K^T 放在 Score 的上边 (垂直对齐)
        k_group.next_to(score_group, UP, buff=BUFF_V, aligned_edge=LEFT)  # 左对齐Score
        k_label.next_to(k_group, UP)

        # 标签：Dot Product
        dot_label = Text("Dot Product", font_size=24, color=GRAY).next_to(
            score_group, DOWN, buff=0.5
        )

        # ==========================================
        # 4. 动画流程
        # ==========================================

        # Phase 1: 入场
        self.play(
            FadeIn(q_group),
            Write(q_label),
            FadeIn(k_group),
            Write(k_label),
            FadeIn(score_group),
            Write(dot_label),
        )
        self.wait(0.5)

        # Phase 2: 逐格计算
        for i in range(s_rows):
            for j in range(s_cols):
                # 获取当前格子
                current_cell_idx = i * s_cols + j
                target_cell = score_cells[current_cell_idx]
                target_sq = target_cell[0]
                target_num = target_cell[1]

                val = scores_data[i][j]
                new_num = Text(str(val), font_size=24).move_to(target_sq)

                # 获取 Q 的整行 和 K 的整列
                # q_cells 结构是扁平的 (i*2 + col)
                # k_cells 结构是扁平的 (row*3 + j)

                q_row_indices = [i * 2, i * 2 + 1]
                k_col_indices = [j, j + 3]  # 0行j列, 1行j列

                highlights = []
                # 高亮 Q 行
                for idx in q_row_indices:
                    hl = SurroundingRectangle(q_cells[idx][0], color=YELLOW, buff=0.05)
                    highlights.append(hl)
                # 高亮 K 列
                for idx in k_col_indices:
                    hl = SurroundingRectangle(k_cells[idx][0], color=YELLOW, buff=0.05)
                    highlights.append(hl)

                # 播放高亮
                self.play(*[Create(h) for h in highlights], run_time=0.1)

                # 连线汇聚 (可选，为了简洁去掉连线，只显示高亮)

                # 显示结果
                self.play(
                    Transform(target_num, new_num),
                    target_sq.animate.set_stroke(WHITE, 2),
                    run_time=0.2,
                )

                # 消失高亮
                self.play(*[FadeOut(h) for h in highlights], run_time=0.1)

        self.wait(0.5)

        # Phase 3: Softmax
        title_sm = Text("Softmax", font_size=30, color=RED).move_to(dot_label)
        self.play(Transform(dot_label, title_sm))

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        probs = np.apply_along_axis(softmax, 1, scores_data)

        for i in range(s_rows):
            # 每一行一起变色
            anims = []
            for j in range(s_cols):
                current_cell_idx = i * s_cols + j
                target_cell = score_cells[current_cell_idx]
                sq, num = target_cell[0], target_cell[1]

                prob = probs[i][j]
                new_color = interpolate_color(WHITE, RED, prob)

                # 数字变小数
                prob_num = Text(f"{prob:.1f}", font_size=18).move_to(sq)

                anims.append(sq.animate.set_fill(new_color, opacity=1))
                anims.append(Transform(num, prob_num))

            self.play(*anims, run_time=0.5)

        self.wait(2)
