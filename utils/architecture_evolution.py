"""
架构演进历史展示：从RNN到Transformer到Mamba的架构发展历程
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import pandas as pd


class ArchitectureEvolutionTimeline:
    """架构演进时间线展示器"""
    
    def __init__(self):
        """初始化架构演进时间线"""
        self.architectures = self._load_architecture_data()
        self.milestones = self._load_milestone_data()
        
    def _load_architecture_data(self) -> List[Dict]:
        """加载架构数据"""
        return [
            {
                "year": 2014,
                "name": "Seq2Seq + RNN",
                "category": "RNN",
                "key_features": ["编码器-解码器结构", "LSTM/GRU单元", "固定长度表示"],
                "complexity": "O(T)",
                "parallelizable": False,
                "long_range": "Poor",
                "paper": "Sutskever et al.",
                "citation": "Sequence to Sequence Learning with Neural Networks",
                "description": "开创性的序列到序列学习框架，使用RNN处理变长序列"
            },
            {
                "year": 2014,
                "name": "Attention Mechanism",
                "category": "Attention",
                "key_features": ["注意力权重", "软对齐", "可解释性"],
                "complexity": "O(T²)",
                "parallelizable": False,
                "long_range": "Good",
                "paper": "Bahdanau et al.",
                "citation": "Neural Machine Translation by Jointly Learning to Align and Translate",
                "description": "引入注意力机制，允许模型动态关注输入序列的不同部分"
            },
            {
                "year": 2015,
                "name": "Pointer Networks",
                "category": "Attention",
                "key_features": ["指针机制", "组合输出", "可变长度输出"],
                "complexity": "O(T²)",
                "parallelizable": False,
                "long_range": "Good",
                "paper": "Vinyals et al.",
                "citation": "Pointer Networks",
                "description": "使用注意力机制作为指针，从输入中选择输出元素"
            },
            {
                "year": 2016,
                "name": "ByteNet",
                "category": "Efficient",
                "key_features": ["因果卷积", "线性复杂度", "深度网络"],
                "complexity": "O(T)",
                "parallelizable": True,
                "long_range": "Good",
                "paper": "Kalchbrenner et al.",
                "citation": "Neural Machine Translation in Linear Time",
                "description": "使用因果卷积实现线性时间的序列建模"
            },
            {
                "year": 2017,
                "name": "Transformer",
                "category": "Transformer",
                "key_features": ["自注意力", "位置编码", "完全并行"],
                "complexity": "O(T²)",
                "parallelizable": True,
                "long_range": "Excellent",
                "paper": "Vaswani et al.",
                "citation": "Attention Is All You Need",
                "description": "革命性的架构，完全基于注意力机制，摒弃了循环结构"
            },
            {
                "year": 2018,
                "name": "Universal Transformer",
                "category": "Transformer",
                "key_features": ["自适应深度", "循环机制", "全局注意力"],
                "complexity": "O(T²D)",
                "parallelizable": True,
                "long_range": "Excellent",
                "paper": "Dehghani et al.",
                "citation": "Universal Transformer",
                "description": "结合了Transformer的并行性和RNN的自适应深度"
            },
            {
                "year": 2018,
                "name": "Transformer-XL",
                "category": "Transformer",
                "key_features": ["段级循环", "相对位置编码", "长距离依赖"],
                "complexity": "O(T²)",
                "parallelizable": True,
                "long_range": "Excellent",
                "paper": "Dai et al.",
                "citation": "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
                "description": "引入段级循环机制，有效建模更长的序列"
            },
            {
                "year": 2019,
                "name": "Sparse Transformer",
                "category": "Efficient",
                "key_features": ["稀疏注意力", "线性复杂度", "可扩展性"],
                "complexity": "O(T√T)",
                "parallelizable": True,
                "long_range": "Very Good",
                "paper": "Child et al.",
                "citation": "Generating Long Sequences with Sparse Transformers",
                "description": "使用稀疏注意力模式降低计算复杂度"
            },
            {
                "year": 2019,
                "name": "Longformer",
                "category": "Efficient",
                "key_features": ["滑动窗口", "全局注意力", "线性复杂度"],
                "complexity": "O(T)",
                "parallelizable": True,
                "long_range": "Excellent",
                "paper": "Beltagy et al.",
                "citation": "Longformer: The Long-Document Transformer",
                "description": "结合局部滑动窗口和全局注意力的高效架构"
            },
            {
                "year": 2020,
                "name": "Reformer",
                "category": "Efficient",
                "key_features": ["LSH注意力", "可逆层", "分块处理"],
                "complexity": "O(T log T)",
                "parallelizable": True,
                "long_range": "Very Good",
                "paper": "Kitaev et al.",
                "citation": "Reformer: The Efficient Transformer",
                "description": "使用局部敏感哈希实现高效的注意力计算"
            },
            {
                "year": 2020,
                "name": "Linformer",
                "category": "Efficient",
                "key_features": ["低秩投影", "线性复杂度", "理论保证"],
                "complexity": "O(T)",
                "parallelizable": True,
                "long_range": "Good",
                "paper": "Wang et al.",
                "citation": "Linformer: Self-Attention with Linear Complexity",
                "description": "通过低秩近似将注意力复杂度降低到线性"
            },
            {
                "year": 2021,
                "name": "Performer",
                "category": "Efficient",
                "key_features": ["随机特征", "核方法", "精确逼近"],
                "complexity": "O(T)",
                "parallelizable": True,
                "long_range": "Good",
                "paper": "Choromanski et al.",
                "citation": "Rethinking Attention with Performers",
                "description": "使用随机特征方法近似注意力矩阵"
            },
            {
                "year": 2021,
                "name": "Linear Transformer",
                "category": "Efficient",
                "key_features": ["核函数", "线性复杂度", "因果掩码"],
                "complexity": "O(T)",
                "parallelizable": True,
                "long_range": "Good",
                "paper": "Katharopoulos et al.",
                "citation": "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
                "description": "将注意力重新表述为核函数，实现线性复杂度"
            },
            {
                "year": 2021,
                "name": "FlashAttention",
                "category": "Efficient",
                "key_features": ["IO感知", "分块计算", "硬件优化"],
                "complexity": "O(T²)",
                "parallelizable": True,
                "long_range": "Excellent",
                "paper": "Dao et al.",
                "citation": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
                "description": "通过IO感知的分块计算大幅提升注意力计算效率"
            },
            {
                "year": 2023,
                "name": "Mamba",
                "category": "SSM",
                "key_features": ["状态空间模型", "选择性机制", "线性复杂度"],
                "complexity": "O(T)",
                "parallelizable": True,
                "long_range": "Excellent",
                "paper": "Gu & Dao",
                "citation": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
                "description": "结合状态空间模型和选择性机制，实现高效的序列建模"
            },
            {
                "year": 2023,
                "name": "Hyena",
                "category": "SSM",
                "key_features": ["长卷积", "隐式参数化", "亚二次复杂度"],
                "complexity": "O(T√T)",
                "parallelizable": True,
                "long_range": "Very Good",
                "paper": "Poli et al.",
                "citation": "Hyena Hierarchy: Towards Larger Convolutional Language Models",
                "description": "使用长卷积和隐式参数化的高效架构"
            },
            {
                "year": 2024,
                "name": "StripedHyena",
                "category": "SSM",
                "key_features": ["混合架构", "多尺度", "高效训练"],
                "complexity": "O(T)",
                "parallelizable": True,
                "long_range": "Excellent",
                "paper": "Fu et al.",
                "citation": "StripedHyena: 7B Fast and Accurate Language Models",
                "description": "结合混合专家和状态空间模型的新架构"
            }
        ]
    
    def _load_milestone_data(self) -> List[Dict]:
        """加载里程碑数据"""
        return [
            {
                "year": 2014,
                "title": "注意力机制诞生",
                "description": "Bahdanau等人引入注意力机制，解决了固定长度表示的瓶颈",
                "impact": "High"
            },
            {
                "year": 2017,
                "title": "Transformer革命",
                "description": "Vaswani等人提出Transformer架构，完全基于注意力机制",
                "impact": "Revolutionary"
            },
            {
                "year": 2018,
                "title": "BERT时代",
                "description": "基于Transformer的预训练模型开始统治NLP领域",
                "impact": "High"
            },
            {
                "year": 2020,
                "title": "效率优化浪潮",
                "description": "大量工作致力于降低Transformer的二次复杂度",
                "impact": "Medium"
            },
            {
                "year": 2023,
                "title": "SSM复兴",
                "description": "Mamba等状态空间模型展现出与Transformer竞争的潜力",
                "impact": "High"
            }
        ]
    
    def create_evolution_timeline(self) -> go.Figure:
        """创建架构演进时间线"""
        fig = go.Figure()
        
        # 按类别分组
        categories = ["RNN", "Attention", "Transformer", "Efficient", "SSM"]
        colors = {
            "RNN": "#FF6B6B",
            "Attention": "#4ECDC4", 
            "Transformer": "#45B7D1",
            "Efficient": "#96CEB4",
            "SSM": "#FECA57"
        }
        
        for category in categories:
            category_archs = [arch for arch in self.architectures if arch["category"] == category]
            
            years = [arch["year"] for arch in category_archs]
            names = [arch["name"] for arch in category_archs]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=[category] * len(names),
                mode='markers',
                name=category,
                marker=dict(
                    size=12,
                    color=colors.get(category, "#95A5A6"),
                    line=dict(width=2, color='white')
                ),
                text=names,
                hovertemplate='<b>%{text}</b><br>年份: %{x}<br>类别: %{y}<extra></extra>'
            ))
        
        # 添加里程碑
        for milestone in self.milestones:
            fig.add_vline(
                x=milestone["year"],
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text=milestone["title"],
                annotation_position="top"
            )
        
        fig.update_layout(
            title="序列建模架构演进时间线 (2014-2024)",
            xaxis_title="年份",
            yaxis_title="架构类别",
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def create_complexity_comparison(self) -> go.Figure:
        """创建复杂度对比图"""
        # 提取代表性架构
        representative_archs = [
            "Seq2Seq + RNN",
            "Attention Mechanism", 
            "Transformer",
            "Transformer-XL",
            "Sparse Transformer",
            "Longformer",
            "Linformer",
            "FlashAttention",
            "Mamba"
        ]
        
        arch_data = {arch["name"]: arch for arch in self.architectures if arch["name"] in representative_archs}
        
        # 创建复杂度对比
        fig = go.Figure()
        
        complexites = []
        names = []
        categories = []
        
        for name in representative_archs:
            if name in arch_data:
                arch = arch_data[name]
                complexites.append(arch["complexity"])
                names.append(name)
                categories.append(arch["category"])
        
        # 分配复杂度数值
        complexity_values = {
            "O(T)": 1,
            "O(T log T)": 2,
            "O(T√T)": 3,
            "O(T²)": 4,
            "O(T²D)": 5
        }
        
        numeric_complexities = [complexity_values.get(comp, 3) for comp in complexites]
        
        fig.add_trace(go.Scatter(
            x=list(range(len(names))),
            y=numeric_complexities,
            mode='markers+lines',
            marker=dict(
                size=[15 if name == "Transformer" else 10 for name in names],
                color=[complexity_values.get(comp, 3) for comp in complexites],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="复杂度等级")
            ),
            text=names,
            hovertemplate='<b>%{text}</b><br>复杂度: %{customdata}<extra></extra>',
            customdata=complexites
        ))
        
        fig.update_layout(
            title="代表性架构的复杂度对比",
            xaxis_title="架构",
            yaxis_title="复杂度等级（数值越高越复杂）",
            height=500,
            xaxis=dict(tickmode='array', tickvals=list(range(len(names))), ticktext=names)
        )
        
        return fig
    
    def create_feature_evolution_chart(self) -> go.Figure:
        """创建特性演化图"""
        # 追踪关键特性的出现时间
        features = {
            "注意力机制": 2014,
            "位置编码": 2017,
            "多头注意力": 2017,
            "残差连接": 2016,
            "层归一化": 2016,
            "稀疏注意力": 2019,
            "线性复杂度": 2020,
            "状态空间模型": 2023,
            "选择性机制": 2023
        }
        
        fig = go.Figure()
        
        for feature, year in features.items():
            fig.add_trace(go.Scatter(
                x=[year],
                y=[feature],
                mode='markers',
                marker=dict(size=20, color='blue'),
                name=feature,
                showlegend=False
            ))
        
        fig.update_layout(
            title="关键特性出现时间线",
            xaxis_title="年份",
            yaxis_title="特性",
            height=600
        )
        
        return fig
    
    def create_architecture_comparison_matrix(self) -> go.Figure:
        """创建架构对比矩阵"""
        # 选择代表性架构进行对比
        selected_archs = [
            "Seq2Seq + RNN",
            "Transformer", 
            "Longformer",
            "Mamba"
        ]
        
        arch_data = {arch["name"]: arch for arch in self.architectures if arch["name"] in selected_archs}
        
        # 创建对比指标
        metrics = ["并行性", "长距离依赖", "计算复杂度", "内存效率", "可解释性"]
        
        # 评分（1-5分）
        scores = {
            "Seq2Seq + RNN": [1, 2, 4, 4, 3],
            "Transformer": [5, 5, 2, 2, 4],
            "Longformer": [5, 4, 4, 3, 4],
            "Mamba": [5, 5, 5, 5, 2]
        }
        
        fig = go.Figure()
        
        for arch in selected_archs:
            if arch in scores:
                fig.add_trace(go.Scatterpolar(
                    r=scores[arch],
                    theta=metrics,
                    fill='toself',
                    name=arch
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            title="架构特性对比雷达图",
            height=600
        )
        
        return fig
    
    def create_evolution_report(self) -> str:
        """生成架构演进报告"""
        report = "# 序列建模架构演进报告\n\n"
        
        # 按时期划分
        periods = {
            "早期 (2014-2016)": ["RNN", "Attention"],
            "Transformer时代 (2017-2019)": ["Transformer"],
            "效率优化期 (2020-2022)": ["Efficient"],
            "新架构探索 (2023-2024)": ["SSM"]
        }
        
        for period, categories in periods.items():
            report += f"## {period}\n\n"
            
            period_archs = [arch for arch in self.architectures 
                          if any(arch["category"] in categories for arch in [arch])]
            
            for arch in period_archs:
                report += f"### {arch['name']} ({arch['year']})\n"
                report += f"**论文**: {arch['paper']} - {arch['citation']}\n\n"
                report += f"**关键特性**: {', '.join(arch['key_features'])}\n\n"
                report += f"**描述**: {arch['description']}\n\n"
                report += f"**复杂度**: {arch['complexity']} | "
                report += f"**并行性**: {'是' if arch['parallelizable'] else '否'} | "
                report += f"**长距离依赖**: {arch['long_range']}\n\n"
                report += "---\n\n"
        
        # 演进趋势分析
        report += """
## 演进趋势分析

### 🔄 主要演进方向

#### 1. 从循环到注意力
- **早期**: RNN/LSTM依赖序列处理，难以并行
- **突破**: Transformer完全基于注意力，实现完全并行
- **影响**: 大幅提升了训练效率和模型规模

#### 2. 从二次复杂度到线性复杂度
- **问题**: Transformer的O(T²)复杂度限制长序列处理
- **解决方案**: 稀疏注意力、低秩近似、核方法等
- **最新进展**: 状态空间模型实现真正的线性复杂度

#### 3. 从固定到选择性
- **传统**: 固定的计算模式和参数
- **创新**: 根据输入动态调整计算（Mamba的选择性机制）
- **未来**: 更加智能和高效的计算策略

### 🎯 技术挑战与解决方案

#### 挑战1: 长序列建模
- **问题**: Transformer的计算和内存需求随序列长度平方增长
- **解决路径**: 
  - 稀疏注意力（Longformer, BigBird）
  - 线性注意力（Linformer, Performer）
  - 状态空间模型（Mamba, S4）

#### 挑战2: 效率与效果平衡
- **问题**: 效率提升往往伴随性能下降
- **解决路径**:
  - 硬件感知优化（FlashAttention）
  - 混合架构（结合不同机制的优势）
  - 自适应计算（根据任务动态调整）

#### 挑战3: 可扩展性
- **问题**: 模型规模增长带来的训练和推理挑战
- **解决路径**:
  - 分布式训练优化
  - 模型压缩和蒸馏
  - 条件计算和专家混合

### 🔮 未来发展方向

#### 1. 更智能的计算策略
- 根据输入内容动态分配计算资源
- 自适应的深度和宽度
- 任务特定的架构优化

#### 2. 跨模态统一架构
- 统一处理文本、图像、音频等模态
- 模态间的有效融合机制
- 高效的多模态预训练

#### 3. 硬件协同设计
- 针对特定架构的硬件优化
- 新的计算范式（光计算、神经形态）
- 能效优先的设计理念

### 📚 学习建议

#### 理论基础
1. **线性代数**: 理解注意力机制的数学原理
2. **信息论**: 理解序列建模的信息瓶颈
3. **优化理论**: 理解不同架构的优化特性

#### 实践技能
1. **架构设计**: 学会根据任务选择合适架构
2. **效率优化**: 掌握各种加速和优化技术
3. **实验分析**: 能够评估和比较不同架构

#### 前沿跟踪
1. **论文阅读**: 关注NeurIPS, ICML, ICLR等顶会
2. **开源项目**: 跟踪Hugging Face, PyTorch等框架更新
3. **工业实践**: 了解大规模模型部署的实际挑战
"""
        
        return report


if __name__ == "__main__":
    # 测试代码
    timeline = ArchitectureEvolutionTimeline()
    
    # 创建时间线
    fig = timeline.create_evolution_timeline()
    print("创建了架构演进时间线")
    
    # 创建复杂度对比
    fig = timeline.create_complexity_comparison()
    print("创建了复杂度对比图")
    
    # 生成报告
    report = timeline.create_evolution_report()
    print(report)
