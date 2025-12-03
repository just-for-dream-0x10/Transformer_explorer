"""
训练动态监控工具：实时监控训练过程中的关键指标
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from collections import deque
import plotly.graph_objects as go
import plotly.express as px


@dataclass
class TrainingMetrics:
    """训练指标数据"""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    grad_norm: float
    param_norm: float
    throughput: float  # samples/second
    memory_usage: float  # MB


@dataclass
class LayerMetrics:
    """层级指标"""
    layer_name: str
    grad_norm: float
    param_norm: float
    update_ratio: float
    activation_sparsity: float
    dead_neurons_ratio: float


class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self, model: nn.Module, window_size: int = 100):
        """
        Args:
            model: 要监控的模型
            window_size: 指标历史记录的窗口大小
        """
        self.model = model
        self.window_size = window_size
        
        # 历史记录
        self.metrics_history = deque(maxlen=window_size)
        self.layer_metrics_history = deque(maxlen=window_size)
        
        # 钩子注册
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        
        # 性能计时
        self.step_times = deque(maxlen=10)
        self.last_step_time = time.time()
        
    def register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(module, input, output):
            self.activations[module] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients[module] = grad_output[0].detach()
        
        # 注册钩子到所有层
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                handle = module.register_forward_hook(forward_hook)
                self.hooks.append(handle)
                
                handle = module.register_backward_hook(backward_hook)
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def compute_grad_norm(self) -> float:
        """计算全局梯度范数"""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def compute_param_norm(self) -> float:
        """计算全局参数范数"""
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def analyze_layer_sparsity(self) -> List[LayerMetrics]:
        """分析各层的激活稀疏性和梯度"""
        layer_metrics = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module in self.activations:
                # 激活稀疏性
                activation = self.activations[module]
                sparsity = (activation.abs() < 1e-6).float().mean().item()
                
                # 死神经元比例（连续多步接近零）
                dead_ratio = self._compute_dead_neurons_ratio(module, activation)
                
                # 梯度范数
                grad_norm = 0
                if module in self.gradients:
                    grad_norm = self.gradients[module].norm(2).item()
                
                # 参数范数
                param_norm = module.weight.data.norm(2).item()
                
                # 更新比例
                update_ratio = grad_norm / (param_norm + 1e-8)
                
                layer_metrics.append(LayerMetrics(
                    layer_name=name,
                    grad_norm=grad_norm,
                    param_norm=param_norm,
                    update_ratio=update_ratio,
                    activation_sparsity=sparsity,
                    dead_neurons_ratio=dead_ratio
                ))
        
        return layer_metrics
    
    def _compute_dead_neurons_ratio(self, module: nn.Module, activation: torch.Tensor) -> float:
        """计算死神经元比例"""
        # 简化版：统计绝对值很小的神经元比例
        # 实际应用中需要跟踪多步的历史
        if len(activation.shape) > 2:
            # 对于多维激活，取最后一个维度
            dead_mask = activation.abs().mean(dim=tuple(range(len(activation.shape)-1))) < 1e-6
        else:
            dead_mask = activation.abs() < 1e-6
        
        return dead_mask.float().mean().item()
    
    def detect_anomalies(self) -> Dict[str, List[str]]:
        """检测训练异常"""
        anomalies = {
            'gradients': [],
            'activations': [],
            'parameters': [],
            'performance': []
        }
        
        if len(self.metrics_history) < 10:
            return anomalies
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # 梯度异常检测
        grad_norms = [m.grad_norm for m in recent_metrics]
        if np.mean(grad_norms) < 1e-6:
            anomalies['gradients'].append("梯度消失 - 可能需要调整学习率或检查模型结构")
        elif np.mean(grad_norms) > 10:
            anomalies['gradients'].append("梯度爆炸 - 考虑梯度裁剪或降低学习率")
        
        # 梯度方差检测
        if np.std(grad_norms) / (np.mean(grad_norms) + 1e-8) > 2:
            anomalies['gradients'].append("梯度不稳定 - 训练可能震荡")
        
        # 性能异常检测
        throughputs = [m.throughput for m in recent_metrics]
        if np.std(throughputs) / np.mean(throughputs) > 0.3:
            anomalies['performance'].append("吞吐量不稳定 - 可能存在资源竞争")
        
        # 损失异常检测
        losses = [m.loss for m in recent_metrics]
        if np.isnan(losses).any():
            anomalies['parameters'].append("损失为 NaN - 检查学习率和数值稳定性")
        elif np.isinf(losses).any():
            anomalies['parameters'].append("损失为 Inf - 可能存在数值溢出")
        
        # 检测损失是否停止下降
        if len(losses) >= 5:
            recent_5 = losses[-5:]
            if np.std(recent_5) / (np.mean(recent_5) + 1e-8) < 0.01:
                anomalies['parameters'].append("损失停止下降 - 可能陷入局部最优或需要调整学习率")
        
        return anomalies
    
    def step(self, step: int, epoch: int, loss: float, learning_rate: float, 
             batch_size: int) -> TrainingMetrics:
        """记录一步训练的指标"""
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.last_step_time = current_time
        
        # 计算吞吐量
        throughput = batch_size / step_time if step_time > 0 else 0
        self.step_times.append(step_time)
        
        # 计算梯度范数和参数范数
        grad_norm = self.compute_grad_norm()
        param_norm = self.compute_param_norm()
        
        # 显存使用
        memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        # 创建指标对象
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            param_norm=param_norm,
            throughput=throughput,
            memory_usage=memory_usage
        )
        
        # 记录历史
        self.metrics_history.append(metrics)
        
        # 分析层级指标
        layer_metrics = self.analyze_layer_sparsity()
        self.layer_metrics_history.append(layer_metrics)
        
        return metrics
    
    def get_training_summary(self) -> Dict:
        """获取训练摘要统计"""
        if not self.metrics_history:
            return {}
        
        metrics = list(self.metrics_history)
        recent_metrics = metrics[-20:] if len(metrics) >= 20 else metrics
        
        return {
            'total_steps': len(metrics),
            'avg_loss': np.mean([m.loss for m in recent_metrics]),
            'avg_grad_norm': np.mean([m.grad_norm for m in recent_metrics]),
            'avg_throughput': np.mean([m.throughput for m in recent_metrics]),
            'avg_memory': np.mean([m.memory_usage for m in recent_metrics]),
            'loss_trend': 'decreasing' if len(recent_metrics) > 1 and recent_metrics[-1].loss < recent_metrics[0].loss else 'increasing',
            'training_stable': self._is_training_stable(recent_metrics)
        }
    
    def _is_training_stable(self, metrics: List[TrainingMetrics]) -> bool:
        """判断训练是否稳定"""
        if len(metrics) < 5:
            return False
        
        losses = [m.loss for m in metrics]
        grad_norms = [m.grad_norm for m in metrics]
        
        # 损失和梯度都应该相对稳定
        loss_stable = np.std(losses) / (np.mean(losses) + 1e-8) < 0.5
        grad_stable = np.std(grad_norms) / (np.mean(grad_norms) + 1e-8) < 1.0
        
        return loss_stable and grad_stable
    
    def visualize_training_curves(self) -> go.Figure:
        """可视化训练曲线"""
        if not self.metrics_history:
            return go.Figure()
        
        metrics = list(self.metrics_history)
        steps = [m.step for m in metrics]
        
        fig = go.Figure()
        
        # 损失曲线
        fig.add_trace(go.Scatter(
            x=steps, y=[m.loss for m in metrics],
            mode='lines', name='Loss',
            line=dict(color='red', width=2)
        ))
        
        # 梯度范数
        fig.add_trace(go.Scatter(
            x=steps, y=[m.grad_norm for m in metrics],
            mode='lines', name='Grad Norm',
            yaxis='y2',
            line=dict(color='blue', width=2)
        ))
        
        # 设置双 y 轴
        fig.update_layout(
            title='训练监控曲线',
            xaxis_title='Step',
            yaxis=dict(title='Loss', side='left'),
            yaxis2=dict(title='Grad Norm', side='right', overlaying='y'),
            height=400
        )
        
        return fig
    
    def visualize_layer_health(self) -> go.Figure:
        """可视化层级健康状况"""
        if not self.layer_metrics_history:
            return go.Figure()
        
        # 获取最新的层级指标
        latest_layer_metrics = self.layer_metrics_history[-1]
        
        layer_names = [m.layer_name for m in latest_layer_metrics]
        update_ratios = [m.update_ratio for m in latest_layer_metrics]
        sparsities = [m.activation_sparsity for m in latest_layer_metrics]
        
        fig = go.Figure()
        
        # 更新比例
        fig.add_trace(go.Bar(
            x=layer_names,
            y=update_ratios,
            name='Update Ratio',
            yaxis='y'
        ))
        
        # 稀疏性
        fig.add_trace(go.Bar(
            x=layer_names,
            y=sparsities,
            name='Activation Sparsity',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='层级健康状况',
            xaxis_title='Layer',
            yaxis=dict(title='Update Ratio', side='left'),
            yaxis2=dict(title='Activation Sparsity', side='right', overlaying='y'),
            barmode='group',
            height=400
        )
        
        return fig


def create_training_report(monitor: TrainingMonitor) -> str:
    """生成训练报告"""
    summary = monitor.get_training_summary()
    anomalies = monitor.detect_anomalies()
    
    report = f"""
# 训练监控报告

## 📊 训练统计
- 总步数: {summary.get('total_steps', 0)}
- 平均损失: {summary.get('avg_loss', 0):.6f}
- 平均梯度范数: {summary.get('avg_grad_norm', 0):.6f}
- 平均吞吐量: {summary.get('avg_throughput', 0):.1f} samples/s
- 平均显存使用: {summary.get('avg_memory', 0):.1f} MB
- 损失趋势: {summary.get('loss_trend', 'unknown')}
- 训练稳定性: {'✅ 稳定' if summary.get('training_stable', False) else '⚠️ 不稳定'}

## ⚠️ 异常检测
"""
    
    for category, issues in anomalies.items():
        if issues:
            report += f"\n### {category.title()}:\n"
            for issue in issues:
                report += f"- {issue}\n"
    
    if not any(anomalies.values()):
        report += "\n✅ 未检测到明显异常\n"
    
    return report


if __name__ == "__main__":
    # 测试代码
    from utils.model_profiler import create_sample_transformer
    
    model = create_sample_transformer()
    monitor = TrainingMonitor(model)
    monitor.register_hooks()
    
    # 模拟训练过程
    for step in range(100):
        # 模拟损失下降
        loss = 10.0 * np.exp(-step * 0.01) + np.random.normal(0, 0.1)
        lr = 0.001 * (0.99 ** step)
        
        metrics = monitor.step(step, 0, loss, lr, batch_size=32)
        
        if step % 20 == 0:
            print(f"Step {step}: Loss={metrics.loss:.4f}, GradNorm={metrics.grad_norm:.4f}")
    
    # 生成报告
    report = create_training_report(monitor)
    print(report)
    
    monitor.remove_hooks()
