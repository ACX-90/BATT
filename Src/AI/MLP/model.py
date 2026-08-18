"""参数 MLP：输入 (I, SOC, T)，输出正的 R0/R1（及可选 C1）。"""

# 开启延迟注解求值（PEP 563）。
# 好处：类型注解里的名字可以后定义，也能写字符串形式的前向引用，避免循环导入问题。
# 现代 PyTorch / 类型提示代码几乎都会加这一行。
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainConfig


def softplus_inv(y: float) -> float:
    """softplus(z) = y 的反函数，用于把最后一层偏置设到参考值附近。"""
    y = max(float(y), 1e-12)
    return math.log(math.expm1(y))


# 继承nn.Module特性
class ParamMLP(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        # 必须调用，否则参数注册、设备迁移等会出问题。
        super().__init__()
        scheme = cfg.scheme.upper()
        if scheme not in {"A", "B", "B+"}:
            raise ValueError(f"未知 scheme={cfg.scheme}，应为 A / B / B+")
        self.scheme = scheme
        self.r0_min = cfg.r0_min
        self.r1_min = cfg.r1_min
        self.c1_min = cfg.c1_min
        self.c1_star = cfg.c1_star

        out_dim = 3 if scheme == "A" else 2
        # *是解包运算符，把列表 cfg.hidden 拆开插入列表
        dims = [3, *cfg.hidden, out_dim]
        # 语法：建立变量layers，是一个空list，每个元素都是nn.Module类型
        layers: list[nn.Module] = []
        # 建立全连接层 Y=a(WX+B)
        for i in range(len(dims) - 2):
            # 线性变换WX+B，需要输入输出大小
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # 非线性a(WX+B)，激活处理
            layers.append(nn.GELU())
            # 可选：随机丢弃权重，避免过拟合，实现正则化
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        # 输出层
        layers.append(nn.Linear(dims[-2], dims[-1]))
        # 把上述层串起来，使用解包运算
        self.net = nn.Sequential(*layers)
        self._init_head(cfg)

        if scheme == "B+":
            self.phi = nn.Parameter(
                torch.tensor(softplus_inv(cfg.c1_star - cfg.c1_min), dtype=torch.float32)
            )
        else:
            self.register_parameter("phi", None)

    def _init_head(self, cfg: TrainConfig) -> None:
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        # 输出层权重清零
        nn.init.zeros_(last.weight)
        # 计算偏置，使用softplus输出
        bias = [
            softplus_inv(cfg.r0_ref - cfg.r0_min),
            softplus_inv(cfg.r1_ref - cfg.r1_min),
        ]
        if self.scheme == "A":
            bias.append(softplus_inv(cfg.c1_star - cfg.c1_min))
        # 偏置设置到神经网络层内
        last.bias.data.copy_(torch.tensor(bias, dtype=last.bias.dtype))

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x_norm: (..., 3) → R0, R1, C1 同形状 (...,)。"""
        # 执行前向传播输出Z，等效与 self.net.forward(x_norm)，自动调用forward方法
        z = self.net(x_norm)
        # z[..., 0]：高级索引。... 表示“前面所有维度都保留”，
        # 只取最后一个维度的第 0 个通道。
        # 这样无论输入是 (batch, 3) 还是 (batch, time, 3) 都能正确工作。
        r0 = self.r0_min + F.softplus(z[..., 0])
        r1 = self.r1_min + F.softplus(z[..., 1])
        if self.scheme == "A":
            c1 = self.c1_min + F.softplus(z[..., 2])
        elif self.scheme == "B+":
            assert self.phi is not None
            c1_val = self.c1_min + F.softplus(self.phi)
            c1 = c1_val + torch.zeros_like(r0)
        else:
            c1 = r0.new_full(r0.shape, self.c1_star)
        return r0, r1, c1
