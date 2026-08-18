# 02-a 用 MLP 估测 ECM 参数：设计思路与反向传播
Powered by SpaceXAI Grok 4.6

> 对象：100 Ah NMC，一阶 Thevenin（OCV + \(R_0\) + \(R_1\parallel C_1\)）  
> 监督信号：只有端电压测量 \(U_t^{\mathrm{meas}}\)，**没有** \(R_0,R_1,C_1\) 标签  
> 结构：MLP 出参数 → 物理 ECM 出电压 → 电压误差反传更新 MLP  
> 与现有代码对齐：`Src/Sim/nmc100ah_ecm_gen.py` 的离散化  
> 阶次：解码器默认 1RC。真值若是 2RC / 更高阶，残差落到哪见 §11；数量级见 `Doc/02-b` §10、模板见 `Doc/01-b` §11

本文只讲设计和推导。实现时用自动微分即可，不必手写全部雅可比；手推是为了看清梯度从哪来、哪些量可观、训练为什么会漂。

---

## 1. 要解决什么问题

解析模型（`NMC100AhECM`）把

\[
(R_0,R_1,C_1)=f_{\mathrm{ana}}(I,T,\mathrm{SOC})
\]

写成乘性因子。真实电芯的形状往往更歪，因子也会耦合。若对每个工况做 HPPC 再拟合 \(R,C\)，贵且和动态工况对不齐。

实验室和车上最容易拿到的是 \(\{I_k,\,T_k,\,s_k,\,U_{\mathrm{ocv},k},\,U_{t,k}^{\mathrm{meas}}\}\)。\(R_0,R_1,C_1\) 本身测不到。因此用 MLP 代替解析映射，**损失只建在电压上**，让物理电路当可微的解码器：

```
(I, SOC, T) ──► MLP ──► (R0, R1, C1) ──► ECM ──► Ût
                                              │
                              Ut_meas ────────┴──► e = Ût − Ut_meas
                                              │
                                    ∂L/∂θ ◄── 反传穿过 ECM
```

这是灰箱 / 物理信息网络：MLP 只负责参数曲面，电路方程不许改。

---

## 2. 整体结构

### 2.1 前向

第 \(k\) 步（步长 \(\Delta t=0.1\,\mathrm{s}\)，放电电流为正）：

1. 输入归一化后的 \(x_k=[I_k,\,s_k,\,T_k]^\top\)
2. MLP 输出无约束向量 \(z_k\in\mathbb{R}^3\)
3. 正性映射得到电路参数
4. 用上一拍极化 \(U_{p,k-1}\) 和本拍 \(I_k,R_{0,k},R_{1,k},C_{1,k}\) 推 \(U_{p,k}\)
5. \(\hat{U}_{t,k}=U_{\mathrm{ocv},k}-I_k R_{0,k}-U_{p,k}\)

OCV、电流、SOC、温度当作**已知外生量**，不进计算图的学习部分。SOC 由安时积分事先算好（或用 BMS 值）。

### 2.2 MLP

三输入、三输出，两到三层全连接足够（参数曲面光滑，不必很深）：

\[
\begin{aligned}
h^{(1)} &= \sigma\!\left(W^{(1)} \tilde{x}_k + b^{(1)}\right) \\
h^{(\ell)} &= \sigma\!\left(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\right),\quad \ell=2,\ldots,L-1 \\
z_k &= W^{(L)} h^{(L-1)} + b^{(L)}
\end{aligned}
\]

\(\tilde{x}\) 按训练集均值方差标准化。\(\sigma\) 用 GELU 或 Tanh。记全部权重为 \(\theta\)。

正性（否则 \(\tau_1=R_1C_1\) 会炸）：

\[
\begin{aligned}
R_{0,k} &= R_{0,\min} + \mathrm{softplus}(z_{k,0}) \\
R_{1,k} &= R_{1,\min} + \mathrm{softplus}(z_{k,1}) \\
C_{1,k} &= C_{1,\min} + \mathrm{softplus}(z_{k,2})
\end{aligned}
\]

\(\mathrm{softplus}(z)=\ln(1+e^{z})\)。100 Ah NMC 量级可取

\[
R_{0,\min}=R_{1,\min}=10^{-5}\,\Omega,\quad C_{1,\min}=10^{2}\,\mathrm{F}
\]

也可用 \(R=R_{\mathrm{ref}}\,e^{z}\) 把输出放在对数域，初值更好定。

### 2.3 ECM（与现有仿真同一套）

连续时间：

\[
\begin{aligned}
\dot{U}_{p} &= -\frac{U_{p}}{R_{1}C_{1}}+\frac{I}{C_{1}} \\
U_{t} &= U_{\mathrm{ocv}}-I R_{0}-U_{p}
\end{aligned}
\]

本拍参数、电流视为常值时，精确离散为

\[
\begin{aligned}
\tau_{k} &= R_{1,k}\,C_{1,k} \\
\alpha_{k} &= \exp(-\Delta t/\tau_{k}) \\
U_{p,k} &= \alpha_{k}\,U_{p,k-1} + R_{1,k}(1-\alpha_{k})\,I_{k} \\
\hat{U}_{t,k} &= U_{\mathrm{ocv},k} - I_{k} R_{0,k} - U_{p,k}
\end{aligned}
\tag{1}
\]

\(U_{p,0}=0\)（或用一段静置预热）。\(U_{p,k}\) 是隐状态，依赖**历史**的 \(R_1,C_1,I\)。反传必须穿过这条递推，不能把每拍当成静态电阻。

### 2.4 损失

主损失是电压均方：

\[
L_{\mathrm{v}}
=\frac{1}{2N}\sum_{k=1}^{N}
\bigl(\hat{U}_{t,k}-U_{t,k}^{\mathrm{meas}}\bigr)^{2}
=\frac{1}{2N}\sum_{k=1}^{N} e_{k}^{2}
\tag{2}
\]

\[
e_{k}=\hat{U}_{t,k}-U_{t,k}^{\mathrm{meas}}
\]

建议加上参数正则，减轻 \(R_1\leftrightarrow C_1\) 的补偿：

\[
\begin{aligned}
L
&= L_{\mathrm{v}}
 + \frac{\lambda_{\tau}}{2N}\sum_{k}(\ln\tau_{k}-\ln\tau_{\mathrm{ref}})^{2}
 + \frac{\lambda_{s}}{2N}\sum_{k}\lVert p_{k}-p_{k-1}\rVert^{2} \\
p_{k} &= (\ln R_{0,k},\,\ln R_{1,k},\,\ln C_{1,k})
\end{aligned}
\tag{3}
\]

第二项把时间常数按在数秒到几十秒（模板 \(\tau_{\mathrm{ref}}\approx 18\,\mathrm{s}\)），第三项禁止相邻拍参数乱跳。\(\lambda\) 先取很小，电压拟合稳了再加。

---

## 3. 为什么电压误差能更新 MLP

自动微分走的就是

\[
\frac{\partial L}{\partial\theta}
=\sum_{k=1}^{N}
\frac{\partial L}{\partial \hat{U}_{t,k}}
\frac{\partial \hat{U}_{t,k}}{\partial p_{k}}
\frac{\partial p_{k}}{\partial z_{k}}
\frac{\partial z_{k}}{\partial\theta}
+\text{（经 }U_{p}\text{ 传到更早步的项）}
\tag{4}
\]

其中 \(p_k=(R_{0,k},R_{1,k},C_{1,k})\)。

没有 \(R,C\) 标签时，\(\partial\hat{U}_{t}/\partial p\) 由电路给出：电流阶跃瞬间的压差主要训 \(R_0\)；指数弯曲和回弹主要训 \(R_1,C_1\)。激励不够（长时间恒流、几乎无静置），\(R_1\) 与 \(C_1\) 会互相补偿，电压仍准、参数不可信。网格数据里要有脉冲和搁置，原因在此。

下面按「先本拍、再沿状态反传」推导 \(\partial\hat{U}_{t}/\partial p\)。

---

## 4. 电压对电路参数的局部导数

由 (1)：

\[
\frac{\partial\hat{U}_{t,k}}{\partial R_{0,k}}=-I_{k},\qquad
\frac{\partial\hat{U}_{t,k}}{\partial U_{p,k}}=-1
\tag{5}
\]

\(R_0\) 不进状态，梯度最干净：\(e_k(-I_k)\) 直接回 MLP。静置 \(I_k=0\) 时本拍不提供 \(R_0\) 信息。

\(R_1,C_1\) 只通过 \(U_{p,k}\) 影响电压：

\[
\frac{\partial\hat{U}_{t,k}}{\partial R_{1,k}}
=-\frac{\partial U_{p,k}}{\partial R_{1,k}},\qquad
\frac{\partial\hat{U}_{t,k}}{\partial C_{1,k}}
=-\frac{\partial U_{p,k}}{\partial C_{1,k}}
\tag{6}
\]

### 4.1 \(\alpha\) 对 \(\tau,R_1,C_1\)

\[
\alpha=\mathrm{e}^{-\Delta t/\tau},\quad
\tau=R_1 C_1
\]

\[
\frac{\partial\alpha}{\partial\tau}=\alpha\cdot\frac{\Delta t}{\tau^{2}}
\tag{7}
\]

\[
\frac{\partial\alpha}{\partial R_{1}}=\alpha\frac{\Delta t}{R_{1}^{2}C_{1}}=\alpha\frac{\Delta t}{R_{1}\tau},\qquad
\frac{\partial\alpha}{\partial C_{1}}=\alpha\frac{\Delta t}{R_{1}C_{1}^{2}}=\alpha\frac{\Delta t}{C_{1}\tau}
\tag{8}
\]

### 4.2 把 \(U_{p,k-1}\) 当常数（本拍局部）

把 (1) 写成

\[
U_{p,k}=\alpha\,U_{p,k-1}+R_{1}I_{k}-\alpha R_{1}I_{k}
\]

\[
\begin{aligned}
\left.\frac{\partial U_{p,k}}{\partial R_{1}}\right|_{\mathrm{loc}}
&=
\frac{\partial\alpha}{\partial R_{1}}(U_{p,k-1}-R_{1}I_{k})
+I_{k}(1-\alpha)
\\[4pt]
\left.\frac{\partial U_{p,k}}{\partial C_{1}}\right|_{\mathrm{loc}}
&=
\frac{\partial\alpha}{\partial C_{1}}(U_{p,k-1}-R_{1}I_{k})
\end{aligned}
\tag{9}
\]

直观：

- \(I_k(1-\alpha)\)：\(R_1\) 增大则稳态极化 \(IR_1\) 增大
- \(\partial\alpha/\partial(\cdot)\)：时间常数变了，本拍走到稳态的比例变了
- \(U_{p,k-1}-R_{1}I_{k}\)：当前状态离新稳态有多远

只在 \(C_1\) 上：它不改变稳态 \(IR_1\)，只改变快慢，所以 (9) 第二式没有单独的 \(I_k\) 项。长时恒流走到稳态后，\(C_1\) 的局部梯度接近 0，必须靠阶跃和回弹来训电容。

### 4.3  sanity：小 \(\Delta t\)

\(\alpha\approx 1-\Delta t/\tau+\cdots\)，\(1-\alpha\approx\Delta t/(R_1 C_1)\)，代入 (9) 回到欧拉离散

\[
U_{p,k}\approx U_{p,k-1}+\Delta t\Bigl(-\frac{U_{p,k-1}}{R_1 C_1}+\frac{I_k}{C_1}\Bigr)
\]

的偏导，说明 (9) 和连续方程一致。

---

## 5. 沿极化状态的 BPTT

\(U_{p,k}\) 还依赖 \(U_{p,k-1}\)，而 \(U_{p,k-1}\) 依赖更早的 \(R_1,C_1\)。令

\[
\frac{\partial U_{p,k}}{\partial U_{p,k-1}}=\alpha_{k}
\tag{10}
\]

定义反传状态（电压损失对隐状态的伴随）

\[
\lambda_{k}\;=\;\frac{\partial L_{\mathrm{v}}}{\partial U_{p,k}}
\tag{11}
\]

由 \(\hat{U}_{t,k}=U_{\mathrm{ocv},k}-I_k R_{0,k}-U_{p,k}\) 以及 \(U_{p,k+1}\) 对 \(U_{p,k}\) 的依赖：

\[
\lambda_{N}=\frac{e_{N}}{N}(-1),\qquad
\lambda_{k}=\frac{e_{k}}{N}(-1)+\lambda_{k+1}\,\alpha_{k+1}
\quad(k=N-1,\ldots,1)
\tag{12}
\]

即从最后一拍往回扫：

\[
\lambda_{k}=-\frac{e_{k}}{N}+\alpha_{k+1}\lambda_{k+1}
\]

（约定 \(\lambda_{N+1}=0\)。）

于是电压损失对第 \(k\) 拍参数的总梯度为局部项乘上本拍伴随：

\[
\begin{aligned}
\frac{\partial L_{\mathrm{v}}}{\partial R_{0,k}}
&=\frac{e_{k}}{N}(-I_{k})
\\[4pt]
\frac{\partial L_{\mathrm{v}}}{\partial R_{1,k}}
&=\lambda_{k}
\left.\frac{\partial U_{p,k}}{\partial R_{1}}\right|_{\mathrm{loc}}
\\[4pt]
\frac{\partial L_{\mathrm{v}}}{\partial C_{1,k}}
&=\lambda_{k}
\left.\frac{\partial U_{p,k}}{\partial C_{1}}\right|_{\mathrm{loc}}
\end{aligned}
\tag{13}
\]

截断 BPTT：每 \(W\) 步把 \(\lambda\) 清零，等价于不让梯度穿过窗口。\(W\) 应覆盖数个 \(\tau_1\)（例如 20–60 s，即 200–600 步）。\(W=1\) 就是只用 (9)、忽略历史，电容会训得很慢。

正则项 (3) 对 \(p_k\) 的梯度直接加在 (13) 上，不经过 \(\lambda\)。

---

## 6. 从电路参数回到 MLP

### 6.1 softplus

\[
\frac{\partial R_{0,k}}{\partial z_{k,0}}=\mathrm{sigmoid}(z_{k,0})
=\frac{1}{1+e^{-z_{k,0}}}
\tag{14}
\]

\(R_1,C_1\) 同理。对数参数化 \(R=R_{\mathrm{ref}}e^{z}\) 时 \(\partial R/\partial z=R\)。

令 \(g_k=\partial L/\partial p_k\in\mathbb{R}^3\)（即 (13) 再加正则），则

\[
\frac{\partial L}{\partial z_{k,i}}=g_{k,i}\cdot\mathrm{sigmoid}(z_{k,i})
\tag{15}
\]

### 6.2 标准反传

记 \(a^{(\ell)}=W^{(\ell)}h^{(\ell-1)}+b^{(\ell)}\)，\(h^{(\ell)}=\sigma(a^{(\ell)})\)，\(h^{(0)}=\tilde{x}\)，\(z=a^{(L)}\)。

\[
\begin{aligned}
\delta^{(L)}&=\frac{\partial L}{\partial z_{k}} \\
\delta^{(\ell)}
&=\bigl(W^{(\ell+1)\top}\delta^{(\ell+1)}\bigr)\odot\sigma'(a^{(\ell)})
\\
\frac{\partial L}{\partial W^{(\ell)}}
&=\delta^{(\ell)}\,h^{(\ell-1)\top},\qquad
\frac{\partial L}{\partial b^{(\ell)}}=\delta^{(\ell)}
\end{aligned}
\tag{16}
\]

一条长度为 \(N\) 的轨迹：对每拍做一次 (16)，把 \(\partial L/\partial W\) 累加（或按 batch 再平均），然后 Adam 更新 \(\theta\)。

输入归一化的雅可比是对角常数，不进学习，但算 \(\partial L/\partial x\)（若以后要做敏感度）时要除回标准差。

### 6.3 一条计算图（单步展开）

```
x̃_k → [MLP_θ] → z_k → softplus → R0,R1,C1
                              ↓
         U_p,k-1 ──► [式 (1)] ──► U_p,k ──► Ût_k
                              ↑
                         I_k, Uocv_k
                              ↓
                    e_k = Ût_k − Ut_meas
                              ↓
                    λ_k ← e_k 与 λ_{k+1}α_{k+1}
                              ↓
                    g_k → ∂L/∂z_k → ∂L/∂θ
```

虚线向左是 \(U_p\) 的时间边，必须保留在计算图里（PyTorch 不要 `detach` \(U_p\)，除非在做截断）。

---

## 7. 训练流程

对一条轨迹 \(\{(I_k,s_k,T_k,U_{\mathrm{ocv},k},U_{t,k}^{\mathrm{meas}})\}_{k=1}^{N}\)：

1. \(U_p\leftarrow 0\)，\(L\leftarrow 0\)，打开计算图
2. 对 \(k=1\ldots N\)：
   - \(R_0,R_1,C_1\leftarrow \mathrm{MLP}_\theta(\tilde{x}_k)\)
   - 用 (1) 更新 \(U_p\)，得到 \(\hat{U}_{t,k}\)
   - 累加 \(e_k^2\) 与正则
3. \(L.\mathrm{backward}()\)（框架自动做第 5–6 节）
4. 梯度裁剪后 `opt.step()`
5. 多条轨迹（不同起始 SOC、温度）组成 mini-batch，损失对 batch 平均

数据：`Data/grid/` 的 25 条波形可作合成预训练；上真实电芯后只用测量电压，OCV 仍用静置表。

建议先在解析 ECM 生成的数据上过拟合一条 HPPC，确认能收回 \(R_0\) 和 \(\tau_1\)，再上网格，最后上实验。

---

## 8. 可观性、耦合与稳定

| 现象 | 原因 | 处理 |
|------|------|------|
| \(R_0\) 收敛快 | 阶跃瞬间 \(\Delta U\approx IR_0\)，梯度 \(-I\) 直接 | 脉冲幅值不要太小 |
| \(R_1\) 与 \(C_1\) 拧在一起 | 电压只看见 \(\tau=R_1C_1\) 和稳态 \(IR_1\)；缺回弹时 \(C_1\) 不可观 | 序列里保留静置；加 \(\ln\tau\) 正则 |
| 静置段训不动 \(R_0\) | \(I=0\Rightarrow\partial\hat{U}/\partial R_0=0\) | 正常；靠有流段更新 |
| 低温 \(\alpha\to 1\)，梯度变小 | \(\tau\) 大，本拍几乎不衰减 | 加长窗口 / 不截太短 |
| \(\tau\to 0\) 数值炸 | \(\Delta t/\tau\) 过大 | softplus 下界；可把 \(\alpha\) 写成 \(\exp(-\mathrm{softplus}(\cdot))\) |
| 只拟合准电压、参数乱 | 过参数化 | 时间平滑 + 初值靠近解析模型 |

初值：用现有 `NMC100AhECM` 在参考点的 \(R_0,R_1,C_1\) 去对 MLP 做几步监督预热（teacher），再切到纯电压损失，比随机初始化稳得多。阶次不够时电压还能贴、参数会漂，见 §11。

---

## 9. 和现有仓库怎么接

```
Src/Sim/nmc100ah_ecm.py          解析映射，作对照与 MLP 预热标签
Src/Sim/nmc100ah_ecm_gen.py      离散 ECM 与单条 CSV（教师 / 合成数据）
Src/Sim/nmc100ah_ecm_gen_grid.py 多 SOC×T 轨迹，作训练集
Data/grid/*.csv                  I, T, SOC, Uocv, Ut_meas
Doc/01-b-NMC100Ah_ECM参数规范.md   电路与因子定义
```

训练时 **不要** 用 CSV 里的 `r0_ohm,r1_ohm,c1_f` 当主损失（那是教师自己的参数）。主损失只用 `u_t_meas_v`（或合成数据里的 `u_t_true_v`）。教师参数仅用于预热或事后对照。

实现要点：

- 前向逐步调用与 (1) 相同的更新，\(U_p\) 留在 tensor 链上
- 长轨迹用截断 BPTT，窗口 \(\ge 2\sim 4\,\tau_1\)
- batch 内各条 \(U_p\) 独立
- 学习率对 \(C_1\) 偏大的通道可单独缩小，或在对数域输出

---

## 10. 公式索引

| 编号 | 内容 |
|------|------|
| (1) | ECM 离散前向（与仿真代码一致） |
| (2)(3) | 电压损失与正则 |
| (5)(6) | \(\hat{U}_t\) 对 \(R_0,U_p,R_1,C_1\) |
| (7)(8) | \(\alpha\) 对 \(\tau,R_1,C_1\) |
| (9) | \(U_p\) 对本拍 \(R_1,C_1\) 的局部导 |
| (10)–(13) | 伴随 \(\lambda_k\) 与 BPTT |
| (14)–(16) | softplus 与 MLP 反传 |
| (17) | 真值 2RC、解码 1RC 时残差的主项 \(U_{p2}\) |

核心就一句：电压误差先变成对 \(U_p\) 的伴随，再经 (9) 分到 \(R_1,C_1\)，\(R_0\) 则直接吃 \(-I\,e\)；这三路梯度乘 softplus 导数后，按普通全连接反传更新 \(\theta\)。真值若多一条慢支路，未建模的 \(U_{p2}\) 会漏进 \(R_1\) 和滤波的 \(s\)，见 §11。

---

## 11. 阶次不足：电压误差会落到哪

本节不推导「MLP 出 \(R_2,C_2\)」的五参数反传。BMS 解码器锁死 1RC（方案 B 更好）。问的是：测量若来自 2RC 或更高阶，1RC 前向的电压误差长什么样、梯度会误打到谁。定量预算见 `Doc/02-b` §10。

### 11.1 真值多一条慢支路

设测量由 2RC 产生（`Doc/01-b` §11 叠加约定），预测仍用 (1)。慢极化不在模型里：

\[
e_k
= \hat U_{t,k}-U_{t,k}^{\mathrm{meas}}
= \bigl(U_{\mathrm{ocv}}-I_k R_0-U_{p1,k}\bigr)
-\bigl(U_{\mathrm{ocv}}-I_k R_0^{\mathrm{tr}}-U_{p1,k}^{\mathrm{tr}}-U_{p2,k}^{\mathrm{tr}}\bigr)
\]

\(R_0\) 对上时，残差的主项是

\[
e_k \approx -\bigl(U_{p1,k}-U_{p1,k}^{\mathrm{tr}}\bigr) + U_{p2,k}^{\mathrm{tr}}
\tag{17}
\]

\(U_{p2}^{\mathrm{tr}}\) 按 \(\tau_2\approx 90\,\mathrm{s}\) 走，1RC 的 \(U_{p1}\) 按 \(\tau_1\approx 18\,\mathrm{s}\) 走。两者在边沿上几乎正交（边沿仍是 \(IR_0\)），在回弹后段共线：1RC 只能用「偏大的 \(R_1\) + 偏长的 \(\tau_1\)」去追 \(U_{p2}\) 的前半截，后半截留下同号慢尾巴。

### 11.2 梯度会误打到谁

对 1RC 解码器，§4–§5 的通道还是那三条：

| 残差形态 | \(\partial L/\partial R_0\) | \(\partial L/\partial R_1\)（及 \(\tau_1\)） | EKF 若同时开 |
|----------|------------------------------|-----------------------------------------------|--------------|
| 边沿尖、回弹对 | 主通道，对的 | 小 | 不要改 \(s\) |
| 回弹前段偏、后段同号慢飘 | 小（静置 \(I=0\)） | 会加大 \(R_1\)、拉长 \(\tau\) 去追 \(U_{p2}\) | 回弹后段啃一点 \(s,U_p\) |
| 长恒流末端慢慢偏 | 与 \(R_1\) 共线，只看见 \(R_0+R_1\) | 和电阻拆不开 | 斜坡易被当成容量 / SOC |
| 静置很久仍常偏 | \(R_0\) 梯度为 0 | 拧 \(R_1\) 补 OCV / \(s_0\)，补不动慢到 \(\tau_2\) 的那截 | 先查 OCV，不要增量 \(R\) |

方案 B 钉死 \(C_1\) 之后，慢尾巴**全部**挤进 \(R_1\)。电压在 \(\sim 2\tau_1\) 内可变好，\(\tau_2\) 量级的残差不会消失。这不是没训够，是 (17) 里 \(U_{p2}\) 不在解码器值域里。

### 11.3 为什么不要对 2RC 做方案 A

若让 MLP 逐步出 \((R_0,R_1,C_1,R_2,C_2)\)，电压对慢支路的局部导与 (9) 同形，只是 \(\tau=\tau_2\)。问题是零空间更大：

\[
R_{1}C_{1}\approx\mathrm{const},\quad
R_{2}C_{2}\approx\mathrm{const},\quad
R_{1}+R_{2}\approx\mathrm{const}\ \text{（长时恒流）}
\]

缺长搁置时，五参数拟合电压可以很准，参数对不上任何 HPPC 拆法。这比方案 A 的 \(R_1\leftrightarrow C_1\) 更扁。离线要对齐 2RC，用 `Doc/01-a` 附录那种双指数（先钉 \(\tau_2\) 或钉 \(C_2\)），不要加两个 MLP 头。

更高阶（3RC、Warburg、CPE）每多一条连续谱，值域里又多一块与 \(U_{p1}\) 慢相关的分量。BPTT 仍只会把它投影到现有的 \(R_0,R_1\) 上；投影残差的时间尺度就是未建模的 \(\tau\)。识别方法：把 \(e^{\mathrm{ol}}\) 在静置段做半对数图——1RC 残差应接近直线，弯折或第二斜率就是 2RC / 更高阶。

### 11.4 和增量、滤波的关系

阶次误差是**结构差**，不是「新电芯 / 老化」。用 \(e^{\mathrm{ol}}\) 的慢尾巴去增量 1RC MLP，等于逼 \(R_1(s,T)\) 兼当 \(R_2\)，旧短脉冲工况上的 \(R_1\) 会漂。正确反应：

- 仿真生成器可以开 2RC（`Doc/01-b` §11），BMS 仍 1RC
- 增量门控：只有慢尾巴、没有边沿时不要拆 \(R_0/R_1\)
- 滤波：允许回弹后段 NIS 略偏，不要为吃掉 \(U_{p2}\) 而加大 \(q_s\)
