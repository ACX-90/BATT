# 02 100 Ah NMC 电芯 ECM 参数规范

> 配套代码：`Src/Sim/nmc100ah_ecm_params.py`、`Src/Sim/nmc100ah_ecm.py`、`Src/Sim/nmc100ah_ecm_demo.py`  
> 特性背景：`Doc/01-NCM电芯ECM参数R0_R1_C1特性.md`  
> 模型：一阶 Thevenin（OCV + \(R_0\) + \(R_1 \parallel C_1\)）  
> 映射：\((I,\, T,\, \mathrm{SOC}) \rightarrow (R_0,\, R_1,\, C_1)\)

本文给出一套可标定、可替换的参数关系。默认数值是 **通用 100 Ah 方壳/软包 NMC 的模板**，用于搭模型、跑通工具链，**不是某一款商用电芯的出厂值**。换真实电芯时只改参数，不改结构。

---

## 1. 电芯假设与符号

| 项目 | 符号 / 值 | 说明 |
|------|-----------|------|
| 体系 | NMC（同 NCM） | 中高镍三元 + 石墨，模板按能量型方壳 |
| 额定容量 | \(Q_n = 100\,\mathrm{Ah}\) | \(1\,\mathrm{C} = 100\,\mathrm{A}\) |
| 电压窗口 | 2.80 / 3.67 / 4.20 V | 下限 / 标称 / 上限 |
| 寿命状态 | BOL | 不含 SOH 维，老化后应重标或整体缩放 |
| 电流符号 | 放电 \(I>0\)，充电 \(I<0\) | 与特性文档一致 |
| 温度 | \(T\) 用 °C 输入，Arrhenius 内部转开尔文 | 指电芯温度，不是环境温度 |
| SOC | \(s \in [0,1]\) | 代码也接受 0–100 百分数 |

端电压（本参数模型不计算，只为对齐 ECM 定义）：

\[
\begin{aligned}
\dot{U}_{p} &= -\frac{U_{p}}{R_{1}C_{1}} + \frac{I}{C_{1}} \\
U_{t} &= U_{\mathrm{ocv}}(s,T) - I R_{0} - U_{p}
\end{aligned}
\]

\[
\tau_{1} = R_{1} C_{1}
\]

---

## 2. 关系结构

三个输出共用同一乘性结构，在参考点处每个因子都是 1：

\[
\begin{aligned}
R_{0}(s,T,I)
  &= R_{0,\mathrm{ref}} \cdot f_{s,0}(s)\cdot f_{\phi,0}(s)\cdot f_{T,0}(T)\cdot f_{I,0}(I)\cdot f_{\mathrm{dir},0}(s,I) \\
R_{1}(s,T,I)
  &= R_{1,\mathrm{ref}} \cdot f_{s,1}(s)\cdot f_{\phi,1}(s)\cdot f_{T,1}(T)\cdot f_{I,1}(I)\cdot f_{\mathrm{dir},1}(s,I) \\
C_{1}(s,T,I)
  &= C_{1,\mathrm{ref}} \cdot f_{s,C}(s)\cdot f_{\phi,C}(s)\cdot f_{T,C}(T)\cdot f_{I,C}(I)\cdot f_{\mathrm{dir},C}(s,I)
\end{aligned}
\]

| 因子 | 含义 | \(R_0\) | \(R_1\) | \(C_1\) |
|------|------|---------|---------|---------|
| \(f_s\) | SOC 大形态（U 形 / 反 U 形） | 浅 U | 深 U | 中段略高 |
| \(f_\phi\) | 高 SOC 相变局部凸起 | 弱 | 中 | 弱、反向 |
| \(f_T\) | Arrhenius 温度 | \(E_a\) 较小 | \(E_a\) 较大 | \(E_a<0\)（升温略增） |
| \(f_I\) | 电流 | 弱软化 | Butler–Volmer | 不修正 |
| \(f_{\mathrm{dir}}\) | 充放不对称 | 弱 | 强 | 无 |

参考点（BOL）：

\[
s_{\mathrm{ref}}=0.50,\quad T_{\mathrm{ref}}=25\,^\circ\mathrm{C},\quad I_{\mathrm{ref}}=+100\,\mathrm{A}\ (1\mathrm{C\ 放电})
\]

在该点：

\[
R_{0}=0.80\,\mathrm{m}\Omega,\quad
R_{1}=0.65\,\mathrm{m}\Omega,\quad
C_{1}=2.80\times 10^{4}\,\mathrm{F},\quad
\tau_{1}=18.2\,\mathrm{s}
\]

对应 10 s 直流内阻量级：

\[
R_{10\mathrm{s}} \approx R_{0}+R_{1}\bigl(1-e^{-10/\tau_{1}}\bigr) \approx 1.07\,\mathrm{m}\Omega
\]

与常见 100 Ah NMC 能量型电芯 10 s DCR（约 0.8–1.5 mΩ）同量级。

---

## 3. 各因子公式

### 3.1 SOC 大形态 \(f_s\)

先写未归一化形状，再除以参考点，保证 \(f_s(s_{\mathrm{ref}})=1\)：

\[
\begin{aligned}
g(s)
  &= 1 + a_{\mathrm{low}}\,e^{-s/s_{\mathrm{low}}}
       + a_{\mathrm{high}}\,e^{-(1-s)/s_{\mathrm{high}}}
       + p\,(s-0.5)^{2} \\
f_{s}(s) &= g(s)\,/\,g(s_{\mathrm{ref}})
\end{aligned}
\]

- \(a_{\mathrm{low}}>0\)：低 SOC 电阻上翘（\(C_1\) 取负，表示低 SOC 电容下降）
- \(a_{\mathrm{high}}>0\)：高 SOC 电阻上翘
- \(p\)：中段二次弯曲

### 3.2 高镍相变凸起 \(f_\phi\)

高镍 NMC 在约 4.15–4.25 V（模板取 \(s\approx 0.93\)）有 H2–H3 相变，电阻局部抬升：

\[
f_{\phi}(s) = 1 + A \exp\left[-\frac{1}{2}\left(\frac{s-s_{\phi}}{w_{\phi}}\right)^{2}\right]
\]

\(s_{\mathrm{ref}}=0.5\) 远离中心，\(f_{\phi}(s_{\mathrm{ref}})\approx 1\)，不必再归一化。

### 3.3 温度 \(f_T\)

\[
f_{T}(T)=\exp\left[\frac{E_{a}}{R}\left(\frac{1}{T+273.15}-\frac{1}{T_{\mathrm{ref}}+273.15}\right)\right]
\]

\(R=8.314462618\,\mathrm{J\,mol^{-1}\,K^{-1}}\)。\(E_a>0\) 时降温参数增大（电阻）；\(E_a<0\) 时降温参数减小（电容）。

### 3.4 电流 \(f_I\)

同样先算原始函数，再对 \(I_{\mathrm{ref}}\) 归一化，使 \(f_I(I_{\mathrm{ref}})=1\)。

**\(R_0\)：弱软化（自热 / 轻微非线性）**

\[
h_{0}(I)=\frac{1}{1+k_{0}\,|I|/I_{1\mathrm{C}}},\qquad
f_{I,0}(I)=\frac{h_{0}(I)}{h_{0}(I_{\mathrm{ref}})}
\]

**\(R_1\)：Butler–Volmer 表观电阻**

\[
h_{1}(I)=\frac{\mathrm{asinh}(|I|/I_{s})}{|I|/I_{s}}\ (I\neq 0),\qquad
h_{1}(0)=1
\]

\[
f_{I,1}(I)=\frac{h_{1}(I)}{h_{1}(I_{\mathrm{ref}})}
\]

\(|I|\ll I_s\) 时 \(h_1\to 1\)（相对 1C 参考点，小电流 \(R_1\) 更大）；大电流表观 \(R_1\) 下降。

**\(C_1\)：不修正**

\[
f_{I,C}(I)=1
\]

### 3.5 充放方向 \(f_{\mathrm{dir}}\)

放电（\(I\ge 0\)）在低 SOC 额外升高，充电（\(I<0\)）在高 SOC 额外升高。两个方向各自用 \(s_{\mathrm{ref}}\) 归一化，因此 **中段 SOC 充、放都回到参考值**，不对称只出现在两端。

\[
\begin{aligned}
d(s) &= 1 + d_{\mathrm{low}}\,e^{-s/s_{d}} \\
c(s) &= 1 + c_{\mathrm{high}}\,e^{-(1-s)/s_{c}} \\
f_{\mathrm{dir}}(s,I)
  &=
  \begin{cases}
    d(s)/d(s_{\mathrm{ref}}), & I \ge 0 \\
    c(s)/c(s_{\mathrm{ref}}), & I < 0
  \end{cases}
\end{aligned}
\]

\(I=0\)（静置）走放电支路。

---

## 4. 默认参数表

### 4.1 参考值与有效域

| 量 | 值 | 单位 |
|----|----|------|
| \(R_{0,\mathrm{ref}}\) | \(8.0\times 10^{-4}\) | \(\Omega\)（0.80 mΩ） |
| \(R_{1,\mathrm{ref}}\) | \(6.5\times 10^{-4}\) | \(\Omega\)（0.65 mΩ） |
| \(C_{1,\mathrm{ref}}\) | \(2.8\times 10^{4}\) | F |
| SOC | \([0.02,\,0.98]\) | 1 |
| \(T\) | \([-20,\,55]\) | °C |
| \(I\) | \([-300,\,300]\) | A（±3C） |

超界时代码默认裁剪并告警；`strict=True` 则抛错。

### 4.2 通道系数

| 系数 | \(R_0\) | \(R_1\) | \(C_1\) |
|------|---------|---------|---------|
| \(a_{\mathrm{low}}\) | 0.55 | 1.80 | −0.20 |
| \(s_{\mathrm{low}}\) | 0.08 | 0.07 | 0.10 |
| \(a_{\mathrm{high}}\) | 0.18 | 0.70 | −0.12 |
| \(s_{\mathrm{high}}\) | 0.07 | 0.06 | 0.08 |
| \(p\) | 0.20 | 0.35 | −0.10 |
| \(A\)（相变） | 0.08 | 0.15 | −0.05 |
| \(s_{\phi}\) | 0.93 | 0.93 | 0.93 |
| \(w_{\phi}\) | 0.03 | 0.03 | 0.03 |
| \(E_a\) / J·mol⁻¹ | 16 000 | 32 000 | −3 000 |
| 电流类型 | `linear_soft` | `bv_asinh` | `identity` |
| \(k_0\) | 0.06 | — | — |
| \(I_s\) / A | — | 50 | — |
| \(d_{\mathrm{low}}\) | 0.10 | 0.45 | 0 |
| \(s_{d}\) | 0.15 | 0.12 | 0.20 |
| \(c_{\mathrm{high}}\) | 0.08 | 0.55 | 0 |
| \(s_{c}\) | 0.12 | 0.10 | 0.20 |

### 4.3 温度倍率（由 \(E_a\) 算出，便于对照实验）

相对 25 °C：

| \(T\) | \(f_{T,R0}\) | \(f_{T,R1}\) | \(f_{T,C1}\) |
|-------|--------------|--------------|--------------|
| −20 °C | 3.150 | 9.921 | 0.806 |
| −10 °C | 2.360 | 5.567 | 0.851 |
| 0 °C | 1.805 | 3.259 | 0.895 |
| 15 °C | 1.251 | 1.565 | 0.959 |
| 25 °C | 1.000 | 1.000 | 1.000 |
| 40 °C | 0.734 | 0.539 | 1.060 |
| 55 °C | 0.554 | 0.307 | 1.117 |

### 4.4 电流倍率（相对 1C）

| \(I\) | \(f_{I,R0}\) | \(f_{I,R1}\) |
|-------|--------------|--------------|
| 0 A | 1.060 | 1.385 |
| 50 A | 1.029 | 1.221 |
| 100 A（参考） | 1.000 | 1.000 |
| 200 A | 0.946 | 0.725 |

小电流下 \(R_1\) 明显高于 1C HPPC 辨出的值，这是 Butler–Volmer 的预期行为。若你的表来自 1C 脉冲、又主要跑 1C，可保持现状；若主要跑小电流工况，应把 \(R_{1,\mathrm{ref}}\) 改成零电流线性电阻，或重标 \(I_s\)。

---

## 5. 手算复核（参考点邻域）

**点 A**（参考点）  
\(s=0.5,\ T=25,\ I=+100\)  
\(R_0=0.800\,\mathrm{m}\Omega,\ R_1=0.650\,\mathrm{m}\Omega,\ C_1=28000\,\mathrm{F}\)

**点 B**（常温、低 SOC、1C 放电）  
\(s=0.10,\ T=25,\ I=+100\)  
默认模板：\(R_0=0.996\,\mathrm{m}\Omega,\ R_1=1.146\,\mathrm{m}\Omega,\ C_1=2.55\times 10^{4}\,\mathrm{F},\ \tau_1=29.3\,\mathrm{s}\)。

**点 C**（−10 °C、中 SOC、1C 放电）  
温度因子单独把 \(R_0\) 乘 2.36、\(R_1\) 乘 5.57，\(C_1\) 乘 0.85。低温功率由 \(R_1\) 主导。

**点 D**（常温、高 SOC、1C 充电）  
\(s=0.90,\ I=-100\)  
充电方向因子抬高 \(R_1\)，并叠加上 SOC 上翘与相变凸起的边缘。

---

## 6. 有效域与裁剪

| 输入 | 声明域 | 超出后 |
|------|--------|--------|
| SOC | 0.02–0.98 | 默认 clip；两端公式仍有定义，但不建议当真实 0%/100% |
| \(T\) | −20–55 °C | Arrhenius 外推误差指数放大，必须补测 |
| \(I\) | −3C–+3C | 更大电流的自热与浓度极化超出 1RC |

模型 **不含**：

- SOH / 循环周次
- 滞后态（除充放分叉外）
- 芯表温差
- 第二 RC（扩散）
- OCV（电压仿真需另给 OCV–SOC 表）

---

## 7. Python 接口

依赖：`numpy`。

```text
Src/Sim/nmc100ah_ecm_params.py   参数数据类、默认值、JSON 读写
Src/Sim/nmc100ah_ecm.py          NMC100AhECM：evaluate / 查表 / 导出
Src/Sim/nmc100ah_ecm_demo.py     参考点复核、典型工况、可选导出
Src/Sim/nmc100ah_ecm_gen.py      充/放/静置时域仿真，输出 Data/*.csv
Src/Plot/plot_ecm_surfaces.py    R0/R1/C1 三维曲面
Src/Plot/plot_sim_waveforms.py   仿真波形
Src/Plot/plot_all.py             一次出曲面+波形
```

### 7.1 单点 / 批量

```python
import sys
sys.path.insert(0, "Src/Sim")
from nmc100ah_ecm import NMC100AhECM

model = NMC100AhECM()

# 标量，返回 (R0[ohm], R1[ohm], C1[F])
R0, R1, C1 = model.evaluate(i_a=100.0, t_celsius=25.0, soc=0.5)

# 百分数 SOC + C 率
R0, R1, C1 = model.evaluate(i_c=1.0, t_celsius=25.0, soc=50, soc_in_percent=True)

# 带因子分解，便于标定对照
out = model.evaluate_full(i_a=100.0, t_celsius=-10.0, soc=0.2)
print(out.R0_mohm, out.R1_mohm, out.C1, out.tau1)
print(out.f_r1)   # soc / phase / temperature / current / direction
```

`evaluate` 对 `numpy` 数组广播，可一次算整张工况面。

### 7.2 查找表

```python
table = model.build_map(
    soc=[0.1, 0.3, 0.5, 0.7, 0.9],
    t_celsius=[-10, 0, 25, 40],
    i_a=[-100, 100],
)
# table["R0"].shape == (5, 4, 2)

model.export_csv("Doc/NMC100Ah_ECM_lookup.csv",
                 soc=[0.05, 0.1, 0.2, 0.5, 0.8, 0.95],
                 t_celsius=[-20, -10, 0, 25, 40, 55],
                 i_a=[-200, -100, 100, 200])
```

### 7.3 替换参数

```python
from nmc100ah_ecm_params import ECMParamSet
from dataclasses import replace

p = ECMParamSet()
p.r0.to_json  # 整集写出：
p.to_json("my_cell.json")

# 改参考内阻（例如 HPPC 测得 0.72 mΩ）
p2 = replace(p, r0=replace(p.r0, ref_value=7.2e-4))
model = NMC100AhECM(p2)

# 或 JSON 往返
model = NMC100AhECM.from_json("my_cell.json")
```

`ParameterChannel` 与 `ECMParamSet` 是 `frozen` 数据类，改值用 `dataclasses.replace`。

### 7.4 命令行

在仓库根目录：

```powershell
python Src/Sim/nmc100ah_ecm_demo.py
python Src/Sim/nmc100ah_ecm_demo.py --json Doc/NMC100Ah_ECM_params.json --csv Doc/NMC100Ah_ECM_lookup.csv
python Src/Sim/nmc100ah_ecm_gen.py
python Src/Plot/plot_all.py
```

---

## 8. 用 HPPC 重标（推荐顺序）

不要一上来拟合全部二十多个系数。按下述拆开，每步只动少数几个数。

1. **参考点**  
   25 °C、SOC=50%、1C 放电脉冲：  
   \(R_{0,\mathrm{ref}}=\Delta U_{\mathrm{instant}}/I\)  
   \(R_{1,\mathrm{ref}}=(\Delta U_{\mathrm{end}}-\Delta U_{\mathrm{instant}})/I\)（扣掉脉冲期内 OCV 随 SOC 的下滑）  
   \(\tau_1\) 由搁置回弹指数拟合，\(C_{1,\mathrm{ref}}=\tau_1/R_{1,\mathrm{ref}}\)。

2. **SOC 形状**  
   固定 25 °C、1C，扫 SOC。先调 \(a_{\mathrm{low}},s_{\mathrm{low}}\) 对低端，再调 \(a_{\mathrm{high}},s_{\mathrm{high}}\) 对高端，最后用 \(p\) 修中段。高镍在 90%+ 仍对不齐时再动 \(A,s_\phi,w_\phi\)。

3. **温度**  
   固定 SOC=50%、1C，用 0 / 25 / 40 °C 三点估 \(E_a\)：

   \[
   E_a = R\cdot
   \frac{\ln\bigl(P(T_1)/P(T_2)\bigr)}{1/T_1-1/T_2}
   \]

   \(R_0\)、\(R_1\)、\(C_1\) 分开估。有 −10 °C 数据务必纳入，不要只靠常温外推。

4. **电流**  
   同一 SOC、温度，用 0.5C / 1C / 2C 脉冲调 \(k_0\) 或 \(I_s\)。  
   注意大电流脉冲的自热：辨识前先看芯温是否已经偏离设定值。

5. **充放**  
   同一 SOC 的充电脉冲与放电脉冲对比，调 \(d_{\mathrm{low}}\)（低 SOC 放电）和 \(c_{\mathrm{high}}\)（高 SOC 充电）。中段应对齐，对不齐先查 OCV 是否分了充放。

6. **冻结形状、只缩放参考值**  
   同体系不同容量时，可先按 \(R \propto 1/Q_n\)、\(C \propto Q_n\) 缩放三个 `ref_value`，再抽检两端 SOC 与低温。

NMC 的 OCV 有斜率，长脉冲必须从 \(\Delta U\) 中减去 \(\Delta U_{\mathrm{ocv}}\)，否则 \(R_1\) 被系统性夸大。

---

## 9. 接入 BMS / 仿真的建议

| 用途 | 建议 |
|------|------|
| 在线 SOC 观测器 | 每步用当前 \(I,T,s\) 取 \(R_0,R_1,C_1\)，再离散化 1RC |
| SOP | \(R_{\mathrm{dyn}}(\Delta t)=R_0+R_1(1-e^{-\Delta t/\tau_1})\) |
| 热功率 | \(I^{2}R_0 + I\cdot U_p\) |
| Simulink | 跑 `export_csv` 做 3 维查表，或用 MATLAB 调本 Python |
| 换电芯 | 只换 JSON / `ref_value` 与 \(E_a\)，不要改乘性结构 |

离散时间（采样周期 \(\Delta t\)，放电电流为正）：

\[
\begin{aligned}
\alpha &= e^{-\Delta t/\tau_{1}} \\
U_{p,k} &= \alpha U_{p,k-1} + R_{1}(1-\alpha)\,I_{k} \\
U_{t,k} &= U_{\mathrm{ocv}}(s_k,T_k) - I_k R_{0} - U_{p,k}
\end{aligned}
\]

---

## 10. 版本

| 项 | 内容 |
|----|------|
| 参数 schema | `1.0.0`（写入 JSON 的 `schema_version`） |
| 默认电芯名 | `NMC-100Ah-Generic` |
| 与特性文档关系 | 本规范把特性文档中的定性规律落成可计算的默认函数 |

标定真实 100 Ah NMC 后，请改 `cell.name`，并在 JSON 里保留测试温度、脉冲倍率、静置时间和拟合残差，避免和本模板数值混用。
