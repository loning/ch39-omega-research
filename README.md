# 《量子计算的几何直觉》第 39 章统一问题研究记录

本仓库保存贝伦德球壳、黎曼零点谱、算术舒尔枢轴、规范耦合阈值与质子选择定则的阶段性推导、反例和复算输入。各节列出模型假设、证明和对应的验证命令。研究中的物理模型与开放问题沿用原记录的证据分级。

- [中文研究总记录](replay/Ω复算说明.md)
- [早期复算说明](replay/复算说明.md)
- [规范动能谱迹与阈值](replay/ch39-规范动能谱迹与阈值.md)
- [球壳三代轮换与质子选择定则](replay/ch39-三代轮换与质子选择定则.md)
- [全部专题、脚本与输入数据](replay/)
- [完整复算包](ch39-omega-math-replay.zip)

## 第 39 章的阶段结论

2026 年 11 月 21 日，Lean FRO 与 Mathlib Initiative 联合确认了 Alpöge 的黎曼猜想形式化命题与证明。本仓库取具体正零点的高度包围区间做有限算术检验；物理作用与匹配条件在各节单列。[Lean 与 Mathlib 联合声明](https://leanprover-community.github.io/blog/posts/riemann-hypothesis/)

十九点贝伦德球壳的两个外层共有十五个点，一次坐标对换只固定其中一个。将壳反射与弱中心作奇投影给出系数 $t=1/15$。在指定的统一群面作用中，它固定三个裸逆耦合的比值为 $24/25:16/15:1$。给定能标和共同标定后，允许三个低端规范输入各浮动 $0.07\%$ 即能达到一组两圈高端目标。固定输入中心时，每个回路至少一支逆耦合仍相差 $0.0287$ 以上。[壳投影与输入范围](https://github.com/loning/ch39-omega-research/pull/29)

对未放大的正面权，从第一个实际零点起，完整四维格点测度中的有界局部场强相关已经指数衰减。这给出了高精度局部匹配与长程 Maxwell 动力学之间的定量障碍。把乘数放入微观作用会改变测度，在完整统一群上加三面圆盘便能测出大于 $0.09222$ 的刚度增量。[连通相关](https://github.com/loning/ch39-omega-research/pull/33) · [微观放大后的圆盘检验](https://github.com/loning/ch39-omega-research/pull/36)

在共轭不变的外部权下，两种有限阶转角的读数可区分前 3200 个实际零点。若只有独立的未放大边界样本，两种相邻候选都要求至少 95% 判对率时，样本数须超过 $10^{22}$。所有结论均附模型假设、书面论证与复算代码。统一能标与绝对标定的来源、长程规范动力学以及引力的定量闭合仍是开放问题。[有限读数](https://github.com/loning/ch39-omega-research/pull/37) · [样本下界](https://github.com/loning/ch39-omega-research/pull/39)

在 `replay/` 或复算包解压目录中运行各节给出的命令。部分整数证书只依赖 Python 标准库。使用 Arb 的证书需要 `python-flint`。例如，以下命令分别核验超荷相位构造与第九个时间节点的舒尔枢轴比较。

```sh
cd replay
python3 ch39-native-hypercharge-filter.py
python3 ch39-heldout-schur-threshold.py
```
