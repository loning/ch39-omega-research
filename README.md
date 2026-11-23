# 《量子计算的几何直觉》第 39 章统一问题研究记录

本仓库保存贝伦德球壳、黎曼零点谱、算术舒尔枢轴、规范耦合阈值与质子选择定则的阶段性推导、反例和复算输入。各节列出模型假设、证明和对应的验证命令。研究中的物理模型与开放问题沿用原记录的证据分级。

- [中文研究总记录](replay/Ω复算说明.md)
- [早期复算说明](replay/复算说明.md)
- [规范动能谱迹与阈值](replay/ch39-规范动能谱迹与阈值.md)
- [球壳三代轮换与质子选择定则](replay/ch39-三代轮换与质子选择定则.md)
- [全部专题、脚本与输入数据](replay/)
- [完整复算包](ch39-omega-math-replay.zip)

在 `replay/` 或复算包解压目录中运行各节给出的命令。部分整数证书只依赖 Python 标准库。使用 Arb 的证书需要 `python-flint`。例如，以下命令分别核验超荷相位构造与第九个时间节点的舒尔枢轴比较。

```sh
cd replay
python3 ch39-native-hypercharge-filter.py
python3 ch39-heldout-schur-threshold.py
```
