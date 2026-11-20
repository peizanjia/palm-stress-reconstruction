# 任务 03　三维压入反演核查基准

沿用任务 02 的三维可压缩 Neo-Hookean 硅胶圆片。圆片半径 25 mm、厚度 5 mm，底面和侧面全固定。剪切模量 30 kPa、泊松比 0.45。压头为半径 4 mm 的无摩擦刚性半球。接触网格目标间距为 0.53 mm，罚系数因子为 140。限制和来源见 [厚度研究](../indentation-study.md)。

两组压头中心在圆片平面的坐标分别为 **(0, 0) mm** 和 **(6, 0) mm**。偏心距离用于构造反演核查工况。每组都保存一个无载参考状态和从 **0.12 到 3.72 mm，每步 0.20 mm** 的压头行程，共 20 个快照。压头行程和由接触斑计算的球冠深度分别记录。圆片网格的细化区随压头位置平移。两组共用圆片坐标原点、固定边界和高度输出网格。

## 文件和字段

本次文件位于 [20261120_reference_v1](../../results/stage-01-forward-model/task-03-inverse-reference/20261120_reference_v1/)。

```text
results/stage-01-forward-model/task-03-inverse-reference/<运行编号>/
  centered/ 或 eccentric/
    manifest.csv              快照索引、行程、球冠深度、合力、反力和数值诊断
    surface_mesh.mat           每组共享的表面网格、采样定义和模型参数
    snapshots/step_NNN.mat     同步保存高度、应力和接触牵引
  logs/                       本次批处理运行日志
  verification.csv            本次归档的数据核查结果
```

`surface_mesh.mat` 中的 `meta` 保存顶面节点的参考坐标 `referenceXYZ_mm`、局部连接 `triangles`、原体网格节点编号 `nodeIds`、固定输出网格的 `x_mm` 和 `y_mm`，以及各面内积分点的重心坐标和权重。每个快照中的 `frame` 包含下列字段。

| 字段 | 含义 |
| --- | --- |
| `height_mm` | 当前坐标上的欧拉高度 $z(x,y)$，$x,y$ 轴各从 −25 到 25 mm，间距 0.25 mm。行对应 `y_mm`，列对应 `x_mm`。圆片投影外为 NaN。未压时高度为 5 mm，如需高度变化可减去 5 mm。 |
| `currentXYZ_mm` | 与参考节点一一对应的当前顶面节点坐标。与 `triangles` 一起提供完整的分片线性表面，无须先栅格化。 |
| `cauchy_MPa` | 硅胶体单元在顶面一侧的 Cauchy 应力，尺寸为三角形 × 7 个面内积分点 × 6 分量。分量为 `xx, yy, zz, xy, xz, yz`，拉应力取正。邻单元应力未做节点平均。 |
| `stressTraction_MPa`、`stressNormal_MPa` | 由顶面外法向与 Cauchy 应力得到的 $\boldsymbol\sigma\mathbf n$ 和 $\mathbf n\cdot\boldsymbol\sigma\mathbf n$。 |
| `contactPressure_MPa`、`contactTraction_MPa` | 求解施加的罚接触压力幅值和三分量接触牵引，按当前面积表示。压力为非负值，法向加载的牵引 $z$ 分量向下。由离散的参考面积罚力除以当前与参考面积之比得到。 |
| `quadratureXYZ_mm`、`currentArea_mm2`、`outwardNormal` | 各应力和接触量的当前采样坐标、三角形面积和外法向。 |
| `contactActive`、`gapPenetration_mm`、`contactForce_N` | 激活的接触积分点、罚接触穿透量和积分得到的接触合力向量。 |
| `travel_mm`、`capDepth_mm`、`force_N`、`support_N`、`minJ`、`residual_N` | 对应行程、球冠深度、正值的压缩力幅值、竖向支座反力、最小体积比和自由自由度残差。 |

做反演核查时，可以用 `height_mm` 作为输入，用 `contactTraction_MPa` 或 `contactPressure_MPa` 核查恢复的接触载荷，用 `cauchy_MPa` 核查顶面材料应力。两类应力量在离散有限元中各有定义。接触边缘和应力峰值受当前网格与罚刚度影响，这批数据代表所选离散模型的合成基准。

## 复现

在工作区根目录运行。

```matlab
directory = run_inverse_reference();  % 新编号，两组依次运行
addpath('tests')
summary = verify_inverse_reference(directory);
```

也可用 `run_inverse_reference('my_run','centered')` 和 `run_inverse_reference('my_run','eccentric')` 分别运行并归在同一编号下。使用新编号可保留已有结果。快照随收敛步写出，整组达到目标行程后才生成该组的 `manifest.csv`。

## 本次运行核查

| 工况 | 快照数 | 末步行程 (mm) | 末步球冠深度 (mm) | 末步压缩力 (N) | 最大竖向力不平衡 (N) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 中心 | 20 | 3.72 | 2.8604 | 4.88989 | $1.07\times 10^{-8}$ |
| 偏心 | 20 | 3.72 | 2.8606 | 4.89385 | $1.68\times 10^{-8}$ |

已回读全部 40 份快照，检查无载平面、栅格高度与当前三角网格的一致性、接触牵引积分与求解反力的一致性，以及偏心工况的初始接触位置。仿射位移试验验证了顶面应力的提取。全程最小单元体积比约 0.401，最大罚穿透约 0.097 mm。 [本次核查汇总](../../results/stage-01-forward-model/task-03-inverse-reference/20261120_reference_v1/verification.csv)
