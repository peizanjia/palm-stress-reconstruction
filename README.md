# TacPalm 硅胶圆片的超弹性有限元与接触力反演

用 MATLAB 求解底面、侧面固定的三维圆片，在给定顶面名义牵引下的准静态有限变形。模型以 TacPalm 论文中的硅胶层为原型。材料、尺寸及 Fig. 3c 的出处见 [论文证据](docs/paper-evidence.md)。


在新设备上从[续作指南](docs/HANDOFF.md)开始，可直接查看已归档的输入数据、正反问题结果和复现入口。
## 运行

在工程根目录运行，已在 MATLAB R2025b 验证，代码使用基础 MATLAB。

```matlab
r = run_forward();
```

参数集中在 [config/forward_config.m](config/forward_config.m)，单位统一为 **mm、N、MPa**。默认圆片半径为 25 mm，以论文手掌半径近似。厚度 5 mm、剪切模量 0.030 MPa、泊松比 0.45 是演示假设。默认从零逐步加载至 0.5 N，参考构形中的圆形载荷斑半径为 3 mm。

可以直接修改配置，也可以在 MATLAB 中覆盖某些设置。例如将载荷移到顶面偏心位置，并使它的合力达到 0.3 N。

```matlab
addpath('config')
cfg = forward_config();
cfg.load.center = [2, 0];
cfg.load.magnitude = 0.3;
r = run_forward(cfg);
```

| 配置 | 含义 |
| --- | --- |
| `geometry.radius`、`geometry.thickness` | 硅胶圆片半径和厚度 |
| `material.mu`、`material.nu` | 小应变剪切模量、泊松比，程序换算第一 Lamé 常数 |
| `boundary.fixSide` | 默认固定侧面全部位移。设为 `false` 可比较底面固定、侧面自由的情况 |
| `mesh.innerRadius`、`innerSpacing`、`outerSpacing`、`axialLayers` | 中心细化半径、内外区目标径向间距、厚向层数 |
| `load.mode` | `"force"` 规定总力 N，`"peak"` 规定单位幅值处的名义牵引 MPa |
| `load.profile` | `"smooth"` 光滑圆斑、`"uniform"` 均布圆斑、`"full"` 全顶面均布、`"custom"` 自定义权重 |
| `load.center`、`load.radius` | 两种圆斑在参考构形中的圆心与半径 |
| `load.custom` | 输入积分点的参考坐标 `[X,Y]`，返回每点的权重列向量，范围为 [0,1] |
| `solver`、`output` | 载荷步和 Newton 容差，输出开关与目录 |

对偏心或窄载荷斑，调整网格使加载区内有多个单元。默认细化以圆片中心为圆心。如果载荷斑远离中心，可减小 `outerSpacing`。均布圆斑的间断边界由积分点采样，边缘应力和近边缘位移应做加密检查。

## 输出

每次成功求解写入配置指定目录，默认是 `results/forward/`。重复运行同一目录会覆盖结果。

- `forward.mat` 保存参考与当前网格、楔体及顶面拓扑、位移、约束反力、单元质心 Cauchy 应力和体积比。
- `top_surface.csv` 保存参考坐标、当前坐标和位移，顶面观测使用当前坐标 `(x,y,z)`。MAT 文件中的 `r.mesh.top`（载入后为 `result.mesh.top`）提供三角连接，可在当前 `(x,y)` 上采样，前提是投影后表面仍为单值。
- `load_history.csv` 保存收敛的载荷比例、Newton 迭代次数、自由自由度的残差范数及高斯点最小 `J`。

MATLAB 中可直接检查网格和三维顶面。

```matlab
trisurf(r.mesh.top, r.currentNodes(:,1), r.currentNodes(:,2), ...
        r.currentNodes(:,3), r.displacement(:,3), 'EdgeColor', 'none');
axis equal tight
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
colorbar
```

## 目录约定

```text
run_forward.m             正问题入口
config/                   工况与数值参数
src/+palm/
  +mesh/                  圆片布点、Delaunay 三角网格、轴向挤出
  +material/              超弹性本构
  +fem/                   单元积分、装配、变形梯度检查
  +load/                  顶面名义牵引离散
  +solve/                 增量 Newton 平衡求解
  +output/                结果整理和导出
docs/                     文献证据、弱式和验证
tests/                    数值回归检查
results/                  自动生成的工况输出
```

验证入口为 `addpath('tests'); verify_forward`。离散公式、近不可压缩限制和数值结果见 [有限元方法](docs/forward-method.md)。

## Fig. 3c 三维球压入

`r = run_indentation();` 求解刚性半球压入，保持完整三维自由度。参数集中于
[config/indentation_config.m](config/indentation_config.m)，可设置厚度、压头半径、偏心位置、位移序列和罚系数。
`run_thickness_study()` 运行多厚度比较，使用 [论文原始数据](data/README.md)。
[三维接触方法和厚度研究](docs/indentation-study.md) 说明接触量、匹配指标和数值可靠性。

## 阶段与任务归档

[阶段 01 正问题建模索引](docs/stage-01-forward-model/README.md) 关联给定牵引、厚度选择和反演核查三个任务。
中心和偏心压入的高度与应力数据由 `run_inverse_reference()` 生成，格式见
[任务 03 数据说明](docs/stage-01-forward-model/task-03-inverse-reference.md)。

[阶段 02 · 任务 01 全位移边界反演](docs/stage-02-inverse/task-01-dirichlet-reconstruction.md) 对照完整顶面位移与仅高度两种逆计算，归档接触力场与载荷–深度曲线。

已知球头和压入位置时，`[summary,details] = run_ball_height_inverse()` 用正算高度库拟合行程与竖向合力，并对归档中心、偏心工况做逐帧留出验证。见[高度场匹配反演](docs/stage-02-inverse/task-01-dirichlet-reconstruction.md#已知球头时的高度场匹配)。

未知压头时，`run_dirichlet_inverse("new_run","centered","surface")` 用栅格高度做[曲面约束反演](docs/stage-02-inverse/task-01-dirichlet-reconstruction.md#未知压头时的曲面约束反演)。能量 PINN、Ritz 与插值的训练和比较见[加速验证](docs/stage-02-inverse/surface-acceleration.md)。

不同压头形状、位置、姿态和深度的采样与整轨迹测试见[高度估力泛化](docs/stage-02-inverse/height-generalization.md)。

仓库已归档 98 条多形状压入轨迹、扩样模型权重、逐帧测试结果和 [PINN 图件](results/stage-02-inverse/surface-generalization/20261122_email_figures/)。从 [续作指南](docs/HANDOFF.md) 可直接复测已训练模型或重新绘图；[交互记录](docs/agent-interactions/README.md) 收录了建模、反演和训练的技术讨论。
