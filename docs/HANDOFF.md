# 换设备续作

在新设备克隆本仓库，于 MATLAB 中切换到仓库根目录。原计算使用 **MATLAB R2025b** 和基础 MATLAB。模型使用 mm、N、MPa，结合各表列名读取结果。已归档的 MAT、CSV、PNG、FIG 和计算日志随普通 Git 克隆获取。

```matlab
addpath('tests')
verify_forward
r = verify_inverse_reference(fullfile('results','stage-01-forward-model', ...
    'task-03-inverse-reference','20261120_reference_v1'));
disp(r)
```

## 当前模型和结论

完整三维圆片半径 25 mm、厚度 5 mm。底面和侧面固定，刚性半球压头半径 4 mm。基准采用可压缩 Neo-Hookean，剪切模量 30 kPa、泊松比 0.45。厚度和材料值均为工作设定，尚未用 TacPalm 原样独立标定。中心压入坐标为 `(0,0)` mm，偏心压入为 `(6,0)` mm。

- [阶段 01 正问题](stage-01-forward-model/README.md) 包含给定牵引、Fig. 3c 厚度比较、中心及偏心合成数据，以及可压缩性对照。中心和偏心各有 20 个高度、应力及接触牵引快照。不可压缩主对照的共同收敛区间到 2.92 mm 行程。
- [阶段 02 全位移边界反演](stage-02-inverse/task-01-dirichlet-reconstruction.md) 用同一批快照比较完整顶面位移与仅高度。完整位移恢复离散节点力。仅高度并冻结横向位移时，虽仍能定位接触斑，两个工况的末步竖向合力高估 37.5% 和 35.4%。
- [反演曲线](../results/stage-02-inverse/task-01-dirichlet-reconstruction/20261120_inverse_v1/figures/load_depth.png) 和 [数值汇总](../results/stage-02-inverse/task-01-dirichlet-reconstruction/20261120_inverse_v1/summary.csv) 可先于重新计算阅读。
- [项目交互记录](agent-interactions/README.md) 包含初始研究说明、主线讨论，以及不可压缩条件和实时计算技术路线的补充讨论。

## 重跑与后续工作

`run_forward()` 用于验证给定牵引模型。运行前可将 `cfg.output.directory` 指向新的结果目录，以保留已发布的牵引案例。`run_indentation()` 进入球压入求解，使用 [默认压入配置](../config/indentation_config.m)。反演快照与三维逆计算使用独立运行编号归档。原始快照目录在 [反演入口](../run_dirichlet_inverse.m) 的 `source` 中指定。新参考数据要先在该入口指定相应源路径。

下一项研究可基于现有反演对照，给顶面切向自由度引入无摩擦条件，再以形状偏差、竖向合力误差和接触区重合度评价。另一条已讨论的路线是用离线有限元样本建立在线合力估计器。实时性能和真实硅胶的反演精度尚待验证。

后续产生的文件按阶段、任务和运行编号保存。历史基准已纳入 Git，新生成的 `results/` 文件默认忽略，选择归档范围后可用 `git add -f <结果路径>` 加入。论文原文与补充材料由 [论文与数据来源](paper-evidence.md) 提供，仓库内保留复算需要的 Fig. 3c 数值表。
