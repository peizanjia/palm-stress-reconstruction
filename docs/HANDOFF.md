# 跨设备续作

从仓库根目录运行。样本、有限元结果、模型权重和图件均可随普通 Git 克隆下载。单位为 mm、N、MPa。重绘中文图件需安装 Noto Sans CJK SC 或微软雅黑。

| 内容 | 入口 |
| --- | --- |
| 建模、反演和 PINN 的讨论 | [项目交互记录](agent-interactions/README.md) |
| 圆片、球头压入和基准观测 | [正问题](stage-01-forward-model/README.md) |
| 高度曲面约束与全阶有限元 | [有限元反演](stage-02-inverse/task-01-dirichlet-reconstruction.md) |
| 球头插值、Ritz 与早期 PINN | [历史对照](stage-02-inverse/surface-acceleration.md) |
| 最终测试协议和从头运行命令 | [盲测流程](stage-02-inverse/height-generalization.md) |
| CPU 模型、数据划分及逐帧预测 | [权重](../results/stage-02-inverse/surface-generalization/20261121_expanded_training/) · [逐帧评估](../results/stage-02-inverse/surface-generalization/20261121_final_expanded/) |
| 对外交流图片 | [总览 PNG](../results/stage-02-inverse/surface-generalization/20261122_email_figures/00_email_overview.png) · [可编辑 draw.io](../results/stage-02-inverse/surface-generalization/20261122_email_figures/05_technical_flow.drawio) · [全部图件](../results/stage-02-inverse/surface-generalization/20261122_email_figures/) |

文件路径沿用既有的权重、评估和图件目录，新内容来自 2026-11-23 的 CPU 重训与仿真盲测。旧开发样本在 `20261121_design`、`20261121_extension`。原先用于调整方案的复合留出样本在 `20261121_combined`。最终高度观测在 `20261123_blind_regular` 和 `20261123_blind_combined`。

## 复测已冻结权重

~~~bash
palm_archive=results/stage-02-inverse/surface-generalization
OPENBLAS_NUM_THREADS=2 python -B tools/train_generalization.py evaluate "$palm_archive/20261123_blind_regular" "$palm_archive/my_cpu_recheck" --extra-data "$palm_archive/20261123_blind_combined" --checkpoint-dir "$palm_archive/20261121_expanded_training"
python -B tools/plot_pinn_email_figures.py
~~~

复测命令写入新目录，已归档图件也可直接重绘。CPU 计时随硬件和系统负载变化。更改求解器、PINN 或测试协议之后需要按 [盲测流程](stage-02-inverse/height-generalization.md) 重新冻结。MATLAB 回归可运行 `addpath('tests'); verify_forward; verify_shaped_indentation`。论文与补充材料入口见 [论文与数据来源](paper-evidence.md)。

新计算按阶段和运行编号落盘。新的 `results/` 内容默认由 Git 忽略，确认归档范围后可使用 `git add -f <结果路径>`。
