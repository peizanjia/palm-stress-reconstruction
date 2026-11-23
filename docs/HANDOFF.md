# 换设备续作

克隆 `main` 分支后，从仓库根目录运行下列命令。已归档的样本、有限元结果、模型权重、预测和图件随普通 Git 克隆下载。

```powershell
git clone https://github.com/peizanjia/palm-stress-reconstruction.git
cd palm-stress-reconstruction
```

有限元使用 MATLAB R2025b，单位为 mm、N、MPa。训练和绘图使用 Python 3.13.5、PyTorch 2.9.0、SciPy 1.15.3、NumPy 2.1.3 和 Matplotlib 3.10.0。装好这些依赖即可在 CPU 上复测和绘图。原 GPU 训练和计时采用 PyTorch CUDA 12.6 版及 RTX 4060 Laptop。重绘中文图件需系统安装支持中文的字体，如微软雅黑或 Noto Sans CJK SC。

## 从已归档结果继续

| 需要查看的内容 | 入口 |
| --- | --- |
| 建模、反演和 PINN 的技术讨论 | [项目交互记录](agent-interactions/README.md) |
| 圆片有限元、球头压入和基准观测 | [阶段 01](stage-01-forward-model/README.md) |
| 旧高度反演、已知球头的高度库和未知压头的曲面约束 | [有限元反演](stage-02-inverse/task-01-dirichlet-reconstruction.md) |
| 插值、Ritz 与单形状 PINN 的早期对照 | [加速验证](stage-02-inverse/surface-acceleration.md) |
| 多形状训练及测试分组和重跑流程 | [泛化验证](stage-02-inverse/height-generalization.md) |
| 扩样权重和逐帧评估 | [pinn.pt](../results/stage-02-inverse/surface-generalization/20261121_expanded_training/pinn.pt) · [评估汇总](../results/stage-02-inverse/surface-generalization/20261121_final_expanded/summary.json) |
| 对外说明的总览与流程图 | [总览 PNG](../results/stage-02-inverse/surface-generalization/20261122_email_figures/00_email_overview.png) · [可编辑 draw.io](../results/stage-02-inverse/surface-generalization/20261122_email_figures/05_technical_flow.drawio) · [全部图件](../results/stage-02-inverse/surface-generalization/20261122_email_figures/) |

`results/stage-02-inverse/surface-generalization/` 下的 `20261121_design`、`20261121_combined`、`20261121_extension` 分别保存主设计、复合留出和扩样数据，共 98 条轨迹、744 个非零帧。`20261121_final_baseline` 与 `20261121_final_expanded` 记录最终的测试。状态、历史失败及选择理由见 [.planning](../.planning/2026-11-21-palm-inverse-acceleration/)。

## 复测权重与重绘

以下是 PowerShell 命令。选用新运行编号，便于对比历史结果。复测读取全部已归档分组，不重新训练。CPU 和 GPU 的计时分别记录。

```powershell
$palmArchive = 'results/stage-02-inverse/surface-generalization'
$palmRun = Get-Date -Format 'yyyyMMdd_HHmmss'
python -B tools/train_generalization.py "$palmArchive/20261121_design" "$palmArchive/${palmRun}_cpu_recheck" --extra-data "$palmArchive/20261121_combined" --extra-data "$palmArchive/20261121_extension" --checkpoint-dir "$palmArchive/20261121_expanded_training" --paths pinn --fast-force --device cpu
$palmFigures = "$palmArchive/${palmRun}_redraw"
New-Item -ItemType Directory -Path "$palmFigures/source_data" -Force | Out-Null
Copy-Item "$palmArchive/20261122_email_figures/source_data/combined_05_fem.mat" "$palmFigures/source_data/"
python -B tools/plot_pinn_email_figures.py --output $palmFigures
```

在 MATLAB 中可独立校验正算、凸压头接触切线、基准快照和已知球头的高度匹配：

```matlab
addpath('tests')
verify_forward
verify_shaped_indentation
r = verify_inverse_reference(fullfile('results','stage-01-forward-model', ...
    'task-03-inverse-reference','20261120_reference_v1'));
disp(r)
[summary,details] = run_ball_height_inverse();
disp(summary)
```

## 当前验证边界

合成模型使用半径 25 mm、厚 5 mm 的圆片，底面及侧面固定，可压缩 Neo-Hookean 的剪切模量为 30 kPa、泊松比为 0.45。材料与边界均需实物标定。无摩擦接触的正算以增量 Newton 求解准静态平衡。整幅高度约束的逆算允许材料点沿曲面移动，结果验证的是当前模型和离散平衡分支的竖向合力。

- 球头中心和偏心的曲面反演，末帧合力误差分别为 −4.86% 和 −4.34%，平均约 4 s/帧。局部单边压力仍待验证。
- 扩样 PINN 在干净复合留出的 6 条轨迹、54 帧上，合力 MAE 为 0.01893 N，最大末帧误差为 3.42%，GPU 推理 P95 为 4.14 ms。计时从高度图已在 GPU 内存开始。
- 最深复合子集和叠加 25 µm 噪声的复合组仍有误差门槛未达。后续可优先验证材料及支承敏感性、有单边接触约束的压力反演、应变能项消融以及从 RGB 重建到实际控制的端到端时延。

新计算按阶段和运行编号落盘，新生成的 `results/` 默认由 Git 忽略。归档前可检查结果范围，再用 `git add -f <结果路径>` 纳入版本历史。论文原文与补充材料入口见 [论文与数据来源](paper-evidence.md)。
