# 掌面高度估力的仿真盲测

圆片半径 25 mm、厚 5 mm，底面和侧面固定。材料为可压缩 Neo-Hookean，剪切模量 30 kPa、泊松比 0.45。有限元接触正算生成当前坐标上的高度图和竖向反力。各估力方法从高度图得到竖向合力，压头几何、位姿、参考反力和实际行程用于离线构造工况与对齐评分。

## 数据划分和冻结

| 集合 | 轨迹 | 非零高度帧 | 用途 |
| --- | ---: | ---: | --- |
| 训练 | 42 | 318 | 拟合权重 |
| 开发 | 14 | 110 | 选择神经模型检查点和核岭超参数 |
| 常规盲测 | 9 | 63 | 新采样的球头、椭球和指数 4 圆角压头，沿用常规工况范围 |
| 复合外推盲测 | 6 | 54 | 指数 3 压头，偏心 8–10 mm、转角 60°–100°、倾角 23°–32°、最大行程 3.12–3.47 mm |

上一轮扩样是在查看旧复合留出误差后决定的。旧常规、复合及单因素测试均归入探索记录。新实验保留已经确定的训练区间和训练设置，先在 CPU 上重训并冻结，再生成新的最终工况。两套盲测均用预先登记的种子独立分层采样。默认展示复合第 05 条末步，选择在测试之前写入配置。

盲测入口核对冻结时的评估代码、正算及反演实现、参考网格、协议和三个模型权重的 SHA-256。MATLAB 样本记录对应的冻结指纹。训练只载入训练与开发轨迹的高度和标签。最终评估逐轨迹检查编号、压头几何与位姿、有效输入高度帧哈希，并要求两批所有指定行程齐全、正算体积比为正、力平衡及残差通过。所有非零载荷步进入主要统计，任何不完整或不满足正算质量标准的轨迹使整次评估中止。

[盲测协议](../../config/generalization_blind.json) · [权重冻结清单](../../results/stage-02-inverse/surface-generalization/20261121_expanded_training/freeze.json) · [逐轨迹测试清单](../../results/stage-02-inverse/surface-generalization/20261121_final_expanded/coverage.csv)

## 比较口径

- 核岭和纯图像模型提供基线。能量 PINN 由局部图与形状统计预测 28 个内部位移系数，使用已知圆片网格和材料对应力积分。两个神经模型使用同样的 120 轮预算和高度噪声增强，用开发集选择检查点。
- 各方法在同一批高度图上预测。标准差 25 µm 的高斯高度噪声使用固定种子。每个测试组的噪声实现跨方法完全相同。复合深压子集直接使用同一批预测中行程超过 3.12 mm 的帧。
- 另对全部 54 帧干净复合高度图运行全阶曲面约束有限元反演。每帧从平坦参考构型开始，0.50 mm 高斯平滑和由高度图决定的网格加密中心均预先设定。反演与 PINN 的合力共同对照该帧的有限元接触正算。若有不收敛则保留失败记录，成对 MAE 在两者都有数值的帧上计算。
- 学习方法的单帧 P95 从高度图已在计算设备上开始，包括预处理，先进行五次预热。归档使用 AMD Ryzen 7 PRO 5850U 和 2 个计算线程。全阶反演为并行离线评估，计时随各帧记录，相机、设备传输和闭环延迟需要单独测量。

## 盲测结果

下表的误差是竖向合力相对于有限元接触正算的 MAE。两个神经模型的冻结权重分别来自纯图像第 32 轮和 PINN 第 110 轮。

| 测试组 | 帧 | 核岭 MAE (N) | 纯图像 MAE (N) | PINN MAE (N) | PINN 最差末帧误差 | PINN CPU P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 常规 | 63 | 0.02310 | 0.03774 | 0.00635 | 1.73% | 6.17 |
| 常规，25 µm 噪声 | 63 | 0.02363 | 0.03792 | 0.03912 | 4.86% | 6.19 |
| 复合外推 | 54 | 0.07115 | 0.14072 | 0.02484 | 3.56% | 5.76 |
| 复合外推，25 µm 噪声 | 54 | 0.07070 | 0.14085 | 0.05068 | 3.68% | 5.98 |
| 复合外推，行程 > 3.12 mm | 6 | 0.21552 | 0.39748 | 0.07901 | 3.56% | 5.80 |

核岭在常规加噪组的 MAE 最低。PINN 的复合加噪和深压子集超过原 0.05 N MAE 目标。本机复合盲测的 CPU P95 高于原 5 ms 目标。

全阶曲面约束反演在 54 帧干净复合盲测上全部收敛。与同一有限元正算参考相比，全阶反演的合力 MAE 为 0.07428 N，PINN 为 0.02484 N，在这批工况上低 66.6%。这个差值比较的是当前反演设置下的竖向合力。

[三种模型合力指标](../../results/stage-02-inverse/surface-generalization/20261121_final_expanded/summary.json) · [同帧全阶反演指标](../../results/stage-02-inverse/surface-generalization/20261121_final_expanded/surface_inverse_summary.json) · [图件目录](../../results/stage-02-inverse/surface-generalization/20261122_email_figures/)

所有结果来自同材料与边界条件的仿真。三维变形和剖面用于显示估力路径。局部压力、接触区、实物材料标定和实时控制需另做验证。

## 从头复现

以下 Bash 命令在仓库根目录执行。需要 MATLAB R2025b、Python 3.11 或以上、PyTorch、SciPy、NumPy、Matplotlib 和中文字体。已归档运行使用 Python 3.14.4、PyTorch 2.15.0 CPU、SciPy 1.18.1、NumPy 2.3.5。Windows 可在 MATLAB 中执行相同的函数调用并调整终端的变量语法。

~~~bash
palm_archive=results/stage-02-inverse/surface-generalization
palm_run="$palm_archive/my_reproduction"
export OPENBLAS_NUM_THREADS=2

python -B tools/train_generalization.py train "$palm_archive/20261121_design" "$palm_run/training" --extra-data "$palm_archive/20261121_extension" --noise-aug

matlab -singleCompThread -batch "run_generalization_samples('$palm_run/regular',1,1,'blind_regular','$palm_run/training/freeze.json'); run_generalization_samples('$palm_run/combined',1,1,'blind_combined','$palm_run/training/freeze.json');"

python -B tools/train_generalization.py evaluate "$palm_run/regular" "$palm_run/evaluation" --extra-data "$palm_run/combined" --checkpoint-dir "$palm_run/training"

mkdir -p "$palm_run/figures/source_data"
matlab -singleCompThread -batch "run_blind_surface_inverse('$palm_run/combined','$palm_run/evaluation/surface_inverse','$palm_run/training/freeze.json'); export_pinn_figure_geometry('$palm_run/combined/blind_combined_05.mat','$palm_run/figures/source_data/blind_combined_05_fem.mat');"

python -B tools/plot_pinn_email_figures.py --checkpoint "$palm_run/training" --evaluation "$palm_run/evaluation" --samples "$palm_run/combined" --output "$palm_run/figures"
~~~

同一冻结清单下，多进程 MATLAB 生成可使用不同的 `worker` 参数将轨迹均分。重新使用已公开的种子是复现。如更改模型设计，应在新协议中预先指定一批新的测试工况。
