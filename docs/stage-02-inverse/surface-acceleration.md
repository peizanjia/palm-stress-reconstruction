# 高度曲面反演的加速验证

目标是用当前坐标上的高度图估计硅胶圆片的向下合力。全阶曲面反演求解时不使用压头形状。加速原型以固定几何和已知 Neo-Hookean 参数为前提，输入不包含压头行程、半径、中心坐标或水平位移。合力和完整牵引场分别验收。

## 文献与方法选择

| 原始资料 | 可用于本项目的方法 | 算例范围 |
| --- | --- | --- |
| [Bai 等，大变形无摩擦接触能量 PINN](https://arxiv.org/html/2411.03671)，第 2–4 节 | 用超弹性应变能和接触势训练，将固定约束写进位移参数化。 | 大变形算例采用平面应变和已指定的接触几何。逐载荷步的训练时间不能作为本项目的在线延迟。 |
| [Sahin、Wolff 与 Popp，三维接触 PINN](https://arxiv.org/html/2412.09022)，第 3.2、4 节 | 混合位移–应力输出，用 Fischer–Burmeister 残差处理间隙–压力互补。 | 小变形弹性。移植时需要换成有限变形的几何和牵引。 |
| [Du、Guo 与 He，有限应变可微混合近似](https://arxiv.org/html/2407.11183)，第 3、5 节 | 将数值基函数和神经参数化放进有限应变弱式，为离散能量训练提供实现依据。 | 论文的反问题辨识材料参数，本项目估计接触力。 |
| [Farhat、Chapman 与 Avery，能量一致超降阶](https://doi.org/10.1002/nme.4820)，论文摘要 | 用带权积分单元近似投影后的非线性残差和切线。 | 文献速度来自结构动力学算例。本次 Ritz 原型仍使用全部楔体。 |

检索日期为 2026-11-21。使用 Agent Reach 的 Exa 和 Codex 学术检索，再按 arXiv 标识或 DOI 阅读原文。关键词包括 `variational deep energy method finite strain hyperelasticity contact inverse`、`physics-informed neural networks frictionless contact mechanics Hertz KKT`、`energy conserving sampling weighting hyper reduction nonlinear finite element`。这是面向实现的定向检索。

## 已执行原型

- **能量 PINN**。以最深处为局部中心，从中心及邻近位置提取 13 个高度特征。MLP 输出 12 个三维位移模式系数。基底和侧缘固定，顶面竖向位移在移动后的平面位置重采样。合力由第一 Piola 应力对顶面竖向虚位移积分得到。以反力误差和三维应变能共同训练，另做应变能权重为零的同结构消融。
- **Ritz 降阶**。使用相同的 12 个位移模式和全网格能量，对每张高度图用 L-BFGS 求系数。该方法不使用力标签。
- **最深点到合力的单调插值**。用中心标定曲线进行 PCHIP 插值，超过标定范围时返回边界值并统计截断。输出为竖向合力。

中心轨迹的 19 个非零帧分为 14 个训练帧、5 个验证帧，验证步号为 3、7、11、15、18。偏心轨迹作为整条测试轨迹。训练用种子 321、Adam、学习率 0.003、90 轮、每轮四个小批量。能量权重为 0.03，按最大下陷深度的平方归一化能量，力误差分母保底 0.15 N。用中心验证集选择检查点。零载单独回放。另以位置 (3,4) mm 的两条新轨迹测试位置和半径变化，球头半径分别为 4 mm 和 5 mm。

同一网格的 FEM 归档终态位移导入 PyTorch 积分后得到 4.889881 N，MATLAB 归档值为 4.889891 N。这项检查核对积分和合力符号。代理输出的最小 J 在体积 Gauss 点统计。完整局部平衡和接触压力精度需单独验证。

### 同半径独立测试

原始 PINN 把位置变化后的测量图放在固定中心细化网格上，在归档偏心轨迹上 MAE 为 0.111 N。将从高度图识别的最深位置移到同一网格的细化中心后，偏心 MAE 降至 0.0146 N。冻结这一修正后，在新生成的 4 mm 球头、(3,4) mm 轨迹上得到下表。GPU 方法的时间包含驻留在 GPU 上的高度图预处理，PCHIP 为内存中的 CPU 高度图。相机和设备传输延迟没有计入。

| 方法 | 新位置 MAE (N) | 新位置最大载荷误差 | 新位置 P95 (ms) | 5 mm 新半径 MAE (N) | 5 mm 最大载荷误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 能量 PINN | 0.00886 | −0.27% | 3.64 | 0.412 | +21.66% |
| 能量权重为零 | 0.00681 | −0.58% | 3.96 | 0.458 | +23.95% |
| Ritz | 0.443 | +5.73% | 523.7 | 0.268 | −2.18% |
| PCHIP | 0.00142 | −0.08% | 0.024 | 0.710 | −28.04% |

单半径训练达到同半径新位置的合力目标。PCHIP 的误差和延迟最低，适用于球头族已标定的控制场景。Ritz 的中浅载荷误差较大。增大球头半径后的误差说明训练需要覆盖接触形状变化。

### 多半径修正

新增 3.5 mm 和 5.5 mm 两条训练轨迹，合并中心 4 mm 数据，共 43 个非零训练帧。4.5 mm 轨迹的 13 个非零帧整条用作开发验证。新测试采用未参与任何拟合的 4.75 mm 球头和 (−4,3) mm 位置，共 13 个非零帧。另回测既有 4 mm 和 5 mm 测试轨迹。四条新采样轨迹均完成预设载荷步，一共 50 个非零帧，单线程生成用时约 172 s。

Ritz 与神经原型共用位移模式和积分网格。Ritz 每帧迭代搜索系数，两种神经版本用一次网络前传提供系数，再做完整的 7,210 楔体反力积分。

以 4.5 mm 开发轨迹选择能量权重，在 0.001、0.003、0.01、0.03 中选中 0.001，入选训练约 9.0 s。4.75 mm 独立轨迹上，PINN 的 MAE 为 0.01016 N、最大绝对误差 0.03207 N、最大载荷误差 −0.11%，GPU 推理 P95 为 3.77 ms。同结构能量权重为零时，MAE 为 0.01482 N、最大绝对误差 0.10225 N。PINN 在此前 5 mm 轨迹的复测中将 MAE 从 0.412 N 降至 0.01252 N，末步由 +21.66% 变为 −0.28%。4.75 mm 是首轮多半径训练前未使用的轨迹。同一条在权重选择后的重复评分用于比较，最终打分未参与权重选择。

[权重选择](../../results/stage-02-inverse/surface-acceleration/20261121_multi_radius_tuned/pinn_selection.json) · [多半径逐工况统计](../../results/stage-02-inverse/surface-acceleration/20261121_multi_radius_tuned/summary.json)

### 高度噪声与修正

未滤波模型在 25 µm 独立像素高斯噪声下的 4.75 mm 轨迹 MAE 超过 0.05 N。校正方案在训练和推理前对高度采用同样的 0.35 mm 高斯滤波，重训时保留能量权重 0.001。用相同的种子 183 和 25 µm 扰动对照，4.75 mm 轨迹的 MAE 从 0.1351 N 降到 0.0330 N，最大载荷误差从 +5.32% 降到 +1.54%，滤波后 P95 为 4.86 ms。滤波模型在干净高度上的相应指标为 0.0153 N、+1.35% 和 4.76 ms。这是特定合成噪声下的力估计验收，浅载峰值相对误差仍需看逐帧结果。空间相关噪声、遮挡和实际控制链路待实测。

[滤波模型与干净测试](../../results/stage-02-inverse/surface-acceleration/20261121_filtered_train/summary.json) · [25 µm 噪声测试](../../results/stage-02-inverse/surface-acceleration/20261121_filtered_noise25/summary.json) · [同种子未滤波对照](../../results/stage-02-inverse/surface-acceleration/20261121_unfiltered_noise183/summary.json)

## 扩展采样与训练方案

1. **标定范围。** 以实测厚度、泊松比和剪切模量作为条件。当前固定 H=5 mm、μ=30 kPa、ν=0.45。首批球头半径取 3.5–5.5 mm，偏心距取 0–10 mm。用面积均匀的尺度采偏心半径和角度，另留同圆周不同角度作对称性检查。压入行程先取 0.05–3.72 mm，按几何、收敛性和顶面单值条件截断。平头、椭球和多点接触使用独立工况族及独立真值。
2. **按轨迹采样和划分。** 首批规划 96 条球头轨迹，采用固定种子的 Sobol 序列或分层设计。整条分为 60 条训练、16 条开发、20 条测试。外推集留出边界半径或偏心段。每条保存零载、0.05–0.40 mm 的 6 个密集浅载步及约 14 个中深步。记录停止原因、翻转和高度失去单值等失败，失败帧不拿来训练。误差大的区间在开发阶段加密，冻结最终测试集。
3. **字段与测量扰动。** 保存欧拉高度、有效掩码、三维节点位移、节点力、原始牵引、三分量合力、接触斑、最小 J、平衡残差。网格差异通过同一空间栅格或显式插值处理。按完整轨迹分组，加载和卸载成对进入同组。实测按试件、压头和批次分组。用 10、25、50 µm 高度噪声、栅格平移、零点漂移和缺像素做开发阶段敏感性测试，再以采集装置的实测噪声校准。
4. **训练和加速。** 先按固定材料和完整高度训练，做能量权重消融，再用局部弱平衡或混合位移–应力输出验证压力。训练积分覆盖接触斑边缘及远场。进一步做超降阶时，仅用训练轨迹拟合非负积分权重，在开发集用完整积分复核。每组保存参数、种子、样本索引、训练时间和权重。
5. **验收。** 此次原型工作目标采用 MAE ≤0.05 N、最大载荷误差 ≤5%、高度就绪后 P95 ≤5 ms。另报全程最大绝对误差和最小 Gauss 点 J。浅载以绝对力误差为主。压力场另比较接触面积 IoU、压力误差、拉力积分和实际接触外牵引。控制验收在目标硬件测相机曝光到控制命令的延迟、P99 和超时率，并以同步测力核对真实加载与卸载。

## 复现

```matlab
run_surface_acceleration_samples(fullfile('results','stage-02-inverse', ...
    'surface-acceleration','my_new_samples'));
run_surface_acceleration_samples(fullfile('results','stage-02-inverse', ...
    'surface-acceleration','my_multi_radius'),'multi_radius');
```

```powershell
python tools/train_surface_surrogate.py results/stage-02-inverse/surface-acceleration/my_training --held-out eccentric --canonical-center
python tools/train_surface_surrogate.py results/stage-02-inverse/surface-acceleration/my_test --held-out eccentric --canonical-center --checkpoint-dir results/stage-02-inverse/surface-acceleration/my_training --extra-test-dir results/stage-02-inverse/surface-acceleration/my_new_samples
```

多半径运行时，用两次 `--train-trajectory` 传入 `train_small.mat` 和 `train_large.mat`，用 `--validation-trajectory` 指定 `validation.mat`，用 `--extra-test` 指定 `new_shape.mat`。`--physics-weight 0.001 0.003 0.01 0.03` 按开发轨迹选权重。`--height-filter-mm 0.35` 在训练和评分时使用同一滤波。用 `--test-noise-um 25 --noise-seed 183` 复现对应噪声测试。

Python 环境需要 NumPy、SciPy 和 PyTorch。默认使用 CUDA，也可指定 `--device cpu`。输出目录包含配置、权重选择记录、模型权重、`summary.json` 和逐帧 CSV。
