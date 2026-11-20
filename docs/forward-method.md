# 三维可压缩 Neo-Hookean 有限元

## 边值问题

参考构形 $\Omega_0=\{(X,Y,Z): X^2+Y^2\le a^2,\ 0\le Z\le H\}$。默认底面和侧面三向位移均为零。忽略体力和惯性。在参考顶面给定位移无关的竖直名义牵引 $\bar{\mathbf T}=-p_0(X,Y)\mathbf e_z$，其单位为参考面积上的力。

令 $\mathbf F=\mathbf I+\nabla_X\mathbf u$，$J=\det\mathbf F>0$。采用与 DOLFINx 超弹性示例一致的对数体积项。 [DOLFINx Hyperelasticity](https://docs.fenicsproject.org/dolfinx/main/cpp/demos/demo_hyperelasticity.html)

$$
W(\mathbf F)=\frac{\mu}{2}(\mathbf F:\mathbf F-3)-\mu\ln J+\frac{\lambda}{2}(\ln J)^2,
\qquad
\mathbf P=\mu\mathbf F+(\lambda\ln J-\mu)\mathbf F^{-T}.
$$

$$
\lambda=\frac{2\mu\nu}{1-2\nu},\qquad
\boldsymbol\sigma=J^{-1}\mathbf P\mathbf F^T.
$$

强式为域内 $\operatorname{Div}_X\mathbf P=0$，顶面 $\mathbf P\mathbf N=\bar{\mathbf T}$。取约束边界上为零的试函数 $\mathbf v$，总拉格朗日弱式为

$$
R(\mathbf u;\mathbf v)=\int_{\Omega_0}\mathbf P:\nabla_X\mathbf v\,\mathrm dV
-\int_{\Gamma_t}\bar{\mathbf T}\cdot\mathbf v\,\mathrm dA=0.
$$

设 $\mathbf Q=\mathbf F^{-T}$，一致材料切线为

$$
\frac{\partial P_{iJ}}{\partial F_{kL}}
=\mu\delta_{ik}\delta_{JL}
+\lambda Q_{iJ}Q_{kL}
-(\lambda\ln J-\mu)Q_{iL}Q_{kJ}.
$$

以该切线装配内力的 Jacobian。当前载荷固定在参考坐标，外力切线为零。若以后输入当前表面的法向压力 $p_c$，相应的名义牵引是 $-p_cJ\mathbf F^{-T}\mathbf N$，需要随构形更新并线性化。若输入 Fig. 3c 的刚性半球位移，需要另外求解非穿透和无摩擦接触条件及反力。

## 网格和求解

在圆盘内按同心环布点，在加载区附近缩小点间距，使用 MATLAB 的 Delaunay 三角剖分。外周是内接多边形。将三角剖分等距挤出成六节点直楔体，形函数在三角截面和轴向分别线性。体积分采用三角面 3 点乘轴向 Gauss 2 点，顶面等效结点力采用 7 点三角积分。三维圆片支持偏心载荷。若只考虑同轴压入，二维轴对称单元也适用，计算成本更低。

默认用光滑的紧支撑载荷斑 $p_0=p_\mathrm{amp}\max(1-r^2/b^2,0)^2$，其中 $r$ 是到载荷中心的参考距离，$b$ 是输入的斑半径。力控制按顶面积分归一化 $p_\mathrm{amp}$。支座反力由固定自由度上的内外力差计算。

每个增量用一致切线 Newton 法求自由自由度上的平衡，残差范数回溯用于缩短更新步，失败时减半载荷增量。对直楔体，$J$ 在三角截面内仿射、沿轴向为二次多项式，因此可求三条轴向棱上的最小值以筛查整个单元内的翻转。正 $J$ 是局部取向检查。Newton 收敛给出从无载构形跟踪到的数值平衡，整体稳定性需要另行验证。

该位移型一阶单元适合建立可压缩基准。$\nu$ 接近 0.5 时应检查体积锁死，尤其是受侧壁约束的压缩问题。若需要近不可压缩的定量预测，采用经验证的混合位移–压力单元或其他抗锁死离散，并重新验证网格收敛。默认的 $\nu=0.45$ 只用于演示。

## 数值验证

在 MATLAB R2025b 运行 `addpath('tests'); verify_forward`。仿射压缩补丁测试检查了常应力、体积比和自由节点内外力平衡。一致切线相对中心差分误差为 $3.26\times10^{-11}$。同时检查了高斯点间的楔体翻转、零载荷、小载荷比例响应、偏心均布圆斑、底部固定和自定义牵引。

用以下代码在相同材料、几何、约束和 0.5 N 载荷下比较网格。三组网格同时缩小面内点间距并增加厚向层数。

```matlab
addpath('config')
base = forward_config();
base.output.write = false;
base.solver.verbose = false;
for level = 0:2
    scale = 0.8^level;
    cfg = base;
    cfg.mesh.innerSpacing = base.mesh.innerSpacing * scale;
    cfg.mesh.outerSpacing = base.mesh.outerSpacing * scale;
    cfg.mesh.axialLayers = ceil(base.mesh.axialLayers / scale);
    r = run_forward(cfg);
    fprintf('%d wedges, depth %.8f mm, min J %.8f, imbalance %.3e N\n', ...
        size(r.mesh.wedges,1), r.maxIndentation, r.minJ, norm(r.forceImbalance));
end
```

| 楔体数 | 厚向层数 | 最大下陷 (mm) | 单元内最小 $J$ | 合力不平衡 (N) |
| ---: | ---: | ---: | ---: | ---: |
| 5,124 | 6 | 1.038594 | 0.724505 | $6.38\times10^{-16}$ |
| 11,160 | 8 | 1.126789 | 0.673165 | $1.97\times10^{-15}$ |
| 20,690 | 10 | 1.204555 | 0.627949 | $2.32\times10^{-14}$ |

末两档的压深仍相差约 7%。三档均满足离散力平衡，默认网格用于快速演示。压深和局部应力的定量使用需要继续加密加载区及厚度方向，再按所需精度比较结果。
