# TacPalm 手掌材料、尺寸与 Fig. 3c

核查日期 2026-11-19。原文为 Zhang, Ren et al., *Nature Communications* 16, 2395 (2025)。核对了工作区正文、出版社补充信息 Fig. S1 和 S3，以及 Source Data 中的 Fig. 3c 工作簿。 [论文与下载入口](https://doi.org/10.1038/s41467-025-57741-6)

| 项目 | 证据与取值 |
| --- | --- |
| 手掌主体弹性层 | 正文 Methods 的 *Palm fabrication and assembly*（p. 10）标注液态 Silicone 00-30，Wesitru Inc.，两组分 1:1。以亚克力板作模具底面直接浇注，50 °C 固化 4 h，脱模时保留底部黏结。随后喷涂灰色硅胶油墨及表面硅油。 |
| 本构参数 | 正文和补充资料未报告这层手掌材料的剪切模量、泊松比、体积模量或 NH 拟合系数。正文 p. 3 的 Ecoflex 00-30（0.07 MPa）和 Dragon Skin 10（0.16 MPa）均属于气动手指。 |
| 半径 | 正文 p. 3 的 Fig. 2c 光学模型给出手掌半径 $r_p=25$ mm。以此作为圆片模型半径。手掌透明弹性层的独立裁切直径和有效夹持半径仍待确认。 |
| 硅胶层厚度 | 正文 Fig. 2b–c、Methods 和补充 Fig. S1 均未给出可靠的硅胶层厚度数值。图中的 24 mm 为相机至亚克力层的距离，Fig. 2a 的 45 mm 标注属于手掌总成。正文 30.08 × 18.80 mm 为有效成像矩形。 |
| Fig. 3c 探头与被压件 | 运动平台经力传感器将 Probe B 沿法向压在装配后的手掌感测表面。B 为半球头，Methods p. 11 给出 $r_f=4$ mm，即头部直径 8 mm。探头杆长、材质与试验中硅胶侧面的固定程度待确认。机械模型采用刚性头这一近似时，决定球面接触几何的是半径。 |
| 另外两种探头 | Probe A 是直径 8 mm 的平头圆柱，见 Fig. 3a–b 和 Methods。直径 5.09 mm 的球用于 RGB–梯度光学校准，见 Fig. 3f 和补充 Fig. S4。 |

[出版社补充信息](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-57741-6/MediaObjects/41467_2025_57741_MOESM1_ESM.pdf)

## 可用的材料信息

杭州微思创的 **C-0030** 产品与论文中的厂商、透明、00-30 硬度、1:1 混合相符，是目前查到的最可能商品型号。正文只写 00-30，确切料号和批次待确认。厂商公开页列出断裂伸长率 350%、比重 1.01 和 2 mm 试片透光率 94%，没有给出应力–应变数据、初始弹性模量或体积响应。Shore 00 硬度无法单独识别可压缩 NH 的两个常数。2 mm 是厂商透光率试片厚度。 [微思创 C-0030 产品页](https://www.wesitru.com/productinfo/1155533.html)

针对 Wesitru、C-0030 和力学试验的公开检索未找到可直接采用的一对 NH 常数。作为软硅胶量级的参照，另一项研究对 **Ecoflex 00-30** 做不可压缩 NH 拟合，报告剪切模量 42.07 kPa。Smooth-On 的 Ecoflex 00-30 数据表列出 100% 伸长应力 10 psi，约 69 kPa，可作为该牌号的大应变拉伸数据点。当前工程的 $\mu=30$ kPa、$\nu=0.45$ 是演示参数，对应初始 $E=87$ kPa、第一 Lamé 常数 $\lambda=270$ kPa、体积模量 $K=290$ kPa。 [Ecoflex 实验拟合](https://doi.org/10.1038/s41467-023-35848-y) · [Smooth-On 技术数据](https://www.smooth-on.com/tb/files/ECOFLEX_SERIES_TB.pdf)

要标定本模型，需先测量实际硅胶厚度，按该手掌的热历程制备试件，用拉伸或剪切响应拟合 $\mu$，结合横向变形或受限压缩的体积响应确定 $K$。其中 $K=\lambda+2\mu/3$ 为初始体积模量。随后将球压入曲线用于装配级核验。若只有一条力–深度曲线，同时放开厚度、剪切模量、可压缩性和侧边条件会引入参数耦合。

## Fig. 3c 数据如何使用

正文报告 Probe B 的考察范围为 0.05–5 N、每组 $n=3$。公开工作簿 `Source Data/Fig. 3c-data-force vs depth.xlsx` 有 12 行统计点，覆盖 0.1–5 N。示例为 0.1 N 对应 $0.529\pm0.153$ mm，1 N 对应 $1.388\pm0.054$ mm，5 N 对应 $2.800\pm0.196$ mm，误差为 SD。图中另有标为 0.05 N 的示例，工作簿无这一行。 [出版社 Source Data ZIP](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-57741-6/MediaObjects/41467_2025_57741_MOESM14_ESM.zip)

论文由图像接触圆半径 $r_b$ 和探头半径 $r_f$ 计算球冠深度 $h_p=r_f-\sqrt{r_f^2-r_b^2}$。将来模拟球压入时，可由求得的接触区计算同一观测量，并单独记录压头行程和表面最大向下位移。默认 MATLAB 算例的 3 mm 固定名义牵引斑是人为输入，Fig. 3c 的接触区会随载荷变化。

检索使用 Wesitru、C-0030、modulus、应力–应变等组合关键词，并核对厂商目录与检索到的实验论文。Europe PMC 的 `/europepmc/webservices/rest/search` 以 `"Wesitru" AND "00-30" AND HAS_FT:Y` 查询返回一条记录，为本论文（PMC11894155）。
