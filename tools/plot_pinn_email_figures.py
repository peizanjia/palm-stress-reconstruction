"""Reproduce the TacPalm contact-force figures from archived FEM and PINN data."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
from scipy.io import loadmat
from scipy.spatial import ConvexHull
import torch

from pinn_flow_diagram import export_drawio, render_flow
from train_generalization import Decoder, Network, Observations
from train_surface_surrogate import MESH, ROOT


INVERSE = ROOT/"results/stage-02-inverse/task-01-dirichlet-reconstruction"
ARCHIVE = ROOT/"results/stage-02-inverse/surface-generalization"
DEFAULT = ARCHIVE/"20261122_email_figures"
CASE = "combined_05"
INK = "#17324A"
MUTED = "#506C80"
FEM = "#215B88"
PINN = "#BD592D"
TEAL = "#117E80"
GRID = "#DCE6E9"
PALE = "#F1F6F8"
FONT = "Microsoft YaHei"
DEPTH_CMAP = LinearSegmentedColormap.from_list("indentation", [
    "#EBF4F4", "#BCD9DA", "#67A9A9", "#3C818C", "#235470", "#173450"
])

plt.rcParams.update({
    "font.family": [FONT, "Noto Sans SC", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.size": 10,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "svg.fonttype": "path",
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
})


@dataclass
class Mesh:
    reference: np.ndarray
    current: np.ndarray
    top: np.ndarray

    def layers(self):
        levels, count = np.unique(self.reference[:, 2], return_counts=True)
        if np.any(count != count[0]):
            raise ValueError("The model plot requires an axially extruded mesh.")
        n = int(count[0])
        return len(levels), n, self.top-(len(levels)-1)*n


def read_table(path):
    return np.genfromtxt(path, names=True, delimiter=",", dtype=None, encoding="utf-8-sig")


def build_figure(size, heading, subtitle):
    fig = plt.figure(figsize=size, facecolor="white")
    fig.add_artist(Line2D([0.038, 0.102], [0.968, 0.968], transform=fig.transFigure,
                          color=TEAL, linewidth=3.1, solid_capstyle="butt"))
    fig.text(0.038, 0.939, heading, fontsize=24, weight="bold", va="top")
    fig.text(0.039, 0.869, subtitle, fontsize=10.4, color=MUTED, va="top")
    return fig


def note(fig, value, y=0.041):
    fig.text(0.04, y, value, color=MUTED, fontsize=8.8, va="center")


def save(fig, dest, name):
    fig.savefig(dest/f"{name}.png", dpi=190, facecolor="white")
    fig.savefig(dest/f"{name}.svg", facecolor="white")
    plt.close(fig)


def chart_style(ax, xgrid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B5CAD3")
    ax.tick_params(length=3, color="#A9BFC9", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="both" if xgrid else "y", color=GRID, lw=0.75)


def title(ax, value, size=12):
    ax.set_title(value, loc="left", color=INK, weight="bold", fontsize=size, pad=11)


def case_rows(table, case):
    rows = table[table["Trajectory"] == case]
    if not len(rows):
        raise ValueError(f"Missing observations for {case}.")
    return rows[np.argsort(rows["Travel_mm"])]


def load_geometry(dest, final_sample, final_fit, device):
    source = dest/"source_data"
    fem = loadmat(source/f"{CASE}_fem.mat", simplify_cells=True)
    sample = loadmat(final_sample, simplify_cells=True, variable_names=["heights", "axis", "spec"])
    height = sample["heights"][:, :, -1]
    ref = Mesh(fem["reference"], fem["deformed"], fem["top"]-1)
    decoder = Decoder(device)
    network = Network(28).to(device).eval()
    network.load_state_dict(torch.load(ARCHIVE/"20261121_expanded_training/pinn.pt",
                                       map_location=device, weights_only=True))
    observations = Observations(device)
    raw = np.where(np.isfinite(height), fem["height_mm"]-height, 0)
    with torch.inference_mode():
        patch, features, depth, canonical = observations(decoder.tensor(raw)[None], physical=True)
        displacement = decoder.displacement(canonical, depth, network(patch, features))
        _, force, min_j = decoder.evaluate_displacement(displacement)
    pin_reference = decoder.nodes.cpu().numpy()
    pin_current = pin_reference+displacement[0].cpu().numpy()
    pin_top = loadmat(MESH, simplify_cells=True, variable_names=["r"])["r"]["mesh"]["top"]-1
    pin = Mesh(pin_reference, pin_current, pin_top)
    force = float(force.item())
    differences = {
        "fem_replay_vs_trajectory_max_N": float(fem["replayError_N"]),
        "fem_replay_vs_prediction_csv_N": abs(float(fem["force_N"])-float(final_fit["Reference_N"])),
        "reconstructed_pinn_vs_prediction_csv_N": abs(force-float(final_fit["Estimate_N"])),
    }
    if max(differences.values())>5e-5 or min_j.item()<=0:
        raise ValueError(f"Model reconstruction did not match the held-out predictions: {differences}")
    if not np.isclose(fem["travel_mm"], final_fit["Travel_mm"], atol=1e-9, rtol=0):
        raise ValueError("The FEM geometry and PINN record must use the same travel step.")
    np.savez_compressed(source/f"{CASE}_pinn.npz", reference=pin.reference,
                        current=pin.current, top=pin.top, force_N=force, min_j=min_j.item())
    (source/"geometry_check.json").write_text(
        json.dumps({"case": CASE, "travel_mm": fem["travel_mm"], "fem_N": fem["force_N"],
                    "pinn_N": force, **differences}, indent=2), encoding="utf-8")
    return fem, sample, ref, pin, force


def project(xyz):
    """Orthographic view with disclosed vertical exaggeration."""
    azim, elev = np.deg2rad([-54, 34])
    right = np.array([-np.sin(azim), np.cos(azim), 0])
    up = np.array([-np.sin(elev)*np.cos(azim), -np.sin(elev)*np.sin(azim), np.cos(elev)])
    view = np.cross(right, up)
    return (xyz*np.array([1, 1, 1.7]))@np.stack([right, up, view], axis=1)


def probe_outline(ax, fem):
    spec = fem["spec"]
    a = np.asarray(spec["axes"])
    power = spec["power"]
    rotation = fem["rotation"]
    origin = fem["probeOrigin"]

    def shape(theta, azimuth):
        signed = lambda v: np.sign(v)*np.abs(v)**(2/power)
        local = np.stack(np.broadcast_arrays(signed(np.cos(theta))*signed(np.cos(azimuth)),
                                             signed(np.cos(theta))*signed(np.sin(azimuth)),
                                             signed(np.sin(theta))), axis=-1)
        return project((local*a)@rotation.T+origin)

    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    azimuth = np.linspace(0, 2*np.pi, 130)
    silhouette = shape(theta[:, None], azimuth[None, :])[..., :2].reshape(-1, 2)
    outline = silhouette[ConvexHull(silhouette).vertices]
    ax.add_patch(Polygon(outline, facecolor="#F5DDC3", edgecolor="#A17451",
                         linewidth=0.9, alpha=0.68, zorder=4))
    for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
        ax.plot(*shape(theta, phi)[:, :2].T, color="#A17451", alpha=0.65,
                lw=0.65, zorder=5)
    for lat in (-0.73, 0, 0.73):
        ax.plot(*shape(lat, azimuth)[:, :2].T, color="#A17451", alpha=0.63,
                lw=0.62, zorder=5)


def disk(ax, mesh, contact_xy, force, color, depth_scale, fem=None):
    layer_count, n, tris = mesh.layers()
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    base = unique[counts == 1]
    side = np.vstack([np.c_[base+k*n, base[:, ::-1]+(k+1)*n] for k in range(layer_count-1)])
    xyz = project(mesh.current)
    top_drop = (mesh.reference[mesh.top, 2]-mesh.current[mesh.top, 2]).mean(axis=1)
    top_colors = DEPTH_CMAP(depth_scale(top_drop))
    side_angles = np.arctan2(mesh.reference[side, 1].mean(axis=1),
                             mesh.reference[side, 0].mean(axis=1))
    side_colors = plt.get_cmap("Blues")(0.28+0.14*(1+np.cos(side_angles+0.3))/2)
    polys = [list(p) for p in xyz[mesh.top]]+[list(p) for p in xyz[side]]
    order = np.argsort([np.asarray(p)[:, 2].mean() for p in polys])
    colors = np.vstack([top_colors, side_colors])
    ax.add_collection(PolyCollection([np.asarray(polys[i])[:, :2] for i in order],
                                     facecolors=colors[order], edgecolor="#628493",
                                     linewidth=0.23, alpha=1.0))
    if fem is not None:
        probe_outline(ax, fem)
        tip_z = float(fem["height_mm"]-fem["travel_mm"])
    else:
        tip_z = float(mesh.current[mesh.top.ravel(), 2].min())
    point = project(np.array([*contact_xy, tip_z+0.35]))[:2]
    tail = project(np.array([*contact_xy, 11.5]))[:2]
    ax.annotate("", xy=point, xytext=tail, zorder=9,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                mutation_scale=15, shrinkA=0, shrinkB=0))
    ax.annotate(f"{force:.2f} N", xy=tail, xytext=(7, 0), textcoords="offset points",
                color=color, fontsize=12, weight="bold", va="center", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))
    ax.set(xlim=(-27, 27), ylim=(-17, 29), aspect="equal")
    ax.axis("off")


def sample_map(ax, sample, contact_xy, height, depth_scale):
    coordinates = sample["axis"]
    drop = height-sample["heights"][:, :, -1]
    plot = ax.imshow(drop, origin="lower", extent=(coordinates[0]-0.125, coordinates[-1]+0.125,
                                                   coordinates[0]-0.125, coordinates[-1]+0.125),
                     cmap=DEPTH_CMAP, norm=depth_scale, interpolation="bilinear")
    cx, cy = contact_xy
    ax.set(xlim=(cx-7, cx+7), ylim=(cy-7, cy+7), xlabel="当前 x (mm)", ylabel="当前 y (mm)")
    ax.set_xticks([round(cx-5), round(cx), round(cx+5)])
    ax.set_yticks([round(cy-5), round(cy), round(cy+5)])
    ax.tick_params(labelsize=8, length=2)
    for spine in ax.spines.values():
        spine.set_color("#9FB9C5")
    return plot


def model_figure(dest, fem, sample, ref, pin, predicted):
    d = fem["travel_mm"]
    spec = fem["spec"]
    center = spec["center_mm"]
    subtitle = (f"复合留出工况 {CASE}  ·  偏心 {np.linalg.norm(center):.2f} mm"
                f"  ·  转角 {spec['yaw_deg']:.1f}°  ·  倾角 {spec['tilt_deg']:.1f}°"
                f"  ·  末步行程 {d:.2f} mm")
    fig = build_figure((15.7, 6.5), "压入后的三维结构与竖向合力", subtitle)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.8, 1.05], left=.045, right=.965,
                          bottom=.21, top=.77, wspace=.20)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    depth_scale = Normalize(0, 3.4, clip=True)
    disk(axes[0], ref, center, fem["force_N"], FEM, depth_scale, fem)
    img = sample_map(axes[1], sample, center, fem["height_mm"], depth_scale)
    disk(axes[2], pin, (0, 0), predicted, PINN, depth_scale)
    for ax, heading in zip(axes, ["有限元正算与压头", "输入高度图的局部区域", "PINN 重构位移（压痕对中）"]):
        ax.set_title(heading, weight="bold", fontsize=12, pad=13)
    cax = fig.add_axes([.425, .118, .160, .017])
    cbar = fig.colorbar(img, cax=cax, orientation="horizontal", ticks=[0, 1, 2, 3])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=2, color=MUTED)
    cbar.set_label("顶面向下位移 / 下陷量 (mm)", fontsize=8, labelpad=3)
    note(fig, "箭头表示向下合力，作用位置为示意。三维视图的竖向比例放大 1.7 倍。PINN 位移用于估力，完整位移场尚未单独验收。")
    save(fig, dest, "01_deformed_model")


def section(ax, mesh, anchor, force, color):
    layer_count, n, tris = mesh.layers()
    points = mesh.reference[:n, :2]
    tri = mtri.Triangulation(points[:, 0], points[:, 1], tris)
    x = anchor[0]+np.linspace(-7, 7, 141)
    y = np.full_like(x, anchor[1])
    curves = []
    for layer in mesh.current.reshape(layer_count, n, 3):
        qx = mtri.LinearTriInterpolator(tri, layer[:, 0])(x, y)-anchor[0]
        qz = mtri.LinearTriInterpolator(tri, layer[:, 2])(x, y)
        curves.append(np.stack((qx, qz), axis=1))
    curves = np.stack(curves)
    for i, (lower, upper) in enumerate(zip(curves[:-1], curves[1:])):
        area = np.vstack([lower, upper[::-1]])
        ax.fill(*area.T, color=plt.get_cmap("Blues")(.14+.055*i), ec="none")
    for column in range(0, curves.shape[1], 10):
        ax.plot(*curves[:, column].T, color="#779BA9", lw=.5)
    for curve in curves:
        ax.plot(*curve.T, color=FEM, lw=.65)
    ax.plot(*curves[-1].T, color=color, lw=2)
    ax.axhline(5, color="#728898", ls="--", lw=1)
    at = np.argmin(curves[-1, :, 1])
    point = curves[-1, at]
    ax.annotate(f"{force:.2f} N", xy=point, xytext=(point[0], 6.7),
                ha="center", va="center", color=color, fontsize=11, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
    chart_style(ax)
    ax.set(xlim=(-7.5, 7.5), ylim=(-.15, 7.15), xlabel="局部横坐标 (mm)", ylabel="高度 (mm)")
    ax.set_aspect("equal", adjustable="box")


def section_figure(dest, fem, ref, pin, predicted):
    fig = build_figure((12.6, 6.6), "压痕下方的硅胶变形剖面",
                       f"同一复合留出工况 {CASE}  ·  末步行程 {fem['travel_mm']:.2f} mm")
    gs = fig.add_gridspec(1, 2, left=.07, right=.965, bottom=.17, top=.75, wspace=.20)
    a, b = (fig.add_subplot(gs[0, i]) for i in range(2))
    section(a, ref, fem["spec"]["center_mm"], fem["force_N"], FEM)
    section(b, pin, (0, 0), predicted, PINN)
    a.set_title("有限元正算剖面", weight="bold", fontsize=13)
    b.set_title("PINN 位移重构剖面", weight="bold", fontsize=13)
    note(fig, "虚线为未变形顶面。连线为各自参考切面上采样材料点的变形后连线，箭头位置为示意。")
    save(fig, dest, "02_deformation_section")


def data_curve(ax, rows, color=FEM, pin_color=PINN, marker="o"):
    ax.plot(np.r_[0, rows["Travel_mm"]], np.r_[0, rows["Reference_N"]],
            color=color, lw=2.0, zorder=2)
    ax.scatter(rows["Travel_mm"], rows["Estimate_N"], s=29, marker=marker,
               facecolor="white", edgecolor=pin_color, linewidth=1.35, zorder=3)


def curve_legend(fig, y, x=.96):
    fig.legend(handles=[
        Line2D([0], [0], color=FEM, lw=2, label="有限元正算"),
        Line2D([0], [0], marker="o", color=PINN, mfc="white", ls="", label="PINN 估计")
    ], bbox_to_anchor=(x, y), loc="upper right", ncol=2, frameon=False,
        fontsize=9, borderaxespad=0)


def load_figure(dest, combined, cases):
    fig = build_figure((13.0, 8.0), "不同压入工况的非线性载荷–行程响应",
                       "六条复合留出轨迹  ·  压头形状、偏心、姿态与行程同时变化")
    curve_legend(fig, .866)
    gs = fig.add_gridspec(2, 3, left=.08, right=.965, bottom=.14, top=.77,
                          wspace=.23, hspace=.43)
    limit = np.ceil(combined["Reference_N"].max()+0.22)
    for i, (key, spec) in enumerate(cases.items()):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])
        chart_style(ax)
        rows = case_rows(combined, key)
        data_curve(ax, rows)
        ax.set(xlim=(-.02, 3.56), ylim=(-.03, limit), xticks=[0, 1, 2, 3],
               yticks=np.arange(0, limit+.1, 2))
        title(ax, key.replace("combined_", "工况 "), 10.2)
        error = 100*(rows["Estimate_N"][-1]/rows["Reference_N"][-1]-1)
        ax.text(.054, .93, f"偏心 {np.linalg.norm(spec['center_mm']):.1f} mm"
                f"  ·  倾角 {spec['tilt_deg']:.0f}°\n末步误差 {error:+.2f}%",
                transform=ax.transAxes, va="top", color=MUTED, fontsize=8.5, linespacing=1.45)
        if row==1:
            ax.set_xlabel("压头行程 (mm)")
        if col==0:
            ax.set_ylabel("向下合力 (N)")
    note(fig, "实线连接有限元载荷步，空心点为同一高度图的 PINN 估计。压头行程用于绘图对照，模型输入为高度场。")
    save(fig, dest, "03_load_travel")


def accuracy_figure(dest, combined, regular, statistics):
    fig = build_figure((13.7, 6.9), "留出轨迹上的合力预测精度",
                       "以有限元正算为参考  ·  扩样及噪声增强版能量 PINN")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.18], left=.075, right=.96,
                          bottom=.15, top=.75, wspace=.34)
    a, b = (fig.add_subplot(gs[0, i]) for i in range(2))
    chart_style(a, xgrid=True)
    lim = np.ceil(max(combined["Reference_N"].max(), combined["Estimate_N"].max()))
    a.plot([0, lim], [0, lim], ls="--", color="#6F8490", lw=1.25, label="完全一致")
    for table, label, color, marker in (
        (regular, "常规联合留出 · 63 帧", TEAL, "o"),
        (combined, "多因素同时外推 · 54 帧", PINN, "D"),
    ):
        a.scatter(table["Reference_N"], table["Estimate_N"], c=color, marker=marker,
                  alpha=.85, s=29, edgecolor="white", lw=.35, label=label, zorder=3)
    a.set(xlim=(0, lim), ylim=(0, lim), aspect="equal", xlabel="有限元参考合力 (N)",
          ylabel="PINN 预测合力 (N)")
    a.legend(loc="upper left", frameon=False, fontsize=8.6)
    title(a, "逐帧参考值与预测值")

    groups = (
        ("test", "常规联合留出"), ("shape", "未见形状"), ("position", "更远偏心"),
        ("yaw", "留出平面转角"), ("tilt", "更高倾角"), ("combined", "多因素同时外推"),
        ("combined_beyond_3p12", "复合外推 >3.12 mm"), ("combined_noise25", "复合外推 +25 µm 噪声"),
    )
    errors = [statistics[key]["MAE_N"] for key, _ in groups]
    indices = np.arange(len(groups))
    colors = [PINN if error>.05 else TEAL for error in errors]
    chart_style(b)
    b.grid(axis="y", visible=False)
    b.grid(axis="x", color=GRID, lw=.7)
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    b.barh(indices, errors, color=colors, height=.63)
    b.axvline(.05, color="#825945", ls="--", lw=1.3)
    for y, error in zip(indices, errors):
        b.text(error+.0014, y, f"{error:.3f}", ha="left", va="center",
               color=INK, fontsize=9)
    b.set(yticks=indices, yticklabels=[label for _, label in groups], xlim=(0, .079),
          ylim=(len(groups)-.53, -.72), xlabel="平均绝对误差 MAE (N)")
    b.set_xticks([0, .02, .04, .06])
    b.text(.05, -.58, "0.05 N 目标", ha="center", va="bottom", fontsize=8.2,
           color="#825945", bbox=dict(facecolor="white", edgecolor="none", pad=.4))
    title(b, "不同留出条件下的合力误差")
    note(fig, "深压入子集含 6 帧。加噪组为同 54 帧复合工况加入标准差 25 µm 的合成高度噪声。")
    save(fig, dest, "04_generalization")


def inverse_figure(dest):
    fig = build_figure((12.5, 6.25), "曲面约束改善高度反演的合力估计",
                       "中心和偏心球头算例  ·  全阶有限元反演  ·  与 PINN 泛化试验分别计时")
    gs = fig.add_gridspec(1, 2, left=.08, right=.965, bottom=.18, top=.75, wspace=.22)
    for i, (case, label) in enumerate((("centered", "中心球头"), ("eccentric", "偏心球头"))):
        ax = fig.add_subplot(gs[i])
        base = read_table(INVERSE/f"20261120_inverse_v1/{case}/height/metrics.csv")
        corrected = read_table(INVERSE/f"20261121_surface_sigma050/{case}/surface/metrics.csv")
        np.testing.assert_allclose(base["Travel_mm"], corrected["Travel_mm"], atol=1e-12)
        np.testing.assert_allclose(base["RefFz_N"], corrected["RefFz_N"], atol=1e-10)
        ref = -corrected["RefFz_N"]
        before = 100*(base["Fz_N"][-1]/corrected["RefFz_N"][-1]-1)
        after = 100*(corrected["Fz_N"][-1]/corrected["RefFz_N"][-1]-1)
        chart_style(ax)
        ax.plot(corrected["Travel_mm"], ref, color=FEM, lw=2.0, label="有限元正算")
        ax.plot(base["Travel_mm"], -base["Fz_N"], color=PINN, ls="--", lw=1.8,
                label="冻结水平位移")
        ax.plot(corrected["Travel_mm"], -corrected["Fz_N"], color=TEAL, lw=1.6, marker="o",
                ms=3.3, mfc="white", markevery=2, label="曲面约束反演")
        ax.set(xlim=(-.025, 3.81), ylim=(-.05, 7.15), xlabel="压头行程 (mm)", ylabel="向下合力 (N)")
        title(ax, label)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.text(.99, .115, f"末步力误差  {before:+.1f}% → {after:+.1f}%\n"
                f"曲面反演平均  {corrected['Solve_s'][1:].mean():.2f} s / 帧",
                transform=ax.transAxes, ha="right", va="center", fontsize=9, color=MUTED,
                linespacing=1.5)
    note(fig, "曲面反演允许顶面切向位移并采用 0.50 mm 高度平滑。此图比较竖向合力。")
    save(fig, dest, "06_fem_inverse_check")


def overview(dest, fem, ref, combined, statistics):
    fig = build_figure((13.3, 7.8), "由掌面高度估计接触合力",
                       "TacPalm 启发的球头与多形状压入仿真  ·  曲面约束有限元反演与能量 PINN")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.09, 1], left=.052, right=.95,
                          bottom=.31, top=.77, wspace=.23)
    a, b = (fig.add_subplot(gs[0, i]) for i in range(2))
    disk(a, ref, fem["spec"]["center_mm"], fem["force_N"], FEM,
         Normalize(0, 3.4, clip=True), fem)
    a.set_title("有限元压入及竖向合力", fontsize=12.5, weight="bold", pad=9)
    a.text(.50, -.02, f"{CASE}  ·  行程 {fem['travel_mm']:.2f} mm  ·  三维图竖向比例放大 1.7 倍",
           ha="center", va="top", color=MUTED, fontsize=8.5, transform=a.transAxes)
    chart_style(b)
    title(b, "典型留出工况的载荷–行程曲线", 12.5)
    data_curve(b, case_rows(combined, "combined_01"), FEM, FEM, "o")
    data_curve(b, case_rows(combined, CASE), PINN, PINN, "s")
    b.set(xlim=(-.04, 3.59), ylim=(-.1, 7.2), xlabel="压头行程 (mm)", ylabel="向下合力 (N)")
    b.legend(handles=[
        Line2D([0], [0], color=FEM, lw=2, label="工况 01"),
        Line2D([0], [0], color=PINN, lw=2, label="工况 05"),
        Line2D([0], [0], marker="o", mfc="white", color=MUTED, ls="", label="PINN 估计")
    ], loc="upper left", frameon=False, fontsize=8.7, ncol=2)

    box = FancyBboxPatch((.043, .099), .914, .132, transform=fig.transFigure,
                         boxstyle="round,pad=0.008,rounding_size=0.016",
                         facecolor=PALE, edgecolor="none", zorder=0)
    fig.add_artist(box)
    s = statistics["combined"]
    metrics = [(f'{s["MAE_N"]:.3f} N', "54 帧复合留出 · 合力 MAE"),
               (f'{s["MaxFinalAPE_pct"]:.2f}%', "最差轨迹末步相对力误差"),
               (f'{s["P95_ms"]:.2f} ms', "GPU 单帧推理 P95")]
    for i, (value, caption) in enumerate(metrics):
        x = [.075, .385, .698][i]
        fig.text(x, .177, value, color=TEAL, weight="bold", fontsize=23, va="center")
        fig.text(x, .125, caption, color=MUTED, fontsize=9.4, va="center")
    note(fig, "实线为有限元正算。精度对照采用合成数据，GPU 计时从高度图就绪开始。")
    save(fig, dest, "00_email_overview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    path = ARCHIVE/"20261121_final_expanded"
    combined = read_table(path/"pinn_combined.csv")
    statistics = json.loads((path/"summary.json").read_text(encoding="utf-8"))["pinn"]
    regular = read_table(path/"pinn_test.csv")
    if not np.isclose(np.mean(abs(combined["Estimate_N"]-combined["Reference_N"])),
                      statistics["combined"]["MAE_N"], atol=2e-7, rtol=0):
        raise ValueError("The combined table and summary disagree.")
    samples = ARCHIVE/"20261121_combined"
    cases = {key: loadmat(samples/f"{key}.mat", simplify_cells=True,
                          variable_names=["spec"])["spec"] for key in sorted(set(combined["Trajectory"]))}
    final_fit = case_rows(combined, CASE)[-1]
    fem, sample, ref, pin, force = load_geometry(dest, samples/f"{CASE}.mat", final_fit,
                                                 torch.device(args.device))
    overview(dest, fem, ref, combined, statistics)
    model_figure(dest, fem, sample, ref, pin, force)
    section_figure(dest, fem, ref, pin, force)
    load_figure(dest, combined, cases)
    accuracy_figure(dest, combined, regular, statistics)
    inverse_figure(dest)
    render_flow(dest, statistics["combined"])
    export_drawio(dest, statistics["combined"])
    print(f"Wrote figures to {dest}")


if __name__ == "__main__":
    main()
