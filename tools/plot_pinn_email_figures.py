"""Reproduce the TacPalm contact-force figures from archived FEM and PINN data."""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
from scipy.io import loadmat
from scipy.spatial import ConvexHull
import torch

from pinn_flow_diagram import export_drawio, render_flow
from train_generalization import Decoder, Network, Observations, sha256
from train_surface_surrogate import MESH, ROOT


ARCHIVE = ROOT/"results/stage-02-inverse/surface-generalization"
DEFAULT = ARCHIVE/"20261122_email_figures"
INK = "#17324A"
MUTED = "#506C80"
FEM = "#2A78D6"
PINN = "#EB6834"
IMAGE = "#1BAF7A"
TEAL = "#117E80"
GRID = "#DCE6E9"
PALE = "#F1F6F8"
installed_fonts = {font.name for font in fontManager.ttflist}
FONT = next((name for name in ("Noto Sans CJK SC", "Microsoft YaHei", "Noto Sans CJK JP",
                              "AR PL UKai CN")
             if name in installed_fonts), "DejaVu Sans")
DEPTH_CMAP = LinearSegmentedColormap.from_list("indentation", [
    "#EBF4F4", "#BCD9DA", "#67A9A9", "#3C818C", "#235470", "#173450"
])

plt.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"],
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


def load_geometry(dest, final_sample, final_fit, checkpoint, device):
    case = final_sample.stem
    source = dest/"source_data"
    fem = loadmat(source/f"{case}_fem.mat", simplify_cells=True)
    sample = loadmat(final_sample, simplify_cells=True, variable_names=["heights", "axis", "spec"])
    height = sample["heights"][:, :, -1]
    ref = Mesh(fem["reference"], fem["deformed"], fem["top"]-1)
    decoder = Decoder(device)
    network = Network(28).to(device).eval()
    network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
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
    np.savez_compressed(source/f"{case}_pinn.npz", reference=pin.reference,
                        current=pin.current, top=pin.top, force_N=force, min_j=min_j.item())
    (source/"geometry_check.json").write_text(
        json.dumps({"case": case, "travel_mm": fem["travel_mm"], "fem_N": fem["force_N"],
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
    subtitle = (f"复合盲测工况 {spec['id']}  ·  偏心 {np.linalg.norm(center):.2f} mm"
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
                       f"同一复合盲测工况 {fem['spec']['id']}  ·  末步行程 {fem['travel_mm']:.2f} mm")
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
                       "六条复合盲测轨迹  ·  压头形状、偏心、姿态与行程同时变化")
    curve_legend(fig, .866)
    gs = fig.add_gridspec(2, 3, left=.08, right=.965, bottom=.14, top=.77,
                          wspace=.23, hspace=.43)
    limit = np.ceil(max(combined["Reference_N"].max(), combined["Estimate_N"].max())+0.22)
    for i, (key, spec) in enumerate(cases.items()):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])
        chart_style(ax)
        rows = case_rows(combined, key)
        data_curve(ax, rows)
        ax.set(xlim=(-.02, 3.56), ylim=(-.03, limit), xticks=[0, 1, 2, 3],
               yticks=np.arange(0, limit+.1, 2))
        title(ax, f"工况 {key[-2:]}", 10.2)
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


def accuracy_figure(dest, combined, regular, report):
    fig = build_figure((14.2, 6.9), "盲测轨迹上的合力预测精度",
                       "以有限元正算为参考  ·  三种方法使用相同的高度帧和训练、开发轨迹")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.30], left=.075, right=.96,
                          bottom=.15, top=.74, wspace=.42)
    a, b = (fig.add_subplot(gs[0, i]) for i in range(2))
    chart_style(a, xgrid=True)
    lim = np.ceil(max(max(t["Reference_N"].max(), t["Estimate_N"].max())
                      for t in (combined, regular)))
    a.plot([0, lim], [0, lim], ls="--", color="#6F8490", lw=1.25, label="完全一致")
    for table, label, color, marker in (
        (regular, "常规盲测", FEM, "o"),
        (combined, "复合盲测", PINN, "D"),
    ):
        a.scatter(table["Reference_N"], table["Estimate_N"], c=color, marker=marker,
                  alpha=.85, s=29, edgecolor="white", lw=.35, label=f"{label} · {len(table)} 帧", zorder=3)
    a.set(xlim=(0, lim), ylim=(0, lim), aspect="equal", xlabel="有限元参考合力 (N)",
          ylabel="PINN 预测合力 (N)")
    a.legend(loc="upper left", frameon=False, fontsize=8.6)
    title(a, "PINN 逐帧参考值与预测值")

    groups = (
        ("test", "常规"), ("test_noise25", "常规 + 25 µm"),
        ("combined", "复合外推"), ("combined_noise25", "复合 + 25 µm"),
        ("combined_beyond_3p12", "复合 > 3.12 mm"),
    )
    methods = (("ridge", "核岭", FEM, "o"), ("pinn", "PINN", PINN, "D"),
               ("image", "纯图像", IMAGE, "s"))
    indices = np.arange(len(groups))
    chart_style(b)
    b.grid(axis="y", visible=False)
    b.grid(axis="x", color=GRID, lw=.7)
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    b.axvline(.05, color="#825945", ls="--", lw=1.2, label="0.05 N 目标")
    for offset, (method, label, color, marker) in zip((-.21, 0, .21), methods):
        values = [report[method][key]["MAE_N"] for key, _ in groups]
        b.scatter(values, indices+offset, color=color, marker=marker, s=48,
                  edgecolor=INK, linewidth=.55, label=label, zorder=3)
    b.set(yticks=indices, yticklabels=[label for _, label in groups],
          ylim=(len(groups)-.54, -.55), xlabel="平均绝对误差 MAE (N)", xlim=(0, None))
    b.legend(loc="lower left", bbox_to_anchor=(0, 1.03), frameon=False,
             ncol=4, fontsize=8.6, columnspacing=1.2, handletextpad=.25, borderaxespad=0)
    b.set_title("各方法在同批盲测高度上的误差", loc="left", y=1.13,
                color=INK, weight="bold", fontsize=11.5)
    note(fig, "25 µm 为合成高度噪声标准差，三种方法共享同一噪声实现。深压子集由预先设定的行程阈值选取。")
    save(fig, dest, "04_generalization")


def load_inverse_comparison(directory, combined, case_ids):
    rows = []
    for key in case_ids:
        with (directory/f"{key}.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows.extend(csv.DictReader(stream))
    keys = [(row["Trajectory"], float(row["Travel_mm"])) for row in rows]
    assert len(keys) == len(combined) and len(set(keys)) == len(keys)
    inverse = np.array([float(row["Estimate_N"]) for row in rows])
    valid = np.array([row["Status"] == "ok" for row in rows])
    residual = np.array([float(row["Residual_N"]) for row in rows])
    minimum_j = np.array([float(row["MinJ"]) for row in rows])
    if not np.all(np.isfinite(inverse[valid]) & (residual[valid] < 2e-9) & (minimum_j[valid] > 0)):
        raise ValueError("A reported successful surface inverse fails the equilibrium checks.")
    inverse[~valid] = np.nan
    np.testing.assert_array_equal([key[0] for key in keys], combined["Trajectory"])
    np.testing.assert_allclose([key[1] for key in keys], combined["Travel_mm"], atol=1e-12, rtol=0)
    reference = combined["Reference_N"]
    np.testing.assert_allclose([float(row["Reference_N"]) for row in rows], reference, atol=1e-6, rtol=0)
    paired = {
        "total_frames": len(valid),
        "fem_solved_frames": int(valid.sum()),
        "fem_inverse_MAE_N": float(np.mean(abs(inverse[valid]-reference[valid]))) if valid.any() else None,
        "pinn_MAE_on_same_frames_N": float(np.mean(abs(combined["Estimate_N"][valid]-reference[valid])))
        if valid.any() else None,
    }
    return inverse, valid, paired


def inverse_figure(dest, combined, cases, inverse, valid, paired):
    fig = build_figure((13.4, 8.05), "同批盲测高度上的有限元反演与 PINN",
                       "六条复合轨迹  ·  各估力方法使用同一帧的高度图  ·  以有限元接触正算为参考")
    gs = fig.add_gridspec(2, 3, left=.077, right=.97, bottom=.16, top=.76,
                          wspace=.24, hspace=.47)
    limit = np.ceil(max(np.max(combined["Reference_N"]), np.max(combined["Estimate_N"]),
                        np.max(inverse[valid], initial=0))+.10)
    for i, key in enumerate(cases):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])
        chart_style(ax)
        selected = combined["Trajectory"] == key
        force = combined[selected]
        data_curve(ax, force)
        ax.plot(force["Travel_mm"], inverse[selected], color=IMAGE, ls="--", lw=1.6,
                marker="s", ms=3.2, markeredgecolor=INK, markeredgewidth=.45)
        ax.set(xlim=(-.02, 3.56), ylim=(-.03, limit), xticks=[0, 1, 2, 3])
        title(ax, f"工况 {key[-2:]}", 10.2)
        if row == 1:
            ax.set_xlabel("压头行程 (mm)")
        if col == 0:
            ax.set_ylabel("向下合力 (N)")
    fig.legend(handles=[
        Line2D([0], [0], color=FEM, lw=2, label="有限元正算参考"),
        Line2D([0], [0], color=PINN, marker="o", mfc="white", ls="", label="PINN 估计"),
        Line2D([0], [0], color=IMAGE, marker="s", ls="--", label="全阶曲面约束反演"),
    ], bbox_to_anchor=(.97, .847), loc="upper right", ncol=3, frameon=False, fontsize=9.5)
    description = f"有限元反演收敛 {paired['fem_solved_frames']}/{paired['total_frames']} 帧"
    if valid.any():
        description += (f"。相同收敛帧的 MAE  ·  全阶反演 {paired['fem_inverse_MAE_N']:.4f} N"
                        f"  ·  PINN {paired['pinn_MAE_on_same_frames_N']:.4f} N")
    note(fig, description)
    save(fig, dest, "06_fem_inverse_check")


def overview(dest, fem, ref, combined, statistics, device_name):
    fig = build_figure((13.3, 7.8), "由掌面高度估计接触合力",
                       "TacPalm 启发的球头与多形状压入仿真  ·  曲面约束有限元反演与能量 PINN")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.09, 1], left=.052, right=.95,
                          bottom=.31, top=.77, wspace=.23)
    a, b = (fig.add_subplot(gs[0, i]) for i in range(2))
    disk(a, ref, fem["spec"]["center_mm"], fem["force_N"], FEM,
         Normalize(0, 3.4, clip=True), fem)
    a.set_title("有限元压入及竖向合力", fontsize=12.5, weight="bold", pad=9)
    case = fem["spec"]["id"]
    a.text(.50, -.02, f"{case}  ·  行程 {fem['travel_mm']:.2f} mm  ·  三维图竖向比例放大 1.7 倍",
           ha="center", va="top", color=MUTED, fontsize=8.5, transform=a.transAxes)
    chart_style(b)
    title(b, "典型留出工况的载荷–行程曲线", 12.5)
    first, chosen = (case_rows(combined, key) for key in ("blind_combined_01", case))
    data_curve(b, first, FEM, FEM, "o")
    data_curve(b, chosen, PINN, PINN, "s")
    load_limit = max(max(rows["Estimate_N"].max(), rows["Reference_N"].max()) for rows in (first, chosen))
    b.set(xlim=(-.04, 3.59), ylim=(-.1, np.ceil(load_limit+.15)), xlabel="压头行程 (mm)", ylabel="向下合力 (N)")
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
    metrics = [(f'{s["MAE_N"]:.3f} N', f'{s["frames"]} 帧复合盲测 · 合力 MAE'),
               (f'{s["MaxFinalAPE_pct"]:.2f}%', "最差轨迹末步相对力误差"),
               (f'{s["P95_ms"]:.2f} ms', f"{device_name} 单帧推理 P95")]
    for i, (value, caption) in enumerate(metrics):
        x = [.075, .385, .698][i]
        fig.text(x, .177, value, color=TEAL, weight="bold", fontsize=23, va="center")
        fig.text(x, .125, caption, color=MUTED, fontsize=9.4, va="center")
    note(fig, "实线为有限元正算。精度对照采用合成数据，计时从高度图就绪开始。")
    save(fig, dest, "00_email_overview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--evaluation", type=Path, default=ARCHIVE/"20261121_final_expanded")
    parser.add_argument("--checkpoint", type=Path, default=ARCHIVE/"20261121_expanded_training")
    parser.add_argument("--samples", type=Path, default=ARCHIVE/"20261123_blind_combined")
    args = parser.parse_args()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    path = args.evaluation
    evaluation = json.loads((path/"evaluation.json").read_text(encoding="utf-8"))
    lock = args.checkpoint/"freeze.json"
    if sha256(lock) != evaluation["freeze_sha256"]:
        raise ValueError("Figures require the same frozen model as the blind evaluation.")
    checkpoint = args.checkpoint/"pinn.pt"
    if sha256(checkpoint) != json.loads(lock.read_text(encoding="utf-8"))["checkpoints"]["pinn.pt"]:
        raise ValueError("PINN checkpoint changed after training.")
    combined = read_table(path/"pinn_combined.csv")
    report = json.loads((path/"summary.json").read_text(encoding="utf-8"))
    statistics = report["pinn"]
    regular = read_table(path/"pinn_test.csv")
    if not np.isclose(np.mean(abs(combined["Estimate_N"]-combined["Reference_N"])),
                      statistics["combined"]["MAE_N"], atol=2e-7, rtol=0):
        raise ValueError("The combined table and summary disagree.")
    samples = args.samples
    cases = {key: loadmat(samples/f"{key}.mat", simplify_cells=True,
                          variable_names=["spec"])["spec"] for key in sorted(set(combined["Trajectory"]))}
    case = evaluation["blind_protocol"]["geometry_case"]
    final_fit = case_rows(combined, case)[-1]
    fem, sample, ref, pin, force = load_geometry(dest, samples/f"{case}.mat", final_fit,
                                                 checkpoint, torch.device(args.device))
    inverse, valid, paired = load_inverse_comparison(path/"surface_inverse", combined, cases)
    (path/"surface_inverse_summary.json").write_text(json.dumps(paired, indent=2)+"\n", encoding="utf-8")
    device_name = "CPU" if evaluation["device"] == "cpu" else "GPU"
    overview(dest, fem, ref, combined, statistics, device_name)
    model_figure(dest, fem, sample, ref, pin, force)
    section_figure(dest, fem, ref, pin, force)
    load_figure(dest, combined, cases)
    accuracy_figure(dest, combined, regular, report)
    inverse_figure(dest, combined, cases, inverse, valid, paired)
    flow_stats = {**statistics["combined"], "device_name": device_name}
    render_flow(dest, flow_stats)
    export_drawio(dest, flow_stats)
    print(f"Wrote figures to {dest}")


if __name__ == "__main__":
    main()
