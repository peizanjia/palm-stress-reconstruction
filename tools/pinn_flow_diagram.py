"""Draw the PINN workflow from a shared SVG and draw.io layout."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WIDTH, HEIGHT = 1540, 918
INK = "#183249"
MUTED = "#4A6376"
LINE = "#56768A"
TEAL = "#087F83"
AMBER = "#B55632"
FONT = "Microsoft YaHei"


@dataclass(frozen=True)
class Card:
    x: float
    y: float
    w: float
    h: float
    title: str
    detail: str
    fill: str = "#FFFFFF"
    stroke: str = "#B5C8D5"


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    start: tuple = (1, 0.5)
    end: tuple = (0, 0.5)
    bends: tuple = ()
    dashed: bool = False


CARDS = {
    "design": Card(59, 168, 238, 80, "设定压入工况", "形状 · 位置 · 姿态 · 行程"),
    "forward": Card(333, 168, 225, 80, "接触有限元正算", "圆柱硅胶 · 超弹性 · 无摩擦"),
    "dataset": Card(603, 141, 257, 73, "保存开发高度与反力", "42 条训练 · 14 条验证", "#E7F4F3", "#96C3C1"),
    "ball": Card(603, 241, 257, 68, "权重冻结后生成盲测", "9 条常规 · 6 条复合"),
    "inverse": Card(907, 241, 294, 68, "曲面约束有限元反演", "材料点可沿观测曲面移动"),
    "femtime": Card(1246, 241, 235, 68, "同帧对照", "同一高度输入与参考合力", "#FEF2EA", "#DAB49D"),
    "height": Card(59, 439, 180, 84, "输入高度场", "仿真顶面采样"),
    "cnn": Card(296, 399, 231, 67, "局部高度图 → CNN", "144 维图像特征"),
    "features": Card(296, 506, 231, 67, "计算形状统计", "深度 · 偏心 · 形状，共 13 维"),
    "mlp": Card(586, 439, 188, 84, "拼接后送入 MLP", "输出 28 个位移系数"),
    "decode": Card(830, 439, 260, 84, "重构三维位移", "结合完整对中高度场", "#E7F4F3", "#96C3C1"),
    "integral": Card(1149, 439, 153, 84, "计算应力", "有限元积分"),
    "estimate": Card(1354, 439, 127, 84, "竖向合力", "PINN 输出", "#E7F4F3", "#96C3C1"),
    "label": Card(59, 725, 260, 65, "力监督项", "正算合力与预测合力的差"),
    "energy": Card(364, 725, 260, 65, "物理能量项", "由重构位移计算应变能"),
    "loss": Card(704, 725, 345, 65, "训练损失", "力监督项 + 0.001 × 归一化能量项"),
    "update": Card(1134, 725, 347, 65, "反向传播", "联合更新 CNN 与 MLP 权重", "#E7F4F3", "#96C3C1"),
}

EDGES = (
    Edge("design", "forward"),
    Edge("forward", "dataset", bends=((581, 208), (581, 177.5))),
    Edge("forward", "ball", bends=((581, 208), (581, 275))),
    Edge("ball", "inverse"),
    Edge("inverse", "femtime"),
    Edge("dataset", "height", (0.96, 1), (0.89, 0),
         ((849.72, 224), (882, 224), (882, 326), (219.2, 326)), True),
    Edge("height", "cnn", bends=((267, 481), (267, 432.5))),
    Edge("height", "features", bends=((267, 481), (267, 539.5))),
    Edge("cnn", "mlp", end=(0, 0.35), bends=((558, 432.5), (558, 468.4))),
    Edge("features", "mlp", end=(0, 0.65), bends=((558, 539.5), (558, 493.6))),
    Edge("mlp", "decode"),
    Edge("height", "decode", (0.5, 1), (0.5, 1), ((149, 617), (960, 617))),
    Edge("decode", "integral"),
    Edge("integral", "estimate"),
    Edge("label", "loss", (0.5, 0), (0.25, 0), ((189, 703), (790.25, 703))),
    Edge("energy", "loss", end=(0, 0.5)),
    Edge("loss", "update"),
)

BANDS = (
    (40, 106, 1460, 224, "有限元样本与全阶反演基线", "#F3F7FA"),
    (40, 354, 1460, 288, "PINN 前向计算", "#F3F8F8"),
    (40, 666, 1460, 144, "离线训练", "#F3F7FA"),
)

def labels(stats):
    result = (f"仿真复合盲测  ·  {stats['trajectories']} 条轨迹 / {stats['frames']} 帧"
              f"  ·  MAE {stats['MAE_N']:.3f} N  ·  {stats['device_name']} 推理 P95 {stats['P95_ms']:.2f} ms")
    return (
        (53, 37, "从掌面高度到竖向接触力", 36, INK, True),
        (55, 77, "TacPalm 启发的接触反演与合力估计", 19, MUTED, False),
        (624, 599, "完整对中高度场", 15, TEAL, False),
        (55, 849, result, 20, INK, True),
        (55, 882, "计时从高度图就绪开始。实物标定与闭环测试待开展。", 15, MUTED, False),
    )


def anchor(card, fraction):
    return (card.x+card.w*fraction[0], card.y+card.h*fraction[1])


def render_flow(destination, statistics):
    fig = plt.figure(figsize=(15.4, 9.18), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.axis("off")
    for x, y, w, h, title, fill in BANDS:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=13",
                                    facecolor=fill, edgecolor="none", zorder=0))
        ax.text(x+20, y+22, title, color=TEAL, fontsize=13, va="center", weight="bold")
    for edge in EDGES:
        points = (anchor(CARDS[edge.source], edge.start), *edge.bends,
                  anchor(CARDS[edge.target], edge.end))
        points = tuple(p for i, p in enumerate(points) if i == 0 or p != points[i-1])
        if len(points)>2:
            ax.plot(*zip(*points[:-1]), c=LINE, lw=1.2, ls="--" if edge.dashed else "-",
                    solid_joinstyle="round", zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     shrinkA=0, shrinkB=0, mutation_scale=13,
                                     lw=1.2, color=LINE, linestyle="--" if edge.dashed else "-",
                                     zorder=1))
    for card in CARDS.values():
        x, y, w, h = card.x, card.y, card.w, card.h
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                                    facecolor=card.fill, edgecolor=card.stroke, lw=1, zorder=2))
        ax.text(x+w/2, y+h/2-12, card.title, fontsize=13, ha="center", va="center",
                color=INK, weight="bold", zorder=3)
        ax.text(x+w/2, y+h/2+14, card.detail, fontsize=9.6, ha="center", va="center",
                color=MUTED, zorder=3)
    for x, y, value, fontsize, color, bold in labels(statistics):
        ax.text(x, y, value, fontsize=fontsize*0.72, color=color, va="center",
                weight="bold" if bold else "normal", zorder=4)
    fig.savefig(destination/"05_technical_flow.svg", facecolor="white")
    fig.savefig(destination/"05_technical_flow.png", dpi=170, facecolor="white")
    plt.close(fig)


def export_drawio(destination, statistics):
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "compressed": "false"})
    diagram = ET.SubElement(mxfile, "diagram", {"id": "pinn-height-force",
                                              "name": "高度反演与PINN"})
    model = ET.SubElement(diagram, "mxGraphModel", {"grid": "0", "page": "1",
                                                   "pageWidth": str(WIDTH),
                                                   "pageHeight": str(HEIGHT),
                                                   "background": "#FFFFFF"})
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    def vertex(key, value, style, x, y, w, h):
        cell = ET.SubElement(root, "mxCell", {"id": key, "parent": "1", "vertex": "1",
                                             "value": value, "style": style})
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w),
                                           "height": str(h), "as": "geometry"})

    def caption(key, x, y, value, size, color, bold):
        style = (f"text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
                 f"spacing=0;fontFamily={FONT};fontSize={size};fontColor={color};fontStyle={int(bold)};")
        vertex(key, value, style, x, y-size/2-4, WIDTH-x-24, size+8)

    for idx, (x, y, w, h, title, fill) in enumerate(BANDS):
        vertex(f"band{idx}", "", f"rounded=1;arcSize=10;fillColor={fill};strokeColor=none;",
               x, y, w, h)
        caption(f"bandtitle{idx}", x+20, y+22, title, 18, TEAL, True)
    for key, card in CARDS.items():
        style = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
                 f"fillColor={card.fill};strokeColor={card.stroke};fontFamily={FONT};fontColor={INK};")
        title = f'<b style="font-size:18px;">{card.title}</b>'
        detail = f'<span style="font-size:13px;color:{MUTED};">{card.detail}</span>'
        vertex(key, title+"<br>"+detail, style, card.x, card.y, card.w, card.h)
    for idx, edge in enumerate(EDGES):
        sx, sy = edge.start
        tx, ty = edge.end
        style = (f"noEdgeStyle=1;endArrow=block;endFill=1;strokeColor={LINE};strokeWidth=1.6;"
                 f"exitX={sx};exitY={sy};exitPerimeter=0;entryX={tx};entryY={ty};entryPerimeter=0;"
                 f"dashed={int(edge.dashed)};")
        cell = ET.SubElement(root, "mxCell", {"id": f"edge{idx}", "parent": "1", "edge": "1",
                                             "source": edge.source, "target": edge.target,
                                             "style": style})
        geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if edge.bends:
            points = ET.SubElement(geo, "Array", {"as": "points"})
            for x, y in edge.bends:
                if (x,y) != anchor(CARDS[edge.target], edge.end):
                    ET.SubElement(points, "mxPoint", {"x": str(x), "y": str(y)})
    for idx, label in enumerate(labels(statistics)):
        caption(f"caption{idx}", *label)
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(destination/"05_technical_flow.drawio",
                                 encoding="utf-8", xml_declaration=True)
