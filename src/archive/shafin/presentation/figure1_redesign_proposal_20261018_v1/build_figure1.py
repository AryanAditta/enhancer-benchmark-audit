"""Create a standalone Figure 1 proposal without changing manuscript assets."""
from pathlib import Path
import os
import json
import hashlib
import math

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
for key in ("MPLCONFIGDIR", "TMP", "TEMP"):
    location = RUNTIME / key.lower()
    location.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(location)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Arc, Circle
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image
from pypdf import PdfReader

W, H = 180.0, 137.0
BLACK = "#222222"
GREY = "#6C747C"
EDGE = "#48525C"
BLUE = "#306B9A"
BLUE_LIGHT = "#DCE9F4"
TEAL = "#277D79"
TEAL_LIGHT = "#DCEFEB"
ORANGE = "#B9632D"
ORANGE_LIGHT = "#F8EBDD"
LIGHT = "#F1F3F5"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": WHITE,
    "figure.facecolor": WHITE,
    "axes.unicode_minus": False,
})
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
TEXT = []


def text(x, y, s, size=6.1, bold=False, ha="center", va="center", color=BLACK):
    item = ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal",
                   ha=ha, va=va, color=color, linespacing=1.25, zorder=5)
    TEXT.append(item)
    return item


def rect(x, y, w, h, fill=WHITE, edge=EDGE, lw=0.55, radius=0):
    if radius:
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge, linewidth=lw)
    else:
        patch = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)
    return patch


def line(points, color=EDGE, lw=0.65):
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
            lw=lw, solid_capstyle="butt", zorder=2)


def arrow(x1, y1, x2, y2, color=EDGE, lw=0.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=5.0, shrinkA=0, shrinkB=0,
                                linewidth=lw, color=color, zorder=3))


def sequence(x, y, sequence_text="AACGTAGC", w=27.2, h=2.7,
             palette=None, letters=True):
    palette = palette or [BLUE_LIGHT, WHITE, "#B9D2E7", "#E7EFF7"]
    cell_w = w / len(sequence_text)
    for i, nt in enumerate(sequence_text):
        rect(x + i * cell_w, y, cell_w, h, palette["ACGT".index(nt)], EDGE, 0.35)
        if letters:
            text(x + (i + .5) * cell_w, y + h / 2, nt, size=5.1)


def pair(cx, cy, color, xspan=8.5, crossed=False):
    for i, yy in enumerate((cy - 2.4, cy + 2.4)):
        ax.add_patch(Circle((cx - xspan / 2, yy), 1.15, facecolor=color,
                            edgecolor=EDGE, linewidth=0.45, zorder=3))
        rect(cx + xspan / 2 - 1.15, yy - 1.15, 2.3, 2.3, WHITE, EDGE, .45)
        text(cx - xspan / 2, yy, "+", size=5)
        text(cx + xspan / 2, yy, "-", size=5)
        line([(cx - xspan / 2 + 1.15, yy), (cx + xspan / 2 - 1.15, yy)], GREY, .45)
    if crossed:
        line([(cx - xspan / 2 + 1.15, cy - 2.4),
              (cx + xspan / 2 - 1.15, cy + 2.4)], GREY, .4)
        line([(cx - xspan / 2 + 1.15, cy + 2.4),
              (cx + xspan / 2 - 1.15, cy - 2.4)], GREY, .4)


def panel(label, title, y):
    text(2.0, y, label, size=8, bold=True, ha="left")
    text(8.0, y, title, size=7, bold=True, ha="left")


# Panel a: data audit and grouped classification.
panel("a", "Legacy enhancer classification", 5.8)
centres = [22, 67, 111, 157]
for x, title in zip(centres, ["Legacy sequences", "Sequence audit", "Grouped classification", "Evaluation"]):
    text(x, 15.2, title, size=6.8, bold=True)
for x1, x2 in [(39.5, 47.0), (86.5, 92.0), (129.5, 136.0)]:
    arrow(x1, 28.2, x2, 28.2)

# Sequence records (schematic DNA, not study observations).
for y, seq in [(22.1, "AACGTAGC"), (27.1, "TGCAATCG"), (32.1, "CGTACGAT")]:
    sequence(8.4, y, seq)

# Identical records and reverse-complement sequence, explicitly labelled.
for y, seq in [(21.7, "AACGTAGC"), (27.0, "AACGTAGC"), (32.3, "GCTACGTT")]:
    sequence(49.3, y, seq, w=26.0)
line([(76.7, 23.05), (79.0, 23.05), (79.0, 28.35), (76.7, 28.35)], BLUE, .6)
text(82.0, 25.7, "=", size=6.4)
text(80.0, 33.6, "RC", size=5.2)

# Fixed five-fold layout. Each row represents a held-out partition.
grid_x, grid_y, cw, ch, gap = 94.3, 23.0, 6.1, 2.0, .5
for j in range(5):
    text(grid_x + j * (cw + gap) + cw / 2, 20.9, str(j + 1), size=5.2)
    for i in range(5):
        rect(grid_x + j * (cw + gap), grid_y + i * (ch + gap), cw, ch,
             BLUE if i == j else LIGHT, EDGE, .35)
rect(96.0, 36.3, 2.4, 1.5, LIGHT, EDGE, .3)
text(99.5, 37.05, "train", size=5.1, ha="left")
rect(110.4, 36.3, 2.4, 1.5, BLUE, EDGE, .3)
text(114.0, 37.05, "held out", size=5.1, ha="left")

# Opposite-label comparison pairs: all pairs versus matching GC count.
text(147.8, 21.3, "All pairs", size=5.4)
text(165.9, 21.3, "Same GC", size=5.4)
pair(147.8, 29.1, BLUE_LIGHT, crossed=True)
pair(165.9, 29.1, BLUE_LIGHT, crossed=False)

for x, label in zip(centres, [
    "Supplied iEnhancer-2L files\n200 bp sequences",
    "Exact / reverse-complement identity\nSimilarity grouping and exclusions",
    "Five fixed grouped folds\nRelated sequences kept together",
    "Raw and exact-GC AUROC\nSingle seeds and ensembles",
]):
    text(x, 42.1, label, size=6.0, va="top")
text(90, 53.0, "DNABERT and DNABERT-2: three seeds each; seven archived baselines", size=6.0)

# Panel b: quantitative reporter activity, preserving validation-only selection.
panel("b", "Measured reporter activity", 62.0)
headers = ["Published lentiMPRA data", "Chromosome partition", "54 training fits", "Validation selection"]
for x, title in zip(centres, headers):
    text(x, 71.3, title, size=6.6, bold=True)
for x1, x2 in [(39.5, 47.0), (86.5, 92.0), (129.5, 136.0)]:
    arrow(x1, 84.8, x2, 84.8)
text(89.3, 92.4, "train / val", size=5.0)

# Published sequence/activity table; no wet-lab experiment is implied.
text(16.9, 77.4, "DNA", size=5.3)
text(30.3, 77.4, "Activity", size=5.3)
activity_palette = [TEAL_LIGHT, WHITE, "#A6D4CC", "#EDF6F3"]
for y, seq, width in [(80.0, "AACGTA", 7.2), (84.6, "TGCATC", 4.0), (89.2, "CGATGC", 9.0)]:
    sequence(7.4, y, seq, w=16.5, h=2.4, palette=activity_palette, letters=False)
    rect(26.0, y + .3, width, 1.8, TEAL, TEAL, .35)

# Separate chromosome partitions; widths are schematic, not proportional counts.
for x, label, fill in [(54.0, "Train", TEAL_LIGHT), (66.8, "Val", "#B8DDD5"), (79.6, "Test", LIGHT)]:
    text(x, 77.4, label, size=5.5)
    for delta, h in [(-2.2, 10.4), (1.4, 8.5)]:
        rect(x + delta - .8, 80.0, 1.6, h / 2 - .4, fill, EDGE, .45, .65)
        rect(x + delta - .8, 80.0 + h / 2 + .4, 1.6, h / 2 - .4, fill, EDGE, .45, .65)

# One 3 x 3 recipe/seed grid per model, repeated across two cell types.
for x, name in [(98.4, "DNABERT-2"), (111.0, "LegNet"), (123.6, "MPRAnn")]:
    text(x, 77.5, name, size=5.2)
    for i in range(3):
        for j in range(3):
            rect(x - 4.3 + j * 2.9, 81.0 + i * 2.9, 2.45, 2.45,
                 TEAL_LIGHT, EDGE, .4)

# A checkpoint symbol and a padlock depict the immutable model-selection lock.
text(144.6, 77.5, "Checkpoints", size=5.2)
text(162.85, 77.5, "Lock", size=5.2)
for i in range(4):
    for j in range(4):
        rect(140.0 + 2.4 * j, 80.7 + 2.4 * i, 2.0, 2.0,
             TEAL_LIGHT if (i + j) % 3 else "#A6D4CC", EDGE, .35)
ax.add_patch(Arc((162.85, 83.9), 6.1, 7.3, theta1=180, theta2=360,
                 edgecolor=EDGE, linewidth=.65))
rect(158.0, 83.2, 9.7, 7.4, TEAL_LIGHT, EDGE, .65, .6)
ax.add_patch(Circle((162.85, 86.6), .65, facecolor=EDGE, edgecolor=EDGE, linewidth=.4))
line([(162.85, 87.0), (162.85, 88.8)], EDGE, .6)

for x, label in zip(centres, [
    "ENCODE; Agarwal et al.\nHepG2 and K562; 200 bp",
    "Train / validation / test\nTargets scaled on training data",
    "3 models × 3 rates\n× 3 seeds × 2 cells",
    "Choose recipes using validation\nLock before test / external scoring",
]):
    text(x, 96.2, label, size=6.0, va="top")

# Model predictions branch after locking; external data enter only external scoring.
line([(157.0, 104.2), (157.0, 110.7), (86.0, 110.7)], EDGE, .7)
arrow(86.0, 110.7, 86.0, 115.2)
arrow(145.0, 110.7, 145.0, 115.2)
text(112.0, 107.8, "Locked models", size=5.5)

# Independent external source and two evaluation endpoints.
rect(4.0, 115.8, 43.0, 17.6, ORANGE_LIGHT, EDGE, .55, .8)
text(25.5, 119.6, "Gosai et al. (2024)", size=6.3, bold=True)
text(25.5, 126.5, "DHS-natural elements\nFixed quality and overlap screens", size=5.8)
arrow(47.0, 124.6, 60.5, 124.6)

line([(61.0, 115.8), (111.0, 115.8)], ORANGE, .7)
text(86.0, 119.6, "External transfer", size=6.6, bold=True)
text(86.0, 126.5, "8,399 elements per cell\nRanking and absolute prediction error", size=5.8)

line([(122.0, 115.8), (176.0, 115.8)], TEAL, .7)
text(149.0, 119.6, "Internal test and duration control", size=6.6, bold=True)
text(149.0, 126.5, "Selected models; DNABERT-2 longer run\nversus its own two-epoch prefix", size=5.8)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
overlap = []
for i, t in enumerate(TEXT):
    b = t.get_window_extent(renderer)
    if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height:
        outside.append(t.get_text())
    for t2 in TEXT[i + 1:]:
        b2 = t2.get_window_extent(renderer)
        if b.overlaps(b2):
            overlap.append([t.get_text(), t2.get_text()])

STEM = "Figure1_Study_workflow_proposed"
pdf_path = ROOT / f"{STEM}.pdf"
svg_path = ROOT / f"{STEM}.svg"
png_path = ROOT / f"{STEM}_600dpi.png"
tiff_path = ROOT / f"{STEM}_600dpi.tiff"
fig.savefig(pdf_path, format="pdf", metadata={
    "Title": "Figure 1. Study design and scientific questions",
    "Subject": "Separate proposed replacement; not inserted into manuscript",
    "Author": "",
})
fig.savefig(svg_path, format="svg", metadata={
    "Title": "Figure 1. Study design and scientific questions",
    "Description": "Standalone proposed redesign. Original study methods preserved.",
})
fig.savefig(png_path, format="png", dpi=600)
with Image.open(png_path) as im:
    rgb = im.convert("RGB")
rgb.save(png_path, dpi=(600, 600))
rgb.save(tiff_path, compression="tiff_lzw", dpi=(600, 600))
preview = rgb.copy()
preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
preview.save(ROOT / f"{STEM}_preview.png", dpi=(254, 254))
plt.close(fig)

reader = PdfReader(pdf_path)
page = reader.pages[0]
fonts = []
for name, ref in page["/Resources"]["/Font"].items():
    font = ref.get_object()
    descendants = font.get("/DescendantFonts", [])
    descriptor = descendants[0].get_object().get("/FontDescriptor") if descendants else font.get("/FontDescriptor")
    desc = descriptor.get_object() if descriptor else {}
    fonts.append({"name": str(font.get("/BaseFont")), "subtype": str(font.get("/Subtype")),
                  "embedded": any(k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3"))})
pdf_text = page.extract_text()
qa = {
    "stage": "automated checks complete; visual inspection pending",
    "width_mm": W, "height_mm": H,
    "pdf_dimensions_mm": [float(page.mediabox.width) * 25.4 / 72, float(page.mediabox.height) * 25.4 / 72],
    "font": "Arial", "resolved_arial": findfont(FontProperties(family="Arial")),
    "ordinary_text_range_pt": [5.0, 7.0], "panel_labels_pt": 8,
    "line_strokes_pt": [0.3, 0.7], "vector_fonts": fonts,
    "all_fonts_embedded": all(f["embedded"] for f in fonts),
    "pdf_has_raster_images": any(obj.get_object().get("/Subtype") == "/Image"
                                  for obj in page["/Resources"].get("/XObject", {}).values()),
    "svg_text_nodes": svg_path.read_text(encoding="utf-8").count("<text"),
    "text_outside_canvas": outside, "text_bbox_overlaps": overlap,
    "required_text_checks": {s: s in pdf_text for s in ["iEnhancer-2L", "54 training fits", "8,399", "DNABERT-2", "MPRAnn", "Gosai", "ENCODE"]},
    "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (pdf_path, svg_path, png_path, tiff_path)},
}
with Image.open(png_path) as im:
    qa["png"] = {"size_px": list(im.size), "dpi": list(im.info.get("dpi", [])), "mode": im.mode}
with Image.open(tiff_path) as im:
    qa["tiff"] = {"size_px": list(im.size), "dpi": [float(v) for v in im.info.get("dpi", [])], "mode": im.mode}
(ROOT / "qa" / "automated_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in qa.items() if k != "outputs_sha256"}, indent=2))
assert not outside, outside
assert not overlap, overlap
assert qa["all_fonts_embedded"]
assert all(qa["required_text_checks"].values())
