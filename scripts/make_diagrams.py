"""Draw the explanatory diagrams used in the notebook and README.

Run from the repository root:  python scripts/make_diagrams.py
Writes PNGs to images/. Only matplotlib and numpy are required.
Values in the data-join diagram are illustrative; every number in the analysis comes from the notebook.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(exist_ok=True)

# Okabe-Ito colour-blind-safe palette (same as the notebook)
BLUE, ORANGE, GREEN, VERMILION, PURPLE, SKY, GREY, INK = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#9AA0A6", "#1F2328")
PAPER = "#FFFFFF"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.facecolor": PAPER})


def box(ax, x, y, w, h, fc, ec=None, r=0.02, lw=1.5, alpha=1.0):
    """Rounded rectangle in axes-fraction coordinates."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=lw, alpha=alpha, transform=ax.transAxes))


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))


def viability(dose, ic50, hill=1.2):
    """Share of cells alive (%) at a given dose for a standard sigmoid dose-response curve."""
    return 100 / (1 + (dose / ic50) ** hill)


# ------------------------------------------------------------------ 1. Banner
def banner():
    fig = plt.figure(figsize=(12, 3.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    box(ax, 0, 0, 1, 1, "#0B2545", r=0)
    ax.text(0.04, 0.72, "Beyond Tissue of Origin", color="white", fontsize=27, weight="bold", transform=ax.transAxes)
    ax.text(0.04, 0.55, "Quantifying the added predictive value of genomic features", color="white", fontsize=15,
            transform=ax.transAxes)
    ax.text(0.04, 0.45, "for anti-cancer drug response", color="white", fontsize=15, transform=ax.transAxes)
    ax.text(0.04, 0.25, "Machine learning on ~700 cancer cell lines  ·  8 drugs  ·  GDSC2 × DepMap",
            color="#B8C4D6", fontsize=12, transform=ax.transAxes)
    ax.text(0.04, 0.11, "Ahmed Shehata", color="#8EA3BF", fontsize=11, transform=ax.transAxes)
    # mini dose-response curves on the right
    ins = fig.add_axes([0.66, 0.16, 0.30, 0.68]); ins.set_facecolor("#0B2545")
    d = np.logspace(-3, 2, 200)
    ins.plot(d, viability(d, 0.03), color=SKY, lw=3, label="responds")
    ins.plot(d, viability(d, 15), color=ORANGE, lw=3, label="resists")
    ins.set_xscale("log"); ins.set_ylim(-5, 105)
    for s in ins.spines.values():
        s.set_visible(False)
    ins.tick_params(colors="#8EA3BF", labelsize=8, length=0)
    ins.set_xticks([]); ins.set_yticks([0, 50, 100]); ins.set_yticklabels(["0%", "50%", "100%"])
    ins.set_xlabel("more drug  →", color="#B8C4D6", fontsize=10)
    ins.set_ylabel("cancer cells alive", color="#B8C4D6", fontsize=10)
    ins.legend(frameon=False, labelcolor="white", fontsize=10, loc="lower left")
    fig.savefig(OUT / "banner.png"); plt.close(fig)


# ------------------------------------------------------------------ 2. Pipeline (project map)
def pipeline():
    steps = [("1", "Collect", "Drug test results\n+ each cell line's genes", BLUE),
             ("2", "Join & clean", "One table: ~700 cell lines,\n8 drugs", BLUE),
             ("3", "Explore", "Are the known\nrules visible?", GREEN),
             ("4", "Describe cells", "Gene activity, broken\ngenes, tissue type", GREEN),
             ("5", "Lock away 20%", "Exam set the model\nnever sees", VERMILION),
             ("6", "Train & tune", "2 models vs 2\n'lazy guess' baselines", ORANGE),
             ("7", "Sit the exam", "Score on the locked\ncells, with error bars", ORANGE),
             ("8", "Check reasoning", "Did it rediscover\nknown biology?", PURPLE)]
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.94, "The project in 8 steps", ha="center", fontsize=17, weight="bold", color=INK, transform=ax.transAxes)
    w, h, gap = 0.205, 0.30, 0.035
    for i, (num, title, sub, col) in enumerate(steps):
        row, col_i = divmod(i, 4)
        x = 0.035 + col_i * (w + gap)
        y = 0.50 if row == 0 else 0.08
        box(ax, x, y, w, h, fc=col, alpha=0.10, ec=col, r=0.025)
        box(ax, x + 0.012, y + h - 0.095, 0.032, 0.06, fc=col, r=0.012)
        ax.text(x + 0.028, y + h - 0.066, num, color="white", ha="center", va="center", fontsize=11,
                weight="bold", transform=ax.transAxes)
        ax.text(x + 0.060, y + h - 0.066, title, color=INK, va="center", fontsize=12.5, weight="bold", transform=ax.transAxes)
        ax.text(x + 0.018, y + 0.105, sub, color="#444", va="center", fontsize=10.5, transform=ax.transAxes,
                linespacing=1.4)
        if col_i < 3:
            arrow(ax, x + w + 0.004, y + h / 2, x + w + gap - 0.004, y + h / 2)
    # connector from step 4 down and back to step 5, routed through the gap between rows
    x4, x5, ymid = 0.035 + 3 * (w + gap) + w / 2, 0.035 + w / 2, 0.44
    ax.plot([x4, x4, x5], [0.50, ymid, ymid], color=GREY, lw=2, transform=ax.transAxes, solid_capstyle="round")
    arrow(ax, x5, ymid, x5, 0.08 + h + 0.004)
    fig.savefig(OUT / "pipeline.png"); plt.close(fig)


# ------------------------------------------------------------------ 3. What IC50 means
def ic50():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    d = np.logspace(-3, 3, 400)
    max_tested = 10
    ax.axvspan(1e-3, max_tested, color=GREY, alpha=0.10)
    ax.text(0.0013, 104, "doses actually tested in the lab", color="#666", fontsize=10, va="bottom")
    for ic, col, lab in [(0.02, BLUE, "Cell line A: responds"), (1.5, GREEN, "Cell line B: in between"),
                         (60, ORANGE, "Cell line C: resists")]:
        ax.plot(d, viability(d, ic), color=col, lw=3, label=lab)
        ax.plot([ic, ic], [0, 50], color=col, ls=":", lw=1.5)
        ax.plot(ic, 50, "o", color=col, ms=8, zorder=5)
    ax.axhline(50, color="#999", lw=1, ls="--")
    ax.text(1.3e-3, 46, "half the cells killed", color="#666", fontsize=10, va="top")
    ax.annotate("C's IC50 lies beyond the highest dose tested:\nit is estimated by extending the curve",
                xy=(60, 50), xytext=(150, 80), fontsize=9.5, color="#8A5A00", ha="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=ORANGE),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE))
    ax.set_xscale("log"); ax.set_ylim(-3, 112); ax.set_xlim(1e-3, 1e3)
    ax.set_xlabel("Drug dose (µM, log scale: each tick = 10× more drug)", fontsize=11)
    ax.set_ylabel("Cancer cells still alive (%)", fontsize=11)
    ax.set_title("IC50 = the dose that kills half the cells.  Lower IC50 → the drug works better.",
                 fontsize=13, weight="bold", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "ic50_explained.png"); plt.close(fig)


# ------------------------------------------------------------------ 4. The data join
def data_join():
    fig = plt.figure(figsize=(13.5, 3.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    def table(x, y, title, sub, header, rows, col, widths):
        ax.text(x, y + 0.13, title, fontsize=13, weight="bold", color=col, transform=ax.transAxes)
        ax.text(x, y + 0.075, sub, fontsize=9.5, color="#555", transform=ax.transAxes)
        tw = sum(widths)
        box(ax, x, y - 0.075, tw, 0.075, fc=col, r=0.008)
        cx = x
        for htxt, wdt in zip(header, widths):
            ax.text(cx + 0.008, y - 0.038, htxt, color="white", fontsize=9.5, weight="bold", va="center", transform=ax.transAxes)
            cx += wdt
        for r_i, row in enumerate(rows):
            yy = y - 0.075 - (r_i + 1) * 0.075
            box(ax, x, yy, tw, 0.075, fc=col if r_i % 2 == 0 else "white", alpha=0.07 if r_i % 2 == 0 else 1, ec=None, r=0.0)
            cx = x
            for val, wdt in zip(row, widths):
                ax.text(cx + 0.008, yy + 0.037, val, fontsize=9.5, color=INK, va="center", transform=ax.transAxes)
                cx += wdt

    table(0.02, 0.72, "Drug test results  (GDSC)", "one row = one cell line × one drug",
          ["cell line", "drug", "IC50 (µM)"],
          [["A375 (skin)", "Dabrafenib", "0.01"], ["A375 (skin)", "Paclitaxel", "0.004"],
           ["HL-60 (blood)", "Venetoclax", "0.05"], ["…", "…", "…"]], BLUE, [0.095, 0.085, 0.07])
    table(0.31, 0.72, "Cell line genetics  (DepMap)", "one row = one cell line",
          ["cell line", "BRAF broken?", "BCL2 level", "tissue", "+19k genes"],
          [["A375", "yes", "low", "skin", "…"], ["HL-60", "no", "high", "blood", "…"],
           ["MCF7", "no", "high", "breast", "…"], ["…", "…", "…", "…", "…"]], GREEN, [0.065, 0.09, 0.075, 0.06, 0.08])
    ax.text(0.2925, 0.50, "+", fontsize=22, ha="center", va="center", color=INK, weight="bold", transform=ax.transAxes)
    arrow(ax, 0.695, 0.50, 0.725, 0.50, color=INK)
    ax.text(0.74, 0.85, "Joined on a shared ID code", fontsize=13, weight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.74, 0.74, "For each drug, one row per cell line:\n\n"
                        "inputs  =  its genes + tissue\n"
                        "answer  =  its IC50 for that drug\n\n"
                        "The model learns the link\nfrom inputs to answer.",
            fontsize=11, color="#333", va="top", transform=ax.transAxes, linespacing=1.45)
    ax.text(0.02, 0.05, "Values shown are illustrative. A375 really is a BRAF-mutant skin (melanoma) cell line.",
            fontsize=8.5, color="#777", transform=ax.transAxes, style="italic")
    fig.savefig(OUT / "data_join.png"); plt.close(fig)


# ------------------------------------------------------------------ 5. The fair test
def fair_test():
    fig = plt.figure(figsize=(12, 2.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.03, 0.86, "The fair test: an exam on cell lines the model has never seen",
            fontsize=15, weight="bold", color=INK, transform=ax.transAxes)
    x0, W, y, h = 0.03, 0.94, 0.40, 0.30
    train_w = W * 0.8
    for k in range(5):
        xk = x0 + k * train_w / 5
        box(ax, xk + 0.003, y, train_w / 5 - 0.006, h, fc=BLUE, alpha=0.18 + 0.1 * (k % 2), ec=BLUE, r=0.01)
        ax.text(xk + train_w / 10, y + h / 2, f"practice\nround {k + 1}", ha="center", va="center", fontsize=10,
                color=INK, transform=ax.transAxes)
    box(ax, x0 + train_w + 0.01, y, W * 0.2 - 0.01, h, fc=VERMILION, alpha=0.9, r=0.01)
    ax.text(x0 + train_w + W * 0.1 + 0.005, y + h / 2, "LOCKED\nfinal exam", ha="center", va="center", fontsize=11,
            color="white", weight="bold", transform=ax.transAxes)
    ax.text(x0, y - 0.08, "80% of cell lines (~570): used to learn and to tune settings by rotating\n"
                         "practice rounds, with each round held back once as a mini-test (5-fold cross-validation)",
            fontsize=10, color="#333", va="top", transform=ax.transAxes)
    ax.text(x0 + train_w + 0.01, y - 0.08, "20% (~140): opened exactly once,\nat the very end",
            fontsize=10, color=VERMILION, va="top", transform=ax.transAxes)
    fig.savefig(OUT / "fair_test.png"); plt.close(fig)


if __name__ == "__main__":
    banner(); pipeline(); ic50(); data_join(); fair_test()
    print("Diagrams written to", OUT)
