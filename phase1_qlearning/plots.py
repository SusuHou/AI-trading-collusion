"""Figures from aggregated POC results.

1. delta_c.png -- normalized profitability Delta^C vs log10 sigma_u
   (paper Fig 2 Panel A analog: session strip + mean +/- p1-p99 band; with
   only two sigma_u cells this is two columns rather than the full U-shape).
2. irf.png -- calibrated-shock impulse responses (paper Fig 3/4 analog):
   price deviation and both speculators' order-flow deviations, means across
   sessions, one panel per result file (requires *_irf.npz from
   classify_mechanism.py).

Usage:
  python -m phase1_qlearning.plots results/*.npz --out-dir results/figures
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1_qlearning.aggregate import collect

# dataviz reference palette (validated set, fixed slot order)
SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e7e6e2"
S1, S2, S3 = "#2a78d6", "#1baf7a", "#eda100"  # blue, aqua, yellow


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def fig_delta_c(rows, out_path):
    cells = {}
    for r in rows:
        cells.setdefault(r["sigma_u"], []).append(r["delta_c"])
    xs = sorted(cells)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    rng = np.random.default_rng(0)
    for su in xs:
        dc = np.asarray(cells[su])
        lx = np.log10(su)
        jitter = rng.uniform(-0.12, 0.12, size=len(dc))
        ax.plot(lx + jitter, dc, "o", ms=4.5, mfc=S1, mec="none", alpha=0.45,
                zorder=2)
        ax.errorbar([lx], [dc.mean()],
                    yerr=[[dc.mean() - np.percentile(dc, 1)],
                          [np.percentile(dc, 99) - dc.mean()]],
                    fmt="o", ms=8, mfc=S1, mec=SURFACE, mew=1.5, color=S1,
                    elinewidth=2, capsize=5, zorder=4)
        ax.annotate(f"mean {dc.mean():.2f}", (lx, dc.mean()),
                    xytext=(14, -4), textcoords="offset points",
                    color=TEXT_1, fontsize=9, fontweight="bold")
    ax.axhline(0.0, color=TEXT_2, lw=1, ls=(0, (4, 3)))
    ax.axhline(1.0, color=TEXT_2, lw=1, ls=(0, (4, 3)))
    x0, x1 = ax.get_xlim()
    ax.text(x0 + 0.02 * (x1 - x0), 0.02, "non-collusive Nash  (ΔC = 0)",
            color=TEXT_2, fontsize=8.5, va="bottom")
    ax.text(x0 + 0.02 * (x1 - x0), 0.98, "perfect cartel  (ΔC = 1)",
            color=TEXT_2, fontsize=8.5, va="top")
    ax.set_xlabel("noise trading risk σ_u  (log-spaced)", color=TEXT_2, fontsize=10)
    ax.set_ylabel("normalized profitability  ΔC", color=TEXT_2, fontsize=10)
    ax.set_title("Supra-competitive profits of Q-learning speculators "
                 "(POC scale)", color=TEXT_1, fontsize=11, loc="left", pad=12)
    ax.set_xticks([np.log10(su) for su in xs])
    ax.set_xticklabels([f"{su:g}" for su in xs])
    ax.set_xlim(min(np.log10(min(xs)) - 0.8, x0), max(np.log10(max(xs)) + 0.8, x1))
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def fig_irf(irf_paths, out_path):
    n = len(irf_paths)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.4), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, path in zip(axes[0], sorted(irf_paths)):
        d = np.load(path)
        p_tilde = d["p_tilde"]            # (T+1, batch)
        x_tilde = d["x_tilde"]            # (T+1, batch, I)
        shock_t = int(d["shock_t"])
        t = np.arange(1, p_tilde.shape[0])
        _style(ax)
        ax.axvline(shock_t, color=GRID, lw=6, zorder=1)
        ax.text(shock_t, 1.02, "shock", transform=ax.get_xaxis_transform(),
                ha="center", color=TEXT_2, fontsize=8.5)
        ax.axhline(0, color=TEXT_2, lw=0.8)
        series = [
            ("price deviation", 100 * np.nanmean(p_tilde[1:], axis=1), S1, "-", "o"),
            ("speculator 1 flow", 100 * np.nanmean(x_tilde[1:, :, 0], axis=1),
             S2, "--", "s"),
            ("speculator 2 flow", 100 * np.nanmean(x_tilde[1:, :, 1], axis=1),
             S3, "-.", "D"),
        ]
        for name, ys, color, ls, mk in series:
            ax.plot(t, ys, ls, marker=mk, color=color, lw=2, ms=5.5,
                    mec=SURFACE, mew=1.0, label=name, zorder=3)
        ax.set_xlabel("period t (shock at t = 3)", color=TEXT_2, fontsize=10)
        ax.set_ylabel("% deviation from paired baseline", color=TEXT_2, fontsize=10)
        ax.set_title(f"σ_u = {float(d['sigma_u']):g}", color=TEXT_1,
                     fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2, loc="upper right")
        ax.set_xlim(0.5, t[-1] + 0.5)
    fig.suptitle("Impulse response to a calibrated 1.2% price shock",
                 color=TEXT_1, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="+", help="run_session .npz files/globs")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args(argv)

    paths = []
    for pattern in args.results:
        paths.extend(glob.glob(pattern) if any(c in pattern for c in "*?[") else [pattern])
    session_paths = [p for p in paths if not p.endswith("_irf.npz")]
    irf_paths = [p for p in paths if p.endswith("_irf.npz")]
    irf_paths += [q for p in session_paths
                  if os.path.exists(q := p.replace(".npz", "_irf.npz"))
                  and q not in irf_paths]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = collect(session_paths)
    out = fig_delta_c(rows, os.path.join(args.out_dir, "delta_c.png"))
    print(f"wrote {out}")
    if irf_paths:
        out = fig_irf(irf_paths, os.path.join(args.out_dir, "irf.png"))
        print(f"wrote {out}")
    else:
        print("no *_irf.npz found -- run classify_mechanism.py first for the IRF figure")


if __name__ == "__main__":
    main()
