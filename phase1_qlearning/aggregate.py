"""Aggregate many run_session output files into a per-session table and
per-parameter-cell summary (mean/median/percentiles of Delta^C, metrics,
convergence stats, and mechanism-classification shares when the matching
*_irf.npz files from classify_mechanism.py exist).

Usage:
  python -m phase1_qlearning.aggregate results/*.npz [--csv results/summary.csv]

(The *_irf.npz files are detected automatically and must not be passed in.)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1_qlearning.classify_mechanism import LABELS

SESSION_FIELDS = ["delta_c", "profit_gain", "chi_hat", "informativeness",
                  "liquidity", "mispricing", "converged_at", "best_streak"]


def collect(paths: list[str]) -> list[dict]:
    """One row per session across all files."""
    rows = []
    for path in sorted(paths):
        if path.endswith("_irf.npz"):
            continue
        d = np.load(path, allow_pickle=False)
        cfg = json.loads(str(d["config"]))
        batch = len(d["delta_c"])
        irf_path = path.replace(".npz", "_irf.npz")
        labels = None
        if os.path.exists(irf_path):
            labels = np.load(irf_path)["label"]
        for b in range(batch):
            row = {
                "file": os.path.basename(path), "session": b,
                "sigma_u": float(d["sigma_u"]), "I": int(d["I"]),
                "rho": float(d["rho"]), "xi": float(d["xi"]),
                "seed": int(d["seed"]), "periods_run": int(d["periods_run"]),
                "beta": cfg["params"].get("beta"),
            }
            for f in SESSION_FIELDS:
                row[f] = float(np.asarray(d[f])[b])
            row["converged"] = row["converged_at"] >= 0
            row["mechanism"] = LABELS[int(labels[b])] if labels is not None else ""
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Group by parameter cell (sigma_u, I, rho, xi)."""
    cells = {}
    for r in rows:
        cells.setdefault((r["sigma_u"], r["I"], r["rho"], r["xi"]), []).append(r)
    out = []
    for key in sorted(cells):
        rs = cells[key]
        dc = np.array([r["delta_c"] for r in rs])
        summary = {
            "sigma_u": key[0], "I": key[1], "rho": key[2], "xi": key[3],
            "n_sessions": len(rs),
            "n_converged": sum(r["converged"] for r in rs),
            "delta_c_mean": dc.mean(), "delta_c_median": np.median(dc),
            "delta_c_p1": np.percentile(dc, 1), "delta_c_p99": np.percentile(dc, 99),
            "profit_gain_mean": np.mean([r["profit_gain"] for r in rs]),
            "chi_hat_mean": np.mean([r["chi_hat"] for r in rs]),
            "informativeness_mean": np.mean([r["informativeness"] for r in rs]),
            "liquidity_mean": np.mean([r["liquidity"] for r in rs]),
            "mispricing_mean": np.mean([r["mispricing"] for r in rs]),
        }
        mechs = [r["mechanism"] for r in rs if r["mechanism"]]
        if mechs:
            for name in LABELS.values():
                summary[f"share_{name}"] = mechs.count(name) / len(mechs)
        out.append(summary)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--csv", default=None, help="write per-session rows here")
    ap.add_argument("--summary-csv", default=None)
    args = ap.parse_args(argv)

    paths = []
    for pattern in args.results:
        paths.extend(glob.glob(pattern) if any(c in pattern for c in "*?[") else [pattern])
    rows = collect(paths)
    if not rows:
        raise SystemExit("no session rows found")
    summaries = summarize(rows)

    for s in summaries:
        line = (f"sigma_u={s['sigma_u']:<8g} I={s['I']} rho={s['rho']} "
                f"xi={s['xi']:g}  n={s['n_sessions']} "
                f"(conv {s['n_converged']})  "
                f"dC mean={s['delta_c_mean']:+.3f} med={s['delta_c_median']:+.3f} "
                f"[p1 {s['delta_c_p1']:+.3f}, p99 {s['delta_c_p99']:+.3f}]  "
                f"gain={s['profit_gain_mean']:.3f}  "
                f"I^C={s['informativeness_mean']:.3g} "
                f"L^C={s['liquidity_mean']:.4g} E^C={s['mispricing_mean']:.4g}")
        if "share_price_trigger" in s:
            line += (f"  mech: PT {s['share_price_trigger']:.0%} / "
                     f"OP {s['share_over_pruning']:.0%} / "
                     f"UC {s['share_unclassified']:.0%}")
        print(line)

    for out_path, data in ((args.csv, rows), (args.summary_csv, summaries)):
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
            print(f"wrote {out_path}")
    return rows, summaries


if __name__ == "__main__":
    main()
