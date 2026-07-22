import glob
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLE = {
    "crys_color": "#E04A3F",
    "irl_color":  "#2B7BB9",
    "intervention_color": "gray",
    "linewidth": 1.6,
    "figsize": (10, 5),
    "dpi": 150,
    "ymin": 0,
    "ymax": None,
}


def from_summary(summary_path):
    with open(summary_path) as f:
        s = json.load(f)
    pmr = s.get("pmr_per_step")
    if not pmr:
        raise ValueError("summary.json missing pmr_per_step. Use --records instead.")
    pmr = sorted(pmr, key=lambda r: r["t"])
    ts       = np.array([r["t"] for r in pmr], dtype=float)
    irl_pmr  = np.array([np.nan if r["irl_pmr"]  is None else r["irl_pmr"]  for r in pmr])
    crys_pmr = np.array([np.nan if r["crys_pmr"] is None else r["crys_pmr"] for r in pmr])
    return {
        "ts": ts, "irl_pmr": irl_pmr, "crys_pmr": crys_pmr,
        "intervention_steps": sorted(set(s.get("intervention_steps", []))),
        "n_samples": s.get("n_processed", s.get("n_samples", 0)),
        "target_ef": s.get("target_formation_energy_per_atom", ""),
    }


def from_records(record_paths, max_steps, target_ef=""):
    if max_steps is None:
        raise ValueError("--records mode requires --max_steps.")

    steps = list(range(1, max_steps + 1))
    irl_sat  = {t: 0 for t in steps}
    irl_seen = {t: 0 for t in steps}
    crys_sat  = {t: 0 for t in steps}
    crys_seen = {t: 0 for t in steps}
    intervention = set()
    n_samples = 0

    for path in record_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_samples += 1
                for ts_str, v in rec.get("irl", {}).items():
                    t = int(ts_str)
                    if t in irl_seen:
                        irl_seen[t] += 1
                        if v:
                            irl_sat[t] += 1
                for ts_str, v in rec.get("crys", {}).items():
                    t = int(ts_str)
                    if t in crys_seen:
                        crys_seen[t] += 1
                        if v:
                            crys_sat[t] += 1
                intervention.update(rec.get("intervention_steps", []))

    if n_samples == 0:
        raise ValueError("No sample records found.")

    def pmr(sat, seen, t):
        return (sat[t] / seen[t] * 100) if seen[t] > 0 else np.nan

    ts       = np.array(steps, dtype=float)
    irl_pmr  = np.array([pmr(irl_sat,  irl_seen,  t) for t in steps])
    crys_pmr = np.array([pmr(crys_sat, crys_seen, t) for t in steps])
    return {
        "ts": ts, "irl_pmr": irl_pmr, "crys_pmr": crys_pmr,
        "intervention_steps": sorted(intervention),
        "n_samples": n_samples,
        "target_ef": target_ef,
    }


def render_plot(data, plot_path, invert_x=False):
    ts        = data["ts"]
    irl_pmr   = data["irl_pmr"]
    crys_pmr  = data["crys_pmr"]
    ivt_steps = data.get("intervention_steps", [])

    fig, ax = plt.subplots(figsize=STYLE["figsize"])

    ax.plot(ts, crys_pmr, color=STYLE["crys_color"], lw=STYLE["linewidth"],
            label="CrysLLMGen (monotonic degradation)")
    ax.plot(ts, irl_pmr, color=STYLE["irl_color"], lw=STYLE["linewidth"],
            label="IRLCrys (sawtooth convergence)")

    for ivt in ivt_steps:
        ax.axvline(x=ivt, color=STYLE["intervention_color"],
                   linestyle="--", lw=0.8, alpha=0.5)

    if invert_x:
        ax.invert_xaxis()

    ax.set_xlabel("Refinement Step", fontsize=11)
    ax.set_ylabel("Formation Energy PMR (%)", fontsize=11)
    ax.set_title(
        "PMR Trajectory during Iterative Refinement",
        fontsize=12,
    )

    ax.set_xlim(left=0)
    ax.legend(loc="upper left", fontsize=9)
    if STYLE["ymax"] is not None:
        ax.set_ylim(STYLE["ymin"], STYLE["ymax"])
    else:
        ax.set_ylim(bottom=STYLE["ymin"])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {plot_path}  (n={data.get('n_samples')})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, default=None,
                   help="path to summary.json (preferred)")
    p.add_argument("--records", type=str, nargs="+", default=None,
                   help="one or more intermediate_records.jsonl (supports wildcards, batch merging)")
    p.add_argument("--max_steps", type=int, default=None,
                   help="required for --records mode: total refinement steps")
    p.add_argument("--target_ef", type=str, default="",
                   help="formation energy label for title in --records mode")
    p.add_argument("--invert_x", action="store_true",
                   help="invert x-axis (T→0 direction); default 0→T matches example plot")
    p.add_argument("--plot_path", type=str, required=True)
    args = p.parse_args()

    if not args.summary and not args.records:
        p.error("Must provide either --summary or --records.")

    if args.summary and Path(args.summary).exists():
        data = from_summary(args.summary)
    else:
        if args.summary:
            print(f"[Plot] {args.summary} not found, using --records instead.")
        paths = []
        for pat in (args.records or []):
            paths.extend(sorted(glob.glob(pat)))
        if not paths:
            p.error("--records did not match any files.")
        print(f"[Plot] Merging {len(paths)} record files to aggregate PMR.")
        data = from_records(paths, args.max_steps, args.target_ef)

    render_plot(data, args.plot_path, invert_x=args.invert_x)


if __name__ == "__main__":
    main()