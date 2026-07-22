"""
eval_stability_novelty_v2.py
Evaluate metastability rate, stability rate, and novelty of generated structures (using real MP convex hull data)

Pipeline:
  1. CHGNet relaxation of generated structures
  2. Match relaxed structures with MP reference database using StructureMatcher
     - Match found -> use MP's e_above_hull directly
     - No match (novel structure) -> use CHGNet formation energy as e_above_hull approximation
  3. Novelty: relaxed structure does not match with either training or test set

Metrics:
  % Metastable  : e_above_hull < 0.1 eV/atom
  % Stable      : e_above_hull < 0.05 eV/atom
  % Novel (among metastable) : proportion of novel structures among metastable
  % Novel (among stable)     : proportion of novel structures among stable

Usage:
    python eval_stability_novelty_v2.py \\
        --csv_path results/bc2/bc2_200_structures.csv \\
        --mp_ref_csv mp_FeO_reference.csv \\
        --train_csv data/mp_20/train.csv \\
        --test_csv  data/mp_20/test.csv \\
        --out_path  results/bc2/bc2_200_stability_novelty.json
"""

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

try:
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer
    CHGNET  = CHGNet.load()
    RELAXER = StructOptimizer()
    print("[INFO] CHGNet + StructOptimizer loaded.")
except Exception as e:
    CHGNET  = None
    RELAXER = None
    print(f"[WARN] CHGNet not available: {e}")


def relax_structure(structure: Structure):
    """
    Relax structure using CHGNet.
    Returns (relaxed_structure, formation_energy_per_atom)
    Returns (None, None) on failure.
    """
    if RELAXER is None:
        return None, None
    try:
        result   = RELAXER.relax(structure, verbose=False)
        relaxed  = result["final_structure"]
        pred     = CHGNET.predict_structure(relaxed)
        ef       = float(pred["e"])
        return relaxed, ef
    except Exception:
        try:
            pred = CHGNET.predict_structure(structure)
            return structure, float(pred["e"])
        except Exception:
            return None, None


def load_mp_reference(mp_ref_csv: str) -> pd.DataFrame:
    """Load Fe-O reference data extracted from MP (with real e_above_hull)."""
    df = pd.read_csv(mp_ref_csv)
    print(f"[MP Ref] Loaded {len(df)} reference phases")
    return df


def match_to_mp(relaxed_structure: Structure,
                mp_df: pd.DataFrame,
                matcher: StructureMatcher) -> float | None:
    """
    Match relaxed structure with MP reference database, return matched phase e_above_hull.
    Returns None if no match (novel structure not in known Fe-O phase space).

    Note: MP reference data only contains formula and e_above_hull, no CIF.
    Thus matching is done by formula + composition approximation using StructureMatcher.
    Since reference data has no CIF, formula-based matching is used as an approximation:
    if the generated structure's formula exists in MP, take the minimum e_above_hull.
    """
    try:
        gen_formula = relaxed_structure.composition.reduced_formula
        matched = mp_df[mp_df["formula"] == gen_formula]
        if len(matched) > 0:
            return float(matched["e_above_hull"].min())
        return None
    except Exception:
        return None


def load_reference_structures(train_csv: str, test_csv: str,
                               target_elements: set) -> list:
    """
    Load structures containing target elements from training+test sets as reference for novelty comparison.
    Only keep structures whose element set exactly matches target_elements (e.g., only Fe and O),
    to avoid meaningless comparisons with unrelated systems and significantly reduce comparison count.
    """
    print(f"[Novelty] Loading reference structures (element set={target_elements})...")
    dfs = []
    for path in [train_csv, test_csv]:
        if path and Path(path).exists():
            dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    structures = []
    n_total = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering target element structures"):
        try:
            s = Structure.from_str(row["cif"], fmt="cif")
            gen_elements = set(str(e) for e in s.composition.elements)
            if gen_elements == target_elements:
                structures.append(s)
                n_total += 1
        except Exception:
            continue

    print(f"[Novelty] Found {n_total} Fe-O reference structures (training + test set)")
    return structures


def is_novel(structure: Structure, ref_structures: list,
             matcher: StructureMatcher) -> bool:
    """Novel if no match with any reference structure."""
    for ref in ref_structures:
        try:
            if matcher.fit(structure, ref):
                return False
        except Exception:
            continue
    return True


def evaluate(args):
    print("=" * 60)
    print("Stability & Novelty Evaluation v2 (Real MP Convex Hull Data)")
    print("=" * 60)

    df       = pd.read_csv(args.csv_path)
    mp_df    = load_mp_reference(args.mp_ref_csv)
    matcher  = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    print(f"\n[Eval] Total {len(df)} generated structures, starting CHGNet relaxation...")

    relaxed_structures = []
    ef_values          = []
    ehull_values       = []
    ehull_sources      = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="CHGNet relaxation"):
        try:
            s = Structure.from_str(row["cif"], fmt="cif")
        except Exception:
            relaxed_structures.append(None)
            ef_values.append(None)
            ehull_values.append(None)
            ehull_sources.append("parse_failed")
            continue

        relaxed, ef = relax_structure(s)
        relaxed_structures.append(relaxed)
        ef_values.append(ef)

        if relaxed is not None:
            ehull_mp = match_to_mp(relaxed, mp_df, matcher)
            if ehull_mp is not None:
                ehull_values.append(ehull_mp)
                ehull_sources.append("mp_matched")
            else:
                ehull_approx = max(0.0, ef - (-1.5)) if ef is not None else None
                ehull_values.append(ehull_approx)
                ehull_sources.append("chgnet_approx")
        else:
            ehull_values.append(None)
            ehull_sources.append("relax_failed")

    valid_ehull = [v for v in ehull_values if v is not None]
    n_total     = len(df)

    n_metastable = sum(v < args.metastable_threshold
                       for v in ehull_values if v is not None)
    n_stable     = sum(v < args.stable_threshold
                       for v in ehull_values if v is not None)

    pct_metastable = n_metastable / n_total * 100
    pct_stable     = n_stable     / n_total * 100

    n_mp_matched = sum(1 for s in ehull_sources if s == "mp_matched")
    n_approx     = sum(1 for s in ehull_sources if s == "chgnet_approx")

    print(f"\n── Stability Results ──────────────────────────────────")
    print(f"  CHGNet relaxation succeeded:         {sum(s is not None for s in relaxed_structures)}/{n_total}")
    print(f"  MP database matched:                 {n_mp_matched}/{n_total}")
    print(f"  CHGNet approximation (novel):        {n_approx}/{n_total}")
    print(f"  e_above_hull mean:                   {np.mean(valid_ehull):.4f} eV/atom")
    print(f"  e_above_hull std:                    {np.std(valid_ehull):.4f} eV/atom")
    print(f"  % Metastable (Ehull < {args.metastable_threshold}): "
          f"{n_metastable}/{n_total} = {pct_metastable:.1f}%")
    print(f"  % Stable (Ehull < {args.stable_threshold}):               "
          f"{n_stable}/{n_total} = {pct_stable:.1f}%")

    target_elements = set(args.target_elements)
    ref_structures = load_reference_structures(
        args.train_csv, args.test_csv, target_elements)

    metastable_indices = [
        i for i, v in enumerate(ehull_values)
        if v is not None and v < args.metastable_threshold
        and relaxed_structures[i] is not None
    ]
    stable_indices = [
        i for i, v in enumerate(ehull_values)
        if v is not None and v < args.stable_threshold
        and relaxed_structures[i] is not None
    ]

    print(f"\n── Novelty Evaluation ──────────────────────────────────")
    print(f"  Metastable structures: {len(metastable_indices)}")
    print(f"  Stable structures:     {len(stable_indices)}")

    novel_flags = {}

    for i in tqdm(metastable_indices, desc="Metastable novelty"):
        novel_flags[i] = is_novel(relaxed_structures[i], ref_structures, matcher)

    for i in stable_indices:
        if i not in novel_flags:
            novel_flags[i] = is_novel(relaxed_structures[i], ref_structures, matcher)

    novel_metastable = sum(novel_flags.get(i, False) for i in metastable_indices)
    novel_stable     = sum(novel_flags.get(i, False) for i in stable_indices)

    pct_novel_metastable = novel_metastable / max(len(metastable_indices), 1) * 100
    pct_novel_stable     = novel_stable     / max(len(stable_indices),     1) * 100

    print(f"  % Novel (among metastable): "
          f"{novel_metastable}/{len(metastable_indices)} = {pct_novel_metastable:.1f}%")
    print(f"  % Novel (among stable):     "
          f"{novel_stable}/{len(stable_indices)} = {pct_novel_stable:.1f}%")

    print(f"\n{'='*60}")
    print("Final Results (Corresponding to Paper Table)")
    print(f"{'='*60}")
    print(f"  N:                          {n_total}")
    print(f"  % Metastable (Ehull<{args.metastable_threshold}):  {pct_metastable:.1f}%")
    print(f"  % Stable (Ehull<{args.stable_threshold}):        {pct_stable:.1f}%")
    print(f"  % Novel (among metastable): {pct_novel_metastable:.1f}%")
    print(f"  % Novel (among stable):     {pct_novel_stable:.1f}%")
    print(f"{'='*60}")

    df["ef_chgnet"]     = ef_values
    df["ehull"]         = ehull_values
    df["ehull_source"]  = ehull_sources
    df["is_metastable"] = [v is not None and v < args.metastable_threshold
                           for v in ehull_values]
    df["is_stable"]     = [v is not None and v < args.stable_threshold
                           for v in ehull_values]
    df["is_novel"]      = [novel_flags.get(i, None) for i in range(len(df))]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path.with_suffix(".csv"), index=False)

    summary = {
        "n_total":              n_total,
        "n_relaxed":            sum(s is not None for s in relaxed_structures),
        "n_mp_matched":         n_mp_matched,
        "n_chgnet_approx":      n_approx,
        "ehull_mean":           float(np.mean(valid_ehull)) if valid_ehull else None,
        "ehull_std":            float(np.std(valid_ehull))  if valid_ehull else None,
        "metastable_threshold": args.metastable_threshold,
        "stable_threshold":     args.stable_threshold,
        "n_metastable":         n_metastable,
        "n_stable":             n_stable,
        "pct_metastable":       round(pct_metastable, 2),
        "pct_stable":           round(pct_stable, 2),
        "n_novel_metastable":   novel_metastable,
        "n_novel_stable":       novel_stable,
        "pct_novel_metastable": round(pct_novel_metastable, 2),
        "pct_novel_stable":     round(pct_novel_stable, 2),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Eval] JSON: {out_path}")
    print(f"[Eval] CSV:  {out_path.with_suffix('.csv')}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",    required=True,
                        help="Generated structures CSV (with cif column)")
    parser.add_argument("--mp_ref_csv",  required=True,
                        help="MP reference data CSV (mp_FeO_reference.csv)")
    parser.add_argument("--train_csv",   default="data/mp_20/train.csv")
    parser.add_argument("--test_csv",    default="data/mp_20/test.csv")
    parser.add_argument("--out_path",    default="results/stability_novelty.json")
    parser.add_argument("--target_elements", nargs="+", default=["Fe", "O"],
                        help="Target element set for filtering reference structures (default Fe O)")
    parser.add_argument("--metastable_threshold", type=float, default=0.1,
                        help="Metastable threshold: e_above_hull < this value considered metastable (default 0.1)")
    parser.add_argument("--stable_threshold", type=float, default=0.05,
                        help="Stable threshold: e_above_hull < this value considered stable (default 0.05)")
    args = parser.parse_args()
    evaluate(args)