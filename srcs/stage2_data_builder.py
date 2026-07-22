import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from pymatgen.core import Structure, Element
from pymatgen.core.lattice import Lattice

from enc_dec import enc, VALID_ELEMENTS

NOISE_STD_CANDIDATES = [0.005, 0.010, 0.015, 0.020, 0.030]
NOISE_STD_WEIGHTS    = [0.15,  0.35,  0.25,  0.15,  0.10]


def sample_noise_std() -> float:
    return random.choices(
        NOISE_STD_CANDIDATES,
        weights=NOISE_STD_WEIGHTS,
        k=1
    )[0]


SIMILAR_ELEMENTS = {
    "Li": ["Na", "K"],   "Na": ["Li", "K"],    "K":  ["Na", "Rb"],
    "Rb": ["K",  "Cs"],  "Cs": ["Rb", "K"],
    "Be": ["Mg", "Ca"],  "Mg": ["Ca", "Be"],   "Ca": ["Sr", "Mg"],
    "Sr": ["Ca", "Ba"],  "Ba": ["Sr", "Ca"],
    "Fe": ["Co", "Ni"],  "Co": ["Fe", "Ni"],   "Ni": ["Co", "Fe"],
    "Cu": ["Ag", "Au"],  "Ag": ["Cu", "Au"],   "Au": ["Cu", "Ag"],
    "Zn": ["Cd", "Hg"],  "Mn": ["Fe", "Cr"],   "Cr": ["Mo", "W"],
    "Mo": ["W",  "Cr"],  "W":  ["Mo", "Cr"],   "Ti": ["Zr", "Hf"],
    "Zr": ["Ti", "Hf"],  "Hf": ["Zr", "Ti"],   "V":  ["Nb", "Ta"],
    "Nb": ["V",  "Ta"],  "Ta": ["Nb", "V"],
    "Al": ["Ga", "In"],  "Ga": ["Al", "In"],   "In": ["Ga", "Tl"],
    "Si": ["Ge", "Sn"],  "Ge": ["Si", "Sn"],   "Sn": ["Ge", "Pb"],
    "Pb": ["Sn", "Ge"],  "C":  ["Si", "Ge"],
    "N":  ["P",  "As"],  "P":  ["N",  "As"],   "As": ["P",  "Sb"],
    "Sb": ["As", "Bi"],  "Bi": ["Sb", "As"],
    "O":  ["S",  "Se"],  "S":  ["O",  "Se"],   "Se": ["S",  "Te"],
    "Te": ["Se", "S"],
    "F":  ["Cl", "Br"],  "Cl": ["F",  "Br"],   "Br": ["Cl", "I"],
    "I":  ["Br", "Cl"],
    "La": ["Ce", "Pr"],  "Ce": ["La", "Nd"],   "Nd": ["Pr", "Sm"],
}


def get_similar_element(symbol: str) -> str:
    candidates = SIMILAR_ELEMENTS.get(symbol, None)
    if candidates:
        return random.choice(candidates)
    try:
        el = Element(symbol)
        z = el.Z
        neighbors = [
            Element.from_Z(z + delta).symbol
            for delta in [-2, -1, 1, 2]
            if 1 <= z + delta <= 94
        ]
        if neighbors:
            return random.choice(neighbors)
    except Exception:
        pass
    return symbol


def perturb_continuous(
    structure: Structure,
    noise_std: float,
) -> Structure:

    frac_coords = structure.frac_coords.copy()
    frac_coords += np.random.normal(0, noise_std, frac_coords.shape)
    frac_coords = frac_coords % 1.0

    params = np.array(structure.lattice.parameters)
    params[:3] += np.random.normal(0, noise_std * 0.5, 3)
    params[3:] += np.random.normal(0, noise_std * 2.0, 3)
    params[:3] = np.clip(params[:3], 1.0, 30.0)
    params[3:] = np.clip(params[3:], 60.0, 120.0)

    try:
        lattice = Lattice.from_parameters(*params.tolist())
        return Structure(
            lattice=lattice,
            species=[site.specie.symbol for site in structure],
            coords=frac_coords,
            coords_are_cartesian=False,
        )
    except Exception:
        return structure


def perturb_composition(
    structure: Structure,
    noise_std: float,
    n_replace: int = 1,
) -> Structure:

    perturbed = perturb_continuous(structure, noise_std)

    species = [site.specie.symbol for site in perturbed]
    n_atoms = len(species)
    if n_atoms == 0:
        return perturbed

    n_replace = min(n_replace, n_atoms)
    replace_indices = random.sample(range(n_atoms), n_replace)
    for idx in replace_indices:
        species[idx] = get_similar_element(species[idx])

    try:
        return Structure(
            lattice=perturbed.lattice,
            species=species,
            coords=perturbed.frac_coords,
            coords_are_cartesian=False,
        )
    except Exception:
        return perturbed


def build_condition_dict(row: dict) -> dict:
    conditions = {}

    try:
        structure = Structure.from_str(row["cif"], fmt="cif")
        conditions["pretty_formula"] = structure.composition.reduced_formula
    except Exception:
        pass

    for key in ["spacegroup.number", "formation_energy_per_atom",
                "band_gap", "e_above_hull"]:
        if key in row and not pd.isna(row.get(key, float("nan"))):
            conditions[key] = row[key]

    return conditions


def _extract_crystal_only(full_text: str) -> str:
    lines = full_text.strip().split("\n")
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                vals = [float(x) for x in parts]
                if all(0.5 < v < 50 for v in vals):
                    return "\n".join(lines[i:])
            except ValueError:
                continue
    return full_text


def build_stage2_samples(
    row: dict,
    composition_perturb_prob: float,
    n_perturb_per_sample: int,
) -> list:

    samples = []

    try:
        target_structure = Structure.from_str(row["cif"], fmt="cif")
    except Exception:
        return []

    conditions = build_condition_dict(row)
    target_text = enc(target_structure, conditions)
    crystal_only_target = _extract_crystal_only(target_text)

    for _ in range(n_perturb_per_sample):
        noise_std = sample_noise_std()

        if random.random() < composition_perturb_prob:
            n_replace = random.randint(1, max(1, len(target_structure) // 3))
            perturbed = perturb_composition(
                target_structure,
                noise_std=noise_std,
                n_replace=n_replace,
            )
            perturb_type = "composition"
        else:
            perturbed = perturb_continuous(
                target_structure,
                noise_std=noise_std,
            )
            perturb_type = "continuous"

        input_text = enc(perturbed, conditions)

        samples.append({
            "input_text":   input_text,
            "target_text":  crystal_only_target,
            "perturb_type": perturb_type,
            "noise_std":    noise_std,
        })

    return samples


def main(args):
    print(f"[Stage2 DataBuilder] Reading training data: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"  Total {len(df)} training samples")
    print(f"\nNoise sampling configuration:")
    for std, w in zip(NOISE_STD_CANDIDATES, NOISE_STD_WEIGHTS):
        print(f"  noise_std={std:.3f}  weight={w:.2f}")
    print(f"  composition_perturb_prob = {args.composition_perturb_prob}")
    print(f"  n_perturb_per_sample     = {args.n_perturb_per_sample}")
    print(f"  Expected output samples: ~{len(df) * args.n_perturb_per_sample}\n")

    all_samples = []
    n_skip = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Stage2 data"):
        samples = build_stage2_samples(
            row=row.to_dict(),
            composition_perturb_prob=args.composition_perturb_prob,
            n_perturb_per_sample=args.n_perturb_per_sample,
        )
        if not samples:
            n_skip += 1
            continue
        all_samples.extend(samples)

    # ── Statistics ──────────────────────────────────────────────────
    print(f"\n[Stage2 DataBuilder] Build complete")
    print(f"  Skipped samples: {n_skip}")
    print(f"  Total training pairs: {len(all_samples)}")

    type_counts = {}
    for s in all_samples:
        t = s["perturb_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Continuous perturbations: {type_counts.get('continuous', 0)}")
    print(f"  Composition perturbations: {type_counts.get('composition', 0)}")

    # noise_std distribution statistics
    noise_counts = {}
    for s in all_samples:
        std = s["noise_std"]
        noise_counts[std] = noise_counts.get(std, 0) + 1
    print(f"\n  noise_std distribution:")
    for std in sorted(noise_counts.keys()):
        pct = noise_counts[std] / len(all_samples) * 100
        print(f"    {std:.3f}: {noise_counts[std]} ({pct:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(all_samples)
    out_df.to_csv(out_path, index=False)
    print(f"\n[Stage2 DataBuilder] Data saved to: {out_path}")

    # ── Sample preview ──────────────────────────────────────────────────
    print("\n=== Sample preview (first 2) ===")
    for i, sample in enumerate(all_samples[:2]):
        print(f"\n--- Sample {i+1} "
              f"[{sample['perturb_type']}, noise_std={sample['noise_std']:.3f}] ---")
        print("INPUT (first 300 chars):")
        print(sample["input_text"][:300] + "...")
        print("TARGET (first 200 chars):")
        print(sample["target_text"][:200] + "...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",
                        type=str,
                        default="data/mp_20/train.csv",
                        help="Path to Stage 1 training CSV")
    parser.add_argument("--out_path",
                        type=str,
                        default="data/mp_20/stage2_train.csv",
                        help="Stage 2 training data output path")
    parser.add_argument("--composition_perturb_prob",
                        type=float,
                        default=0.3,
                        help="Probability of triggering composition perturbation (0-1)")
    parser.add_argument("--n_perturb_per_sample",
                        type=int,
                        default=3,
                        help="Number of perturbed samples per target structure")
    args = parser.parse_args()
    main(args)
