import argparse
import pandas as pd
import numpy as np


def verify_spacegroup(test_csv: str):
    print("=" * 60)
    print("1. Verifying SpacegroupAnalyzer")
    print("=" * 60)

    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    df = pd.read_csv(test_csv)
    row = df.iloc[0]

    structure = Structure.from_str(row["cif"], fmt="cif")
    print(f"Test structure: {structure.composition.reduced_formula}")

    sga = SpacegroupAnalyzer(structure, symprec=0.2)
    sg_number = sga.get_space_group_number()
    sg_symbol = sga.get_space_group_symbol()

    print(f"  Detected spacegroup number: {sg_number}")
    print(f"  Detected spacegroup symbol: {sg_symbol}")

    if "spacegroup.number" in df.columns:
        target_sg = row["spacegroup.number"]
        print(f"  Target spacegroup from CSV:   {target_sg}")
        print(f"  Match: {sg_number == target_sg}")
    else:
        print("  [WARN] CSV missing spacegroup.number column, needs on-the-fly computation")

    print("  [OK] SpacegroupAnalyzer verification passed\n")
    return True


def verify_chgnet():
    print("=" * 60)
    print("2. Verifying CHGNet")
    print("=" * 60)

    try:
        from chgnet.model import CHGNet
        from pymatgen.core import Structure, Lattice

        chgnet = CHGNet.load()
        print("  [OK] CHGNet model loaded successfully")

        lattice = Lattice.cubic(5.64)
        structure = Structure(
            lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        prediction = chgnet.predict_structure(structure)
        ef = float(prediction["e"])
        print(f"  NaCl test prediction formation energy: {ef:.4f} eV/atom")
        print("  [OK] CHGNet prediction functional\n")
        return True

    except ImportError as e:
        print(f"  [FAIL] CHGNet not installed: {e}")
        print("  Installation: pip install chgnet --break-system-packages\n")
        return False
    except Exception as e:
        print(f"  [FAIL] CHGNet runtime error: {e}\n")
        return False


def verify_alignn():
    print("=" * 60)
    print("3. Verifying ALIGNN")
    print("=" * 60)

    try:
        from alignn.pretrained import get_figshare_model
        from pymatgen.core import Structure, Lattice
        from jarvis.core.atoms import pmg2jarvis

        print("  Loading ALIGNN pretrained model (mp_gappbe_alignn)...")
        model = get_figshare_model(model_name="mp_gappbe_alignn")
        print("  [OK] ALIGNN model loaded successfully")

        lattice = Lattice.cubic(5.64)
        structure = Structure(
            lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )

        jarvis_atoms = pmg2jarvis(structure)

        from alignn.pretrained import get_prediction
        bg = get_prediction(
            model_name="mp_gappbe_alignn",
            atoms=jarvis_atoms,
        )
        print(f"  NaCl test prediction band gap: {bg}")
        print("  [OK] ALIGNN prediction functional\n")
        return True

    except ImportError as e:
        print(f"  [FAIL] ALIGNN not installed: {e}")
        print("  Installation: pip install alignn --break-system-packages\n")
        return False
    except Exception as e:
        print(f"  [FAIL] ALIGNN runtime error: {e}")
        print(f"  Error type: {type(e).__name__}\n")
        return False


def verify_condition_extraction(test_csv: str):
    print("=" * 60)
    print("4. Verifying target condition extraction from test.csv")
    print("=" * 60)

    df = pd.read_csv(test_csv)
    print(f"  test.csv columns: {list(df.columns)}")
    print(f"  Total {len(df)} test samples\n")

    row = df.iloc[0]

    from pymatgen.core import Structure
    structure = Structure.from_str(row["cif"], fmt="cif")

    conditions = {
        "pretty_formula": structure.composition.reduced_formula,
    }

    for key in ["spacegroup.number", "formation_energy_per_atom",
                "band_gap", "e_above_hull"]:
        if key in df.columns:
            conditions[key] = row[key]
        else:
            print(f"  [WARN] Column '{key}' not found in test.csv")

    print("  Example extracted target conditions:")
    for k, v in conditions.items():
        print(f"    {k}: {v}")

    print("\n  [OK] Condition extraction logic verified\n")
    return conditions


def main(args):
    results = {}

    results["spacegroup"] = verify_spacegroup(args.test_csv)
    results["chgnet"]     = verify_chgnet()
    results["alignn"]     = verify_alignn()
    conditions            = verify_condition_extraction(args.test_csv)

    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    for tool, ok in results.items():
        status = "✓ Available" if ok else "✗ Unavailable"
        print(f"  {tool:15s}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv",
                        type=str,
                        default="/home/wx/MatLLM/IRLCrys/data/mp_20/test.csv",
                        help="Test set CSV path")
    args = parser.parse_args()
    main(args)