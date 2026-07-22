import re
import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.core.lattice import Lattice

VALID_ELEMENTS = set([el.symbol for el in Element])

PROMPT_LOOKUP = {
    "pretty_formula":            "The chemical formula is",
    "spacegroup.number":         "The spacegroup number is",
    "formation_energy_per_atom": "The formation energy per atom is",
    "band_gap":                  "The band gap is",
    "e_above_hull":              "The energy above the convex hull is",
    "elements":                  "The elements are",
}

PROMPT_SUFFIX = (
    "Generate a description of the lengths and angles of the lattice vectors "
    "and then the element type and coordinates for each atom within the lattice:\n"
)


def enc(
    structure: Structure,
    conditions: dict = None,
) -> str:
    """
    Encode a pymatgen Structure object into CIF-style text string.

    Args:
        structure: pymatgen Structure object
        conditions: conditional property dictionary, e.g.:
            {
                "pretty_formula": "Fe2O3",
                "spacegroup.number": 167,
                "formation_energy_per_atom": -1.2,
                "band_gap": 2.2,
            }
            Pass None or empty dict for unconditional prompt.

    Returns:
        Complete text string (conditional prompt + crystal structure sequence)
    """
    prompt = "Below is a description of a bulk material. "

    if conditions:
        for key, val in conditions.items():
            if key not in PROMPT_LOOKUP:
                continue
            prefix = PROMPT_LOOKUP[key]
            if key == "pretty_formula":
                prompt += f"{prefix} {val}. "
            elif key == "spacegroup.number":
                prompt += f"{prefix} {int(val)}. "
            elif key in ("formation_energy_per_atom", "band_gap", "e_above_hull"):
                prompt += f"{prefix} {round(float(val), 4)}. "
            else:
                prompt += f"{prefix} {val}. "

    prompt += PROMPT_SUFFIX

    lengths = structure.lattice.parameters[:3]
    angles  = structure.lattice.parameters[3:]

    lattice_str = (
        " ".join([f"{x:.1f}" for x in lengths]) + "\n" +
        " ".join([str(int(round(x))) for x in angles])
    )

    atom_lines = []
    for site in structure:
        symbol = site.specie.symbol
        fc = site.frac_coords % 1.0
        coord_str = " ".join([f"{x:.2f}" for x in fc])
        atom_lines.append(f"{symbol}\n{coord_str}")

    crystal_str = lattice_str + "\n" + "\n".join(atom_lines)

    return prompt + crystal_str


def enc_from_triple(
    atom_types: list,
    frac_coords: np.ndarray,
    lengths: np.ndarray,
    angles: np.ndarray,
    conditions: dict = None,
) -> str:
    """
    Encode directly from structure triple (A, X, L) without constructing pymatgen Structure first.
    Suitable for tensor data read from .pt files.

    Args:
        atom_types:  list of atom types, either element symbols or atomic numbers
        frac_coords: fractional coordinates np.ndarray, shape=(N, 3)
        lengths:     lattice lengths np.ndarray, shape=(3,)
        angles:      lattice angles np.ndarray, shape=(3,)
        conditions:  conditional property dictionary (same as enc)
    """
    if len(atom_types) > 0 and isinstance(atom_types[0], (int, np.integer)):
        from pymatgen.core import Element as PyEl
        atom_types = [PyEl.from_Z(int(z)).symbol for z in atom_types]

    try:
        lattice = Lattice.from_parameters(*lengths.tolist(), *angles.tolist())
        structure = Structure(
            lattice=lattice,
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
    except Exception as e:
        raise ValueError(f"[Enc] Failed to construct Structure: {e}")

    return enc(structure, conditions)


class DecodeResult:
    """Return value of Dec, containing parse results and validation status."""
    def __init__(self):
        self.success     = False
        self.atom_types  = None
        self.frac_coords = None
        self.lengths     = None
        self.angles      = None
        self.structure   = None
        self.error       = None


def dec(text: str, min_dist: float = 0.5) -> DecodeResult:
    """
    Parse LLM-generated text string back to structure triple with physical validity validation.

    Args:
        text:     LLM output text string (may include conditional prompt prefix)
        min_dist: minimum allowed interatomic distance in Angstrom, default 0.5

    Returns:
        DecodeResult object
            .success     = True/False
            .atom_types  = List[str]
            .frac_coords = np.ndarray (N, 3)
            .lengths     = np.ndarray (3,)
            .angles      = np.ndarray (3,)
            .structure   = pymatgen Structure (valid when success=True)
            .error       = error message string (valid when success=False)
    """
    result = DecodeResult()

    crystal_text = _extract_crystal_section(text)
    if crystal_text is None:
        result.error = "Failed to locate crystal structure text section"
        return result

    lines = [l.strip() for l in crystal_text.strip().split("\n") if l.strip()]
    if len(lines) < 4:
        result.error = f"Insufficient text lines ({len(lines)} lines), cannot parse"
        return result

    try:
        lengths = np.array([float(x) for x in lines[0].split()])
        angles  = np.array([float(x) for x in lines[1].split()])
        assert len(lengths) == 3 and len(angles) == 3
    except Exception as e:
        result.error = f"Failed to parse lattice parameters: {e}"
        return result

    atom_types, frac_coords = [], []
    i = 2
    while i < len(lines):
        symbol = lines[i].strip()
        if i + 1 >= len(lines):
            result.error = f"Missing coordinate line for atom {symbol}"
            return result
        try:
            coords = [float(x) for x in lines[i + 1].split()]
            assert len(coords) == 3
        except Exception as e:
            result.error = f"Failed to parse coordinates for atom {symbol}: {e}"
            return result
        atom_types.append(symbol)
        frac_coords.append(coords)
        i += 2

    if len(atom_types) == 0:
        result.error = "No atoms parsed"
        return result

    frac_coords = np.array(frac_coords) % 1.0

    for sym in atom_types:
        if sym not in VALID_ELEMENTS:
            result.error = f"Invalid element symbol: {sym}"
            return result

    if not all(l > 0 for l in lengths):
        result.error = f"Lattice lengths contain non-positive values: {lengths}"
        return result
    if not all(0 < a < 180 for a in angles):
        result.error = f"Lattice angles out of range: {angles}"
        return result

    try:
        lattice   = Lattice.from_parameters(*lengths.tolist(), *angles.tolist())
        structure = Structure(
            lattice=lattice,
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
    except Exception as e:
        result.error = f"Failed to construct Structure: {e}"
        return result

    if not _check_min_distance(structure, min_dist):
        result.error = f"Interatomic distance < {min_dist} A detected"
        return result

    result.success     = True
    result.atom_types  = atom_types
    result.frac_coords = frac_coords
    result.lengths     = lengths
    result.angles      = angles
    result.structure   = structure
    return result


def _extract_crystal_section(text: str) -> str | None:
    """
    Extract crystal structure section from full LLM output.
    Strategy: find the first line that "looks like" lattice parameters (3 floats), start from there.
    """
    lines = text.strip().split("\n")
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                vals = [float(x) for x in parts]
                if all(0.5 < v < 50 for v in vals):
                    return "\n".join(lines[i:])
            except ValueError:
                continue
    return None


def _check_min_distance(structure: Structure, min_dist: float) -> bool:
    """Check if all interatomic distances are greater than min_dist."""
    try:
        dist_matrix = structure.distance_matrix
        np.fill_diagonal(dist_matrix, np.inf)
        return float(dist_matrix.min()) >= min_dist
    except Exception:
        return False


if __name__ == "__main__":
    from pymatgen.core import Structure, Lattice

    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    conditions = {
        "pretty_formula": "NaCl",
        "spacegroup.number": 225,
        "formation_energy_per_atom": -1.78,
        "band_gap": 8.5,
    }

    text = enc(structure, conditions)
    print("=== Enc Output ===")
    print(text)
    print()

    crystal_only = "\n".join(text.split("\n")[-5:])
    result = dec(text)
    print("=== Dec Result ===")
    print(f"success:     {result.success}")
    print(f"atom_types:  {result.atom_types}")
    print(f"lengths:     {result.lengths}")
    print(f"angles:      {result.angles}")
    print(f"frac_coords:\n{result.frac_coords}")
    if result.error:
        print(f"error:       {result.error}")