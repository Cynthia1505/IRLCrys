import os
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure

MP_API_KEY = "MP-API-Key"
OUTPUT_DIR = "data/mp_20"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        num_sites=(1, 20),
        fields=["material_id", "formula_pretty", "structure", 
                "formation_energy_per_atom", "band_gap", "spacegroup"]
    )

records = []
for doc in docs:
    records.append({
        "material_id": doc.material_id,
        "formula": doc.formula_pretty,
        "structure": doc.structure.to(fmt="cif"),
        "formation_energy": doc.formation_energy_per_atom,
        "band_gap": doc.band_gap,
        "space_group": doc.spacegroup.symbol if doc.spacegroup else None
    })

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "mp_20_full.csv"), index=False)

train, val, test = np.split(df.sample(frac=1, random_state=42), 
                            [int(0.6*len(df)), int(0.8*len(df))])
train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print(f"   train: {len(train)}, val: {len(val)}, test: {len(test)}")