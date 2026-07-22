import os
import csv
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data, Batch

from transformers import (
    LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, modeling_utils
)
from peft import PeftModel, prepare_model_for_kbit_training

from pymatgen.core import Structure, Element, Composition
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from enc_dec import enc, dec, VALID_ELEMENTS
from models_ddpm.diffusion import CSPDiffusion
from models_ddpm.data_utils import (
    lattice_params_to_matrix_torch,
    lattices_to_params_shape,
)
from data_utils import process_one

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or \
        modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

MAX_LENGTH = 2048

try:
    from chgnet.model import CHGNet
    CHGNET = CHGNet.load()
    print("[INFO] CHGNet loaded.")
except ImportError:
    CHGNET = None
    print("[WARN] CHGNet not available.")


def load_llm(args):
    print(f"\n[IRLCrys] Loading base model: {args.base_model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlamaForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )

    print(f"[IRLCrys] Loading tokenizer: {args.stage1_lora_path}")
    tokenizer = LlamaTokenizer.from_pretrained(
        args.stage1_lora_path,
        model_max_length=MAX_LENGTH,
        padding_side="left",
        use_fast=False,
        local_files_only=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    print(f"[IRLCrys] Loading Stage 1 LoRA: {args.stage1_lora_path}")
    model = PeftModel.from_pretrained(model, args.stage1_lora_path)

    print(f"[IRLCrys] Loading Stage 2 LoRA: {args.stage2_lora_path}")
    model.load_adapter(args.stage2_lora_path, adapter_name="stage2")
    model.set_adapter("stage2")

    model.eval()
    return model, tokenizer


def load_diffusion(args):
    print(f"\n[IRLCrys] Loading refinement module: {args.diffusion_ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = CSPDiffusion(args.timesteps, "sample").to(device)
    checkpoint = torch.load(args.diffusion_ckpt, map_location=device)
    diffusion.load_state_dict(checkpoint["model"])
    diffusion.eval()
    return diffusion, device


def llm_generate(model, tokenizer, prompt: str, max_new_tokens: int = 400,
                  temperature: float = 0.7, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_LENGTH).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def build_condition_prompt(conditions: dict, constraint_level: str) -> str:
    target_elements = conditions["target_elements"]
    elements_str = ", ".join(target_elements)

    prompt = "Below is a description of a bulk material. "
    prompt += f"The material is composed only of the following elements: {elements_str}. "

    if constraint_level in ("bc2", "sc"):
        prompt += f"The spacegroup number is {int(conditions['spacegroup.number'])}. "

    if constraint_level == "sc":
        prompt += (f"The formation energy per atom is "
                   f"{round(float(conditions['formation_energy_per_atom']), 4)}. ")

    prompt += (
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )
    return prompt


def build_enc_compatible_conditions(conditions: dict, constraint_level: str) -> dict:
    elements_str = "-".join(conditions["target_elements"])
    enc_conditions = {"pretty_formula": elements_str}

    if constraint_level in ("bc2", "sc"):
        enc_conditions["spacegroup.number"] = conditions["spacegroup.number"]
    if constraint_level == "sc":
        enc_conditions["formation_energy_per_atom"] = conditions["formation_energy_per_atom"]

    return enc_conditions


def check_composition_mismatch(structure: Structure, target_elements: list) -> bool:
    try:
        gen_elements = set(s.specie.symbol for s in structure)
        target_set = set(target_elements)
        return gen_elements != target_set
    except Exception:
        return True


def check_spacegroup_mismatch(structure: Structure, target_sg: int,
                               symprec: float = 0.2) -> bool:
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        return sga.get_space_group_number() != int(target_sg)
    except Exception:
        return True


def check_formation_energy_deviation(structure: Structure, target_ef: float) -> float:
    if CHGNET is None:
        return float("inf")
    try:
        pred = CHGNET.predict_structure(structure)
        return abs(float(pred["e"]) - target_ef)
    except Exception:
        return float("inf")


def structure_to_data(structure: Structure, device) -> Batch:
    cif_str = structure.to(fmt="cif")
    frac_coords, atom_types, lengths, angles, num_atoms, edge_indices, to_jimages, data_dict = \
        process_one(cif_str, niggli=True, primitive=False, graph_method="crystalnn")

    data = Data(
        num_atoms=torch.LongTensor([data_dict["n_atom"]]),
        num_nodes=data_dict["n_atom"],
        num_bonds=data_dict["edge_indices"].shape[0],
        lengths=data_dict["length"],
        angles=data_dict["angle"],
        frac_coords=torch.Tensor(data_dict["x_coord"]),
        atom_types=torch.LongTensor(data_dict["a_type"]),
        edge_index=torch.LongTensor(data_dict["edge_indices"].T).contiguous(),
        to_jimages=torch.LongTensor(data_dict["to_jimages"]),
    )
    batch = Batch.from_data_list([data]).to(device)
    return batch


def data_to_structure(atom_types: torch.Tensor, frac_coords: torch.Tensor,
                       lattices: torch.Tensor) -> Structure | None:
    try:
        lengths, angles = lattices_to_params_shape(lattices)
        lengths = lengths.flatten().detach().cpu().numpy()
        angles  = angles.flatten().detach().cpu().numpy()
        species = [Element.from_Z(int(z)).symbol for z in atom_types]
        lattice = Lattice.from_parameters(*lengths.tolist(), *angles.tolist())
        return Structure(
            lattice=lattice, species=species,
            coords=frac_coords.detach().cpu().numpy(),
            coords_are_cartesian=False,
        )
    except Exception:
        return None


def irlcrys_generate_one(
    model, tokenizer, diffusion, device,
    conditions: dict,
    constraint_level: str,
    theta_ef: float,
    max_steps: int,
    reset_delta_t: int,
    check_interval: int = 50,
    symprec: float = 0.2,
) -> dict:
    target_elements = conditions["target_elements"]
    target_sg       = conditions.get("spacegroup.number")
    target_ef       = conditions.get("formation_energy_per_atom")

    check_sg = constraint_level in ("bc2", "sc")
    check_ef = constraint_level == "sc"

    condition_prompt = build_condition_prompt(conditions, constraint_level)
    enc_conditions = build_enc_compatible_conditions(conditions, constraint_level)

    gen_text = llm_generate(model, tokenizer, condition_prompt)
    full_text_for_dec = condition_prompt + gen_text
    decode_result = dec(full_text_for_dec)

    if not decode_result.success:
        return {"success": False, "error": f"Initial generation decode failed: {decode_result.error}",
                "n_llm_interventions": 0}

    initial_structure = decode_result.structure

    try:
        batch = structure_to_data(initial_structure, device)
    except Exception as e:
        return {"success": False, "error": f"Initial structure graph construction failed: {e}",
                "n_llm_interventions": 0}

    atom_types  = batch.atom_types
    frac_coords = batch.frac_coords
    lattices    = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    num_atoms   = batch.num_atoms

    n_interventions = 0
    pmr_trajectory  = []
    prev_state      = None
    t = max_steps

    while t > 0:
        with torch.no_grad():
            frac_coords, lattices = diffusion.sample_step(
                batch, atom_types, frac_coords, lattices, num_atoms, t
            )

        should_check = (t % check_interval == 0) or (t == 1)
        if not should_check:
            t -= 1
            continue

        cur_structure = data_to_structure(atom_types, frac_coords, lattices)
        if cur_structure is None:
            t -= 1
            continue

        comp_mismatch = check_composition_mismatch(cur_structure, target_elements)
        sg_mismatch   = check_spacegroup_mismatch(cur_structure, target_sg, symprec) \
                        if check_sg else False
        ef_deviation  = check_formation_energy_deviation(cur_structure, target_ef) \
                        if check_ef else 0.0
        struct_mismatch = sg_mismatch or (check_ef and ef_deviation > theta_ef)

        cur_state = (comp_mismatch, sg_mismatch, round(ef_deviation, 3))

        if cur_state != prev_state:
            pmr_trajectory.append({
                "step": t,
                "ef_deviation": ef_deviation,
                "sg_mismatch": sg_mismatch,
                "comp_mismatch": comp_mismatch,
            })
            prev_state = cur_state

        if comp_mismatch or struct_mismatch:
            if comp_mismatch:
                cur_text = enc(cur_structure, enc_conditions)
            else:
                current_n_atoms = int(atom_types.shape[0])
                species = [Element.from_Z(int(z)).symbol for z in atom_types]
                species_str = ", ".join(species)
                cur_text = enc(cur_structure, enc_conditions)
                constraint_note = (
                    f" The structure must contain exactly {current_n_atoms} atoms "
                    f"with the following element sequence: {species_str}."
                    f" Only adjust the lattice parameters and fractional coordinates,"
                    f" do not change the number or type of atoms."
                )
                cur_text = cur_text.replace(
                    "Generate a description of the lengths and angles",
                    constraint_note + " Generate a description of the lengths and angles"
                )

            corrected_text = llm_generate(model, tokenizer, cur_text)
            full_corrected = cur_text + "\n" + corrected_text
            corrected_result = dec(full_corrected)

            if corrected_result.success:
                n_interventions += 1

                if comp_mismatch:
                    try:
                        batch = structure_to_data(corrected_result.structure, device)
                        atom_types  = batch.atom_types
                        frac_coords = batch.frac_coords
                        lattices    = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
                        num_atoms   = batch.num_atoms
                        t = min(max_steps, t + reset_delta_t)
                    except Exception:
                        pass
                else:
                    try:
                        corrected_n_atoms = len(corrected_result.structure)
                        current_n_atoms = int(atom_types.shape[0])
                        if corrected_n_atoms != current_n_atoms:
                            print(f"[WARN] Mild correction atom count mismatch: "
                                  f"corrected={corrected_n_atoms} vs current={current_n_atoms}, "
                                  f"skipping this correction", flush=True)
                        else:
                            new_batch = structure_to_data(corrected_result.structure, device)
                            rebuilt_structure = data_to_structure(
                                atom_types,
                                new_batch.frac_coords,
                                lattice_params_to_matrix_torch(new_batch.lengths, new_batch.angles)
                            )
                            if rebuilt_structure is not None:
                                batch = structure_to_data(rebuilt_structure, device)
                                atom_types  = batch.atom_types
                                frac_coords = batch.frac_coords
                                lattices    = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
                                num_atoms   = batch.num_atoms
                    except Exception as e:
                        print(f"[WARN] Mild correction failed: {e}, keeping current state", flush=True)

        elif not comp_mismatch and not struct_mismatch:
            break

        t -= 1

    final_structure = data_to_structure(atom_types, frac_coords, lattices)
    final_comp = check_composition_mismatch(final_structure, target_elements) \
                 if final_structure else None
    final_sg   = check_spacegroup_mismatch(final_structure, target_sg, symprec) \
                 if (final_structure and check_sg) else False
    final_ef   = check_formation_energy_deviation(final_structure, target_ef) \
                 if (final_structure and check_ef) else 0.0

    return {
        "success": True,
        "final_structure": final_structure,
        "n_llm_interventions": n_interventions,
        "pmr_trajectory": pmr_trajectory,
        "final_comp_mismatch": final_comp,
        "final_sg_mismatch": final_sg,
        "final_ef_deviation": final_ef,
    }


def main(args):
    print("=" * 60)
    print(f"IRLCrys Inference  (constraint level: {args.constraint_level.upper()})")
    print("=" * 60)

    conditions = {"target_elements": args.target_elements}
    if args.constraint_level in ("bc2", "sc"):
        if args.target_spacegroup is None:
            raise ValueError("constraint_level=bc2/sc requires --target_spacegroup")
        conditions["spacegroup.number"] = args.target_spacegroup
    if args.constraint_level == "sc":
        if args.target_ef is None:
            raise ValueError("constraint_level=sc requires --target_ef")
        conditions["formation_energy_per_atom"] = args.target_ef

    print(f"\n[IRLCrys] Fixed target conditions: {conditions}")
    print(f"[IRLCrys] Generating {args.n_samples} samples\n")

    model, tokenizer = load_llm(args)
    diffusion, device = load_diffusion(args)

    results = []
    n_success, n_all_satisfied = 0, 0

    for i in tqdm(range(args.n_samples), desc=f"IRLCrys inference [{args.constraint_level}]"):
        result = irlcrys_generate_one(
            model, tokenizer, diffusion, device,
            conditions=conditions,
            constraint_level=args.constraint_level,
            theta_ef=args.theta_ef,
            max_steps=args.max_steps,
            reset_delta_t=args.reset_delta_t,
            check_interval=args.check_interval,
        )

        if result["success"]:
            n_success += 1
            check_sg = args.constraint_level in ("bc2", "sc")
            check_ef = args.constraint_level == "sc"

            all_satisfied = not result["final_comp_mismatch"]
            if check_sg:
                all_satisfied = all_satisfied and not result["final_sg_mismatch"]
            if check_ef:
                all_satisfied = all_satisfied and (result["final_ef_deviation"] <= args.theta_ef)

            if all_satisfied:
                n_all_satisfied += 1

            result_out = {
                "success": True,
                "all_satisfied": all_satisfied,
                "n_llm_interventions": result["n_llm_interventions"],
                "final_comp_mismatch": result["final_comp_mismatch"],
                "final_sg_mismatch": result["final_sg_mismatch"],
                "final_ef_deviation": result["final_ef_deviation"],
                "pmr_trajectory": result.get("pmr_trajectory", []),
            }
            if result["final_structure"] is not None:
                result_out["final_structure_cif"] = result["final_structure"].to(fmt="cif")
            results.append(result_out)
        else:
            results.append(result)

    print(f"\n{'='*60}")
    print(f"Inference complete: {args.n_samples} generations  (constraint level: {args.constraint_level.upper()})")
    print(f"  Successful generations (including decode success): {n_success}")
    pmr = n_all_satisfied / args.n_samples * 100
    print(f"  PMR (all constraints satisfied):    {n_all_satisfied}/{args.n_samples} ({pmr:.2f}%)")
    avg_interventions = np.mean([r.get("n_llm_interventions", 0) for r in results])
    print(f"  Average LLM interventions:        {avg_interventions:.2f}")
    print(f"{'='*60}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_path.with_name(out_path.stem + "_structures.csv")
    structure_rows = []
    for idx, r in enumerate(results):
        if r.get("success") and "final_structure_cif" in r:
            structure_rows.append({
                "sample_idx": idx,
                "all_satisfied": r.get("all_satisfied"),
                "n_llm_interventions": r.get("n_llm_interventions"),
                "final_comp_mismatch": r.get("final_comp_mismatch"),
                "final_sg_mismatch": r.get("final_sg_mismatch"),
                "final_ef_deviation": r.get("final_ef_deviation"),
                "cif": r["final_structure_cif"],
            })
            r.pop("final_structure_cif", None)

    if structure_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(structure_rows[0].keys()))
            writer.writeheader()
            writer.writerows(structure_rows)
        print(f"[IRLCrys] Generated {len(structure_rows)} structures saved to: {csv_path}")

    with open(out_path, "w") as f:
        json.dump({
            "constraint_level": args.constraint_level,
            "target_conditions": conditions,
            "n_samples": args.n_samples,
            "n_success": n_success,
            "n_all_satisfied": n_all_satisfied,
            "pmr": pmr,
            "avg_interventions": float(avg_interventions),
            "structures_csv": str(csv_path),
            "results": results,
        }, f, indent=2, default=str)
    print(f"[IRLCrys] Statistical results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str,
                        default="/home/wx/MatLLM/Llama-2-7b-hf")
    parser.add_argument("--stage1_lora_path", type=str,
                        default="/home/wx/MatLLM/IRLCrys/exp/7b-mp-attr/checkpoint-27136")
    parser.add_argument("--stage2_lora_path", type=str,
                        default="/home/wx/MatLLM/IRLCrys/exp/stage2-7b-4bit-mp/checkpoint-4750")
    parser.add_argument("--diffusion_ckpt", type=str,
                        default="/home/wx/MatLLM/IRLCrys/out/mp_20/05062026/094741/model_final.pt")

    parser.add_argument("--constraint_level", type=str, default="bc1",
                        choices=["bc1", "bc2", "sc"],
                        help="bc1=element types only, bc2=+spacegroup, sc=+formation energy")
    parser.add_argument("--target_elements", type=str, nargs="+", required=True,
                        help="target element set, e.g. --target_elements Ga Ni")
    parser.add_argument("--target_spacegroup", type=int, default=None,
                        help="target spacegroup number, required for bc2/sc")
    parser.add_argument("--target_ef", type=float, default=None,
                        help="target formation energy (eV/atom), required for sc")

    parser.add_argument("--n_samples", type=int, default=20,
                        help="number of repeated generations for fixed target conditions")
    parser.add_argument("--theta_ef", type=float, default=0.2,
                        help="formation energy deviation threshold (eV/atom)")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--reset_delta_t", type=int, default=100)
    parser.add_argument("--check_interval", type=int, default=50,
                        help="interval for triggering deviation detection + LLM correction (default 50). "
                             "Refinement steps run every step, only detection and LLM calls are controlled by this. "
                             "1000 steps + check_interval=50 -> at most 20 LLM calls triggered")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--out_path", type=str,
                        default="results/irlcrys_results.json")
    args = parser.parse_args()
    main(args)