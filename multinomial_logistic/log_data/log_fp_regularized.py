import os
import json
import numpy as np
from state_evolution.full_recursion import state_evolution_full_recursion
from typing import Literal
from configs.R_initiation import get_R_00
import fcntl


def run_and_log_fp_regularized(
    k_0: int,
    k: int,
    alphas: np.ndarray,
    lambda_regs: np.ndarray,
    type_3: str,
    tol: float = 1e-5,
    max_iter: int = 80,
    integral_mesh_size: int = 8,
    integral_size: float = 4.5,
):

    base_filepath = os.path.join(
        os.path.dirname(__file__),
        "Oct_data", "fp_reg_solution", f"FP_reg_solutions_(k={k},k0={k_0})_{type_3}.json",
    )
    R_00_values = np.array([get_R_00(k, type_3)])

    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    if os.path.exists(base_filepath):
        with open(base_filepath, "r") as f:
            results = json.load(f).get("results", {})
    else:
        results = {}

    def persist():
        with open(base_filepath, "w") as f:
            json.dump(
                {"metadata": {"k": k, "k_0": k_0}, "results": results}, f, indent=2
            )

    def find_existing(R_list, alpha_val, lambda_val):
        for key, entry in results.items():
            meta = json.loads(key)
            if (
                meta["R_00"] == R_list
                and abs(meta["alpha"] - alpha_val) < 1e-12
                and abs(meta["lambda_reg"] - lambda_val) < 1e-10
            ):
                print(f"Found existing entry for alpha={alpha_val}, lambda={lambda_val}; found lambda_reg = {meta['lambda_reg']}")
                return key, entry
        return None, None

    for R_00 in R_00_values:
        R_list = R_00.tolist()
        for lambda_reg in lambda_regs:
            lambda_val = float(lambda_reg)
            if lambda_val >= 0.8:
                continue
            schur, R_01, S = np.array(R_00), np.zeros((k, k_0)), np.eye(k)
            diverged_flag = False

            for alpha in alphas:
                alpha_val = float(alpha)
                existing_key, existing_entry = find_existing(
                    R_list, alpha_val, lambda_val
                )
                if existing_entry is not None:
                    print(
                        f"Skipping alpha={alpha_val}, lambda={lambda_val} (≈ existing) for R_00={R_list}"
                    )
                    schur = np.array(existing_entry["schur"])
                    R_01 = np.array(existing_entry["R_01"])
                    S = np.array(existing_entry["S"])
                    diverged_flag = bool(existing_entry.get("diverged"))
                    continue

                key = json.dumps(
                    {"R_00": R_list, "alpha": alpha_val, "lambda_reg": lambda_val}
                )

                if diverged_flag:
                    results[key] = {
                        "schur": np.zeros((k, k)).tolist(),
                        "R_01": np.zeros((k, k_0)).tolist(),
                        "S": np.zeros((k, k)).tolist(),
                        "diverged": True,
                    }
                    persist()
                    continue

                print(f"\nProcessing alpha={alpha_val}, lambda={lambda_val}")
                schur, R_01, S, diverged = state_evolution_full_recursion(
                    R_00=R_00,
                    schur_0=schur,
                    R_01_0=R_01,
                    S_0=S,
                    lambda_reg=lambda_val,
                    alpha=alpha_val,
                    k=k,
                    k_0=k_0,
                    tol=tol,
                    max_iter=max_iter,
                    integral_mesh_size=integral_mesh_size,
                    integral_size=integral_size,
                )

                results[key] = {
                    "schur": schur.tolist(),
                    "R_01": R_01.tolist(),
                    "S": S.tolist(),
                    "diverged": bool(diverged),
                }
                diverged_flag = bool(diverged)
                persist()

    return base_filepath


def refine_logged_regulairzed_fp(
    k_0: int,
    k: int,
    type_3: Literal[
        False,
        "symmetric",
        "two_classes_close",
        "three_classes_close",
        "two_vs_two_vs_one",
    ] = False,
    tol: float = 1e-5,
    max_iter: int = 80,
    integral_mesh_size: int = 8,
    modified_alpha: float = 10,
    integral_size: float = 4.5,
):
    """Refine previously logged regularized fixed points by rerunning state evolution."""

    if type_3 == False:
        base_filepath = os.path.join(
            os.path.dirname(__file__),
            "data",
            "fp_solution",
            f"fp_reg_data_k{k}_k0{k_0}.json",
        )
        print("Refining regularized symmetric FP data")
    else:
        base_filepath = os.path.join(
            os.path.dirname(__file__),
            "Oct_data",
            "fp_reg_solution",
            f"FP_reg_solutions_(k={k},k0={k_0})_{type_3}.json",
        )
        print(f"Refining regularized FP data for type_3 = {type_3}")

    if not os.path.exists(base_filepath):
        raise FileNotFoundError(
            f"No logged regularized FP data found at {base_filepath}"
        )

    with open(base_filepath, "r") as f:
        data = json.load(f)

    results = data.get("results", {})

    if not results:
        print("No results to refine in", base_filepath)
        return base_filepath

    def _persist_results():
        temp_filepath = base_filepath + ".tmp"
        with open(temp_filepath, "w") as tmp_file:
            fcntl.flock(tmp_file.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, tmp_file, indent=2)
            finally:
                fcntl.flock(tmp_file.fileno(), fcntl.LOCK_UN)
        os.replace(temp_filepath, base_filepath)

    # Sort entries by R_00, lambda_reg, then alpha for deterministic processing
    sorted_items = []
    for key, entry in results.items():
        key_data = json.loads(key)
        sorted_items.append((key_data, entry, key))

    sorted_items.sort(
        key=lambda item: (
            np.array(item[0]["R_00"]).tolist(),  # ensure grouping by R_00 structure
            float(item[0]["lambda_reg"]),
            float(item[0]["alpha"]),
        )
    )

    for key_data, entry, key in sorted_items:
        R_00 = np.array(key_data["R_00"])
        alpha_value = float(key_data["alpha"])
        lambda_reg = float(key_data["lambda_reg"])

        print(f"\nRefining alpha = {alpha_value}, lambda = {lambda_reg}")

        schur = np.array(entry["schur"])
        R_01 = np.array(entry["R_01"])
        S = np.array(entry["S"])
        if alpha_value == modified_alpha:
            schur_refined, R_01_refined, S_refined, diverged = (
                state_evolution_full_recursion(
                    R_00=R_00,
                    schur_0=schur,
                    R_01_0=R_01,
                    S_0=S,
                    lambda_reg=lambda_reg,
                    alpha=alpha_value,
                    k=k,
                    k_0=k_0,
                    tol=tol,
                    max_iter=max_iter,
                    integral_mesh_size=integral_mesh_size,
                    integral_size=integral_size,
                ))
        else:
            schur_refined, R_01_refined, S_refined, diverged = schur, R_01, S, False # not refining for other alphas

        entry["schur"] = schur_refined.tolist()
        entry["R_01"] = R_01_refined.tolist()
        entry["S"] = S_refined.tolist()
        entry["diverged"] = bool(diverged)
        entry.pop("error", None)

        results[key] = entry

        _persist_results()

    return base_filepath


def get_regularized_fp_data(k, k_0, R_00, type_3='symmetric'):
    base_filepath = os.path.join(
        os.path.dirname(__file__),
        "Oct_data",
        "fp_reg_solution",
        f"FP_reg_solutions_(k={k},k0={k_0})_{type_3}.json",
    )

    with open(base_filepath, "r") as f:
        data = json.load(f)

    alphas, lambda_regs, S_matrices, schur_matrices, R_01_matrices, diverged_flags = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Process each result
    for key, result in data["results"].items():

        key_data = json.loads(key)
        if not np.array_equal(np.array(key_data["R_00"]), R_00):
            raise ValueError(f"R_00 mismatch for key: {key}")

        alpha = key_data["alpha"]
        lambda_reg = key_data["lambda_reg"]

        # Store the values
        alphas.append(alpha)
        lambda_regs.append(lambda_reg)
        S_matrices.append(np.array(result["S"]))
        schur_matrices.append(np.array(result["schur"]))
        R_01_matrices.append(np.array(result["R_01"]))
        diverged_flags.append(result["diverged"])

    # Convert lists to numpy arrays
    alphas = np.array(alphas)
    lambda_regs = np.array(lambda_regs)
    S_matrices = np.array(S_matrices)
    schur_matrices = np.array(schur_matrices)
    R_01_matrices = np.array(R_01_matrices)
    diverged_flags = np.array(diverged_flags)

    # Sort everything by alpha and lambda_reg
    sort_idx = np.lexsort((alphas, lambda_regs))
    alphas = alphas[sort_idx]
    lambda_regs = lambda_regs[sort_idx]
    S_matrices = S_matrices[sort_idx]
    schur_matrices = schur_matrices[sort_idx]
    R_01_matrices = R_01_matrices[sort_idx]
    diverged_flags = diverged_flags[sort_idx]
    return (
        alphas,
        lambda_regs,
        S_matrices,
        schur_matrices,
        R_01_matrices,
        diverged_flags,
    )
