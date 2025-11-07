# logging the theoretical test errors, train errors, F_norm
# using the results from the fp_solution folder
# for different values of alpha, k, k_0, R_00

import json
import os
from typing import Literal

import numpy as np
from scipy.linalg import sqrtm

from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error
from multinomial_logistic.evaluation.misclassification_test_error import (
    misclassification_test_error,
)
from multinomial_logistic.log_data.log_fp import read_fp_results
from multinomial_logistic.log_data.log_fp_regularized import get_regularized_fp_data


def run_and_log_fp_tests_regularized(
    k_0,
    k,
    R_00,
    type_3
):
    """
    Run and log regularized FP tests, saving test errors for different alpha and lambda values.
    """
    print("running and logging fp tests regularized...")
    alphas, lambda_regs, S_matrices, schur_matrices, R_01_matrices, diverged_flags = (
        get_regularized_fp_data(k, k_0, R_00, type_3)
    )

    # Create data directory if it doesn't exist
    if type_3 == False:
        input(f"Creating new file: {type_3}?...")
        data_dir = os.path.join(
            os.path.dirname(__file__), "data", "fp_tests_regularized"
        )
    elif type_3 != False:
        data_dir = os.path.join(
            os.path.dirname(__file__),
            "Oct_data",
            "fp_tests_regularized"
        )

    os.makedirs(data_dir, exist_ok=True)

    # Create filename based on parameters
    if type_3 == False:
        filename = f"fp_reg_tests_k{k}_k0{k_0}.json"
    elif type_3 != False:
        filename = f"FP_reg_evals_k{k}_k0{k_0}_{type_3}.json"
    filepath = os.path.join(data_dir, filename)

    # Load existing results if file exists
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_data = json.load(f)
            results = existing_data["results"]
        print(f"Loading existing results from {filepath}")
    else:
        results = {}
        print(f"Creating new results file: {filename}")

    unique_alphas = np.unique(alphas)

    inv_R00 = np.linalg.inv(R_00)
    sqrtm_R00_inv = np.linalg.inv(sqrtm(R_00))

    for alpha in unique_alphas:

        # Get data for this alpha
        alpha_mask = alphas == alpha

        # Get lambda values and corresponding matrices for this alpha
        lambda_values = lambda_regs[alpha_mask]
        S_vals = S_matrices[alpha_mask]
        schur_vals = schur_matrices[alpha_mask]
        R_01_vals = R_01_matrices[alpha_mask]

        # Sort by lambda values (in descending order)
        sort_idx = np.argsort(lambda_values)[::-1]
        lambda_values = lambda_values[sort_idx]
        S_vals = S_vals[sort_idx]
        schur_vals = schur_vals[sort_idx]
        R_01_vals = R_01_vals[sort_idx]

        # Filter out lambda values >= 0.6
        valid_mask = lambda_values < 0.8
        lambda_values = lambda_values[valid_mask]
        S_vals = S_vals[valid_mask]
        schur_vals = schur_vals[valid_mask]
        R_01_vals = R_01_vals[valid_mask]

        alpha_key = str(alpha)
        existing_entry = results.get(alpha_key)
        if existing_entry:
            stored_lambda = np.array(existing_entry.get("lambda_values", []), dtype=float)
            stored_metrics = {
                "test_errors": list(existing_entry.get("test_errors", [])),
                "train_errors": list(existing_entry.get("train_errors", [])),
                "f_norms": list(existing_entry.get("f_norms", [])),
                "misclassification_test_errors": list(
                    existing_entry.get("misclassification_test_errors", [])
                ),
            }
        else:
            stored_lambda = np.array([])
            stored_metrics = {
                "test_errors": [],
                "train_errors": [],
                "f_norms": [],
                "misclassification_test_errors": [],
            }

        print(f"Computing all metrics for new alpha={alpha}")
        test_errors = []
        train_errors = []
        f_norms = []
        misclassification_test_errors = []

        for j, lambda_val in enumerate(lambda_values):
            lambda_float = float(lambda_val)
            reuse_idx = (
                np.where(np.isclose(stored_lambda, lambda_float, atol=1e-6))[0]
                if stored_lambda.size
                else []
            )
            if len(reuse_idx):
                idx = reuse_idx[0]
                print(
                    f"Skipping alpha={alpha}, lambda={lambda_float} (already exists); reusing stored metrics"
                )
                test_errors.append(float(stored_metrics["test_errors"][idx]))
                train_errors.append(float(stored_metrics["train_errors"][idx]))
                f_norms.append(float(stored_metrics["f_norms"][idx]))
                misclassification_test_errors.append(
                    float(stored_metrics["misclassification_test_errors"][idx])
                )
                continue

            R_11 = schur_vals[j] + R_01_vals[j] @ inv_R00 @ R_01_vals[j].T
            test_err = test_error(
                R_00, schur_vals[j], R_01=R_01_vals[j], alpha=alpha, k=k, k_0=k_0
            )
            print(
                "test_err for alpha", alpha, "lambda", lambda_float, "is", test_err
            )
            train_err = train_error(
                R_00=R_00,
                schur=schur_vals[j],
                R_01=R_01_vals[j],
                S=S_vals[j],
                alpha=alpha,
                k=k,
                k_0=k_0,
            )
            f_norm = (
                np.trace(R_00)
                + np.trace(R_11)
                - np.trace(R_01_vals[j])
                - np.trace(R_01_vals[j].T)
            )
            print(
                "test_err for alpha",
                alpha,
                "lambda",
                lambda_float,
                "is",
                test_err,
                "train_err",
                train_err,
                "f_norm",
                f_norm,
            )

            A = R_01_vals[j] @ sqrtm_R00_inv
            misclassification_test_err = misclassification_test_error(
                S=S_vals[j],
                R_00=R_00,
                schur_t=schur_vals[j],
                A_t=A,
                alpha=alpha,
                k=k,
                k_0=k_0,
            )
            print(
                "misclassification_test_err for alpha",
                alpha,
                "lambda",
                lambda_float,
                "is",
                misclassification_test_err,
            )

            misclassification_test_errors.append(misclassification_test_err)
            test_errors.append(test_err)
            train_errors.append(train_err)
            f_norms.append(f_norm)

        results[alpha_key] = {
            "lambda_values": lambda_values.tolist(),
            "test_errors": [float(x) for x in test_errors],
            "train_errors": [float(x) for x in train_errors],
            "f_norms": [float(x) for x in f_norms],
            "misclassification_test_errors": [
                float(x) for x in misclassification_test_errors
            ],
        }

        # Save after each alpha to prevent data loss
        data = {
            "metadata": {"k": k, "k_0": k_0, "R_00": R_00.tolist()},
            "results": results,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Results for alpha={alpha} saved")
        if alpha_key in results:
            print(
                "[storing] for alpha",
                alpha,
                "misclassification_test_errors:",
                results[alpha_key]["misclassification_test_errors"],
            )

    print(f"All results saved to {filepath}")
    return filepath


def refine_logged_fp_tests_regularized(
    k_0,
    k,
    R_00,
    type_3,
    metric_name: Literal['test_errors', 'train_errors', 'f_norms', 'misclassification_test_errors'],
    modified_alpha,
):
    """Recompute a stored regularized FP evaluation metric for a specific alpha."""

    valid_metrics = {"test_errors": "test_errors",
                    "train_errors": "train_errors",
                    "f_norms": "f_norms",
                    "misclassification_test_errors": "misclassification_test_errors"}
    if metric_name not in valid_metrics:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Choose from {sorted(valid_metrics)}."
        )

    alpha_val = float(modified_alpha)


    data_dir = os.path.join(
        os.path.dirname(__file__), "Oct_data", "fp_tests_regularized"
    )
    filename = f"FP_reg_evals_k{k}_k0{k_0}_{type_3}.json"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No regularized FP test file found at {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    results = data.get("results", {})


    def _match_alpha_key(target, keys):
        for key in keys:
            if abs(float(key) - target) < 1e-8:
                return key
        return None

    alpha_key = _match_alpha_key(alpha_val, results.keys())


    stored_lambda = np.array(results[alpha_key]["lambda_values"], dtype=float)

    (
        alphas,
        lambda_regs,
        S_matrices,
        schur_matrices,
        R_01_matrices,
        diverged_flags,
    ) = get_regularized_fp_data(k, k_0, R_00, type_3)

    mask_alpha = np.isclose(alphas, alpha_val, atol=1e-8)
    if not mask_alpha.any():
        raise ValueError(f"Alpha {alpha_val} not present in FP data file.")

    lambda_alpha = lambda_regs[mask_alpha]
    S_alpha = S_matrices[mask_alpha]
    schur_alpha = schur_matrices[mask_alpha]
    R_01_alpha = R_01_matrices[mask_alpha]

    # Order data in descending lambda to mirror stored ordering
    order = np.argsort(lambda_alpha)[::-1]
    lambda_alpha = lambda_alpha[order]
    S_alpha = S_alpha[order]
    schur_alpha = schur_alpha[order]
    R_01_alpha = R_01_alpha[order]

    # Filter out lambda >= 0.8
    valid = lambda_alpha < 0.8
    lambda_alpha = lambda_alpha[valid]
    S_alpha = S_alpha[valid]
    schur_alpha = schur_alpha[valid]
    R_01_alpha = R_01_alpha[valid]

    inv_R00 = np.linalg.inv(R_00)
    sqrtm_R00_inv = np.linalg.inv(sqrtm(R_00))

    recompute_funcs = {
        "test_errors": lambda idx: test_error(
            R_00,
            schur_alpha[idx],
            R_01=R_01_alpha[idx],
            alpha=alpha_val,
            k=k,
            k_0=k_0,
        ),
        "train_errors": lambda idx: train_error(
            R_00=R_00,
            schur=schur_alpha[idx],
            R_01=R_01_alpha[idx],
            S=S_alpha[idx],
            alpha=alpha_val,
            k=k,
            k_0=k_0,
        ),
        "f_norms": lambda idx: (
            np.trace(R_00)
            + np.trace(
                schur_alpha[idx] + R_01_alpha[idx] @ inv_R00 @ R_01_alpha[idx].T
            )
            - np.trace(R_01_alpha[idx])
            - np.trace(R_01_alpha[idx].T)
        ),
        "misclassification_test_errors": lambda idx: misclassification_test_error(
            S=S_alpha[idx],
            R_00=R_00,
            schur_t=schur_alpha[idx],
            A_t=R_01_alpha[idx] @ sqrtm_R00_inv,
            alpha=alpha_val,
            k=k,
            k_0=k_0,
        ),
    }

    old_metric_values = results[alpha_key].get(metric_name, [])
    recomputed_values = [
        float(old_metric_values[i]) if i < len(old_metric_values) else None
        for i in range(len(stored_lambda))
    ]

    lambda_order = np.argsort(stored_lambda)

    for pos in lambda_order:
        lambda_val = stored_lambda[pos]
        matches = np.where(np.isclose(lambda_alpha, lambda_val, atol=1e-6))[0]
        if matches.size == 0:
            if recomputed_values[pos] is None:
                if pos < len(old_metric_values):
                    recomputed_values[pos] = float(old_metric_values[pos])
            continue

        idx = matches[0]
        recomputed_values[pos] = float(recompute_funcs[metric_name](idx))
        print(
            f"for alpha: {alpha_val}, lambda: {lambda_val}, {metric_name} is {recomputed_values[pos]}"
        )

    final_values = []
    for i, val in enumerate(recomputed_values):
        if val is None:
            if i < len(old_metric_values):
                final_values.append(float(old_metric_values[i]))
        else:
            final_values.append(float(val))



    results[alpha_key][metric_name] = final_values

    with open(filepath, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=4)

    print(
        f"Updated {metric_name} for alpha={alpha_key} in {filepath}: {recomputed_values}"
    )

    return filepath





def read_fp_tests_regularized(k, k_0, R_00, type_3:Literal[False, 'symmetric', 'two_classes_close', 'three_classes_close', 'two_vs_two_vs_one'] = False):
    """
    Read regularized FP test results from file.

    Args:
        k (int): Number of classes
        k_0 (int): Number of observed classes
        R_00 (numpy.ndarray): The R_00 matrix

    Returns:
        dict: Dictionary containing alphas as keys and tuples of (lambda_values, test_errors) as values,
              or None if file not found
    """
    # Get filepath
    if type_3 == False:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "fp_tests_regularized")
        filename = f"fp_reg_tests_k{k}_k0{k_0}.json"
    elif type_3 != False:
        data_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "fp_tests_regularized")
        filename = f"FP_reg_evals_k{k}_k0{k_0}_{type_3}.json"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"No regularized FP test results found at {filepath}")
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    # Verify metadata matches
    metadata = data["metadata"]
    if (
        metadata["k"] != k
        or metadata["k_0"] != k_0
        or not np.allclose(np.array(metadata["R_00"]), R_00)
    ):
        print("Metadata mismatch in stored results")
        return None

    # Convert results back to numpy arrays
    results = {}
    for alpha_str, alpha_data in data["results"].items():
        alpha = float(alpha_str)
        lambda_values = np.array(alpha_data["lambda_values"])
        test_errors = np.array(alpha_data["test_errors"])
        train_errors = np.array(alpha_data["train_errors"])
        f_norms = np.array(alpha_data["f_norms"])
        misclassification_test_errors = np.array(
            alpha_data["misclassification_test_errors"]
        )
        results[alpha] = {
            "lambda_values": lambda_values,
            "test_errors": test_errors,
            "train_errors": train_errors,
            "misclassification_test_errors": misclassification_test_errors,
            "f_norms": f_norms,
        }

    return results


from typing import Literal


###########################################################################
def run_and_log_fp_tests(
    k_0,
    k,
    non_symmetric=False,
    lambda_reg=0,
    two_classes_close=False,
    type_3: Literal[
        False, "symmetric", "two_classes_close", "three_classes_close"
    ] = False,
):
    print("****************running and logging fp tests...")
    # Create base filename
    if type_3 == False and two_classes_close:
        base_filename = (
            f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}_two_classes_close.json"
        )
    elif type_3 == False and non_symmetric:
        base_filename = (
            f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric.json"
        )
    elif type_3 != False:
        base_filename = f"FP_evals_(k={k},k0={k_0},lambda={lambda_reg})_{type_3}.json"

    print("base_filename", base_filename)
    if type_3 != False:
        base_filepath = os.path.join(
            os.path.dirname(__file__), "Oct_data", "fp_tests", base_filename
        )
    else:
        base_filepath = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_tests", base_filename
        )
    print("base_filepath", base_filepath)

    # Create data/fp_tests directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)

    # Check for existing files with matching parameters
    data_dir = os.path.dirname(base_filepath)
    existing_files = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        if filename == base_filename:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                existing_files.append((filename, data))

    # If matching file exists, use it
    if existing_files:
        filename, existing_data = existing_files[0]
        filepath = os.path.join(data_dir, filename)
        results = existing_data["results"]
        print(f"Appending to existing file: {filename}")
    else:
        # Create new file
        filepath = base_filepath
        results = {}
        print(f"Creating new file: {os.path.basename(filepath)}")

    # Read available alphas and R_00 values from FP solution files
    if type_3 == False and two_classes_close:
        fp_data_dir = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_solution_two_classes_close"
        )
        fp_filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_two_classes_close.json"
        fp_filepath = os.path.join(fp_data_dir, fp_filename)
    elif type_3 == False and non_symmetric:
        fp_data_dir = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_solution_nonsym"
        )
        fp_filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric.json"
        fp_filepath = os.path.join(fp_data_dir, fp_filename)
    elif type_3 != False:
        fp_data_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "fp_solution")
        fp_filename = f"FP_solutions_(k={k},k0={k_0},lambda={lambda_reg})_{type_3}.json"
        fp_filepath = os.path.join(fp_data_dir, fp_filename)
    else:
        fp_data_dir = os.path.join(os.path.dirname(__file__), "tempdata", "fp_solution")
        fp_filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
        fp_filepath = os.path.join(fp_data_dir, fp_filename)

    if not os.path.exists(fp_filepath):
        print(f"No FP solution file found: {fp_filename}")
        return None

    with open(fp_filepath, "r") as f:
        fp_data = json.load(f)

    # Extract unique R_00 values and alphas from FP solutions
    R_00_values = []
    alphas = set()

    for R_00_str in fp_data["results"].keys():
        R_00 = np.array(json.loads(R_00_str))
        R_00_values.append(R_00)
        alphas.update(float(alpha) for alpha in fp_data["results"][R_00_str].keys())

    alphas = sorted(list(alphas), reverse=True)  # Sort in decreasing order
    print("R_00_values:", R_00_values, "alphas:", alphas)
    print("\n\n")

    for R_00 in R_00_values:
        if R_00.tolist() == [[1, 0], [0, 1]]:
            continue
        for _alpha in alphas:
            if (
                _alpha == int(_alpha)
                and str(_alpha) not in fp_data["results"][R_00_str].keys()
            ):
                alpha_str = str(int(_alpha))
            else:
                alpha_str = str(_alpha)

            R_00_str = str(R_00.tolist())

            # Skip if we already have results for this alpha and R_00
            if R_00_str in results and alpha_str in results[R_00_str]:
                print(f"Skipping alpha={_alpha} for R_00={R_00} (already exists)")
                continue

            print(f"\nProcessing alpha = {_alpha} (alpha_str: {alpha_str})")
            result = fp_data["results"][R_00_str][alpha_str]
            print("->result", result)

            schur = np.array(result["schur"]).reshape(k, k)
            R_01 = np.array(result["R_01"]).reshape(k_0, k)
            S = np.array(result["S"]).reshape(k, k)
            # Calculate test error, train error, and F_norm
            R_11 = schur + R_01 @ np.linalg.inv(R_00) @ R_01.T
            test_err = test_error(R_00, schur, R_01=R_01, alpha=_alpha, k=k, k_0=k_0)
            train_err = train_error(
                R_00=R_00, schur=schur, R_01=R_01, S=S, alpha=_alpha, k=k, k_0=k_0
            )
            A = R_01.T @ sqrtm(np.linalg.inv(R_00))

            misclassification_test_err = misclassification_test_error(
                S=S, R_00=R_00, schur_t=schur, A_t=A, alpha=_alpha, k=k, k_0=k_0
            )
            f_norm = np.trace(R_00) + np.trace(R_11) - np.trace(R_01) - np.trace(R_01.T)

            # Initialize R_00 dict if it doesn't exist
            if R_00_str not in results:
                results[R_00_str] = {}

            # Store results
            results[R_00_str][alpha_str] = {
                "test_error": float(test_err),
                "train_error": float(train_err),
                "misclassification_test_error": float(misclassification_test_err),
                "F_norm": float(f_norm),
                "diverged": False,
                "actual_alpha": float(_alpha),
            }

            print(
                "storing",
                results[R_00_str][alpha_str],
                "test:",
                test_err,
                "train:",
                train_err,
                "F_norm:",
                f_norm,
                "misclass:",
                misclassification_test_err,
            )

            # Save after each iteration
            with open(filepath, "w") as f:
                json.dump(
                    {
                        "metadata": {"k": k, "k_0": k_0, "lambda_reg": lambda_reg},
                        "results": results,
                    },
                    f,
                    indent=2,
                )

    return filepath


def refine_logged_fp_tests(
    k_0,
    k,
    lambda_reg=0,
    non_symmetric=False,
    two_classes_close=False,
    type_3: Literal[
        False, "symmetric", "two_classes_close", "three_classes_close"
    ] = False,
):
    """Recompute stored FP test metrics for all logged alphas."""

    if type_3 == False and two_classes_close:
        base_filename = (
            f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}_two_classes_close.json"
        )
        base_dir = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_tests"
        )
        fp_dir = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_solution_two_classes_close"
        )
        fp_filename = (
            f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_two_classes_close.json"
        )
    elif type_3 == False and non_symmetric:
        base_filename = (
            f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric.json"
        )
        base_dir = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_tests"
        )
        fp_dir = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_solution_nonsym"
        )
        fp_filename = (
            f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric.json"
        )
    elif type_3 != False:
        base_filename = f"FP_evals_(k={k},k0={k_0},lambda={lambda_reg})_{type_3}.json"
        base_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "fp_tests")
        fp_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "fp_solution")
        fp_filename = f"FP_solutions_(k={k},k0={k_0},lambda={lambda_reg})_{type_3}.json"
    else:
        base_filename = f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
        base_dir = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_tests"
        )
        fp_dir = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_solution"
        )
        fp_filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"

    filepath = os.path.join(base_dir, base_filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No FP test file found at {filepath}")

    fp_filepath = os.path.join(fp_dir, fp_filename)
    if not os.path.exists(fp_filepath):
        raise FileNotFoundError(f"No FP solution file found at {fp_filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    with open(fp_filepath, "r") as f:
        fp_data = json.load(f)

    metadata = data.get("metadata", {})
    results = data.get("results", {})
    fp_results = fp_data.get("results", {})

    if not results:
        print("No stored FP test results to refine.")
        return filepath

    def _match_alpha_key(value, candidates):
        for key in candidates:
            try:
                if abs(float(key) - value) < 1e-8:
                    return key
            except ValueError:
                continue
        return None

    updated = False

    for R_00_str, alpha_dict in results.items():
        fp_alpha_dict = fp_results.get(R_00_str)
        if fp_alpha_dict is None:
            print(f"No FP data found for R_00 key {R_00_str}; skipping.")
            continue

        try:
            R_00 = np.array(json.loads(R_00_str))
        except Exception:
            R_00 = np.array(eval(R_00_str))

        inv_R00 = np.linalg.inv(R_00)
        sqrtm_R00_inv = sqrtm(inv_R00)

        for alpha_key, entry in alpha_dict.items():
            alpha_val = float(entry.get("actual_alpha", alpha_key))
            fp_alpha_key = _match_alpha_key(alpha_val, fp_alpha_dict.keys())
            if fp_alpha_key is None:
                print(
                    f"Alpha {alpha_val} not found in FP data for R_00 key {R_00_str}; skipping."
                )
                continue

            fp_entry = fp_alpha_dict[fp_alpha_key]
            schur = np.array(fp_entry["schur"]).reshape(k, k)
            R_01 = np.array(fp_entry["R_01"]).reshape(k_0, k)
            S = np.array(fp_entry["S"]).reshape(k, k)

            R_11 = schur + R_01 @ inv_R00 @ R_01.T
            test_err = test_error(
                R_00,
                schur,
                R_01=R_01,
                alpha=alpha_val,
                k=k,
                k_0=k_0,
            )
            train_err = train_error(
                R_00=R_00,
                schur=schur,
                R_01=R_01,
                S=S,
                alpha=alpha_val,
                k=k,
                k_0=k_0,
            )
            A = R_01.T @ sqrtm_R00_inv
            misclassification_test_err = misclassification_test_error(
                S=S,
                R_00=R_00,
                schur_t=schur,
                A_t=A,
                alpha=alpha_val,
                k=k,
                k_0=k_0,
            )
            f_norm = np.trace(R_00) + np.trace(R_11) - np.trace(R_01) - np.trace(R_01.T)

            entry.update(
                {
                    "test_error": float(test_err),
                    "train_error": float(train_err),
                    "misclassification_test_error": float(misclassification_test_err),
                    "F_norm": float(f_norm),
                    "diverged": bool(entry.get("diverged", False)),
                    "actual_alpha": float(alpha_val),
                }
            )
            updated = True

    if updated:
        with open(filepath, "w") as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=2)
        print(f"Refined FP test metrics saved to {filepath}")
    else:
        print("No metrics were updated.")

    return filepath


def read_fp_test_results(alpha, k, k_0, R_00, lambda_reg=0):
    """
    Read test results for the closest available alpha value.

    Args:
        alpha (float): The alpha value to look for
        k (int): Dimension of the system
        k_0 (int): Dimension of g_0
        R_00 (ndarray): Initial covariance matrix
        lambda_reg (float): Regularization parameter

    Returns:
        tuple: (test_error, train_error, F_norm, R_01, diverged) if found, None if not found
               where R_01 is the cross-correlation matrix (always zeros in current implementation)
    """
    data_dir = os.path.join(os.path.dirname(__file__), "tempdata", "fp_tests")

    if not os.path.exists(data_dir):
        print("No data directory found")
        return None

    filename = f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(
            f"No file found matching parameters k={k}, k_0={k_0}, lambda={lambda_reg}"
        )
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    R_00_str = str(R_00.tolist())
    if R_00_str not in data["results"]:
        print(f"No results found for R_00={R_00}")
        return None

    available_alphas = [float(a) for a in data["results"][R_00_str].keys()]

    if not available_alphas:
        print(f"No valid results found for R_00={R_00}")
        return None

    available_alphas = np.array(available_alphas)
    closest_alpha = available_alphas[np.argmin(np.abs(available_alphas - alpha))]
    closest_alpha_str = str(closest_alpha)

    result = data["results"][R_00_str][closest_alpha_str]

    if result["diverged"]:
        return None, None, None, np.zeros((k, k_0)), True

    print(f"Found results in file: {filename}")
    print(f"Using alpha={closest_alpha} (requested alpha={alpha})")
    return (
        result["test_error"],
        result["train_error"],
        result["F_norm"],
        np.zeros((k, k_0)),
        result["diverged"],
    )


from configs.R_initiation import get_R_00


def get_fp_statistics(
    k,
    k_0,
    lambda_reg=0,
    non_symmetric=False,
    two_classes_close=False,
    type_3: Literal[
        False,
        "symmetric",
        "two_classes_close",
        "three_classes_close",
        "two_vs_two_vs_one",
    ] = False,
):

    if type_3 == False and non_symmetric:
        filename = f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric.json"
        R_00 = np.array([[1, -1 / 2], [-1 / 2, 1]])
    elif type_3 == False and two_classes_close:
        filename = (
            f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}_two_classes_close.json"
        )
        R_00 = np.array([[1, 0.9], [0.9, 1]])
    elif type_3 != False:
        filename = f"FP_evals_(k={k},k0={k_0},lambda={lambda_reg})_{type_3}.json"
        R_00 = get_R_00(k, type_3)
    else:
        filename = f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
        R_00 = np.array([[1, 1 / 2], [1 / 2, 1]])
    # filepath = os.path.join(os.path.dirname(__file__), "newdata", "fp_tests", filename) #changed from newdata to tempdata
    if type_3 != False:
        filepath = os.path.join(
            os.path.dirname(__file__), "Oct_data", "fp_tests", filename
        )
    else:
        filepath = os.path.join(
            os.path.dirname(__file__), "tempdata", "fp_tests", filename
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No FP test file found: {filename}")

    # Read the file
    with open(filepath, "r") as f:
        data = json.load(f)

    # Check if we have results for this R_00

    R_00_str = str(R_00.tolist())
    if R_00_str not in data["results"]:
        raise ValueError(f"No results found for R_00={R_00}")

    # Initialize lists to store results
    alphas, test_errors, train_errors, F_norms, misclassification_test_errors = (
        [],
        [],
        [],
        [],
        [],
    )

    # Process each alpha value
    for alpha_str, alpha_data in data["results"][R_00_str].items():

        # Convert strings to float/arrays
        alpha = float(alpha_str)
        test_error = alpha_data["test_error"]
        train_error = alpha_data["train_error"]
        F_norm = alpha_data["F_norm"]
        misclassification_test_error = alpha_data["misclassification_test_error"]

        # Store values
        alphas.append(alpha)
        test_errors.append(test_error)
        train_errors.append(train_error)
        F_norms.append(F_norm)
        misclassification_test_errors.append(misclassification_test_error)
    # Sort everything by alpha values
    sorted_indices = np.argsort(alphas)
    alphas = np.array(alphas)[sorted_indices]
    test_errors = np.array(test_errors)[sorted_indices]
    train_errors = np.array(train_errors)[sorted_indices]
    F_norms = np.array(F_norms)[sorted_indices]
    misclassification_test_errors = np.array(misclassification_test_errors)[
        sorted_indices
    ]

    return alphas, test_errors, train_errors, F_norms, misclassification_test_errors


def get_fp_misclassification_statistics(
    k, k_0, lambda_reg=0, non_symmetric=False, two_classes_close=False
):
    # Get the filepath
    if non_symmetric:
        filename = f"fp_misclassification_test_error_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric.json"
        R_00 = np.array([[1, -1 / 2], [-1 / 2, 1]])
    else:
        filename = (
            f"fp_misclassification_test_error_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
        )
        R_00 = np.array([[1, 1 / 2], [1 / 2, 1]])
    filepath = os.path.join(
        os.path.dirname(__file__),
        "data",
        "fp_tests",
        "misclassification_test_error",
        filename,
    )

    if not os.path.exists(filepath):
        print(f"No FP misclassification test file found: {filename}")
        return None

    # Read the file
    with open(filepath, "r") as f:
        data = json.load(f)

    # Check if we have results for this R_00
    R_00_str = str(R_00.tolist())
    if R_00_str not in data["results"]:
        print(f"No results found for R_00={R_00}")
        return None

    # Initialize lists to store results
    alphas = []
    misclassification_errors = []

    # Process each alpha value
    for alpha_str, alpha_data in data["results"][R_00_str].items():
        # Skip if diverged
        if alpha_data["diverged"]:
            continue

        # Convert strings to float/arrays
        alpha = float(alpha_str)
        misclassification_error = alpha_data["misclassification_test_error"]

        # Store values
        alphas.append(alpha)
        misclassification_errors.append(misclassification_error)

    # Sort everything by alpha values
    sorted_indices = np.argsort(alphas)
    alphas = np.array(alphas)[sorted_indices]
    misclassification_errors = np.array(misclassification_errors)[sorted_indices]

    return alphas, misclassification_errors
