import numpy as np
import os
import json
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline


def run_and_log_mle_regularized(
    k_0,
    k,
    R_00_values,
    type_3,
    d=250,
    n_trials=100,
    lambda_regs=np.linspace(0.001, 0.6, 10),
    alpha_values=[1.5, 3, 5, 10],
):

    # Create base filename without timestamp, but with d and n_trials
    if type_3 == False:
        base_filename = f"mle_reg_k{k}_k0{k_0}_d{d}_ntrials{n_trials}.json"
        base_filepath = os.path.join(
            os.path.dirname(__file__), "newdata", "mle_empirical", base_filename
        )
    elif type_3 != False:
        base_filename = f"MLE_reg_evals_(k={k},k0={k_0})_{type_3}.json"
        base_filepath = os.path.join(
            os.path.dirname(__file__), "Oct_data", "mle_empirical", base_filename
        )

    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    data_dir = os.path.dirname(base_filepath)
    existing_files = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        if (
            f"mle_reg_k{k}_k0{k_0}_d{d}_ntrials{n_trials}"
            or f"MLE_reg_evals_(k={k},k0={k_0})_{type_3}.json" in filename
        ):
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
        print(f"Creating new  mle reg file: {os.path.basename(filepath)}")

    n_trials = 100
    d = 250

    for R_00 in R_00_values:
        for lambda_reg in lambda_regs:

            for alpha in alpha_values:  # Using specific alpha values
                # Create composite key
                key = json.dumps(
                    {
                        "R_00": R_00.tolist(),
                        "alpha": float(alpha),
                        "lambda_reg": float(lambda_reg),
                    }
                )

                # Skip if we already have results for this combination
                if key in results:
                    print(
                        f"Skipping alpha={alpha}, lambda={lambda_reg} for R_00={R_00} (already exists)"
                    )
                    continue

                print(f"\nProcessing alpha={alpha}, lambda={lambda_reg}")

                (
                    Theta_hats,
                    norms,
                    test_errors,
                    train_errors,
                    misclassification_test_errors,
                ) = fit_mle_baseline(
                    alpha=alpha,
                    k=k,
                    lambda_reg=lambda_reg * 2,  # remember the *2
                    R_00=R_00,
                    n_trials=n_trials,
                    d=d,
                    return_full_results=True,
                )

                results[key] = {
                    "test_errors": test_errors.tolist(),
                    "train_errors": train_errors.tolist(),
                    "norms": norms.tolist(),
                    "misclassification_test_errors": misclassification_test_errors.tolist(),
                }

                print(
                    "done mle for alpha=",
                    alpha,
                    "lambda=",
                    lambda_reg,
                    "avg test error=",
                    np.mean(test_errors),
                    "avg train error=",
                    np.mean(train_errors),
                    "avg norm=",
                    np.mean(norms),
                    "avg misclassification test error=",
                    np.mean(misclassification_test_errors),
                )
                # Save after each computation
                with open(base_filepath, "w") as f:
                    json.dump(
                        {"metadata": {"k": k, "k_0": k_0}, "results": results},
                        f,
                        indent=2,
                    )

    return filepath