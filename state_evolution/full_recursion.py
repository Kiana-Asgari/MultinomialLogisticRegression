
import torch

from state_evolution.recursion_parts.S_recursion import S_recursion
from state_evolution.recursion_parts.full_R_recursion import R_recursion
from state_evolution.utils import get_primary_device


div_prox = False


def state_evolution_full_recursion(R_00, schur_0, R_01_0, lambda_reg, alpha, k, k_0, S_0=None, 
tol=1e-5, max_iter=300, seed=42, integral_mesh_size=None, integral_size=None, dtype=torch.float64):

    torch.manual_seed(seed)

    device = get_primary_device()
    dtype = torch.float64 if dtype is None else dtype  



    R_00_tensor = torch.as_tensor(R_00, dtype=dtype, device=device)
    schur_0_tensor = torch.as_tensor(schur_0, dtype=dtype, device=device)
    R_01_0_tensor = torch.as_tensor(R_01_0, dtype=dtype, device=device)
    lambda_tensor = torch.as_tensor(lambda_reg, dtype=dtype, device=device)
    alpha_tensor = torch.as_tensor(alpha, dtype=dtype, device=device)
    tol_tensor = torch.as_tensor(tol, dtype=dtype, device=device)

    R_00_sqrtm_inv, R_01_t, schur_t, S_t, divergence, errors = _initialize_state_evolution(
        R_00_tensor,
        schur_0_tensor,
        R_01_0_tensor,
        lambda_tensor,
        alpha_tensor,
        k,
        k_0,
        S_0=S_0,
        max_iter=max_iter,
        alpha_init=alpha
    )

    div_tol = torch.linalg.norm(R_00_tensor) * 5.0e4
    R00_sqrt = _matrix_sqrt(R_00_tensor)



    for t in range(max_iter):
        S_next, schur_next, R_01_next = _compute_next_state_variables(
            S_t,
            R_00_tensor,
            R00_sqrt,
            schur_t,
            R_01_t,
            lambda_tensor,
            alpha_tensor,
            k,
            k_0,
            R_00_sqrtm_inv,
            integral_mesh_size=integral_mesh_size,
            integral_size=integral_size,
        )


        current_errors = torch.stack(
            [
                torch.linalg.norm(R_01_next - R_01_t),
                torch.linalg.norm(schur_next - schur_t),
                torch.linalg.norm(S_next - S_t),
            ]
        )
        errors[t] = current_errors

        _print_stats(t, alpha_tensor, lambda_tensor, errors)
        print('         S_next:', S_next.flatten())
        print('         R_01_next:', R_01_next.flatten())
        print('         schur_next:', schur_next.flatten())



        # if torch.all(current_errors < 1e-4):
        #     integral_mesh_size = 15
        #     integral_size = 5.5

        if torch.all(current_errors < tol_tensor):
            divergence = False
            break

        schur_t, R_01_t, S_t = schur_next, R_01_next, S_next

    return schur_t, R_01_t, S_t, divergence


def _compute_next_state_variables(
    S_t,
    R_00,
    R00_sqrt,
    schur_t,
    R_01_t,
    lambda_reg,
    alpha,
    k,
    k_0,
    R_00_sqrtm_inv,
    integral_mesh_size=10,
    integral_size=5.5,
): 
    while False:
        S_next = S_recursion(
            S_t_tensor=S_t,
            R_00_tensor=R_00,
            R00_sqrt=R00_sqrt,
            schur_tensor=schur_t,
            R_01_tensor=R_01_t,
            lambda_tensor=lambda_reg,
            alpha_tensor=alpha,
            k=k,
            k_0=k_0,
            R_00_sqrtm_inv_tensor=R_00_sqrtm_inv,
            integral_mesh_size=10,
            integral_size=integral_size,
        )
        print(f"   [midstep] S_next -S_t norm: {torch.linalg.norm(S_next - S_t)}")
        if torch.linalg.norm(S_t - S_next) < 1e-1:
            S_t = S_next
            break
        S_t = S_next
        


    S_next = S_recursion(
        S_t_tensor=S_t,
        R_00_tensor=R_00,
        R00_sqrt=R00_sqrt,
        schur_tensor=schur_t,
        R_01_tensor=R_01_t,
        lambda_tensor=lambda_reg,
        alpha_tensor=alpha,
        k=k,
        k_0=k_0,
        R_00_sqrtm_inv_tensor=R_00_sqrtm_inv,
        integral_mesh_size=integral_mesh_size,
        integral_size=integral_size,
    )
   # print(f"   [final] S_next -S_t norm: {torch.linalg.norm(S_next - S_t)}")

    while False:
        R_01_next, schur_next = R_recursion(
            S_t_tensor=S_t,
            S_next_tensor=S_next,
            R_00_tensor=R_00,
            schur_tensor=schur_t,
            R_01_tensor=R_01_t,
            lambda_tensor=lambda_reg,
            alpha_tensor=alpha,
            k=k,
            k_0=k_0,
            R_00_sqrtm_inv_tensor=R_00_sqrtm_inv,
            integral_mesh_size=8,
            integral_size=integral_size,
        )
        print(f"   [midstep] schur_next -schur_t norm: {torch.linalg.norm(schur_next - schur_t)}")
        if torch.linalg.norm(schur_next - schur_t) < 1e-1:
            R_01_t = R_01_next
            schur_t = schur_next
            break
        R_01_t = R_01_next
        schur_t = schur_next

    R_01_next, schur_next = R_recursion(
            S_t_tensor=S_t,
            S_next_tensor=S_next,
            R_00_tensor=R_00,
            schur_tensor=schur_t,
            R_01_tensor=R_01_t,
            lambda_tensor=lambda_reg,
            alpha_tensor=alpha,
            k=k,
            k_0=k_0,
            R_00_sqrtm_inv_tensor=R_00_sqrtm_inv,
            integral_mesh_size=integral_mesh_size,
            integral_size=integral_size,
        )


    return S_next, schur_next, R_01_next


def check_for_divergence(
    S_next,
    S_t,
    schur_next,
    schur_t,
    R_01_next,
    R_01_t,
    div_tol,
    iteration,
    errors,
):
    global div_prox
    divergence = False

    if div_prox:
        print("[DIVERGENCE] Halting state evolution due to [prox] divergence")
        divergence = True

    for i in range(1, iteration):
        if torch.all(errors[i] - errors[i - 1] > 1e-5):
            print("[DIVERGENCE] Halting state evolution due to [all errors increase > 0]")
            print(f"Error jump detected: {(errors[i] - errors[i - 1]).tolist()}")
            divergence = True
            break

        if torch.any(errors[i] > div_tol / 2):
            print("[DIVERGENCE] Halting state evolution due to one [error] too large")
            divergence = True
            break

        if torch.any((errors[i] - errors[i - 1] > 0.5) & (errors[i] > 5)):
            print("[DIVERGENCE] Halting state evolution due to one [error] too large and growing")
            divergence = True
            break

    if (
        torch.linalg.norm(S_next) > div_tol
        or torch.linalg.norm(schur_next) > div_tol
        or torch.linalg.norm(R_01_next) > div_tol
    ):
        print("[DIVERGENCE] Halting state evolution due to [norm] divergence")
        divergence = True
    return divergence


def _initialize_state_evolution(
    R_00,
    schur_0,
    R_01_0,
    lambda_reg,
    alpha,
    k,
    k_0,
    S_0=None,
    max_iter=300,
    alpha_init=0
):



    print(f"{'*'*20} State evolution (α={alpha:5.2f}, λ={lambda_reg}, R_00 = [{R_00[0,0]:.2f}  {R_00[0,1]:.2f}  {R_00[1,0]:.2f}  {R_00[1,1]:.2f}]) {'*'*20}")
    dtype = R_00.dtype
    device = R_00.device
    R_00_sqrtm_inv = torch.linalg.inv(_matrix_sqrt(R_00))
    if R_01_0 is not None:
        R_01_t = torch.as_tensor(R_01_0, dtype=dtype, device=device)
    else:
        R_01_t = torch.zeros_like(R_00, device=device, dtype=dtype)
    if schur_0 is not None:
        schur_t = torch.as_tensor(schur_0, dtype=dtype, device=device)
    else:
        schur_t = R_00.clone()  
    if S_0 is not None:
        S_t = torch.as_tensor(S_0, dtype=dtype, device=device)
    else:
        S_t = torch.eye(k, dtype=dtype, device=device)
    divergence = False
    errors = torch.zeros((max_iter, 3), dtype=dtype, device=device)

    return R_00_sqrtm_inv, R_01_t, schur_t, S_t, divergence, errors


def _print_stats(t, alpha, lambda_reg, errors):
    print(f"[Iter {t + 1}]     R01 error: {errors[t, 0].item():.4f},   Schur error: {errors[t, 1].item():.4f},   S error: {errors[t, 2].item():.4f}\n")

def _matrix_sqrt(matrix):
    symmetric_matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix)
    eigenvalues_clamped = torch.clamp(eigenvalues, min=0.0)
    sqrt_eigenvalues = torch.sqrt(eigenvalues_clamped)
    return eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)



