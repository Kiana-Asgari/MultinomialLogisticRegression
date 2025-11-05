
import torch

from state_evolution.recursion_parts.S_recursion import S_recursion
from state_evolution.recursion_parts.full_R_recursion import R_recursion
from state_evolution.utils import get_primary_device


div_prox = False


def state_evolution_full_recursion(R_00, schur_0, R_01_0, lambda_reg, alpha, k, k_0, S_0=None, 
tol=1e-5, max_iter=300, seed=42, integral_mesh_size=10  , integral_size=5.5):

    torch.manual_seed(seed)

    device = get_primary_device()
    dtype = torch.float32



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

        _print_stats(t, alpha_tensor, errors)

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

    print("\n", "-" * 40, "\n")
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

    print("*************state evolution iteration started*************")
    print(
        f"     [initial parameters] lambda: {lambda_reg}, alpha: {alpha}, k: {k}, R_00 norm: {torch.linalg.norm(R_00)}"
    )
    dtype = R_00.dtype
    device = R_00.device
    R_00_sqrtm_inv = torch.linalg.inv(_matrix_sqrt(R_00))
    R_01_t = torch.zeros_like(R_00, device=device, dtype=dtype)
    schur_t = R_00.clone()
    S_t = torch.eye(k, dtype=dtype, device=device)
    divergence = False
    errors = torch.zeros((max_iter, 3), dtype=dtype, device=device)
    import math
    if math.fabs(alpha_init - 4.6) < 1e-5:
        schur_t, R_01_t, S_t = _initialize_alpha6(device)
        input("shur = " + str(schur_t.tolist()) + "Press Enter to continue...")
    return R_00_sqrtm_inv, R_01_t, schur_t, S_t, divergence, errors


def _print_stats(t, alpha, errors):
    print(f"state evolution iteration {t + 1} Done (alpha: {alpha.item()})")
    print(f"     ** R_01 RESIDUAL IS {errors[t, 0].item()}**")
    print(f"     ** SCHUR RESIDUAL IS {errors[t, 1].item()}**")
    print(f"     ** S RESIDUAL IS {errors[t, 2].item()}**\n\n")


def _matrix_sqrt(matrix):
    symmetric_matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix)
    eigenvalues_clamped = torch.clamp(eigenvalues, min=0.0)
    sqrt_eigenvalues = torch.sqrt(eigenvalues_clamped)
    return eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)



def _initialize_alpha6(device):
    schur=torch.tensor([[
    18.373910903930664,
    9.648628234863281,
    9.234729766845703,
    9.238040924072266
    ],
    [
    9.648633003234863,
    18.46535873413086,
    9.42496109008789,
    9.421724319458008
    ],
    [
    9.234728813171387,
    9.424952507019043,
    19.927770614624023,
    11.850250244140625
    ],
    [
    9.238055229187012,
    9.421735763549805,
    11.850273132324219,
    19.94683265686035]], dtype=torch.float32, device=device)

    R_01=torch.tensor([[
    2.078552484512329, 
    2.022347927093506,
    1.553566575050354,
    1.5539439916610718],[
    1.9897136688232422,
    2.0501620769500732,
    2.0202648639678955,
    2.0065078735351562],[1.5743502378463745,
    1.9869319200515747,
    6.359892845153809,
    6.301181793212891], [1.5738719701766968,
    1.9699735641479492,
    6.282260417938232,
    6.339254379272461]], dtype=torch.float32, device=device)

    S=torch.tensor([[
    8.31820011138916,
    4.46047306060791,
    4.224239349365234,
    4.225575923919678
    ],
    [
    4.460474014282227,
    8.51653003692627,
    4.31066370010376,
    4.309206485748291
    ],
    [
    4.224238872528076,
    4.310661792755127,
    9.028604507446289,
    5.416868686676025
    ],
    [
    4.225578784942627,
    4.309208869934082,
    5.416873455047607,
    9.051876068115234
    ]], dtype=torch.float32, device=device)
    return schur, R_01, S
      