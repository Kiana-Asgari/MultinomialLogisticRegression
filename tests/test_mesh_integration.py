import importlib.util
import pathlib

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "multinomial_logistic" / "evaluation" / "torch_eval_utils.py"

spec = importlib.util.spec_from_file_location("torch_eval_utils", MODULE_PATH)
torch_eval_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(torch_eval_utils)

mesh_integration = torch_eval_utils.mesh_integration


def _reference_mesh_integral(integrand, args, ndim, n_mesh, size, dtype):
    dx = 2.0 * size / n_mesh
    axis = torch.linspace(
        -size + dx / 2.0,
        size - dx / 2.0,
        n_mesh,
        dtype=dtype,
    )
    grids = torch.meshgrid(*([axis] * ndim), indexing="ij")
    points = torch.stack([grid.reshape(-1) for grid in grids], dim=-1)
    return integrand(points, *args).sum(dim=0) * dx ** ndim


def test_mesh_integration_matches_reference():
    torch.manual_seed(0)

    ndim = 3
    n_mesh = 5
    size = 1.25
    dtype = torch.float64
    base_device = torch.device("cpu")

    weight = torch.linspace(0.5, 1.5, ndim, dtype=dtype)

    def integrand(points, w):
        linear = points @ w
        gaussian = torch.exp(-0.5 * (points ** 2).sum(dim=1))
        return torch.stack([linear, gaussian], dim=1)

    args = (weight,)

    expected = _reference_mesh_integral(integrand, args, ndim, n_mesh, size, dtype)
    result = mesh_integration(
        integrand,
        args,
        ndim,
        n_mesh,
        size,
        seed=0,
        dtype=dtype,
        base_device=base_device,
        batch_size=7,
    )

    torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-10)

