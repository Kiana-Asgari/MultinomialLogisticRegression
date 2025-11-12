import numpy as np
import torch
from cubature import cubature
import threading
import gc
import math
import contextlib


_ALLOWED_CUDA_INDICES =  (3,4)


def _get_preferred_cuda_devices(max_devices=None):
    """Return available CUDA devices restricted to indices 1-4."""
    if max_devices is None:
        max_devices = len(_ALLOWED_CUDA_INDICES)

    preferred_devices = []

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for idx in _ALLOWED_CUDA_INDICES:
            if idx < num_devices:
                preferred_devices.append(torch.device(f"cuda:{idx}"))
                if len(preferred_devices) >= max_devices:
                    break

    return preferred_devices


def get_primary_device():
    """Expose the primary compute device obeying the cuda:1-4 policy."""
    preferred_devices = _get_preferred_cuda_devices(max_devices=1)
    if preferred_devices:
        return preferred_devices[0]
    return torch.device("cpu")


def _is_forbidden_cuda(device):
    return device is not None and device.type == "cuda" and (device.index is None or device.index == 0)


def _is_allowed_cuda(device):
    return (
        device is not None
        and device.type == "cuda"
        and device.index is not None
        and device.index in _ALLOWED_CUDA_INDICES
    )


def _sanitize_cuda_device(device):
    """Return a CUDA device obeying policy or None if unavailable."""
    if not torch.cuda.is_available():
        return None

    if device is None or _is_forbidden_cuda(device) or not _is_allowed_cuda(device):
        preferred_devices = _get_preferred_cuda_devices(max_devices=1)
        return preferred_devices[0] if preferred_devices else None

    if device.type != "cuda":
        return None

    return device


def integration(integrand, S, R_00, schur_t, A_t, alpha, k, k_0, R_01_t=None, seed=42, fdim=None):
    # Set numpy random seed before cubature call
    np.random.seed(seed)
    
    if fdim is None:
        fdim = k*k
    ndim = k+k_0
    if R_01_t is not None:
        args = (S, R_00, schur_t, A_t, alpha, k, k_0, R_01_t)
    else:
        args = (S, R_00, schur_t, A_t, alpha, k, k_0)
    expectations, err = cubature(integrand, 
                                   args=args, 
                                   ndim=ndim,
                                   vectorized=True,
                                   fdim=fdim,
                                   xmin=[-5]*ndim, 
                                   xmax=[5]*ndim, 
                                   relerr=1e-5,
                                   abserr=1e-6,
                                   maxEval=3_000_000, 
                                   norm=1)

    if np.max(err) > 1e-4:
        print('     **Error in integration is too large**', np.max(err))
    
    if fdim == k*k:
        return expectations.reshape(k, k)
    else: # reshape into two matrices of size k*k and k*k_0
        R_01 = expectations[:k*k].reshape(k, k)
        schur = expectations[k*k:].reshape(k, k)
        return R_01, schur






def _create_batch_points(start_idx, end_idx, target_device, input_dim, n_mesh, axis_cache):
    # ---------- Helper: build batch points on target device ----------
    batch_range = torch.arange(start_idx, end_idx, device=target_device, dtype=torch.long)
    coords = torch.empty((batch_range.shape[0], input_dim), dtype=torch.long, device=target_device)
    remainder = batch_range
    for dim in range(input_dim - 1, -1, -1):
        coords[:, dim] = remainder % n_mesh
        remainder = remainder // n_mesh
    axis_target = axis_cache[target_device]
    return axis_target[coords]  # (batch, input_dim) on target_device

def _prepare_arg(arg, target_device, target_dtype):
    if isinstance(arg, torch.Tensor):
        tensor = arg.to(device=target_device, non_blocking=True)
        if torch.is_floating_point(tensor) and tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        return tensor
    if isinstance(arg, np.ndarray):
        return torch.tensor(arg, dtype=target_dtype, device=target_device)
    return arg

def _set_device_and_dtype(integrand_args, dtype=None):
    # ---------- Device / dtype discovery (unchanged pattern) ----------
    preferred_devices = _get_preferred_cuda_devices()
    default_device = preferred_devices[0] if preferred_devices else torch.device("cpu")
    default_dtype = torch.float64 if dtype is None else dtype

    source_tensors = [arg for arg in integrand_args if isinstance(arg, torch.Tensor)]
    if source_tensors:
        first = next((t for t in source_tensors if torch.is_floating_point(t)), source_tensors[0])
        device = first.device
        dtype = first.dtype if torch.is_floating_point(first) else default_dtype
        if _is_forbidden_cuda(device) or not _is_allowed_cuda(device):
            device = default_device
    else:
        device = default_device
        dtype = default_dtype

    if _is_forbidden_cuda(device) or not _is_allowed_cuda(device):
        device = default_device

    return device, dtype

#######################################################################
## mesh integration helpers
#######################################################################

def _get_available_devices(base_device, max_devices=None):
    """Get list of available devices honoring the cuda:1-4 policy."""
    if max_devices is None:
        max_devices = len(_ALLOWED_CUDA_INDICES)

    preferred_devices = _get_preferred_cuda_devices(max_devices=max_devices)

    sanitized_base = base_device
    if sanitized_base is None:
        sanitized_base = get_primary_device()
    elif _is_forbidden_cuda(sanitized_base) or not _is_allowed_cuda(sanitized_base):
        sanitized_base = get_primary_device()

    if sanitized_base is None:
        sanitized_base = torch.device("cpu")

    devices = []
    if sanitized_base not in devices:
        devices.append(sanitized_base)

    for dev in preferred_devices:
        if dev not in devices:
            devices.append(dev)
        if len(devices) >= max_devices:
            break

    if not devices:
        devices.append(torch.device("cpu"))

    return devices


def _prepare_device_caches(integrand_args, axis, available_devices, dtype):
    """Prepare per-device caches for arguments and axis."""
    args_cache = {
        dev: tuple(_prepare_arg(arg, dev, dtype) for arg in integrand_args)
        for dev in available_devices
    }
    axis_cache = {dev: axis.to(device=dev) for dev in available_devices}
    return args_cache, axis_cache


def _mesh_integration_worker(
    dev_index,
    target_device,
    integrand,
    args_cache,
    axis_cache,
    partials,
    num_devices,
    n_batches,
    batch_size,
    n_points,
    input_dim,
    n_mesh,
):
    """Worker function for processing batches on a single device."""
    batch_args = args_cache[target_device]
    local_sum = partials[target_device]
    
    # Setup CUDA stream if device is CUDA
    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        stream = torch.cuda.Stream(device=target_device)
        ctx = torch.cuda.stream(stream)
    else:
        stream = None
        ctx = torch.no_grad()
    
    with ctx:
        # Strided schedule: dev i processes batches i, i+num_devices, ...
        for batch_index in range(dev_index, n_batches, num_devices):
            start_idx = batch_index * batch_size
            end_idx = min((batch_index + 1) * batch_size, n_points)
            
            # Build points and evaluate integrand on this device
            batch_points = _create_batch_points(
                start_idx, end_idx, target_device, input_dim, n_mesh, axis_cache
            )
            batch_result = integrand(batch_points, *batch_args)
            
            # Accumulate locally
            local_sum.add_(batch_result.sum(dim=0))
            
            del batch_points, batch_result
        
        if stream is not None:
            stream.synchronize()
    
    # Clean up device memory
    if target_device.type == "cuda":
        torch.cuda.empty_cache()


def _launch_worker_threads(
    available_devices,
    integrand,
    args_cache,
    axis_cache,
    partials,
    n_batches,
    batch_size,
    n_points,
    input_dim,
    n_mesh,
):
    """Launch worker threads for each device."""
    num_devices = len(available_devices)
    threads = []
    
    for dev_index, target_device in enumerate(available_devices):
        t = threading.Thread(
            target=_mesh_integration_worker,
            args=(
                dev_index,
                target_device,
                integrand,
                args_cache,
                axis_cache,
                partials,
                num_devices,
                n_batches,
                batch_size,
                n_points,
                input_dim,
                n_mesh,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
    
    # Wait for all threads to complete
    for t in threads:
        t.join()


def _reduce_partial_sums(partials, base_device, output_dim, dtype):
    """Reduce partial sums from all devices to base device."""
    expectations = torch.zeros(output_dim, dtype=dtype, device=base_device)
    with torch.no_grad():
        for dev, partial in partials.items():
            if dev == base_device:
                expectations.add_(partial)
            else:
                expectations.add_(partial.to(base_device, non_blocking=True))
    return expectations


#######################################################################
## mesh integration
#######################################################################

def mesh_integration(
    integrand,
    *integrand_args,
    input_dim,
    output_dim,
    seed=42,
    n_mesh=20,
    size=6,
    min_size=None,
    batch_size = 200_000,
    dtype=None,
):
    """Multi-threaded mesh integration across multiple GPUs."""
    torch.manual_seed(seed)
    device, dtype = _set_device_and_dtype(integrand_args, dtype=dtype)
    
    # Grid setup
    dx = 2.0 * size / n_mesh
    if min_size is not None:
        lower_bound = min_size + dx / 2.0
        upper_bound = size - dx / 2.0
        axis1 = torch.linspace(lower_bound, upper_bound, n_mesh, device=device, dtype=dtype)
        axis2 = torch.linspace(-upper_bound, -lower_bound, n_mesh, device=device, dtype=dtype)
        axis = torch.cat([axis1, axis2], dim=0)
    else:
        lower_bound = -size + dx / 2.0
        upper_bound = size - dx / 2.0
        axis = torch.linspace(lower_bound, upper_bound, n_mesh, device=device, dtype=dtype)
        
    n_points = n_mesh ** input_dim
    
    n_batches = (n_points + batch_size - 1) // batch_size
    print(f"  --n_batches: {n_batches}, n_points: {n_points}")
    
    # Setup devices and caches
    available_devices = _get_available_devices(device)
    args_cache, axis_cache = _prepare_device_caches(integrand_args, axis, available_devices, dtype)
    partials = {
        dev: torch.zeros(output_dim, dtype=dtype, device=dev)
        for dev in available_devices
    }
    
    
    # Launch worker threads
    _launch_worker_threads(
        available_devices,
        integrand,
        args_cache,
        axis_cache,
        partials,
        n_batches,
        batch_size,
        n_points,
        input_dim,
        n_mesh,
    )
    
    # Reduce results
    expectations = _reduce_partial_sums(partials, device, output_dim, dtype)
    
    # Apply volume element
    volume_element = dx ** input_dim
    expectations = expectations * volume_element
    
    # Memory cleanup
    gpu_barrier(device)
    gpu_clear_cache(device)
    gpu_reset_memstats(device)
    
    return expectations


#######################################################################
## sphere integration helpers
#######################################################################


def _build_spherical_axes(input_dim, n_polar, n_azimuth, device, dtype):
    if input_dim < 2:
        raise ValueError("sphere_mesh_integration requires input_dim >= 2.")
    if n_polar <= 0:
        raise ValueError("n_polar must be a positive integer.")
    if n_azimuth is not None and n_azimuth <= 0:
        raise ValueError("n_azimuth must be a positive integer when provided.")

    num_angles = input_dim - 1
    azimuth_count = n_azimuth if n_azimuth is not None else max(4, 2 * n_polar)
    counts = [n_polar] * max(num_angles - 1, 0) + [azimuth_count]

    axes = []
    steps = []
    for angle_idx, count in enumerate(counts):
        step = math.pi / count if angle_idx < num_angles - 1 else 2.0 * math.pi / count
        centers = (torch.arange(count, device=device, dtype=dtype) + 0.5) * step
        if angle_idx == num_angles - 1:
            centers = torch.remainder(centers, 2.0 * math.pi)
        axes.append(centers)
        steps.append(step)

    return axes, steps, counts


def _build_radial_axis(radius_max, n_radius, device, dtype, radius_min=0.0):
    if n_radius <= 0:
        raise ValueError("n_radius must be a positive integer.")
    if radius_min < 0:
        raise ValueError("radius_min must be non-negative.")
    if radius_max <= radius_min:
        raise ValueError("radius must be greater than radius_min.")

    delta_r = (radius_max - radius_min) / n_radius
    centers = radius_min + (torch.arange(n_radius, device=device, dtype=dtype) + 0.5) * delta_r
    return centers, delta_r


def _prepare_spherical_device_caches(integrand_args, angle_axes, radius_axis, available_devices, dtype):
    args_cache = {
        dev: tuple(_prepare_arg(arg, dev, dtype) for arg in integrand_args)
        for dev in available_devices
    }
    angle_cache = {
        dev: tuple(axis.to(device=dev) for axis in angle_axes)
        for dev in available_devices
    }
    radius_cache = {
        dev: radius_axis.to(device=dev)
        for dev in available_devices
    }
    return args_cache, angle_cache, radius_cache


def _create_spherical_batch(start_idx, end_idx, target_device, counts, radius_axis, angle_axes):
    batch_range = torch.arange(start_idx, end_idx, device=target_device, dtype=torch.long)
    num_dims = len(counts)
    coords = torch.empty((batch_range.shape[0], num_dims), dtype=torch.long, device=target_device)

    remainder = batch_range
    for dim in range(num_dims - 1, -1, -1):
        base = counts[dim]
        coords[:, dim] = remainder % base
        remainder = remainder // base

    radial_idx = coords[:, 0]
    angle_indices = coords[:, 1:]

    radii = radius_axis[radial_idx]

    if angle_axes:
        angles = [angle_axes[dim][angle_indices[:, dim]] for dim in range(len(angle_axes))]
        angles = torch.stack(angles, dim=-1)
    else:
        angles = torch.empty((batch_range.shape[0], 0), dtype=radius_axis.dtype, device=target_device)

    return radii, angles


def _angles_to_cartesian(radius, angles):
    if angles.numel() == 0:
        return radius.unsqueeze(-1)

    num_angles = angles.shape[1]
    input_dim = num_angles + 1
    dtype = angles.dtype
    device = angles.device

    points = torch.empty((angles.shape[0], input_dim), dtype=dtype, device=device)
    sin_prefix = torch.ones(angles.shape[0], dtype=dtype, device=device)

    for coord_idx in range(input_dim):
        if coord_idx < num_angles:
            angle = angles[:, coord_idx]
            points[:, coord_idx] = radius * sin_prefix * torch.cos(angle)
            sin_prefix = sin_prefix * torch.sin(angle)
        else:
            points[:, coord_idx] = radius * sin_prefix

    return points


def _compute_spherical_weights(radius, angles, sin_exponents, angular_step_tensor, radial_step_tensor, input_dim):
    if angles.numel() == 0:
        return torch.zeros_like(radius)

    weights = (radius ** (input_dim - 1))
    weights = weights * angular_step_tensor * radial_step_tensor

    for idx, exponent in enumerate(sin_exponents):
        if exponent <= 0:
            continue
        weights = weights * torch.sin(angles[:, idx]).pow(exponent)

    return weights


def _sphere_integration_worker(
    dev_index,
    target_device,
    integrand,
    args_cache,
    angle_cache,
    radius_cache,
    partials,
    num_devices,
    n_batches,
    batch_size,
    n_points,
    counts,
    sin_exponents,
    angular_step_tensors,
    radial_step_tensors,
    input_dim,
):
    batch_args = args_cache[target_device]
    local_sum = partials[target_device]
    angles_axes = angle_cache[target_device]
    radius_axis = radius_cache[target_device]
    angular_step_tensor = angular_step_tensors[target_device]
    radial_step_tensor = radial_step_tensors[target_device]

    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        stream = torch.cuda.Stream(device=target_device)
        stream_ctx = torch.cuda.stream(stream)
    else:
        stream = None
        stream_ctx = contextlib.nullcontext()

    with stream_ctx:
        with torch.no_grad():
            for batch_index in range(dev_index, n_batches, num_devices):
                start_idx = batch_index * batch_size
                end_idx = min((batch_index + 1) * batch_size, n_points)
                if start_idx >= end_idx:
                    continue

                radius_values, angles = _create_spherical_batch(
                    start_idx, end_idx, target_device, counts, radius_axis, angles_axes
                )
                points = _angles_to_cartesian(radius_values, angles)
                weights = _compute_spherical_weights(
                    radius_values,
                    angles,
                    sin_exponents,
                    angular_step_tensor,
                    radial_step_tensor,
                    input_dim,
                ).unsqueeze(-1)

                batch_result = integrand(points, *batch_args)
                if batch_result.ndim == 1:
                    batch_result = batch_result.unsqueeze(-1)
                batch_result = batch_result.to(local_sum.dtype)
                weights = weights.to(local_sum.dtype)

                local_sum.add_((batch_result * weights).sum(dim=0))

                del angles, points, weights, batch_result

        if stream is not None:
            stream.synchronize()

    if target_device.type == "cuda":
        torch.cuda.empty_cache()


def _launch_spherical_worker_threads(
    available_devices,
    integrand,
    args_cache,
    angle_cache,
    radius_cache,
    partials,
    n_batches,
    batch_size,
    n_points,
    counts,
    sin_exponents,
    angular_step_tensors,
    radial_step_tensors,
    input_dim,
):
    num_devices = len(available_devices)
    threads = []

    for dev_index, target_device in enumerate(available_devices):
        t = threading.Thread(
            target=_sphere_integration_worker,
            args=(
                dev_index,
                target_device,
                integrand,
                args_cache,
                angle_cache,
                radius_cache,
                partials,
                num_devices,
                n_batches,
                batch_size,
                n_points,
                counts,
                sin_exponents,
                angular_step_tensors,
                radial_step_tensors,
                input_dim,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


#######################################################################
## sphere integration
#######################################################################


def sphere_mesh_integration(
    integrand,
    *integrand_args,
    input_dim,
    output_dim,
    batch_size, # increase if working with n_polar <=5 and n_radius <=14
    n_radius, #16 worked, 14 worked
    n_polar, #6 worked, 5 barely worked
    radius,
    seed=42,
    n_azimuth=None,
    min_radius=None, #0.0
    normalize=False, #False
    dtype=None,
):
    """Riemann integration over a ball using spherical coordinates."""

    torch.manual_seed(seed)
    device, dtype = _set_device_and_dtype(integrand_args, dtype=dtype)

    radius_min = 0.0 if min_radius is None else float(min_radius)
    angle_axes, angle_steps, angle_counts = _build_spherical_axes(
        input_dim, n_polar, n_azimuth, device, dtype
    )
    radius_axis, radial_step = _build_radial_axis(
        radius, n_radius, device, dtype, radius_min
    )

    counts = [radius_axis.numel()] + angle_counts
    n_points = 1
    for count in counts:
        n_points *= count

    if n_points == 0:
        raise ValueError(
            "Empty spherical grid; check n_radius, n_polar, and n_azimuth values."
        )

    n_batches = (n_points + batch_size - 1) // batch_size
    print(f"  --sphere_n_batches: {n_batches}, n_points: {n_points}")
    available_devices = _get_available_devices(device)
    args_cache, angle_cache, radius_cache = _prepare_spherical_device_caches(
        integrand_args, angle_axes, radius_axis, available_devices, dtype
    )

    partials = {
        dev: torch.zeros(output_dim, dtype=dtype, device=dev)
        for dev in available_devices
    }

    angular_step = 1.0
    for step in angle_steps:
        angular_step *= step

    angular_step_tensors = {
        dev: torch.tensor(angular_step, dtype=dtype, device=dev)
        for dev in available_devices
    }

    radial_step_tensors = {
        dev: torch.tensor(radial_step, dtype=dtype, device=dev)
        for dev in available_devices
    }

    sin_exponents = [max(input_dim - idx - 2, 0) for idx in range(input_dim - 1)]
    sin_exponents = tuple(sin_exponents)
    counts = tuple(counts)

    _launch_spherical_worker_threads(
        available_devices,
        integrand,
        args_cache,
        angle_cache,
        radius_cache,
        partials,
        n_batches,
        batch_size,
        n_points,
        counts,
        sin_exponents,
        angular_step_tensors,
        radial_step_tensors,
        input_dim,
    )

    expectations = _reduce_partial_sums(partials, device, output_dim, dtype)

    if normalize:
        def _ball_volume(dim, r):
            return math.pi ** (dim / 2.0) / math.gamma(dim / 2.0 + 1.0) * (r ** dim)

        volume_outer = _ball_volume(input_dim, radius)
        volume_inner = _ball_volume(input_dim, radius_min) if radius_min > 0 else 0.0
        total_volume = volume_outer - volume_inner
        expectations = expectations / expectations.new_tensor(total_volume)

    gpu_barrier(device)
    gpu_clear_cache(device)
    gpu_reset_memstats(device)

    return expectations


#######################################################################
## memory clean up
#######################################################################

def gpu_barrier(device=None):
    """Synchronize CUDA operations on specified device."""
    cuda_device = _sanitize_cuda_device(device)
    if cuda_device is None:
        return
    torch.cuda.synchronize(cuda_device)


def gpu_clear_cache(device=None):
    """Clear GPU cache and run garbage collection."""
    cuda_device = _sanitize_cuda_device(device)
    if cuda_device is not None:
        torch.cuda.set_device(cuda_device)
        torch.cuda.synchronize(cuda_device)
    gc.collect()
    if cuda_device is not None:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_reset_memstats(device=None):
    """Reset peak memory statistics for specified device."""
    cuda_device = _sanitize_cuda_device(device)
    if cuda_device is None:
        return
    torch.cuda.reset_peak_memory_stats(cuda_device)