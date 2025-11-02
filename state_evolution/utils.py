import numpy as np
import torch
from cubature import cubature





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


import numpy as np
import torch
import threading


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

def _set_device_and_dtype(integrand_args):
    # ---------- Device / dtype discovery (unchanged pattern) ----------
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_dtype = torch.float64

    source_tensors = [arg for arg in integrand_args if isinstance(arg, torch.Tensor)]
    if source_tensors:
        first = next((t for t in source_tensors if torch.is_floating_point(t)), source_tensors[0])
        device = first.device
        dtype = first.dtype if torch.is_floating_point(first) else default_dtype
    else:
        device = default_device
        dtype = default_dtype
    return device, dtype

#######################################################################
## mesh integration
#######################################################################z
def mesh_integration(
    integrand,
    *integrand_args,
    input_dim,
    output_dim,
    seed=42,
    n_mesh=20,
    size=6,
):
    torch.manual_seed(seed)
    device, dtype = _set_device_and_dtype(integrand_args)
        
    # ---------- Grid setup (unchanged math) ----------
    dx = 2.0 * size / n_mesh
    axis = torch.linspace(-size + dx / 2.0, size - dx / 2.0, n_mesh, device=device, dtype=dtype)
    n_points = n_mesh ** input_dim

    # Tune batch size as needed
    batch_size = 400_000
    n_batches = (n_points + batch_size - 1) // batch_size

    # Choose up to 5 GPUs 
    available_devices = []
    if torch.cuda.is_available():
        num_devices_tot = torch.cuda.device_count()
        for idx in range(min(5, num_devices_tot)):
            available_devices.append(torch.device(f"cuda:{idx}"))
    if not available_devices:
        # CPU fallback: single-threaded
        available_devices = [device]

    # Per-device cached args and axis
    args_cache = {
        dev: tuple(_prepare_arg(arg, dev, dtype) for arg in integrand_args)
        for dev in available_devices
    }
    axis_cache = {dev: axis.to(device=dev) for dev in available_devices}




    # Output accumulator on base device 
    base_device = device
    expectations = torch.zeros(output_dim, dtype=dtype, device=base_device)
    print(f"  --n_batches: {n_batches}, n_points: {n_points}, devices: {len(available_devices)}")

    num_devices = len(available_devices)

    #  Per-device partial sums      
    partials = {dev: torch.zeros(output_dim, dtype=dtype, device=dev) for dev in available_devices}

    #  Worker: strided batches for true simultaneous start 
    def worker(dev_index, target_device):
        # Bind this thread to its CUDA device (no-op on CPU)
        if target_device.type == "cuda":
            torch.cuda.set_device(target_device)
            stream = torch.cuda.Stream(device=target_device)
        else:
            stream = None

        batch_args = args_cache[target_device]
        local_sum = partials[target_device]

        ctx_stream = torch.cuda.stream(stream) if stream is not None else torch.no_grad()
        with ctx_stream, torch.no_grad():
            # Strided schedule: dev i processes batches i, i+num_devices, ...
            for batch_index in range(dev_index, n_batches, num_devices):
                start_idx = batch_index * batch_size
                end_idx = min((batch_index + 1) * batch_size, n_points)

                # Build points and evaluate integrand on this device
                batch_points_device = _create_batch_points(start_idx, end_idx, target_device, input_dim, n_mesh, axis_cache)
                batch_result = integrand(batch_points_device, *batch_args)  # (batch, output_dim)

                # Accumulate locally (no cross-device traffic here)
                local_sum.add_(batch_result.sum(dim=0))
                partials[target_device].add_(batch_result.sum(0))

                del batch_points_device, batch_result  # drop refs so they can be freed



            # Ensure the device work is all enqueued & finished
            if stream is not None:
                stream.synchronize()

        # Optional allocator clean-up on this device
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    #  Launch one thread per device 
    threads = []
    for dev_index, target_device in enumerate(available_devices):
        t = threading.Thread(target=worker, args=(dev_index, target_device), daemon=True)
        t.start()
        threads.append(t)

    #  Wait for all devices 
    for t in threads:
        t.join()

    #  Single, final reduction to base device 
    with torch.no_grad():
        for dev in available_devices:
            if dev == base_device:
                expectations.add_(partials[dev])
            else:
                expectations.add_(partials[dev].to(base_device, non_blocking=True))

    volume_element = dx ** input_dim
    expectations = expectations * volume_element

    # Memory hygiene
    gpu_barrier(base_device)
    gpu_clear_cache(base_device) 
    gpu_reset_memstats(base_device)

    return expectations


#######################################################################
## memory clean up
#######################################################################
import torch, gc

def gpu_barrier(device=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

def gpu_clear_cache(device=None):
    gpu_barrier(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def gpu_reset_memstats(device=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)