import numpy as np
import torch
import time

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Matrix dimensions
n = 5000
m = 5000
k = 5000

print(f"\nMatrix dimensions: A({n}x{k}) @ B({k}x{m}) = C({n}x{m})")

# Create matrices
print("\nCreating matrices...")
np.random.seed(42)
A_np = np.random.randn(n, k).astype(np.float32)
B_np = np.random.randn(k, m).astype(np.float32)

# Convert to torch tensors
A_torch = torch.from_numpy(A_np).to(device)
B_torch = torch.from_numpy(B_np).to(device)

# Warm-up GPU (first run can be slower)
if device.type == 'cuda':
    _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()

# NumPy matrix multiplication (CPU)
print("\nRunning NumPy (CPU) matrix multiplication...")
num_runs = 10
numpy_times = []
for i in range(num_runs):
    start = time.time()
    C_np = np.dot(A_np, B_np)
    numpy_times.append(time.time() - start)

numpy_avg = np.mean(numpy_times)
numpy_std = np.std(numpy_times)

# PyTorch matrix multiplication (GPU)
print("Running PyTorch (GPU) matrix multiplication...")
torch_times = []
for i in range(num_runs):
    start = time.time()
    C_torch = torch.matmul(A_torch, B_torch)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU to finish
    torch_times.append(time.time() - start)

torch_avg = np.mean(torch_times)
torch_std = np.std(torch_times)

# Results
print("\n" + "="*60)
print("BENCHMARK RESULTS")
print("="*60)
print(f"NumPy (CPU) average time: {numpy_avg:.4f} ± {numpy_std:.4f} seconds")
print(f"PyTorch (GPU) average time: {torch_avg:.4f} ± {torch_std:.4f} seconds")
print(f"\nSpeedup: {numpy_avg / torch_avg:.2f}x faster with GPU")
print("="*60)

# Verify results match (optional)
if device.type == 'cuda':
    C_torch_cpu = C_torch.cpu().numpy()
    max_diff = np.max(np.abs(C_np - C_torch_cpu))
    print(f"\nMax difference between NumPy and PyTorch results: {max_diff:.2e}")
    print("(Small differences are expected due to floating-point precision)")

