import torch
import torch.nn as nn
from astra.pruners.sparsegpt import SparseGPT

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def compute_H(X, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i : i + batch_size].to(device=DEVICE, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
M, K = W0.shape
N = X_cpu.shape[0]

print(f"W: {W0.shape}, X: {X_cpu.shape}")

print("Computing H...")
H = compute_H(X_cpu)

# Create a dummy layer to use with SparseGPT
layer = nn.Linear(K, M, bias=False)
layer.weight.data = W0.clone()
layer = layer.to(DEVICE)

sgpt = SparseGPT(layer)
# Manually set H and nsamples since we precomputed them
sgpt.H = H.clone()
sgpt.nsamples = N

# Run SparseGPT with 2:4 sparsity (prune 2 out of every 4)
sgpt.fasterprune(sparsity=0.5, prune_n=2, prune_m=4, blocksize=128, percdamp=0.01)


W_quant = layer.weight.data

# Compute loss
dW = W_quant - W0
loss = ((dW @ H) * dW).sum().item()

print(f"Loss: {loss * N}")
