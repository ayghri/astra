import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch import nn
import numpy as np



device = torch.device("cuda")
dtype = torch.bfloat16

k_dim = 128
v_dim = 128
seq_len = 4096
batch_size = 4
num_heads = 8
keys = nn.Parameter(
    torch.randn(batch_size, num_heads, seq_len, k_dim).to(device, dtype=dtype),
    requires_grad=True,
)
values = nn.Parameter(
    torch.randn(batch_size, num_heads, seq_len, v_dim).to(device, dtype=dtype),
    requires_grad=True,
)
queries = nn.Parameter(
    torch.randn(batch_size, num_heads, seq_len, k_dim).to(device, dtype=dtype),
    requires_grad=True,
)
# output shape is (batch_size, num_heads, seq_len, v_dim) = (4, 8, 4096, 128) bf16 = 32 MB
# Attention score shape is (batch_size, num_heads, seq_len, seq_len) = (4, 8, 4096, 4096) = 512 MB


outs = {}


def benchmark_sdpa_kernel():
    import time

    n_iters = 100

    for name, backend in [
        ("flash_attention", SDPBackend.FLASH_ATTENTION),
        ("efficient_attention", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]:
        for is_causal in [False, True]:
            print(f"Benchmarking {name} with is_causal={is_causal}")

            torch.cuda.synchronize()

            start_mem = torch.cuda.memory_allocated()
            forward_times = 0
            backward_times = 0
            forward_mems = []
            backward_mems = []

            with sdpa_kernel(backend):
                for _ in range(n_iters):
                    torch.cuda.empty_cache()
                    start_mem = torch.cuda.memory_allocated()
                    torch.cuda.synchronize()
                    start = time.time()
                    outs[("name", is_causal)] = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=is_causal,
                    )
                    torch.cuda.synchronize()
                    end = time.time()

                    end_mem = torch.cuda.memory_allocated()
                    forward_times += end - start
                    forward_mems.append(end_mem - start_mem)

                    loss = outs[("name", is_causal)].sum()

                    torch.cuda.synchronize()
                    start_mem = torch.cuda.memory_allocated()
                    start = time.time()
                    loss.backward()
                    torch.cuda.synchronize()
                    end = time.time()
                    end_mem = torch.cuda.memory_allocated()

                    backward_mems.append(end_mem - start_mem)
                    backward_times += end - start
                    torch.cuda.empty_cache()

                torch.cuda.synchronize()

                for p in [keys, values, queries]:
                    p.grad = None

                print(
                    f"{name} forward time: {forward_times / n_iters:.4f} seconds per iteration"
                )
                print(
                    f"{name} forward memory usage: {np.max(forward_mems) / (1024**2):.2f} MB"
                )
                print(
                    f"{name} backward time: {backward_times / n_iters:.4f} seconds per iteration"
                )
                print(
                    f"{name} backward memory usage: {np.max(backward_mems) / (1024**2):.2f} MB"
                )

                torch.cuda.empty_cache()
                print("-" * 25)
        print("=" * 50)

    for name, backend in [
        ("flash_attention", SDPBackend.FLASH_ATTENTION),
        ("efficient_attention", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]:
        for is_causal in [False, True]:
            print(f"Benchmarking {name} with is_causal={is_causal}")

            torch.cuda.synchronize()

            start_mem = torch.cuda.memory_allocated()
            forward_times = 0
            forward_mems = []

            with sdpa_kernel(backend):
                for _ in range(n_iters):
                    torch.cuda.empty_cache()
                    start_mem = torch.cuda.memory_allocated()
                    torch.cuda.synchronize()
                    start = time.time()
                    with torch.no_grad():
                        outs[("name", is_causal)] = (
                            scaled_dot_product_attention(
                                queries,
                                keys,
                                values,
                                attn_mask=None,
                                dropout_p=0.0,
                                is_causal=is_causal,
                            )
                        )
                    torch.cuda.synchronize()
                    end = time.time()

                    end_mem = torch.cuda.memory_allocated()
                    forward_times += end - start
                    forward_mems.append(end_mem - start_mem)

                    torch.cuda.empty_cache()

                torch.cuda.synchronize()

                for p in [keys, values, queries]:
                    p.grad = None

                print(
                    f"{name} forward time: {forward_times / n_iters:.4f} seconds per iteration"
                )
                print(
                    f"{name} forward memory usage: {np.max(forward_mems) / (1024**2):.2f} MB"
                )

                torch.cuda.empty_cache()
                print("-" * 25)
        print("=" * 50)


if __name__ == "__main__":
    benchmark_sdpa_kernel()
