#!/usr/bin/env python3
"""
Benchmark scaled_dot_product_attention at different scales.
Compares uncompiled vs compiled versions.
Fixes batch_size=8, no multi-head (single head).
Iterates over d_k in [16, 32, 64, 128] and seq_len in [256, 1024, 4096, 8192, 16384].
"""
import argparse
import timeit
import torch
import itertools
from contextlib import nullcontext
from cs336_basics.lang_model import scaled_dot_product_attention

D_KS    = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
BATCH_SIZE = 8
WARMUP_STEPS = 5
NUM_STEPS = 100

def sync():
    torch.cuda.synchronize()

def benchmark_attention(d_k, seq_len, device):
    """Benchmark forward and backward pass of attention for given d_k and seq_len."""

    # create random Q, K, V — no head dimension (single head)
    # shape: (batch, seq_len, d_k)
    Q = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)

    # causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # warmup
    for _ in range(WARMUP_STEPS):
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        out.sum().backward()
        Q.grad = K.grad = V.grad = None
        sync()

    # --- time forward pass ---
    forward_times = []
    for _ in range(NUM_STEPS):
        start = timeit.default_timer()
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        sync()
        end = timeit.default_timer()
        forward_times.append(end - start)
        Q.grad = K.grad = V.grad = None

    # --- measure memory before backward ---
    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    sync()
    memory_before_backward = torch.cuda.memory_allocated()

    # --- time backward pass ---
    backward_times = []
    for _ in range(NUM_STEPS):
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        sync()
        start = timeit.default_timer()
        out.sum().backward()
        sync()
        end = timeit.default_timer()
        backward_times.append(end - start)
        Q.grad = K.grad = V.grad = None

    fwd_mean = sum(forward_times) / len(forward_times) * 1000
    fwd_std  = torch.tensor(forward_times).std().item() * 1000
    bwd_mean = sum(backward_times) / len(backward_times) * 1000
    bwd_std  = torch.tensor(backward_times).std().item() * 1000

    return fwd_mean, fwd_std, bwd_mean, bwd_std, memory_before_backward

def main():
    parser = argparse.ArgumentParser(description="Benchmark attention")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE} | Warmup: {WARMUP_STEPS} | Steps: {NUM_STEPS}")

    # compiled version
    compiled_attention = torch.compile(scaled_dot_product_attention)

    versions = [
        ("uncompiled", scaled_dot_product_attention),
        ("compiled",   compiled_attention),
    ]

    for version_name, attn_fn in versions:
        print(f"\n{'='*90}")
        print(f"Version: {version_name}")
        print(f"{'='*90}")
        print(f"{'d_k':<6} {'seq_len':<10} {'fwd (ms)':<20} {'bwd (ms)':<20} {'mem before bwd (MB)':<22}")
        print("-" * 90)

        for d_k, seq_len in itertools.product(D_KS, SEQ_LENS):
            try:
                # free memory between runs
                torch.cuda.empty_cache()

                fwd_mean, fwd_std, bwd_mean, bwd_std, mem = benchmark_attention(
                    d_k, seq_len, device
                )
                mem_mb = mem / (1024 ** 2)

                print(
                    f"{d_k:<6} {seq_len:<10} "
                    f"{fwd_mean:.2f} ± {fwd_std:.2f}ms  "
                    f"{bwd_mean:.2f} ± {bwd_std:.2f}ms  "
                    f"{mem_mb:.1f} MB"
                )

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{d_k:<6} {seq_len:<10} {'OOM':^16} {'OOM':^16} {'OOM':^22}")

            except Exception as e:
                print(f"{d_k:<6} {seq_len:<10} ERROR: {e}")

if __name__ == "__main__":
    main()