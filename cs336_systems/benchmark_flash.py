#!/usr/bin/env python3
"""
Benchmark FlashAttention-2 (Triton) vs standard PyTorch attention.
Uses triton.testing.do_bench for accurate GPU timing.
"""
import torch
import triton
import triton.testing
import math
import itertools
import traceback
from contextlib import nullcontext

from cs336_basics.lang_model import (
    FlashAttentionTriton,
    FlashAttentionPyTorch,
    scaled_dot_product_attention,
)

# benchmark configurations
SEQ_LENS   = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_HEADS    = [16, 32, 64, 128]
PRECISIONS = [torch.bfloat16, torch.float32]
BATCH_SIZE = 1
IS_CAUSAL  = True

def make_inputs(seq_len, d_head, dtype, device="cuda"):
    """Generate random Q, K, V inputs."""
    shape = (BATCH_SIZE, seq_len, d_head)
    Q = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    return Q, K, V

def pytorch_attention(Q, K, V, is_causal):
    """Standard PyTorch attention using our scaled_dot_product_attention."""
    seq_len = Q.shape[1]
    mask = torch.tril(torch.ones(seq_len, seq_len, 
                                  dtype=torch.bool, device=Q.device))
    return scaled_dot_product_attention(Q, K, V, mask=mask if is_causal else None)

def benchmark_forward(fn, Q, K, V, is_causal):
    """Benchmark forward pass only."""
    def fwd():
        with torch.no_grad():
            return fn(Q, K, V, is_causal)
    return triton.testing.do_bench(fwd, warmup=25, rep=100)

def benchmark_backward(fn, Q, K, V, is_causal):
    """Benchmark backward pass only."""
    # run forward once to get output
    Q_ = Q.detach().requires_grad_(True)
    K_ = K.detach().requires_grad_(True)
    V_ = V.detach().requires_grad_(True)

    out = fn(Q_, K_, V_, is_causal)
    dO = torch.randn_like(out)

    def bwd():
        # zero grads
        if Q_.grad is not None: Q_.grad.zero_()
        if K_.grad is not None: K_.grad.zero_()
        if V_.grad is not None: V_.grad.zero_()
        out.backward(dO, retain_graph=True)

    return triton.testing.do_bench(bwd, warmup=25, rep=100)

def benchmark_fwd_bwd(fn, Q, K, V, is_causal):
    """Benchmark end-to-end forward + backward."""
    dO = torch.randn(BATCH_SIZE, Q.shape[1], V.shape[2],
                     dtype=Q.dtype, device=Q.device)

    def fwd_bwd():
        Q_ = Q.detach().requires_grad_(True)
        K_ = K.detach().requires_grad_(True)
        V_ = V.detach().requires_grad_(True)
        out = fn(Q_, K_, V_, is_causal)
        out.backward(dO)

    return triton.testing.do_bench(fwd_bwd, warmup=25, rep=100)

def format_ms(val):
    if val is None:
        return "OOM"
    return f"{val:.3f}"

def main():
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {BATCH_SIZE} | Causal: {IS_CAUSAL}")
    print()

    # implementations to benchmark
    implementations = {
        "triton": lambda Q, K, V, c: FlashAttentionTriton.apply(Q, K, V, c),
        "pytorch": pytorch_attention,
    }

    # header
    col_w = 12
    header = (
        f"{'seq_len':<10} {'d_head':<8} {'dtype':<12} "
        f"{'fwd_triton':>{col_w}} {'fwd_pytorch':>{col_w}} "
        f"{'bwd_triton':>{col_w}} {'bwd_pytorch':>{col_w}} "
        f"{'e2e_triton':>{col_w}} {'e2e_pytorch':>{col_w}} "
        f"{'speedup_fwd':>{col_w}} {'speedup_e2e':>{col_w}}"
    )
    sep = "-" * len(header)

    print(header)
    print(sep)

    for dtype, seq_len, d_head in itertools.product(PRECISIONS, SEQ_LENS, D_HEADS):
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"

        try:
            torch.cuda.empty_cache()
            Q, K, V = make_inputs(seq_len, d_head, dtype, device)

            # benchmark each implementation
            results = {}
            for name, fn in implementations.items():
                try:
                    fwd = benchmark_forward(fn, Q, K, V, IS_CAUSAL)
                    bwd = benchmark_backward(fn, Q, K, V, IS_CAUSAL)
                    e2e = benchmark_fwd_bwd(fn, Q, K, V, IS_CAUSAL)
                    results[name] = (fwd, bwd, e2e)
                except torch.cuda.OutOfMemoryError:
                    results[name] = (None, None, None)
                    torch.cuda.empty_cache()

            t_fwd, t_bwd, t_e2e = results.get("triton", (None, None, None))
            p_fwd, p_bwd, p_e2e = results.get("pytorch", (None, None, None))

            # compute speedups
            speedup_fwd = f"{p_fwd/t_fwd:.2f}x" if t_fwd and p_fwd else "N/A"
            speedup_e2e = f"{p_e2e/t_e2e:.2f}x" if t_e2e and p_e2e else "N/A"

            print(
                f"{seq_len:<10} {d_head:<8} {dtype_str:<12} "
                f"{format_ms(t_fwd):>{col_w}} {format_ms(p_fwd):>{col_w}} "
                f"{format_ms(t_bwd):>{col_w}} {format_ms(p_bwd):>{col_w}} "
                f"{format_ms(t_e2e):>{col_w}} {format_ms(p_e2e):>{col_w}} "
                f"{speedup_fwd:>{col_w}} {speedup_e2e:>{col_w}}"
            )

        except Exception as e:
            print(f"{seq_len:<10} {d_head:<8} {dtype_str:<12} ERROR: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    print(sep)
    print("All times in ms. Speedup = pytorch / triton (higher = triton faster)")

if __name__ == "__main__":
    main()