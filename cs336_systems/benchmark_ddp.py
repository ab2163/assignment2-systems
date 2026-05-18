#!/usr/bin/env python3
"""
Benchmark naive DDP training step for XL model.
Measures total time per step and time spent on gradient communication.

Usage:
    torchrun --nproc_per_node=2 benchmark_ddp.py
"""
import os
import math
import time
import torch
import torch.distributed as dist
from torch.profiler import record_function

from cs336_basics.lang_model import (
    TransformerLM,
    cross_entropy,
    AdamW,
)

from cs336_systems.ddp import NaiveDDP

# XL model config (Section 2.1.2)
XL_CONFIG = {
    "d_model":    2560,
    "d_ff":       10240,
    "num_layers": 32,
    "num_heads":  32,
    "vocab_size": 32000,
    "context_length": 512,
    "theta":      10000.0,
}

BATCH_SIZE   = 4
WARMUP_STEPS = 3
BENCH_STEPS  = 5
LR           = 3e-4

def sync():
    torch.cuda.synchronize()

def benchmark_ddp(rank, world_size):
    # setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\nXL Model DDP Benchmark")
        print(f"World size: {world_size} | Batch size: {BATCH_SIZE} | Context: {XL_CONFIG['context_length']}")
        print(f"Warmup: {WARMUP_STEPS} | Measurement: {BENCH_STEPS}")

    # random inputs — each rank gets BATCH_SIZE // world_size examples
    local_batch = BATCH_SIZE // world_size
    inputs = torch.randint(0, XL_CONFIG["vocab_size"],
                           (local_batch, XL_CONFIG["context_length"]),
                           device=device)
    targets = torch.randint(0, XL_CONFIG["vocab_size"],
                            (local_batch, XL_CONFIG["context_length"]),
                            device=device)

    def aggregate(times):
        t = torch.tensor(times, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
        return t.mean().item(), t.std().item(), t.min().item(), t.max().item()

    for flat in [False, True]:
        label = "flat" if flat else "individual"

        # rebuild model and optimizer fresh for each variant
        model = TransformerLM(
            vocab_size=XL_CONFIG["vocab_size"],
            context_length=XL_CONFIG["context_length"],
            d_model=XL_CONFIG["d_model"],
            num_layers=XL_CONFIG["num_layers"],
            num_heads=XL_CONFIG["num_heads"],
            d_ff=XL_CONFIG["d_ff"],
            theta=XL_CONFIG["theta"],
        ).to(device)

        ddp_model = NaiveDDP(model, flat_gradients=flat)
        optimizer = AdamW(ddp_model.parameters(), lr=LR)

        def run_step(measure_comm=False):
            """Run one full training step, optionally measuring comm time."""
            optimizer.zero_grad()

            # forward pass
            logits = ddp_model(inputs)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # backward pass
            loss.backward()

            # measure communication time separately
            comm_time = 0.0
            if measure_comm:
                sync()
                comm_start = time.perf_counter()
                ddp_model.after_backward()
                sync()
                comm_time = (time.perf_counter() - comm_start) * 1000
            else:
                ddp_model.after_backward()

            # optimizer step
            optimizer.step()
            sync()

            return loss.item(), comm_time

        # warmup
        if rank == 0:
            print(f"\nWarming up ({label})...")
        for _ in range(WARMUP_STEPS):
            run_step(measure_comm=False)

        # benchmark total step time
        if rank == 0:
            print(f"Benchmarking total step time ({label})...")
        total_times = []
        for _ in range(BENCH_STEPS):
            sync()
            start = time.perf_counter()
            loss, _ = run_step(measure_comm=False)
            sync()
            total_times.append((time.perf_counter() - start) * 1000)

        # benchmark communication time
        if rank == 0:
            print(f"Benchmarking communication time ({label})...")
        comm_times = []
        for _ in range(BENCH_STEPS):
            _, comm_time = run_step(measure_comm=True)
            comm_times.append(comm_time)

        # aggregate and print results
        total_mean, total_std, total_min, total_max = aggregate(total_times)
        comm_mean, comm_std, comm_min, comm_max     = aggregate(comm_times)

        if rank == 0:
            comm_pct = (comm_mean / total_mean) * 100
            print(f"\n{'='*60}")
            print(f"Results ({label} gradients, world_size={world_size})")
            print(f"{'='*60}")
            print(f"\nTotal step time:")
            print(f"  mean: {total_mean:.2f}ms | std: {total_std:.2f}ms | "
                  f"min: {total_min:.2f}ms | max: {total_max:.2f}ms")
            print(f"\nGradient communication time (after_backward):")
            print(f"  mean: {comm_mean:.2f}ms | std: {comm_std:.2f}ms | "
                  f"min: {comm_min:.2f}ms | max: {comm_max:.2f}ms")
            print(f"\nCommunication as % of total: {comm_pct:.1f}%")
            print(f"Compute time (total - comm): {total_mean - comm_mean:.2f}ms")
            print(f"{'='*60}")

    dist.destroy_process_group()

def main():
    # detect torchrun or use mp.spawn
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        benchmark_ddp(rank, world_size)
    else:
        import torch.multiprocessing as mp
        world_size = 2
        mp.spawn(
            benchmark_ddp,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

if __name__ == "__main__":
    main()