#!/usr/bin/env python3
"""
Benchmark all-reduce operation across different data sizes and number of processes.
Debug locally with Gloo on CPU, benchmark with NCCL on GPU.

Usage:
  # CPU debug (Gloo)
  python benchmark_allreduce.py --backend gloo --world_size 2

  # GPU benchmark (NCCL) - run with torchrun
  torchrun --nproc_per_node=2 benchmark_allreduce.py --backend nccl
  torchrun --nproc_per_node=4 benchmark_allreduce.py --backend nccl
  torchrun --nproc_per_node=6 benchmark_allreduce.py --backend nccl
"""
import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# data sizes in bytes
DATA_SIZES = {
    "1MB":   1 * 1024 * 1024,
    "10MB":  10 * 1024 * 1024,
    "100MB": 100 * 1024 * 1024,
    "1GB":   1024 * 1024 * 1024,
}

WARMUP_STEPS = 5
BENCH_STEPS  = 10

def setup(rank, world_size, backend, master_port="29500"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def sync(device):
    """Synchronize device."""
    if device.type == "cuda":
        torch.cuda.synchronize()

def benchmark_allreduce(rank, world_size, backend, results_queue=None):
    """Run all-reduce benchmark for all data sizes."""
    setup(rank, world_size, backend)

    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")

    if rank == 0:
        print(f"\nBackend: {backend} | World size: {world_size} | Device: {device}")
        print(f"Warmup: {WARMUP_STEPS} | Measurement: {BENCH_STEPS}")
        print(f"\n{'Size':<10} {'World Size':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 70)

    local_results = {}

    for size_name, size_bytes in DATA_SIZES.items():
        # number of float32 elements
        n_elements = size_bytes // 4
        data = torch.zeros(n_elements, dtype=torch.float32, device=device)

        # warmup
        for _ in range(WARMUP_STEPS):
            dist.all_reduce(data, async_op=False)
            sync(device)

        # measurement
        timings = []
        for _ in range(BENCH_STEPS):
            # reset data each step
            data.fill_(float(rank))
            sync(device)

            start = time.perf_counter()
            dist.all_reduce(data, async_op=False)
            sync(device)
            end = time.perf_counter()

            timings.append((end - start) * 1000)  # ms

        # collect timings from all ranks using all_gather_object
        all_timings = [None] * world_size
        dist.all_gather_object(all_timings, timings)

        if rank == 0:
            # flatten all timings across ranks
            flat_timings = [t for rank_timings in all_timings for t in rank_timings]
            mean_t = sum(flat_timings) / len(flat_timings)
            std_t  = (sum((t - mean_t)**2 for t in flat_timings) / len(flat_timings)) ** 0.5
            min_t  = min(flat_timings)
            max_t  = max(flat_timings)

            local_results[size_name] = {
                "mean": mean_t, "std": std_t,
                "min": min_t, "max": max_t,
                "world_size": world_size,
            }

            print(
                f"{size_name:<10} {world_size:<12} "
                f"{mean_t:<12.2f} {std_t:<12.2f} "
                f"{min_t:<12.2f} {max_t:<12.2f}"
            )

    if results_queue is not None and rank == 0:
        results_queue.put((world_size, local_results))

    cleanup()

def run_with_spawn(world_size, backend):
    """Launch benchmark using mp.spawn (for Gloo/CPU debugging)."""
    import multiprocessing
    results_queue = multiprocessing.Manager().Queue()
    mp.spawn(
        fn=benchmark_allreduce,
        args=(world_size, backend, results_queue),
        nprocs=world_size,
        join=True,
    )
    results = {}
    while not results_queue.empty():
        ws, res = results_queue.get()
        results[ws] = res
    return results

def run_with_torchrun(backend):
    """Launch benchmark using torchrun (for NCCL/GPU)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")

    if rank == 0:
        print(f"\nBackend: {backend} | World size: {world_size} | Device: {device}")
        print(f"Warmup: {WARMUP_STEPS} | Measurement: {BENCH_STEPS}")
        print(f"\n{'Size':<10} {'World Size':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 70)

    for size_name, size_bytes in DATA_SIZES.items():
        n_elements = size_bytes // 4
        data = torch.zeros(n_elements, dtype=torch.float32, device=device)

        # warmup
        for _ in range(WARMUP_STEPS):
            dist.all_reduce(data, async_op=False)
            if backend == "nccl":
                torch.cuda.synchronize()

        # measurement
        timings = []
        for _ in range(BENCH_STEPS):
            data.fill_(float(rank))
            if backend == "nccl":
                torch.cuda.synchronize()

            start = time.perf_counter()
            #print("Before allreduce")
            dist.all_reduce(data, async_op=False)
            #print("After allreduce")
            if backend == "nccl":
                torch.cuda.synchronize()
            end = time.perf_counter()

            timings.append((end - start) * 1000)

        # instead of gathering across ranks, just use rank 0's timings
        # bugfix: avoid using "all_gather_object"
        if rank == 0:
            mean_t = sum(timings) / len(timings)
            std_t  = (sum((t - mean_t)**2 for t in timings) / len(timings)) ** 0.5
            min_t  = min(timings)
            max_t  = max(timings)
            print(
                f"{size_name:<10} {world_size:<12} "
                f"{mean_t:<12.5f} {std_t:<12.5f} "
                f"{min_t:<12.5f} {max_t:<12.5f}"
        )
        
        dist.barrier()
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Benchmark all-reduce")
    parser.add_argument("--backend", type=str, default="gloo",
                        choices=["gloo", "nccl"],
                        help="Communication backend")
    parser.add_argument("--world_size", type=int, default=2,
                        help="Number of processes (only used with mp.spawn/gloo)")
    args = parser.parse_args()

    # detect if launched with torchrun
    if "RANK" in os.environ:
        run_with_torchrun(args.backend)
    else:
        # use mp.spawn for local debugging
        for world_size in [2, 4, 6] if args.world_size == 0 else [args.world_size]:
            run_with_spawn(world_size, args.backend)

if __name__ == "__main__":
    main()
