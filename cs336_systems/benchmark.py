#!/usr/bin/env python3
import argparse
import math
import statistics
import timeit
import torch
import numpy as np
from contextlib import nullcontext
from cs336_basics.lang_model import TransformerLM, AdamW, cross_entropy

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer LM")

    # model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # benchmarking
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward", "forward_backward", "full"],
        default="full",
        help="forward: forward only, forward_backward: forward+backward, full: forward+backward+optimizer step",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", action="store_true", 
        help="Use bf16 mixed precision via torch.autocast")
    parser.add_argument("--memory_profile", action="store_true",
        help="Run memory profiler and save snapshot")
    parser.add_argument("--memory_profile_path", type=str, 
        default="memory_snapshot.pickle",
        help="Output path for memory snapshot")
    
    # checkpointing
    parser.add_argument("--use_checkpoint", action="store_true",
        help="Use activation checkpointing per transformer block")
    parser.add_argument("--checkpoint_blocks", type=int, default=1,
        help="Number of blocks per checkpoint (1=per block, 2=every 2 blocks etc.)")
    
    # compilation
    parser.add_argument("--compile", action="store_true",
        help="Compile model with torch.compile")
    parser.add_argument("--compile_backend", type=str, default="inductor",
        help="Backend for torch.compile (inductor, aot_eager etc.)")

    return parser.parse_args()

def sync(device: str):
    """Synchronize device to ensure accurate timing."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    # mps does not have a synchronize call so we use a dummy tensor operation
    elif device == "mps":
        torch.mps.synchronize()

def run_step(model, optimizer, inputs, targets, mode: str, device: str, mixed_precision: bool = False, memory_profile=False, 
             memory_profile_path="memory_snapshot.pickle"):
    """Run a single step based on mode."""

    # use autocast if mixed precision, otherwise nullcontext (no-op)
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=torch.bfloat16, cache_enabled=False)
        if mixed_precision
        else nullcontext()
    )

    if memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    if mode == "forward":
        with torch.no_grad():
            with autocast_ctx:
                logits = model(inputs)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    elif mode == "forward_backward":
        optimizer.zero_grad()
        with autocast_ctx:
            logits = model(inputs)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

    elif mode == "full":
        optimizer.zero_grad()
        with autocast_ctx:
            logits = model(inputs)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

    sync(device)

    if memory_profile:
        torch.cuda.memory._dump_snapshot(memory_profile_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot saved to {memory_profile_path}")

    return loss.item()

def main():
    args = parse_args()
    device = args.device

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # build model
    d_ff = args.d_ff or (math.ceil((8 / 3 * args.d_model) / 64) * 64)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        theta=args.rope_theta,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    # compile if requested
    if args.compile:
        print(f"Compiling model with backend={args.compile_backend}...")
        model = torch.compile(model, backend=args.compile_backend)
        print("Compilation done.")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4)

    # generate random batch
    inputs = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=device
    )
    targets = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=device
    )

    # set model to appropriate mode
    if args.mode == "forward":
        model.eval()
    else:
        model.train()

    # warmup steps
    print(f"\nRunning {args.warmup_steps} warmup steps...")
    for _ in range(args.warmup_steps):
        run_step(model, optimizer, inputs, targets, args.mode, device, args.mixed_precision)

    # measurement steps
    print(f"Running {args.num_steps} measurement steps...")
    timings = []
    for step in range(args.num_steps):
        memory_profile = args.memory_profile and step == 0 # only profile first step
        memory_profile_path = args.memory_profile_path if memory_profile else "memory_snapshot.pickle"
        start = timeit.default_timer()
        loss = run_step(model, optimizer, inputs, targets, args.mode, device, args.mixed_precision, memory_profile=memory_profile,
            memory_profile_path=args.memory_profile_path,)
        end = timeit.default_timer()
        elapsed = end - start
        timings.append(elapsed)
        print(f"  step {step+1:3d} | loss {loss:.4f} | time {elapsed*1000:.2f}ms")
    
    print(f"Compiled: {'yes (' + args.compile_backend + ')' if args.compile else 'no'}")
    print(f"Precision: {'bf16 mixed' if args.mixed_precision else 'fp32 full'}")

    # summary statistics
    mean_time = statistics.mean(timings)
    std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(f"\nResults ({args.mode}):")
    print(f"  mean: {mean_time*1000:.2f}ms")
    print(f"  std:  {std_time*1000:.2f}ms")
    print(f"  min:  {min(timings)*1000:.2f}ms")
    print(f"  max:  {max(timings)*1000:.2f}ms")

if __name__ == "__main__":
    main()