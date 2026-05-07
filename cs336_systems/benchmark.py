#!/usr/bin/env python3
import argparse
import math
import statistics
import timeit

import torch
import numpy as np

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
    parser.add_argument("--device", type=str, default="mps")

    return parser.parse_args()


def sync(device: str):
    """Synchronize device to ensure accurate timing."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    # mps does not have a synchronize call so we use a dummy tensor operation
    elif device == "mps":
        torch.mps.synchronize()


def run_step(model, optimizer, inputs, targets, mode: str, device: str):
    """Run a single step based on mode."""
    if mode == "forward":
        with torch.no_grad():
            logits = model(inputs)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    elif mode == "forward_backward":
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

    elif mode == "full":
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

    sync(device)
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
    ).to(device)

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
        run_step(model, optimizer, inputs, targets, args.mode, device)

    # measurement steps
    print(f"Running {args.num_steps} measurement steps...")
    timings = []
    for step in range(args.num_steps):
        start = timeit.default_timer()
        loss = run_step(model, optimizer, inputs, targets, args.mode, device)
        end = timeit.default_timer()
        elapsed = end - start
        timings.append(elapsed)
        print(f"  step {step+1:3d} | loss {loss:.4f} | time {elapsed*1000:.2f}ms")

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