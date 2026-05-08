#!/usr/bin/env python3
import argparse
import math
import time
import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.lang_model import TransformerLM, AdamW, cross_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Transformer LM with nsys")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--mode", type=str,
                        choices=["forward", "forward_backward", "full"],
                        default="forward")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def run_step(model, optimizer, inputs, targets, mode, device):
    """Run a single annotated step."""

    nvtx.range_push("step")

    nvtx.range_push("zero_grad")
    optimizer.zero_grad()
    nvtx.range_pop()

    nvtx.range_push("forward")
    logits = model(inputs)
    loss = cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    nvtx.range_pop()  # forward

    if mode in ["forward_backward", "full"]:
        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()  # backward

    if mode == "full":
        nvtx.range_push("optimizer_step")
        optimizer.step()
        nvtx.range_pop()  # optimizer_step

    torch.cuda.synchronize()
    nvtx.range_pop()  # step

    return loss.item()


def main():
    args = parse_args()
    device = args.device

    print(f"Device: {device} | Mode: {args.mode}")

    # build model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = AdamW(model.parameters(), lr=3e-4)

    # random batch
    inputs = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
        dtype=torch.long
    )
    targets = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
        dtype=torch.long
    )

    if args.mode == "forward":
        model.eval()
    else:
        model.train()

    # warmup steps — not annotated so nsys doesn't capture them
    print(f"Running {args.warmup_steps} warmup steps...")
    for _ in range(args.warmup_steps):
        run_step(model, optimizer, inputs, targets, args.mode, device)

    # measurement steps — annotated with NVTX
    print(f"Running {args.num_steps} profiled steps...")
    torch.cuda.cudart().cudaProfilerStart()  # tell nsys to start capturing

    for step in range(args.num_steps):
        nvtx.range_push(f"measurement_step_{step}")
        loss = run_step(model, optimizer, inputs, targets, args.mode, device)
        nvtx.range_pop()
        print(f"  step {step+1} | loss {loss:.4f}")

    torch.cuda.cudart().cudaProfilerStop()  # tell nsys to stop capturing
    print("Done.")


if __name__ == "__main__":
    main()