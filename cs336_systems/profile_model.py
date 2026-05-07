#!/usr/bin/env python3
import argparse
import math
import torch
import time
from torch.profiler import profile, ProfilerActivity, record_function
from cs336_basics.lang_model import TransformerLM, AdamW, cross_entropy

def parse_args():
    parser = argparse.ArgumentParser(description="Profile Transformer LM")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--mode", type=str,
                        choices=["forward", "forward_backward", "full"],
                        default="forward")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output_dir", type=str, default="./profiles")
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device

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

    optimizer = AdamW(model.parameters(), lr=3e-4)

    # random batch
    inputs = torch.randint(0, args.vocab_size,
                           (args.batch_size, args.context_length),
                           device=device)
    targets = torch.randint(0, args.vocab_size,
                            (args.batch_size, args.context_length),
                            device=device)

    # warmup
    print("Warming up...")
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if args.mode in ["forward_backward", "full"]:
            loss.backward()
        if args.mode == "full":
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()

    # profile
    print(f"Profiling mode: {args.mode}")
    activities = [ProfilerActivity.CPU]

    # time the whole block with perf_counter
    start_time = time.perf_counter()

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        optimizer.zero_grad()

        logits = model(inputs)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        if args.mode in ["forward_backward", "full"]:
            loss.backward()

        if args.mode == "full":
            optimizer.step()

        if device == "mps":
            torch.mps.synchronize()

    end_time = time.perf_counter()
    total_wall_time = (end_time - start_time) * 1000  # convert to ms

    # print results
    print("\n" + "=" * 80)
    print(f"Profiling Results — mode={args.mode}, "
          f"d_model={args.d_model}, context_length={args.context_length}")
    print("=" * 80)

    # (a/b/c) total time and operator breakdown
    print(f"\nTotal wall-clock time: {total_wall_time:.2f}ms")
    print("\nOperator summary (sorted by CPU time):")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=100
    ))

    # (d) matrix multiplication fraction of total time
    print("\nMatrix multiplication fraction of total time:")

    # get all aten events sorted by time
    all_aten = sorted(
        [e for e in prof.key_averages() if e.key.startswith("aten::")],
        key=lambda x: x.cpu_time_total,
        reverse=True
    )
    total_aten_time = sum(e.cpu_time_total for e in all_aten)

    # matmul fraction
    #matmul_ops = ["aten::mm", "aten::bmm", "aten::matmul", "aten::linear", "aten::addmm"]
    matmul_ops = ["aten::bmm"]
    matmul_time = sum(e.cpu_time_total for e in all_aten if e.key in matmul_ops)

    print(f"\n  Total aten time:  {total_aten_time/1000:.2f}ms")
    print(f"  Matmul time:      {matmul_time/1000:.2f}ms  "
        f"({100*matmul_time/total_aten_time:.1f}% of total)")
    
    # (e) softmax vs matmul in self-attention
    print("\n[e] Self-attention softmax vs matmul:")
    key_avgs = prof.key_averages()

    regions = ["attn_scores_matmul", "attn_softmax", "attn_values_matmul"]
    times = {}
    for region in regions:
        events = [e for e in key_avgs if e.key == region]
        times[region] = sum(e.cpu_time_total for e in events)
        count = sum(e.count for e in events)
        print(f"  {region:<25} {times[region]/1000:.2f}ms  count: {count}")

    # matmul vs softmax comparison
    matmul_time = times["attn_scores_matmul"] + times["attn_values_matmul"]
    softmax_time = times["attn_softmax"]

    if softmax_time > 0:
        print(f"\n  Runtime ratio (matmul/softmax): {matmul_time/softmax_time:.1f}x")

    # manual FLOPs calculation
    # scores matmul: 2 * batch * heads * seq * seq * d_k
    scores_flops = 2 * args.batch_size * args.num_heads * args.context_length ** 2 * (args.d_model // args.num_heads)
    # values matmul: same
    values_flops = scores_flops
    # softmax FLOPs: ~3 * batch * heads * seq * seq (exp + sum + divide)
    softmax_flops = 3 * args.batch_size * args.num_heads * args.context_length ** 2

    total_matmul_flops = scores_flops + values_flops
    flops_ratio = total_matmul_flops / softmax_flops

    print(f"\n  Matmul FLOPs:  {total_matmul_flops:,}")
    print(f"  Softmax FLOPs: {softmax_flops:,}")
    print(f"  FLOPs ratio (matmul/softmax): {flops_ratio:.1f}x")
    print(f"\n  If runtime ratio << FLOPs ratio → softmax is memory-bandwidth bound")

if __name__ == "__main__":
    main()