#!/usr/bin/env python3
import subprocess
import re
import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODELS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    #"large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    #"xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

MODES = ["forward", "forward_backward", "full"]

PRECISIONS = [False, True] # False = fp32 and True = bf16 mixed

def run_benchmark(model_name, config, mode, device="cuda", mixed_precision=False, mixed_precision, memory_profile=False):
    """Run benchmark.py for a given model config and mode, return mean and std in ms."""

    output_name = f"memory_profiles/memory_{model_name}_{mode}.pickle"
    cmd = [
        "python", "benchmark.py",
        "--mode", mode,
        "--warmup_steps", "1",
        "--num_steps", "10",
        "--device", device,
        "--d_model",    str(config["d_model"]),
        "--d_ff",       str(config["d_ff"]),
        "--num_layers", str(config["num_layers"]),
        "--num_heads",  str(config["num_heads"]),
    ]
    if mixed_precision:
        cmd.append("--mixed_precision")
    if memory_profile:
        cmd.extend(["--memory_profile", "--memory_profile_path", output_name])

    precision_str = "bf16" if mixed_precision else "fp32"
    print(f"  Running {model_name} / {mode} / {precision_str}...", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout

        # parse mean and std from output
        mean_match = re.search(r"mean:\s+([\d.]+)ms", output)
        std_match  = re.search(r"std:\s+([\d.]+)ms",  output)

        if mean_match and std_match:
            return float(mean_match.group(1)), float(std_match.group(1))
        else:
            print(f"    WARNING: could not parse output for {model_name}/{mode}")
            print(f"    stdout: {output[-500:]}")
            print(f"    stderr: {result.stderr[-200:]}")
            return None, None

    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: {model_name}/{mode} exceeded 600s")
        return None, None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None

def format_cell(mean, std):
    """Format a mean/std pair as a table cell."""
    if mean is None:
        return "OOM/ERR"
    return f"{mean:.1f} ± {std:.1f}"

def print_table(results):
    """Print results as a formatted table."""
    col_width = 22
    name_width = 10

    # header
    header = f"{'Model':<{name_width}}"
    for mode in MODES:
        label = mode.replace("_", "+")
        header += f"  {label:^{col_width}}"
    print("\n" + "=" * (name_width + (col_width + 2) * len(MODES)))
    print("Benchmark Results (mean ± std in ms, warmup=5, steps=10)")
    print("=" * (name_width + (col_width + 2) * len(MODES)))
    print(header)
    print("-" * (name_width + (col_width + 2) * len(MODES)))

    # rows
    for model_name in MODELS:
        row = f"{model_name:<{name_width}}"
        for mode in MODES:
            mean, std = results.get((model_name, mode), (None, None))
            cell = format_cell(mean, std)
            row += f"  {cell:^{col_width}}"
        print(row)

    print("=" * (name_width + (col_width + 2) * len(MODES)))

def print_comparison_table(results):
    col_width = 20
    name_width = 10

    print(f"\n{'='*80}")
    print("Mixed Precision Benchmark Results (mean ± std in ms)")
    print(f"{'='*80}")

    for mode in MODES:
        print(f"\nMode: {mode}")
        print(f"{'Model':<{name_width}} {'FP32':^{col_width}} {'BF16 Mixed':^{col_width}} {'Speedup':^10}")
        print("-" * (name_width + col_width * 2 + 10))

        for model_name in MODELS:
            fp32_mean, fp32_std = results.get((model_name, mode, False), (None, None))
            bf16_mean, bf16_std = results.get((model_name, mode, True), (None, None))

            if fp32_mean and bf16_mean:
                speedup = fp32_mean / bf16_mean
                fp32_str = f"{fp32_mean:.1f} ± {fp32_std:.1f}"
                bf16_str = f"{bf16_mean:.1f} ± {bf16_std:.1f}"
                print(f"{model_name:<{name_width}} {fp32_str:^{col_width}} {bf16_str:^{col_width}} {speedup:^10.2f}")
            else:
                print(f"{model_name:<{name_width}} {'OOM/ERR':^{col_width}} {'OOM/ERR':^{col_width}} {'N/A':^10}")

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    print(f"Running benchmarks on device: {device}")
    print("This may take a while for larger models...\n")

    results = {}
    for model_name, config in MODELS.items():
        print(f"\n--- {model_name} ---")
        for mode in MODES:
            for mixed_precision in PRECISIONS:
                key = (model_name, mode, mixed_precision)
                mean, std = run_benchmark(model_name, config, mode, device, mixed_precision)
                results[key] = (mean, std)

    print_comparison_table(results)

if __name__ == "__main__":
    main()