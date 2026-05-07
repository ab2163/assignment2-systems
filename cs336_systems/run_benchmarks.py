#!/usr/bin/env python3
import subprocess
import re
import sys

MODELS = {
    "small": {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium":{"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
}

MODES = ["forward", "forward_backward", "full"]

def run_benchmark(model_name, config, mode, device="mps"):
    """Run benchmark.py for a given model config and mode, return mean and std in ms."""
    cmd = [
        "uv", "run", "python", "benchmark.py",
        "--mode", mode,
        "--warmup_steps", "1",
        "--num_steps", "10",
        "--device", device,
        "--d_model",    str(config["d_model"]),
        "--d_ff",       str(config["d_ff"]),
        "--num_layers", str(config["num_layers"]),
        "--num_heads",  str(config["num_heads"]),
    ]

    print(f"  Running {model_name} / {mode}...", flush=True)
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


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    print(f"Running benchmarks on device: {device}")
    print("This may take a while for larger models...\n")

    results = {}
    for model_name, config in MODELS.items():
        print(f"\n--- {model_name} ---")
        for mode in MODES:
            mean, std = run_benchmark(model_name, config, mode, device)
            results[(model_name, mode)] = (mean, std)

    print_table(results)


if __name__ == "__main__":
    main()