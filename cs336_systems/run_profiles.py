#!/usr/bin/env python3
import subprocess
import sys
import os

MODELS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
}

MODES = ["forward", "forward_backward", "full"]
CONTEXT_LENGTHS = [256, 512, 1024]

def run_nsys_profile(model_name, config, mode, context_length, device="cuda", batch_size=8):
    output_dir = f"nsys_profiles/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    output_name = f"{output_dir}/{mode}_ctx{context_length}"

    # nsys command
    nsys_cmd = [
        "nsys", "profile",
        "--output", output_name,
        "--trace", "cuda,nvtx",   # capture CUDA kernels, NVTX ranges
        "--capture-range", "cudaProfilerApi",  # only capture between cudaProfilerStart/Stop
        "--capture-range-end", "stop",
        "--force-overwrite", "true",
    ]

    # python command
    python_cmd = [
        "python", "profile_model.py",
        "--mode", mode,
        "--context_length", str(context_length),
        "--batch_size", str(batch_size),
        "--device", device,
        "--warmup_steps", "5",
        "--num_steps", "10",
        "--d_model",    str(config["d_model"]),
        "--d_ff",       str(config["d_ff"]),
        "--num_layers", str(config["num_layers"]),
        "--num_heads",  str(config["num_heads"]),
    ]

    cmd = nsys_cmd + python_cmd
    label = f"{model_name} | {mode} | ctx={context_length}"
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Output:  {output_name}.nsys-rep")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, timeout=600, text=True)
        if result.returncode != 0:
            print(f"  FAILED with return code {result.returncode}")
            return False
        print(f"  SUCCESS — open {output_name}.nsys-rep in Nsight Systems GUI")
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    results = {}
    for model_name, config in MODELS.items():
        for mode in MODES:
            for context_length in CONTEXT_LENGTHS:
                key = (model_name, mode, context_length)
                success = run_nsys_profile(
                    model_name, config, mode,
                    context_length, device, batch_size
                )
                results[key] = "OK" if success else "FAILED"

    # summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Mode':<20} {'Context':<10} {'Status'}")
    print(f"{'-'*60}")
    for (model_name, mode, context_length), status in results.items():
        print(f"{model_name:<10} {mode:<20} {context_length:<10} {status}")


if __name__ == "__main__":
    main()