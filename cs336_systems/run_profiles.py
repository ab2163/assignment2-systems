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

def run_profile(model_name, config, mode, context_length, device="mps", batch_size=8):
    """Run profile_model.py for a given configuration."""
    output_dir = f"profiles/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "profile_model.py",
        "--mode", mode,
        "--context_length", str(context_length),
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_dir", output_dir,
        "--d_model",    str(config["d_model"]),
        "--d_ff",       str(config["d_ff"]),
        "--num_layers", str(config["num_layers"]),
        "--num_heads",  str(config["num_heads"]),
    ]

    label = f"{model_name} | {mode} | ctx={context_length}"
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            timeout=300,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED with return code {result.returncode}")
            return False
        return True

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: exceeded 300s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print(f"Device: {device} | Batch size: {batch_size}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Modes: {MODES}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")

    # track results
    results = {}
    for model_name, config in MODELS.items():
        for mode in MODES:
            for context_length in CONTEXT_LENGTHS:
                key = (model_name, mode, context_length)
                success = run_profile(
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