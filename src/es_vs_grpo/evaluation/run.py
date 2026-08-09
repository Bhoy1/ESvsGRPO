#!/usr/bin/env python3
"""Batch evaluation for sequential ES or GRPO checkpoints.

Iterates through checkpoints in an experiment folder, runs the isolated vLLM
worker on each, and combines the results into performance matrices.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m es_vs_grpo.evaluation.run --folder runs/experiment_name/
"""

import os
import sys
import json
import argparse
import subprocess
from collections import defaultdict


def find_checkpoints(folder):
    """Find all checkpoint folders with metadata.json, sorted by task order."""
    checkpoints = []

    for name in os.listdir(folder):
        ckpt_path = os.path.join(folder, name)
        metadata_path = os.path.join(ckpt_path, "metadata.json")

        if os.path.isdir(ckpt_path) and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            checkpoints.append({
                "path": ckpt_path,
                "name": name,
                "metadata": metadata,
            })

    # Sort by cycle, then task_idx (use -1 as default for baseline which lacks task_idx)
    checkpoints.sort(key=lambda x: (x["metadata"].get("cycle", -1), x["metadata"].get("task_idx", -1)))
    return checkpoints


def run_eval(checkpoint_path, output_path, eval_id, script_path, test_size=2000, verbose_logging=True, num_samples_to_log=100, save_traces=False):
    """Run the isolated evaluation worker on a single checkpoint."""
    cmd = [
        sys.executable, script_path,
        "--checkpoint", checkpoint_path,
        "--output", output_path,
        "--eval_id", str(eval_id),
        "--test_size", str(test_size),
        "--num_samples_to_log", str(num_samples_to_log),
    ]
    if verbose_logging:
        cmd.append("--verbose_logging")
    if save_traces:
        cmd.append("--save_traces")

    print(f"\n{'='*60}")
    print(f"Running eval {eval_id}: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    sys.stdout.flush()

    result = subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    return result.returncode == 0


def print_performance_matrix(matrix, title, tasks, checkpoints):
    """Print performance matrix in formatted table."""
    print(f"\n{title}:")

    # Header
    header = f"{'Task':<20}"
    for ckpt in checkpoints:
        col_name = ckpt["name"][:12]
        header += f"{col_name:<14}"
    print(header)
    print("-" * len(header))

    # Rows
    for task_name in tasks:
        row = f"{task_name:<20}"
        for ckpt in checkpoints:
            val = matrix.get(task_name, {}).get(ckpt["name"], 0.0)
            row += f"{val:<14.3f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for sequential ES or GRPO checkpoints")
    parser.add_argument("--folder", type=str, required=True, help="Experiment folder with checkpoints")
    parser.add_argument("--eval_script", type=str, default=None, help="Path to the evaluation worker (auto-detected if not specified)")
    parser.add_argument("--test_size", type=int, default=2000, help="Maximum test examples per task (MATH is capped at 500)")
    parser.add_argument("--verbose_logging", action="store_true", default=True, help="Enable verbose logging")
    parser.add_argument("--num_samples_to_log", type=int, default=100, help="Number of samples to show detailed logs for")
    parser.add_argument("--save_traces", action="store_true", default=False, help="Save eval traces to eval_traces/ in each checkpoint")
    parser.add_argument("--force", action="store_true", help="Recompute checkpoints that already contain eval_results.json")
    args = parser.parse_args()

    # Find eval script
    if args.eval_script:
        eval_script = args.eval_script
    else:
        eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")

    if not os.path.exists(eval_script):
        print(f"Error: eval script not found: {eval_script}")
        sys.exit(1)

    print(f"[BatchEval] Experiment folder: {args.folder}")
    print(f"[BatchEval] Eval script: {eval_script}")
    print(f"[BatchEval] Settings: test_size={args.test_size}, verbose_logging={args.verbose_logging}, num_samples_to_log={args.num_samples_to_log}")

    # Find checkpoints
    checkpoints = find_checkpoints(args.folder)
    if not checkpoints:
        print(f"Error: No checkpoints found in {args.folder}")
        print("  (Looking for folders with metadata.json)")
        sys.exit(1)

    print(f"[BatchEval] Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        meta = ckpt["metadata"]
        print(f"  - {ckpt['name']} (cycle={meta['cycle']}, task={meta['task_name']})")

    # Run eval on each checkpoint
    eval_results = []
    for eval_id, ckpt in enumerate(checkpoints):
        ckpt_path = ckpt["path"]
        output_path = os.path.join(ckpt_path, "eval_results.json")

        # Check if already evaluated
        if os.path.exists(output_path) and not args.force:
            print(f"\n[BatchEval] Skipping {ckpt['name']} - already evaluated")
            with open(output_path, 'r') as f:
                result = json.load(f)
            eval_results.append({"checkpoint": ckpt, "result": result})
            continue

        # Run eval
        success = run_eval(ckpt_path, output_path, eval_id, eval_script, args.test_size, args.verbose_logging, args.num_samples_to_log, args.save_traces)

        if success and os.path.exists(output_path):
            with open(output_path, 'r') as f:
                result = json.load(f)
            eval_results.append({"checkpoint": ckpt, "result": result})
        else:
            print(f"[BatchEval] Warning: Eval failed for {ckpt['name']}")
            eval_results.append({"checkpoint": ckpt, "result": None})

    # Build performance matrices
    train_matrix = defaultdict(dict)  # task_name -> {ckpt_name -> score}
    test_matrix = defaultdict(dict)
    all_tasks = set()

    for eval_data in eval_results:
        ckpt = eval_data["checkpoint"]
        result = eval_data["result"]

        if result and result.get("success"):
            for task_name, score in result.get("train_results", {}).items():
                train_matrix[task_name][ckpt["name"]] = score
                all_tasks.add(task_name)

            for task_name, score in result.get("test_results", {}).items():
                test_matrix[task_name][ckpt["name"]] = score
                all_tasks.add(task_name)

    all_tasks = sorted(all_tasks)

    # Print matrices
    print(f"\n{'='*60}")
    print("PERFORMANCE MATRICES")
    print(f"{'='*60}")

    print_performance_matrix(train_matrix, "TRAIN Performance Matrix", all_tasks, checkpoints)
    print_performance_matrix(test_matrix, "TEST Performance Matrix", all_tasks, checkpoints)

    # Save matrices
    train_matrix_path = os.path.join(args.folder, "train_performance_matrix.json")
    test_matrix_path = os.path.join(args.folder, "test_performance_matrix.json")

    # Convert defaultdict to regular dict for JSON
    with open(train_matrix_path, 'w') as f:
        json.dump(dict(train_matrix), f, indent=2)
    with open(test_matrix_path, 'w') as f:
        json.dump(dict(test_matrix), f, indent=2)

    # Save combined results
    all_results_path = os.path.join(args.folder, "all_eval_results.json")
    all_results = {
        "evaluation_settings": {
            "test_size_cap": args.test_size,
            "math_test_size_cap": min(args.test_size, 500),
            "tasks": ["countdown", "math", "sciknoweval_chemistry", "boolq"],
        },
        "checkpoints": [
            {
                "name": e["checkpoint"]["name"],
                "metadata": e["checkpoint"]["metadata"],
                "train_results": e["result"].get("train_results", {}) if e["result"] else {},
                "test_results": e["result"].get("test_results", {}) if e["result"] else {},
            }
            for e in eval_results
        ],
        "train_matrix": dict(train_matrix),
        "test_matrix": dict(test_matrix),
    }
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[BatchEval] Results saved:")
    print(f"  - {train_matrix_path}")
    print(f"  - {test_matrix_path}")
    print(f"  - {all_results_path}")
    print(f"\n[BatchEval] Done!")


if __name__ == "__main__":
    main()
