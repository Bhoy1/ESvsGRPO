#!/usr/bin/env python3
"""Isolated vLLM worker for sequential checkpoint evaluation.

The worker evaluates the four paper tasks on both train and test splits and
writes one checkpoint-level JSON result. It runs in a subprocess so GPU memory
is released between checkpoints.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m es_vs_grpo.evaluation.worker --checkpoint /path/to/ckpt --output /path/to/results.json --eval_id 0
"""

import os
import sys
import json
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--eval_id", type=int, required=True, help="Evaluation ID")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--test_size", type=int, default=2000, help="Maximum test examples per task (MATH is capped at 500)")
    parser.add_argument("--verbose_logging", action="store_true", default=True, help="Enable verbose logging")
    parser.add_argument("--num_samples_to_log", type=int, default=100, help="Number of samples to show detailed logs for")
    parser.add_argument("--save_traces", action="store_true", default=False, help="Save eval traces (per-sample responses/rewards) to eval_traces/ dir")
    args = parser.parse_args()

    print(f"[EvalWorker] Starting eval {args.eval_id}")
    print(f"[EvalWorker] Checkpoint: {args.checkpoint}")
    print(f"[EvalWorker] Output: {args.output}")
    print(f"[EvalWorker] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"[EvalWorker] Settings: test_size={args.test_size}, max_new_tokens={args.max_new_tokens}, verbose_logging={args.verbose_logging}, num_samples_to_log={args.num_samples_to_log}")
    sys.stdout.flush()

    start_time = time.time()

    try:
        # Import here to ensure clean CUDA init
        import torch
        print(f"[EvalWorker] torch.cuda.device_count(): {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"[EvalWorker] torch.cuda.current_device(): {torch.cuda.current_device()}")
            print(f"[EvalWorker] torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")
        sys.stdout.flush()

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        from es_vs_grpo.tasks import create_task

        # Fixed four-task paper protocol. MATH contains 500 configured test
        # examples; the other three tasks use the 2,000-example default cap.
        test_sz = args.test_size
        TASK_CONFIGS = [
            {'type': 'countdown', 'name': 'countdown', 'train_size': 200, 'test_size': test_sz},
            {'type': 'math', 'name': 'math', 'train_size': 200, 'test_size': min(test_sz, 500), 'stratify_by_level': True},
            {'type': 'sciknoweval_chemistry', 'name': 'sciknoweval_chemistry', 'train_size': 200, 'test_size': test_sz},
            {'type': 'boolq', 'name': 'boolq', 'train_size': 200, 'test_size': test_sz},
        ]

        # Load task data (both train and test)
        print("[EvalWorker] Loading task data...")
        sys.stdout.flush()
        tasks_train_data = {}
        tasks_test_data = {}
        for cfg in TASK_CONFIGS:
            task = create_task(cfg)
            train_data, test_data = task.load_data()
            tasks_train_data[task.name] = (train_data, task)
            tasks_test_data[task.name] = (test_data, task)
            print(f"[EvalWorker]   {task.name}: {len(train_data)} train, {len(test_data)} test examples")
        sys.stdout.flush()

        # Load tokenizer
        print("[EvalWorker] Loading tokenizer...")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load vLLM
        print("[EvalWorker] Loading vLLM model...")
        sys.stdout.flush()
        llm = LLM(
            model=args.checkpoint,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )
        print("[EvalWorker] vLLM model loaded successfully!")
        sys.stdout.flush()

        sampling_params = SamplingParams(temperature=0, seed=42, max_tokens=args.max_new_tokens)

        # Setup trace saving
        traces_dir = None
        if args.save_traces:
            traces_dir = os.path.join(os.path.dirname(args.output), "eval_traces")
            os.makedirs(traces_dir, exist_ok=True)
            print(f"[EvalWorker] Saving traces to {traces_dir}")
            sys.stdout.flush()

        def eval_split(split_name, tasks_data):
            """Evaluate a split (train or test) and return results dict."""
            results = {}
            for task_name, (eval_data, task) in tasks_data.items():
                if len(eval_data) == 0:
                    print(f"[EvalWorker]   {task_name} ({split_name}): (no data)")
                    results[task_name] = 0.0
                    continue

                print(f"[EvalWorker]   Evaluating {task_name} ({split_name})...")
                sys.stdout.flush()

                # Format prompts with chat template (matching training format)
                prompts = []
                for item in eval_data:
                    if task.system_prompt:
                        messages = [
                            {"role": "system", "content": task.system_prompt},
                            {"role": "user", "content": item['context']}
                        ]
                    else:
                        messages = [{"role": "user", "content": item['context']}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)

                outputs = llm.generate(prompts, sampling_params)

                rewards = []
                responses = []
                for output, item in zip(outputs, eval_data):
                    response = output.outputs[0].text
                    responses.append(response)
                    reward_dict = task.compute_reward('', response, item)
                    rewards.append(reward_dict['reward'])

                avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                results[task_name] = avg_reward
                print(f"[EvalWorker]     {task_name} ({split_name}): {avg_reward:.3f}")
                sys.stdout.flush()

                # Verbose logging
                if args.verbose_logging:
                    num_to_show = min(args.num_samples_to_log, len(eval_data))
                    for s_idx in range(num_to_show):
                        print(f"\n  {'='*60}")
                        print(f"  [{split_name.title()} Sample {s_idx+1}] Prompt:\n{eval_data[s_idx]['context']}")
                        print(f"  [{split_name.title()} Sample {s_idx+1}] Response:\n{responses[s_idx]}")
                        print(f"  [{split_name.title()} Sample {s_idx+1}] Reward: {rewards[s_idx]:.3f}")
                        print(f"  {'='*60}")
                    sys.stdout.flush()

                # Save traces
                if traces_dir:
                    trace_entries = []
                    for item, response, reward in zip(eval_data, responses, rewards):
                        trace_entries.append({
                            "task_name": task_name,
                            "split": split_name,
                            "item": item,
                            "prompt": item['context'],
                            "final_response": response,
                            "reward": reward,
                            "conversation": [
                                {"role": "system", "content": task.system_prompt} if task.system_prompt else None,
                                {"role": "user", "content": item['context']},
                                {"role": "assistant", "content": response},
                            ],
                        })
                        # Remove None system message if no system prompt
                        trace_entries[-1]["conversation"] = [m for m in trace_entries[-1]["conversation"] if m is not None]

                    trace_path = os.path.join(traces_dir, f"{task_name}_{split_name}.json")
                    with open(trace_path, 'w', encoding='utf-8') as f:
                        json.dump(trace_entries, f, indent=2, ensure_ascii=False)
                    print(f"[EvalWorker]     Traces saved: {trace_path}")
                    sys.stdout.flush()

            return results

        # Evaluate both splits
        print("[EvalWorker] Evaluating on TRAIN sets...")
        sys.stdout.flush()
        train_results = eval_split("train", tasks_train_data)

        print("[EvalWorker] Evaluating on TEST sets...")
        sys.stdout.flush()
        test_results = eval_split("test", tasks_test_data)

        elapsed = time.time() - start_time
        print(f"[EvalWorker] Eval {args.eval_id} completed in {elapsed:.1f}s")
        sys.stdout.flush()

        # Write results (both train and test)
        output_data = {
            "eval_id": args.eval_id,
            "train_results": train_results,
            "test_results": test_results,
            "evaluation_settings": {
                "test_size_cap": args.test_size,
                "math_test_size_cap": min(args.test_size, 500),
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0,
                "seed": 42,
                "tasks": [cfg["name"] for cfg in TASK_CONFIGS],
            },
            "elapsed": elapsed,
            "success": True,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"[EvalWorker] Results written to {args.output}")
        sys.stdout.flush()

    except Exception as e:
        import traceback
        print(f"[EvalWorker] Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()

        elapsed = time.time() - start_time
        # Write error result
        output_data = {
            "eval_id": args.eval_id,
            "train_results": {},
            "test_results": {},
            "elapsed": elapsed,
            "error": str(e),
            "success": False,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
