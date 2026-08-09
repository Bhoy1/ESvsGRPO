"""Sequential evolution-strategies training with optional KL measurement.

This is the baseline used by the experiments: tasks are trained in order with
no replay buffer and no KL penalty. KL divergence is measured against a frozen
reference model but does not affect the ES update.
"""

import argparse
import os


from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
try:
    from vllm.utils import get_ip, get_open_port
except ImportError:
    from vllm.utils.network_utils import get_open_port, get_ip

from es_vs_grpo.tasks import create_task


WORKER_EXTENSION = "es_vs_grpo.train.es_worker.WorkerExtension"


def check_disk_space(path, required_gb=20):
    """Check if there's enough disk space for checkpoint. Returns (has_space, free_gb)."""
    try:
        free_gb = shutil.disk_usage(path).free / (1024**3)
        return free_gb >= required_gb, free_gb
    except Exception:
        return True, 0


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class EpochDataLoader:
    """
    Simple dataloader that shuffles data at the start of each epoch
    and yields sequential batches until all data is seen.
    """

    def __init__(self, data, seed=42):
        self.data = data
        self.indices = list(range(len(data)))
        self.position = 0
        self.epochs_completed = 0
        self.rng = random.Random(seed)
        self._shuffle()

    def _shuffle(self):
        """Shuffle indices for a new epoch."""
        self.rng.shuffle(self.indices)
        self.position = 0

    def get_batch(self, size):
        """Get next `size` items. Reshuffles automatically when epoch ends."""
        if self.position >= len(self.indices):
            self._shuffle()
            self.epochs_completed += 1

        end_pos = min(self.position + size, len(self.indices))
        batch_indices = self.indices[self.position:end_pos]
        self.position = end_pos

        return [self.data[i] for i in batch_indices]

    def get_epoch_progress(self):
        """Returns (current_position, total, epochs_completed) for logging."""
        return self.position, len(self.indices), self.epochs_completed


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Disable torch compile and CUDA graphs to avoid cache conflicts with Ray
        kwargs["enforce_eager"] = True
        super().__init__(*args, **kwargs)


def launch_training_engines(num_engines, model_name, gpu_memory_utilization=0.20):
    """Launch training engines only (these will have weights updated)."""
    print(f"[DEBUG] Creating {num_engines} placement groups for training engines...")
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    print(f"[DEBUG] Waiting for placement groups to be ready...")
    ray.get([pg.ready() for pg in pgs])
    print(f"[DEBUG] Placement groups ready.")

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    print(f"[DEBUG] Launching {num_engines} training engine actors...")
    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls=WORKER_EXTENSION,
            dtype="bfloat16",
            enable_prefix_caching=False,
            enforce_eager=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        for strategy in strategies
    ]
    print(f"[DEBUG] Training engine actors launched (loading models in background).")
    return engines, pgs


def launch_reference_engine(model_name, gpu_memory_utilization=0.20):
    """Launch a single frozen reference engine for KL computation."""
    print(f"[DEBUG] Creating placement group for reference engine...")
    pg = placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
    print(f"[DEBUG] Waiting for reference engine placement group to be ready...")
    ray.get(pg.ready())
    print(f"[DEBUG] Reference engine placement group ready.")

    strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )

    # Use same class but this engine will never join the collective or get weight updates
    print(f"[DEBUG] Launching reference engine actor...")
    ref_engine = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
        model=model_name,
        tensor_parallel_size=1,
        distributed_executor_backend="ray",
        worker_extension_cls=WORKER_EXTENSION,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enforce_eager=False,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print(f"[DEBUG] Reference engine actor launched (loading model in background).")

    return ref_engine, pg


def format_prompts(batch, task, tokenizer):
    """Format prompts with system prompt and chat template."""
    prompts = []
    for item in batch:
        if task.system_prompt:
            messages = [
                {"role": "system", "content": task.system_prompt},
                {"role": "user", "content": item['context']}
            ]
        else:
            messages = [{"role": "user", "content": item['context']}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def evaluate_batch_async(llm, batch, task, tokenizer, max_tokens=512):
    """Async evaluation - returns handle instead of blocking. No KL penalty, just reward."""
    prompts = format_prompts(batch, task, tokenizer)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        seed=None,
        max_tokens=max_tokens,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, batch, task, time.time()


def process_batch_result(handle_data):
    """Process the result from async evaluation."""
    handle, batch, task, start_time = handle_data

    outputs = ray.get(handle)

    rewards = []
    responses = []
    prompts = []
    for output, data in zip(outputs, batch):
        response = output.outputs[0].text
        responses.append(response)
        prompts.append(data.get('context', ''))
        r = task.compute_reward('', response, data)
        rewards.append(r["reward"])

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    elapsed = time.time() - start_time

    return {
        "fitness": avg_reward,  # No KL, fitness = reward
        "reward": avg_reward,
        "time": elapsed,
        "prompts": prompts,
        "responses": responses,
        "rewards_list": rewards,
    }


def score_with_reference(ref_engine, prompts, responses, tokenizer):
    """
    Get log probabilities from reference model for the generated responses.

    We feed prompt+response as a single sequence and extract logprobs for
    the response portion using prompt_logprobs.
    """
    # Combine prompt and response for each sample
    full_sequences = [p + r for p, r in zip(prompts, responses)]

    # Get prompt token lengths to know where response starts
    prompt_lengths = []
    for p in prompts:
        tokens = tokenizer.encode(p, add_special_tokens=False)
        prompt_lengths.append(len(tokens))

    # Use sampling params that return prompt_logprobs
    scoring_params = SamplingParams(
        max_tokens=1,  # We don't need to generate, just score
        prompt_logprobs=0,  # Get logprobs for all prompt tokens
    )

    outputs = ray.get(ref_engine.generate.remote(full_sequences, scoring_params, use_tqdm=False))

    # Extract logprobs for the response portion only
    ref_logprobs_list = []
    for output, prompt_len in zip(outputs, prompt_lengths):
        if output.prompt_logprobs is None:
            ref_logprobs_list.append([])
            continue

        # prompt_logprobs includes all tokens; we want those after prompt_len
        ref_lps = []
        for i in range(prompt_len, len(output.prompt_logprobs)):
            lp_dict = output.prompt_logprobs[i]
            if lp_dict is not None:
                token_id = output.prompt_token_ids[i]
                if token_id in lp_dict:
                    ref_lps.append(lp_dict[token_id].logprob)
        ref_logprobs_list.append(ref_lps)

    return ref_logprobs_list


def compute_iteration_kl(engine, ref_engine, batch, task, tokenizer, max_tokens, debug=False):
    """
    Compute KL divergence between current model and reference model.

    KL(current || ref) = mean(current_logprob - ref_logprob) per token, averaged across samples.

    Returns dict with kl_mean, kl_std, kl_max, kl_min.
    """
    if ref_engine is None:
        return {"kl_mean": 0.0, "kl_std": 0.0, "kl_max": 0.0, "kl_min": 0.0}

    if debug:
        print(f"[DEBUG KL] Formatting {len(batch)} prompts...")
    prompts = format_prompts(batch, task, tokenizer)

    # Generate with current model and collect logprobs
    if debug:
        print(f"[DEBUG KL] Generating with current model...")
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        logprobs=1,  # Collect token logprobs
    )
    outputs = ray.get(engine.generate.remote(prompts, sampling_params, use_tqdm=False))
    if debug:
        print(f"[DEBUG KL] Current model generation done.")

    # Extract responses and current model logprobs
    responses = []
    current_lps_list = []
    for output in outputs:
        responses.append(output.outputs[0].text)
        if output.outputs[0].logprobs:
            lps = []
            for lp_dict, token_id in zip(output.outputs[0].logprobs, output.outputs[0].token_ids):
                if lp_dict and token_id in lp_dict:
                    lps.append(lp_dict[token_id].logprob)
            current_lps_list.append(lps)
        else:
            current_lps_list.append([])

    if debug:
        print(f"[DEBUG KL] Extracted {len(responses)} responses, scoring with reference model...")
    # Score same responses with reference model
    ref_lps_list = score_with_reference(ref_engine, prompts, responses, tokenizer)
    if debug:
        print(f"[DEBUG KL] Reference model scoring done.")

    # Compute KL per sample
    kl_values = []
    for curr_lps, ref_lps in zip(current_lps_list, ref_lps_list):
        if curr_lps and ref_lps:
            min_len = min(len(curr_lps), len(ref_lps))
            if min_len > 0:
                # KL = mean(current_lp - ref_lp) per token
                kl = (sum(curr_lps[:min_len]) - sum(ref_lps[:min_len])) / min_len
                kl_values.append(kl)

    if not kl_values:
        return {"kl_mean": 0.0, "kl_std": 0.0, "kl_max": 0.0, "kl_min": 0.0}

    return {
        "kl_mean": float(np.mean(kl_values)),
        "kl_std": float(np.std(kl_values)),
        "kl_max": float(np.max(kl_values)),
        "kl_min": float(np.min(kl_values)),
    }


def evaluate_task_performance_parallel(engines, data, task, tokenizer, max_tokens=512):
    """Evaluate model on task data using all engines in parallel."""
    if len(data) == 0:
        return 0.0

    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=max_tokens,
    )

    # Split data across engines
    chunks = np.array_split(data, len(engines))

    # Launch all engines in parallel
    handles = []
    for engine, chunk in zip(engines, chunks):
        if len(chunk) == 0:
            continue
        prompts = format_prompts(list(chunk), task, tokenizer)
        handle = engine.generate.remote(prompts, sampling_params, use_tqdm=False)
        handles.append((handle, list(chunk)))

    # Gather results from all engines
    all_rewards = []
    for handle, chunk in handles:
        outputs = ray.get(handle)
        for output, data_item in zip(outputs, chunk):
            response = output.outputs[0].text
            reward_dict = task.compute_reward('', response, data_item)
            all_rewards.append(reward_dict['reward'])

    return float(np.mean(all_rewards)) if all_rewards else 0.0


def main(config_path):
    training_start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Load config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Extract settings from config
    model_name = config['model']['name']
    cuda_devices = config.get('cuda_devices', '0,1,2,3')
    gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)
    seed = config.get('seed', 42)

    # KL tracking config
    kl_config = config.get('kl_tracking', {})
    kl_enabled = kl_config.get('enabled', False)

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure local Ray
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Extract global ES hyperparameters (can be overridden per-task)
    es_hp = config['es_hyperparameters']

    # Logging directory - includes "baseline" to distinguish from replay
    exp_name = config.get('experiment_name', 'multitask')
    logging_dir = f"{exp_name}_ES_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(logging_dir, exist_ok=True)
    log_file = os.path.join(logging_dir, "log.txt")
    metrics_file = os.path.join(logging_dir, "metrics_baseline.jsonl")
    debug_samples_file = os.path.join(logging_dir, "debug_samples.jsonl")

    with open(log_file, "w") as f:
        pass  # Create empty log file

    # Save config to logging dir
    with open(f"{logging_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    print(f"\nLoading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save baseline checkpoint (matching GRPO format for KL matrix compatibility)
    baseline_ckpt_dir = f"{logging_dir}/baseline"
    os.makedirs(baseline_ckpt_dir, exist_ok=True)
    shutil.copy(f"{base_model_path}/config.json", f"{baseline_ckpt_dir}/config.json")
    # Copy model weights file
    for weights_file in ["model.safetensors", "pytorch_model.bin"]:
        src = f"{base_model_path}/{weights_file}"
        if os.path.exists(src):
            shutil.copy(src, f"{baseline_ckpt_dir}/{weights_file}")
            break
    tokenizer.save_pretrained(baseline_ckpt_dir)
    metadata = {
        "cycle": -1,
        "task_idx": -1,
        "task_name": "baseline",
        "forward_passes": 0,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(baseline_ckpt_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Baseline checkpoint saved: {baseline_ckpt_dir}")

    print(f"\nCreating task sequence:")
    tasks = [create_task(task_config) for task_config in config['tasks']]
    for idx, task in enumerate(tasks):
        print(f"  Task {idx}: {task.name}")

    task_names = [t.name for t in tasks]

    # Initialize performance matrices
    performance_matrix = {t.name: [] for t in tasks}
    test_performance_matrix = {t.name: [] for t in tasks}

    # Launch all engines together (training + reference if KL enabled)
    ref_engine = None
    ref_pg = None

    total_engines = es_hp['num_engines'] + (1 if kl_enabled else 0)
    print(f"\nLaunching {total_engines} engines ({es_hp['num_engines']} training{' + 1 reference' if kl_enabled else ''})...")

    # Create ALL placement groups at once to avoid resource contention
    print(f"[DEBUG] Creating {total_engines} placement groups...")
    all_pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(total_engines)]
    print(f"[DEBUG] Waiting for all {total_engines} placement groups to be ready...")
    ray.get([pg.ready() for pg in all_pgs])
    print(f"[DEBUG] All placement groups ready.")

    # Split placement groups: first N for training, last one for reference
    pgs = all_pgs[:es_hp['num_engines']]
    ref_pg = all_pgs[-1] if kl_enabled else None

    # Create strategies for training engines
    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    # Launch training engines
    print(f"[DEBUG] Launching {es_hp['num_engines']} training engine actors...")
    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=base_model_path,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls=WORKER_EXTENSION,
            dtype="bfloat16",
            enable_prefix_caching=False,
            enforce_eager=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        for strategy in strategies
    ]
    print(f"[DEBUG] Training engine actors created. Models loading in background...")

    # Launch reference engine if KL enabled
    if kl_enabled:
        ref_strategy = PlacementGroupSchedulingStrategy(
            placement_group=ref_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        print(f"[DEBUG] Launching reference engine actor...")
        ref_engine = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=ref_strategy)(ESNcclLLM).remote(
            model=base_model_path,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls=WORKER_EXTENSION,
            dtype="bfloat16",
            enable_prefix_caching=False,
            enforce_eager=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print(f"[DEBUG] Reference engine actor created. Model loading in background...")

    # Initialize collective for training engines only (must be parallel - all engines join simultaneously)
    print(f"\n[DEBUG] Initializing collective for {es_hp['num_engines']} training engines...")
    master_address = get_ip()
    master_port = get_open_port()
    print(f"[DEBUG] Master address: {master_address}, port: {master_port}")
    print(f"[DEBUG] Launching collective init for all engines in parallel...")
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, es_hp['num_engines'])
        )
        for i in range(es_hp['num_engines'])
    ])
    print(f"[DEBUG] All training engines collective initialized.")
    # ref_engine does NOT join collective - stays frozen

    print(f"\nBASELINE MODE: No replay buffer, no KL penalty")
    print(f"KL Tracking: {'ENABLED' if kl_enabled else 'DISABLED'}")
    print(f"ES Hyperparameters:")
    print(f"  sigma: {es_hp['sigma']}")
    print(f"  alpha: {es_hp['alpha']}")
    print(f"  population_size: {es_hp['population_size']}")
    print(f"  num_engines: {es_hp['num_engines']}")

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        if ref_engine is not None:
            try:
                ray.kill(ref_engine)
            except Exception:
                pass
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        if ref_pg is not None:
            try:
                remove_placement_group(ref_pg)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Store all train and test sets
    all_train_sets = {}
    all_test_sets = {}

    for task_idx, task in enumerate(tasks):
        train_data, test_data = task.load_data()
        task_config = config['tasks'][task_idx]
        all_train_sets[task.name] = (train_data, task, task_config)
        all_test_sets[task.name] = (test_data, task, task_config)
        print(f"  {task.name}: {len(train_data)} train, {len(test_data)} test examples")

    global_step = 0
    total_forward_passes = 0
    num_cycles = config.get('num_cycles', 1)

    # Get max_tokens from config (default 512)
    max_tokens = config.get('model', {}).get('max_new_tokens', 512)

    # Debug settings (from config or defaults)
    debug_config = config.get('debug', {})
    debug_print_every = debug_config.get('print_every_n_steps', 5)
    debug_num_samples = debug_config.get('num_samples_to_print', 5)

    # Helper to log metrics to JSONL
    def log_metrics(metrics_dict):
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics_dict) + "\n")

    # ============================================================================
    # BASELINE EVALUATION (before any training) - PARALLEL
    # ============================================================================
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION (before training)")
    print(f"{'='*60}")

    print("\n[Train Set Evaluation]")
    for eval_task_name, (eval_train_data, eval_task, eval_task_config) in all_train_sets.items():
        avg_reward = evaluate_task_performance_parallel(engines, eval_train_data, eval_task, tokenizer, max_tokens)
        performance_matrix[eval_task_name].append(avg_reward)
        print(f"  {eval_task_name}: {avg_reward:.3f}")
        with open(log_file, "a") as f:
            f.write(f"[Baseline Train] {eval_task_name}: {avg_reward:.3f}\n")
        log_metrics({
            "type": "eval", "eval_type": "train", "task": eval_task_name,
            "global_step": 0, "cycle": -1, "current_task": "baseline",
            "reward": avg_reward
        })

    print("\n[Test Set Evaluation]")
    for eval_task_name, (eval_test_data, eval_task, eval_task_config) in all_test_sets.items():
        if len(eval_test_data) == 0:
            print(f"  {eval_task_name}: (no test data)")
            test_performance_matrix[eval_task_name].append(0.0)
            continue
        avg_reward = evaluate_task_performance_parallel(engines, eval_test_data, eval_task, tokenizer, max_tokens)
        test_performance_matrix[eval_task_name].append(avg_reward)
        print(f"  {eval_task_name}: {avg_reward:.3f}")
        with open(log_file, "a") as f:
            f.write(f"[Baseline Test] {eval_task_name}: {avg_reward:.3f}\n")
        log_metrics({
            "type": "eval", "eval_type": "test", "task": eval_task_name,
            "global_step": 0, "cycle": -1, "current_task": "baseline",
            "reward": avg_reward
        })

    # ============================================================================
    # MAIN CONTINUAL LEARNING LOOP (NO REPLAY BUFFER)
    # ============================================================================

    for cycle in range(num_cycles):
        print(f"\n{'#'*60}")
        print(f"CYCLE {cycle + 1}/{num_cycles}")
        print(f"{'#'*60}")

        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"CYCLE {cycle + 1} | TASK {task_idx}: {task.name}")
            print(f"{'='*60}")

            # Get task data and config
            train_data, _, task_config = all_train_sets[task.name]

            # Per-task hyperparameters (inherit from global, override if specified)
            task_batch_size = task_config.get('batch_size', len(train_data))
            task_num_iterations = task_config.get('num_iterations', task_config.get('num_epochs', 1) * (len(train_data) // task_batch_size + 1))
            task_sigma = task_config.get('sigma', es_hp['sigma'])
            task_alpha = task_config.get('alpha', es_hp['alpha'])
            task_population_size = task_config.get('population_size', es_hp['population_size'])
            task_max_tokens = task_config.get('max_new_tokens', max_tokens)

            # Calculate forward passes for this task
            task_forward_passes_per_step = task_population_size * task_batch_size

            print(f"Training examples: {len(train_data)}")
            print(f"Iterations: {task_num_iterations}, Batch size: {task_batch_size}")
            print(f"ES params: sigma={task_sigma}, alpha={task_alpha}, pop={task_population_size}")
            print(f"Forward passes/iter: {task_forward_passes_per_step}")

            dataloader = EpochDataLoader(train_data, seed=seed + cycle * 1000 + task_idx)

            # Train on this task for num_iterations
            for local_iter in range(task_num_iterations):
                iter_start = time.time()

                # Get full batch from dataloader (no replay)
                batch = dataloader.get_batch(task_batch_size)

                seeds = [random.randint(0, 1_000_000) for _ in range(task_population_size)]
                seeds_perf = {}

                seed_iter = iter(seeds)
                inflight = {}

                for eng_idx, llm in enumerate(engines):
                    try:
                        seed_val = next(seed_iter)
                    except StopIteration:
                        break

                    ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed_val, task_sigma, False)))

                    handle_data = evaluate_batch_async(llm, batch, task, tokenizer, task_max_tokens)
                    handle = handle_data[0]
                    inflight[handle] = {
                        "engine": llm,
                        "engine_idx": eng_idx,
                        "seed": seed_val,
                        "handle_data": handle_data,
                    }

                while inflight:
                    done, _ = ray.wait(list(inflight.keys()), num_returns=1)
                    h = done[0]
                    meta = inflight.pop(h)

                    result = process_batch_result(meta["handle_data"])
                    seeds_perf[meta["seed"]] = result

                    llm = meta["engine"]
                    ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], task_sigma)))

                    try:
                        next_seed = next(seed_iter)
                    except StopIteration:
                        continue

                    ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, task_sigma, False)))

                    handle_data = evaluate_batch_async(llm, batch, task, tokenizer, task_max_tokens)
                    handle = handle_data[0]
                    inflight[handle] = {
                        "engine": llm,
                        "engine_idx": meta["engine_idx"],
                        "seed": next_seed,
                        "handle_data": handle_data,
                    }

                all_fitnesses = [v["fitness"] for v in seeds_perf.values()]
                mean_fitness = float(np.mean(all_fitnesses))
                std_fitness = float(np.std(all_fitnesses))
                max_fitness = float(np.max(all_fitnesses))
                min_fitness = float(np.min(all_fitnesses))

                all_rewards = [v["reward"] for v in seeds_perf.values()]
                mean_reward = float(np.mean(all_rewards))

                for k in seeds_perf:
                    seeds_perf[k]["norm_fitness"] = (seeds_perf[k]["fitness"] - mean_fitness) / (std_fitness + 1e-8)

                per_seed_coeffs = [
                    (seed_val, (task_alpha / task_population_size) * float(seeds_perf[seed_val]["norm_fitness"]))
                    for seed_val in seeds
                ]

                handles = []
                for seed_val, coeff in per_seed_coeffs:
                    handles.append(engines[0].collective_rpc.remote("perturb_self_weights", args=(seed_val, coeff, False)))
                ray.get(handles)

                ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])

                # ================================================================
                # COMPUTE KL DIVERGENCE (after weight update, before next iter)
                # ================================================================
                kl_metrics = {"kl_mean": 0.0, "kl_std": 0.0, "kl_max": 0.0, "kl_min": 0.0}
                if kl_enabled and ref_engine is not None:
                    if local_iter == 0:
                        print(f"[DEBUG] Computing first KL divergence...")
                    kl_metrics = compute_iteration_kl(
                        engines[0], ref_engine, batch, task, tokenizer, task_max_tokens,
                        debug=(local_iter == 0)  # Debug output for first iteration only
                    )
                    if local_iter == 0:
                        print(f"[DEBUG] First KL computation done: {kl_metrics['kl_mean']:.4f}")

                # Update counters
                global_step += 1
                total_forward_passes += task_forward_passes_per_step
                iter_time = time.time() - iter_start

                # Get epoch progress for logging
                pos, total, epochs_done = dataloader.get_epoch_progress()

                # Console logging
                print(f"\n[C{cycle}/{task.name}] Iter {local_iter + 1}/{task_num_iterations} (Epoch {epochs_done})")
                print(f"  Batch: {len(batch)} (no replay)")
                print(f"  Fitness: {mean_fitness:.4f} (std={std_fitness:.4f}) | Reward: {mean_reward:.3f}")
                if kl_enabled:
                    print(f"  KL: {kl_metrics['kl_mean']:.4f} (std={kl_metrics['kl_std']:.4f})")
                print(f"  FP: {total_forward_passes:,} | Time: {iter_time:.1f}s")

                # Log to file
                with open(log_file, "a") as f:
                    kl_str = f" | KL: {kl_metrics['kl_mean']:.4f}" if kl_enabled else ""
                    f.write(f'[C{cycle}/{task.name}] Iter {local_iter + 1} | Batch: {len(batch)} | '
                            f'Fitness: {mean_fitness:.4f} | Reward: {mean_reward:.3f}{kl_str}\n')

                # Log metrics to JSONL for graphing
                metrics_to_log = {
                    "type": "train",
                    "global_step": global_step,
                    "forward_passes": total_forward_passes,
                    "cycle": cycle,
                    "task": task.name,
                    "local_iter": local_iter + 1,
                    "fitness_mean": mean_fitness,
                    "fitness_std": std_fitness,
                    "fitness_max": max_fitness,
                    "fitness_min": min_fitness,
                    "reward": mean_reward,
                    "batch_size": len(batch),
                    "epoch": epochs_done,
                    "iter_time": iter_time,
                }
                # Add KL metrics if enabled
                if kl_enabled:
                    metrics_to_log["kl_mean"] = kl_metrics["kl_mean"]
                    metrics_to_log["kl_std"] = kl_metrics["kl_std"]
                    metrics_to_log["kl_max"] = kl_metrics["kl_max"]
                    metrics_to_log["kl_min"] = kl_metrics["kl_min"]

                log_metrics(metrics_to_log)

                # Debug logging: show sample prompts and responses
                if local_iter % debug_print_every == 0:
                    first_seed = seeds[0]
                    first_result = seeds_perf[first_seed]
                    prompts_list = first_result.get("prompts", [])
                    responses = first_result.get("responses", [])
                    rewards_list = first_result.get("rewards_list", [])

                    print(f"\n[DEBUG] Iter {local_iter + 1} - Sample prompts/responses (seed {first_seed}):")
                    debug_samples = []
                    for i in range(min(debug_num_samples, len(responses))):
                        prompt = prompts_list[i] if i < len(prompts_list) else ""
                        resp = responses[i]
                        rew = rewards_list[i] if i < len(rewards_list) else 0.0
                        print(f"\n  [{i}] reward={rew:.3f}")
                        print(f"  PROMPT: {prompt}")
                        print(f"  RESPONSE: {resp}")
                        debug_samples.append({
                            "index": i,
                            "reward": rew,
                            "prompt": prompt,
                            "response": resp
                        })

                    # Write debug samples to JSONL
                    with open(debug_samples_file, "a") as f:
                        f.write(json.dumps({
                            "cycle": cycle,
                            "task": task.name,
                            "iteration": local_iter + 1,
                            "seed": first_seed,
                            "samples": debug_samples
                        }) + "\n")

            # After finishing this task, evaluate on ALL tasks (backward transfer) - PARALLEL
            print(f"\n--- Backward Transfer Evaluation (after C{cycle}/{task.name}) ---")

            print("[Train Set]")
            for eval_task_name, (eval_train_data, eval_task, eval_task_config) in all_train_sets.items():
                avg_reward = evaluate_task_performance_parallel(engines, eval_train_data, eval_task, tokenizer, max_tokens)
                performance_matrix[eval_task_name].append(avg_reward)
                print(f"  {eval_task_name}: {avg_reward:.3f}")
                with open(log_file, "a") as f:
                    f.write(f"[Eval Train after C{cycle}/{task.name}] {eval_task_name}: {avg_reward:.3f}\n")
                log_metrics({
                    "type": "eval", "eval_type": "train", "task": eval_task_name,
                    "global_step": global_step, "forward_passes": total_forward_passes,
                    "cycle": cycle, "current_task": task.name,
                    "reward": avg_reward
                })

            print("[Test Set]")
            for eval_task_name, (eval_test_data, eval_task, eval_task_config) in all_test_sets.items():
                if len(eval_test_data) == 0:
                    print(f"  {eval_task_name}: (no test data)")
                    test_performance_matrix[eval_task_name].append(0.0)
                    continue
                avg_reward = evaluate_task_performance_parallel(engines, eval_test_data, eval_task, tokenizer, max_tokens)
                test_performance_matrix[eval_task_name].append(avg_reward)
                print(f"  {eval_task_name}: {avg_reward:.3f}")
                with open(log_file, "a") as f:
                    f.write(f"[Eval Test after C{cycle}/{task.name}] {eval_task_name}: {avg_reward:.3f}\n")
                log_metrics({
                    "type": "eval", "eval_type": "test", "task": eval_task_name,
                    "global_step": global_step, "forward_passes": total_forward_passes,
                    "cycle": cycle, "current_task": task.name,
                    "reward": avg_reward
                })

            # Save checkpoint after completing task (matching GRPO format)
            has_space, free_gb = check_disk_space(logging_dir, required_gb=20)
            if has_space:
                ckpt_dir = f"{logging_dir}/c{cycle}_t{task_idx}_{task.name}"
                os.makedirs(ckpt_dir, exist_ok=True)
                ray.get(engines[0].collective_rpc.remote(
                    "save_self_weights_to_disk",
                    args=(f"{ckpt_dir}/pytorch_model.bin",)
                ))
                shutil.copy(f"{base_model_path}/config.json", f"{ckpt_dir}/config.json")
                tokenizer.save_pretrained(ckpt_dir)
                metadata = {
                    "cycle": cycle,
                    "task_idx": task_idx,
                    "task_name": task.name,
                    "forward_passes": total_forward_passes,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(os.path.join(ckpt_dir, "metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Checkpoint saved: {ckpt_dir}")
            else:
                print(f"WARNING: Skipping checkpoint save - only {free_gb:.1f} GB free (need 20 GB)")

    # ============================================================================
    # FINAL CONTINUAL LEARNING ANALYSIS
    # ============================================================================

    print("\n" + "="*60)
    print("CONTINUAL LEARNING SUMMARY (BASELINE - NO REPLAY)")
    print("="*60)

    print(f"\nTRAIN Performance Matrix ({num_cycles} cycles x {len(tasks)} tasks):")
    print(f"{'Task':<15}{'Base':<10}", end='')
    for c in range(num_cycles):
        for t in tasks:
            col_name = f"C{c}_{t.name[:6]}"
            print(f"{col_name:<12}", end='')
    print()

    for task_name, perfs in performance_matrix.items():
        print(f"{task_name:<15}", end='')
        for p in perfs:
            print(f"{p:<12.3f}", end='')
        print()

    print(f"\nTEST Performance Matrix ({num_cycles} cycles x {len(tasks)} tasks):")
    print(f"{'Task':<15}{'Base':<10}", end='')
    for c in range(num_cycles):
        for t in tasks:
            col_name = f"C{c}_{t.name[:6]}"
            print(f"{col_name:<12}", end='')
    print()

    for task_name, perfs in test_performance_matrix.items():
        print(f"{task_name:<15}", end='')
        for p in perfs:
            print(f"{p:<12.3f}", end='')
        print()

    # Save results
    with open(f"{logging_dir}/train_performance_matrix.json", 'w') as f:
        json.dump(performance_matrix, f, indent=2)
    with open(f"{logging_dir}/test_performance_matrix.json", 'w') as f:
        json.dump(test_performance_matrix, f, indent=2)
    print(f"\nResults saved to {logging_dir}/")
    print(f"Metrics log: {metrics_file}")

    # Save final model (matching GRPO format)
    has_space, free_gb = check_disk_space(model_saves_dir, required_gb=20)
    if has_space:
        final_model_path = f"{model_saves_dir}/final_model"
        os.makedirs(final_model_path, exist_ok=True)
        ray.get(
            engines[0].collective_rpc.remote(
                "save_self_weights_to_disk", args=(f"{final_model_path}/pytorch_model.bin",)
            )
        )
        shutil.copy(f"{base_model_path}/config.json", f"{final_model_path}/config.json")
        tokenizer.save_pretrained(final_model_path)
        print(f"Final model saved: {final_model_path}\n")
    else:
        print(f"WARNING: Skipping final model save - only {free_gb:.1f} GB free (need 20 GB)\n")

    # Print timing info
    training_end_time = time.time()
    end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_elapsed = training_end_time - training_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Start time: {start_timestamp}")
    print(f"End time:   {end_timestamp}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Total forward passes: {total_forward_passes:,}")

    with open(log_file, "a") as f:
        f.write(f"\nStart time: {start_timestamp}\n")
        f.write(f"End time: {end_timestamp}\n")
        f.write(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s\n")
        f.write(f"Total forward passes: {total_forward_passes:,}\n")

    cleanup()


def parse_args():
    """Parse the ES training command line."""
    parser = argparse.ArgumentParser(
        description="Train the four-task sequential ES baseline."
    )
    parser.add_argument("config", help="Path to an ES experiment JSON file.")
    return parser.parse_args()


if __name__ == "__main__":
    config_path = parse_args().config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    main(config_path)
