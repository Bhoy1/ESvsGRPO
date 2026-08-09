"""Sequential GRPO training with a live-synchronized vLLM rollout server.

Tasks are trained in order with no replay. The optimization budget is measured
in response forward passes, and the updated DDP policy is synchronized to the
rollout server after every optimizer step.
"""

import argparse
import os
import math
import random
import json
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from es_vs_grpo.rollout import VLLMClient
from es_vs_grpo.tasks import create_task
import shutil


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


def create_scheduler(optimizer, scheduler_type, warmup_ratio, total_steps):
    """Create LR scheduler for a task.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: None, "constant", "linear", or "cosine"
        warmup_ratio: Fraction of total_steps for warmup
        total_steps: Total optimizer steps for this task

    Returns:
        scheduler or None
    """
    if scheduler_type is None:
        return None

    warmup_steps = int(total_steps * warmup_ratio)

    if scheduler_type == "constant":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# --------------------------------------
# DDP-Compatible Epoch DataLoader
# --------------------------------------
class DistributedEpochDataLoader:
    """
    DataLoader that:
    - Shuffles data at the start of each epoch (same seed across ranks for consistent sharding)
    - Shards data across DDP ranks (each rank sees different subset)
    - Supports get_batch() interface
    """
    def __init__(self, data, rank, world_size, seed=42):
        self.data = data
        self.rank = rank
        self.world_size = world_size
        self.base_seed = seed
        self.epochs_completed = 0
        self._shuffle()

    def _shuffle(self):
        """Shuffle and shard indices for a new epoch."""
        rng = random.Random(self.base_seed + self.epochs_completed)
        indices = list(range(len(self.data)))
        rng.shuffle(indices)
        self.rank_indices = [indices[i] for i in range(self.rank, len(indices), self.world_size)]
        self.position = 0

    def get_batch(self, size):
        """Get next batch for this rank. Auto-advances epoch when exhausted."""
        if self.position >= len(self.rank_indices):
            self.epochs_completed += 1
            self._shuffle()

        end_pos = min(self.position + size, len(self.rank_indices))
        batch_indices = self.rank_indices[self.position:end_pos]
        self.position = end_pos

        return [self.data[i] for i in batch_indices]

    def get_epoch_progress(self):
        """Returns (position, total_for_rank, epochs_completed)."""
        return self.position, len(self.rank_indices), self.epochs_completed


# --------------------------------------
# Main Function
# --------------------------------------
def main(config_path):
    # --------------------------------------
    # Load Config
    # --------------------------------------
    config = load_config(config_path)

    training_start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # --------------------------------------
    # Distributed Setup
    # --------------------------------------
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    master_process = rank == 0

    # Set random seeds
    base_seed = config.get('seed', 42)
    seed = base_seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --------------------------------------
    # Extract Config Values
    # --------------------------------------
    exp_name = config.get('experiment_name', 'grpo_multitask')
    num_cycles = config.get('num_cycles', 1)
    model_name = config['model']['name']
    default_max_new_tokens = config['model']['max_new_tokens']

    # GRPO hyperparameters
    grpo_hp = config['grpo_hyperparameters']
    K = grpo_hp.get('K', 8)
    use_dr_grpo = grpo_hp.get('use_dr_grpo', False)
    ppo_clip_range = grpo_hp.get('ppo_clip_range', 0.2)
    kl_coef = grpo_hp.get('kl_coef', 0.001)
    track_kl = grpo_hp.get('track_kl', False)  # Track KL even if kl_coef=0
    grad_accum_steps = grpo_hp.get('grad_accum_steps', 1)
    train_micro_batch_size = grpo_hp.get('train_micro_batch_size', None)
    infer_micro_batch_size = grpo_hp.get('infer_micro_batch_size', train_micro_batch_size)
    gen_micro_batch_size = grpo_hp.get('gen_micro_batch_size', None)

    # Optimizer settings
    opt_config = config['optimizer']
    optimizer_type = opt_config.get('type', 'adamw')
    lr = opt_config.get('lr', 1e-5)
    weight_decay = opt_config.get('weight_decay', 0.0)
    max_grad_norm = opt_config.get('max_grad_norm', 1.0)

    # Scheduler settings (global defaults, can be overridden per-task)
    scheduler_config = config.get('scheduler', {})
    default_scheduler_type = scheduler_config.get('type', None)  # None, "constant", "linear", "cosine"
    default_warmup_ratio = scheduler_config.get('warmup_ratio', 0.1)

    # Evaluation settings
    eval_config = config.get('evaluation', {})
    eval_batch_size = eval_config.get('eval_batch_size', 64)

    # Debug settings
    debug_config = config.get('debug', {})
    debug_print_every = debug_config.get('print_every_n_steps', 10)
    debug_num_samples = debug_config.get('num_samples_to_print', 4)

    # vLLM settings
    vllm_config = config.get('vllm', {})
    vllm_host = vllm_config.get('host', 'localhost')
    vllm_port = vllm_config.get('port', 8000)
    vllm_group_port = vllm_config.get('group_port', 51216)
    vllm_connection_timeout = vllm_config.get('connection_timeout', 120.0)

    # --------------------------------------
    # Load Model & Tokenizer
    # --------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------
    # Model Setup
    # --------------------------------------
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(local_rank)
    policy_model.gradient_checkpointing_enable()
    policy_model.train()

    # Load ref model if using KL penalty OR if tracking KL without penalty
    if kl_coef > 0 or track_kl:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(local_rank)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        if master_process:
            if kl_coef > 0:
                print(f"Reference model loaded (KL penalty enabled, coef={kl_coef})")
            else:
                print(f"Reference model loaded (KL tracking only, no penalty)")
    else:
        ref_model = None

    policy_model = DDP(policy_model, device_ids=[local_rank], output_device=local_rank)

    # Setup optimizer
    param_dict = {pn: p for pn, p in policy_model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(optim_groups, lr=lr)
    elif optimizer_type == "sgd_momentum":
        optimizer = torch.optim.SGD(optim_groups, lr=lr, momentum=0.9)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(optim_groups, lr=lr, momentum=0.0)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    if master_process:
        print(f"Optimizer: {optimizer_type} (lr={lr})")

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------
    # vLLM Client Setup (master process only)
    # --------------------------------------
    vllm_client = None
    if master_process:
        print(f"\n[vLLM] Connecting to server at {vllm_host}:{vllm_port}...")
        vllm_client = VLLMClient(
            host=vllm_host,
            server_port=vllm_port,
            group_port=vllm_group_port,
            connection_timeout=vllm_connection_timeout,
        )
        vllm_client.init_communicator(device=f"cuda:{local_rank}")
        print(f"[vLLM] Connected and NCCL communicator initialized!")

    dist.barrier()  # Wait for vLLM setup

    # --------------------------------------
    # Prepare Logging
    # --------------------------------------
    logging_dir = None
    log_file = None
    metrics_file = None
    debug_samples_file = None
    checkpoint_dir = None

    if master_process:
        logging_dir = f"{exp_name}_GRPO_vllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(logging_dir, exist_ok=True)
        log_file = os.path.join(logging_dir, "log.txt")
        metrics_file = os.path.join(logging_dir, "metrics_grpo_baseline.jsonl")
        debug_samples_file = os.path.join(logging_dir, "debug_samples.jsonl")
        checkpoint_dir = os.path.join(logging_dir, "async_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        with open(log_file, "w") as f:
            pass  # Create empty log file

        # Save config to logging dir
        with open(f"{logging_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)

    # Helper to log metrics to JSONL
    def log_metrics(metrics_dict):
        if master_process and metrics_file:
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics_dict) + "\n")

    # --------------------------------------
    # Create Tasks and Load Data
    # --------------------------------------
    task_configs = config['tasks']
    tasks = []
    all_task_data = {}
    all_test_data = {}

    for task_cfg in task_configs:
        task = create_task(task_cfg)
        tasks.append(task)
        train_data, test_data = task.load_data()
        all_task_data[task.name] = (train_data, task, task_cfg)
        all_test_data[task.name] = (test_data, task, task_cfg)
        if master_process:
            print(f"Loaded task '{task.name}': {len(train_data)} train, {len(test_data)} test examples")

    # --------------------------------------
    # Print training config
    # --------------------------------------
    if master_process:
        print(f"\n{'='*60}")
        print(f"GRPO CONTINUAL LEARNING BASELINE (NO REPLAY) + vLLM")
        print(f"{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"Model: {model_name}")
        print(f"World size: {world_size} GPUs (training)")
        print(f"vLLM: {vllm_host}:{vllm_port} (group_port={vllm_group_port})")
        print(f"Tasks: {[t.name for t in tasks]}")
        print(f"Cycles: {num_cycles}")
        print(f"GRPO: K={K}, clip={ppo_clip_range}, kl_coef={kl_coef}, track_kl={track_kl}, dr_grpo={use_dr_grpo}")
        print(f"Optimizer: {optimizer_type} (lr={lr}, wd={weight_decay})")
        print(f"Grad accum: {grad_accum_steps}")
        if train_micro_batch_size:
            print(f"Train micro-batch: {train_micro_batch_size}")
        print(f"{'='*60}\n")

    # --------------------------------------
    # Save Baseline Checkpoint (before any training)
    # --------------------------------------
    dist.barrier()
    if master_process:
        baseline_ckpt_dir = f"{logging_dir}/baseline"
        print(f"Saving baseline checkpoint (pre-training)...")
        policy_model.module.save_pretrained(baseline_ckpt_dir)
        tokenizer.save_pretrained(baseline_ckpt_dir)

        # Save metadata for run_all_evals_baseline.py
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
    dist.barrier()

    # --------------------------------------
    # Training Loop
    # --------------------------------------
    total_forward_passes_all = 0

    for cycle in range(num_cycles):
        if master_process:
            print(f"\n{'#'*60}")
            print(f"CYCLE {cycle + 1}/{num_cycles}")
            print(f"{'#'*60}")
        dist.barrier()

        for task_idx, task in enumerate(tasks):
            task_cfg = task_configs[task_idx]
            train_data, _, _ = all_task_data[task.name]

            # Per-task settings (inherit from global if not set)
            task_batch_size = task_cfg.get('batch_size', 16)
            task_total_forward_passes = task_cfg.get('total_forward_passes', 90000)
            task_max_new_tokens = task_cfg.get('max_new_tokens', default_max_new_tokens)

            # Per-task GRPO hyperparameters (inherit from grpo_hyperparameters if not set)
            task_K = task_cfg.get('K', K)
            task_kl_coef = task_cfg.get('kl_coef', kl_coef)
            task_ppo_clip_range = task_cfg.get('ppo_clip_range', ppo_clip_range)
            task_use_dr_grpo = task_cfg.get('use_dr_grpo', use_dr_grpo)

            # Per-task micro batch sizes (inherit from grpo_hyperparameters if not set)
            task_train_micro_batch_size = task_cfg.get('train_micro_batch_size', train_micro_batch_size)
            task_infer_micro_batch_size = task_cfg.get('infer_micro_batch_size', infer_micro_batch_size)
            task_gen_micro_batch_size = task_cfg.get('gen_micro_batch_size', gen_micro_batch_size)

            # Per-task learning rate (inherit from optimizer if not set)
            task_lr = task_cfg.get('lr', lr)

            # Per-task scheduler settings (inherit from global scheduler config if not set)
            task_scheduler_type = task_cfg.get('scheduler_type', default_scheduler_type)
            task_warmup_ratio = task_cfg.get('warmup_ratio', default_warmup_ratio)

            # Update learning rate if different from current
            if task_lr != lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = task_lr
                if master_process:
                    print(f"Updated learning rate to {task_lr} for task {task.name}")

            if master_process:
                print(f"\n{'='*60}")
                print(f"CYCLE {cycle + 1} | TASK {task_idx}: {task.name}")
                print(f"{'='*60}")
            dist.barrier()

            dataloader = DistributedEpochDataLoader(train_data, rank, world_size, seed=base_seed + cycle * len(tasks) + task_idx)

            # Compute forward passes per step
            prompts_per_step = task_batch_size * grad_accum_steps * world_size
            forward_passes_per_step = prompts_per_step * task_K

            # Calculate total steps for this task (all processes need this for scheduler)
            task_total_steps = task_total_forward_passes // forward_passes_per_step

            # Create scheduler for this task
            task_scheduler = create_scheduler(optimizer, task_scheduler_type, task_warmup_ratio, task_total_steps)

            if master_process:
                print(f"Training: {len(train_data)} examples")
                print(f"Target: {task_total_forward_passes:,} forward passes (~{task_total_steps} steps)")
                print(f"Batch: {task_batch_size}/gpu x {grad_accum_steps} accum x {world_size} gpus = {prompts_per_step} prompts/step")
                print(f"Forward passes/step: {prompts_per_step} x {task_K} = {forward_passes_per_step}")
                print(f"GRPO: K={task_K}, clip={task_ppo_clip_range}, kl_coef={task_kl_coef}, dr_grpo={task_use_dr_grpo}, lr={task_lr}")
                if task_scheduler is not None:
                    warmup_steps = int(task_total_steps * task_warmup_ratio)
                    print(f"Scheduler: {task_scheduler_type}, warmup={warmup_steps} steps ({task_warmup_ratio*100:.0f}%)")
                else:
                    print(f"Scheduler: None (fixed LR)")

            local_step = 0
            task_forward_passes = 0
            step_start_time = time.time()

            while task_forward_passes < task_total_forward_passes:
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                accum_grpo_loss = 0.0
                accum_kl_loss = 0.0
                accum_reward = 0.0
                accum_advantage_mean = 0.0
                accum_advantage_std = 0.0
                accum_prompts = []
                accum_responses = []
                accum_rewards_list = []

                for accum_step in range(grad_accum_steps):
                    batch_items = dataloader.get_batch(task_batch_size)

                    if len(batch_items) == 0:
                        continue

                    # Format prompts with chat template
                    if task.system_prompt:
                        messages_list = [
                            [
                                {"role": "system", "content": task.system_prompt},
                                {"role": "user", "content": item['context']}
                            ]
                            for item in batch_items
                        ]
                    else:
                        messages_list = [
                            [{"role": "user", "content": item['context']}]
                            for item in batch_items
                        ]

                    prompts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages_list]
                    curr_batch_size = len(prompts)

                    # --------------------------------------
                    # vLLM Generation: Gather -> Generate -> Broadcast
                    # --------------------------------------
                    # Gather prompts and items from all ranks to rank 0
                    all_prompts_gathered = [None] * world_size
                    all_items_gathered = [None] * world_size
                    dist.all_gather_object(all_prompts_gathered, prompts)
                    dist.all_gather_object(all_items_gathered, batch_items)

                    # Flatten gathered data
                    global_prompts = [p for shard in all_prompts_gathered for p in shard]
                    global_items = [item for shard in all_items_gathered for item in shard]
                    global_batch_size = len(global_prompts)

                    # Rank 0 generates using vLLM
                    if master_process:
                        vllm_result = vllm_client.generate(
                            prompts=global_prompts,
                            n=task_K,
                            temperature=1.0,
                            top_p=0.95,
                            max_tokens=task_max_new_tokens,
                            generation_kwargs={
                                "stop_token_ids": [tokenizer.eos_token_id],
                            },
                        )
                        # vLLM returns: prompt_ids (per prompt), completion_ids (n*prompts completions)
                        all_prompt_ids = vllm_result['prompt_ids']  # list of prompt token lists
                        all_completion_ids = vllm_result['completion_ids']  # list of completion token lists
                    else:
                        all_prompt_ids = None
                        all_completion_ids = None

                    # Broadcast results from rank 0 to all ranks
                    broadcast_data = [all_prompt_ids, all_completion_ids]
                    dist.broadcast_object_list(broadcast_data, src=0)
                    all_prompt_ids, all_completion_ids = broadcast_data

                    # Each rank extracts its shard of completions
                    # vLLM returns completions in order: [prompt0_k0, prompt0_k1, ..., prompt0_kK-1, prompt1_k0, ...]
                    my_start_prompt = rank * curr_batch_size
                    my_end_prompt = my_start_prompt + curr_batch_size
                    my_start_completion = my_start_prompt * task_K
                    my_end_completion = my_end_prompt * task_K

                    my_prompt_ids = all_prompt_ids[my_start_prompt:my_end_prompt]
                    my_completion_ids = all_completion_ids[my_start_completion:my_end_completion]

                    # Build full sequences (prompt + completion) for each of the K samples
                    # Use LEFT-padding for prompts to match v2 behavior (uniform prompt_len)
                    max_prompt_len = max(len(p) for p in my_prompt_ids)

                    full_sequences = []
                    for i, prompt_ids in enumerate(my_prompt_ids):
                        # Left-pad prompt to max_prompt_len
                        prompt_pad_len = max_prompt_len - len(prompt_ids)
                        padded_prompt = [tokenizer.pad_token_id] * prompt_pad_len + prompt_ids

                        for k in range(task_K):
                            comp_idx = i * task_K + k
                            completion_ids = my_completion_ids[comp_idx]
                            full_seq = padded_prompt + completion_ids
                            full_sequences.append(full_seq)

                    # Right-pad full sequences to uniform length (for variable completion lengths)
                    max_seq_len = max(len(seq) for seq in full_sequences)
                    padded_sequences = []
                    for seq in full_sequences:
                        pad_len = max_seq_len - len(seq)
                        padded_seq = seq + [tokenizer.pad_token_id] * pad_len
                        padded_sequences.append(padded_seq)

                    explore_generations = torch.tensor(padded_sequences, dtype=torch.long, device=local_rank)

                    # Uniform prompt_len for all sequences (like v2)
                    prompt_len = max_prompt_len

                    # Compute masks & labels (using uniform prompt_len like v2)
                    batch_attention_mask = (explore_generations != tokenizer.pad_token_id).long()
                    batch_action_mask = batch_attention_mask.clone()
                    batch_action_mask[:, :prompt_len] = 0  # Zero out prompt region (including left-pad)

                    labels = explore_generations.clone()
                    labels[batch_action_mask == 0] = -100

                    # Compute old logprobs (chunked)
                    total_seqs = curr_batch_size * task_K
                    labels_shifted = labels[:, 1:].contiguous()

                    if task_infer_micro_batch_size is not None:
                        infer_chunk_size = task_infer_micro_batch_size
                    else:
                        infer_chunk_size = total_seqs

                    num_infer_chunks = (total_seqs + infer_chunk_size - 1) // infer_chunk_size
                    logprobs_old_list = []

                    policy_model.eval()
                    with torch.no_grad():
                        for chunk_idx in range(num_infer_chunks):
                            start_idx = chunk_idx * infer_chunk_size
                            end_idx = min(start_idx + infer_chunk_size, total_seqs)

                            chunk_gens = explore_generations[start_idx:end_idx]
                            chunk_mask = batch_attention_mask[start_idx:end_idx]
                            chunk_labels = labels_shifted[start_idx:end_idx]

                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                out_old = policy_model(chunk_gens, chunk_mask, use_cache=False)
                            logits_old = out_old.logits[:, :-1, :].contiguous()
                            chunk_logprobs = -F.cross_entropy(
                                logits_old.view(-1, logits_old.shape[-1]),
                                chunk_labels.view(-1),
                                reduction='none',
                                ignore_index=-100
                            ).view(logits_old.shape[0], -1)
                            logprobs_old_list.append(chunk_logprobs)
                            del out_old, logits_old

                    policy_model.train()
                    logprobs_old = torch.cat(logprobs_old_list, dim=0)
                    logprobs_old = logprobs_old.view(curr_batch_size, task_K, -1)

                    # Compute reference logprobs (if ref model exists - for KL penalty or tracking)
                    if ref_model is not None:
                        logprobs_ref_list = []
                        with torch.no_grad():
                            for chunk_idx in range(num_infer_chunks):
                                start_idx = chunk_idx * infer_chunk_size
                                end_idx = min(start_idx + infer_chunk_size, total_seqs)

                                chunk_gens = explore_generations[start_idx:end_idx]
                                chunk_mask = batch_attention_mask[start_idx:end_idx]
                                chunk_labels = labels_shifted[start_idx:end_idx]

                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    out_ref = ref_model(chunk_gens, chunk_mask, use_cache=False)
                                logits_ref = out_ref.logits[:, :-1, :].contiguous()
                                chunk_logprobs = -F.cross_entropy(
                                    logits_ref.view(-1, logits_ref.shape[-1]),
                                    chunk_labels.view(-1),
                                    reduction='none',
                                    ignore_index=-100
                                ).view(logits_ref.shape[0], -1)
                                logprobs_ref_list.append(chunk_logprobs)
                                del out_ref, logits_ref

                        logprobs_ref = torch.cat(logprobs_ref_list, dim=0)
                        logprobs_ref = logprobs_ref.view(curr_batch_size, task_K, -1)

                    # Compute rewards - use uniform prompt_len (like v2)
                    batch_responses_ids = explore_generations[:, prompt_len:]
                    batch_responses = tokenizer.batch_decode(batch_responses_ids, skip_special_tokens=True)

                    all_items_K = [item for item in batch_items for _ in range(task_K)]

                    rewards_list = []
                    for response, item in zip(batch_responses, all_items_K):
                        reward_dict = task.compute_reward('', response, item)
                        rewards_list.append(reward_dict['reward'])

                    # Store for debug logging
                    if accum_step == 0:
                        accum_prompts = prompts[:debug_num_samples]
                        accum_responses = batch_responses[:debug_num_samples * task_K]
                        accum_rewards_list = rewards_list[:debug_num_samples * task_K]

                    batch_rewards = torch.tensor(rewards_list, dtype=torch.bfloat16, device=local_rank)
                    batch_rewards = batch_rewards.view(curr_batch_size, task_K)

                    # Compute advantages
                    if task_use_dr_grpo:
                        batch_advantages = batch_rewards - batch_rewards.mean(dim=-1, keepdim=True)
                    else:
                        batch_advantages = (batch_rewards - batch_rewards.mean(dim=-1, keepdim=True)) / batch_rewards.std(dim=-1, keepdim=True).clamp_min(1e-6)

                    # Track advantage stats
                    accum_advantage_mean += batch_advantages.mean().item()
                    accum_advantage_std += batch_advantages.std().item()

                    batch_advantages = batch_advantages.unsqueeze(2).expand_as(logprobs_old)

                    # Forward pass with gradient accumulation (chunked)
                    is_last_accum = (accum_step == grad_accum_steps - 1)

                    logprobs_old_flat = logprobs_old.view(total_seqs, -1)
                    batch_advantages_flat = batch_advantages.view(total_seqs, -1)
                    valid_mask_flat = batch_action_mask[:, 1:].contiguous().float()
                    labels_flat = labels[:, 1:].contiguous()

                    if ref_model is not None:
                        logprobs_ref_flat = logprobs_ref.view(total_seqs, -1)

                    if task_train_micro_batch_size is not None and task_train_micro_batch_size < total_seqs:
                        chunk_size = task_train_micro_batch_size
                    else:
                        chunk_size = total_seqs

                    num_train_chunks = (total_seqs + chunk_size - 1) // chunk_size
                    grpo_loss_accum = 0.0
                    kl_loss_accum = 0.0
                    valid_tokens_accum = 0

                    for chunk_idx in range(num_train_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, total_seqs)
                        chunk_weight = (end_idx - start_idx) / total_seqs

                        chunk_generations = explore_generations[start_idx:end_idx]
                        chunk_attention_mask = batch_attention_mask[start_idx:end_idx]
                        chunk_logprobs_old = logprobs_old_flat[start_idx:end_idx]
                        chunk_advantages = batch_advantages_flat[start_idx:end_idx]
                        chunk_valid_mask = valid_mask_flat[start_idx:end_idx]
                        chunk_labels = labels_flat[start_idx:end_idx]

                        # Only sync on last chunk of last accum step (matches train_single_task1.py)
                        is_last_chunk = (chunk_idx == num_train_chunks - 1)
                        need_sync = is_last_accum and is_last_chunk
                        sync_context = nullcontext() if need_sync else policy_model.no_sync()

                        with sync_context:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                out_new = policy_model(chunk_generations, chunk_attention_mask, use_cache=False)

                            logits_new = out_new.logits[:, :-1, :].contiguous()
                            logprobs_new = -F.cross_entropy(
                                logits_new.view(-1, logits_new.shape[-1]),
                                chunk_labels.view(-1),
                                reduction='none',
                                ignore_index=-100
                            ).view(logits_new.shape[0], -1)

                            ratio = torch.exp(logprobs_new - chunk_logprobs_old)
                            ratio_clipped = torch.clamp(ratio, 1.0 - task_ppo_clip_range, 1.0 + task_ppo_clip_range)
                            individual_ppo_reward = torch.min(ratio * chunk_advantages, ratio_clipped * chunk_advantages)

                            # Compute KL if ref model exists (for penalty or tracking)
                            if ref_model is not None:
                                chunk_logprobs_ref = logprobs_ref_flat[start_idx:end_idx]
                                ratio_ref_log = chunk_logprobs_ref - logprobs_new
                                ratio_ref = torch.exp(ratio_ref_log)
                                individual_kl_penalty = ratio_ref - ratio_ref_log - 1
                                chunk_kl = (individual_kl_penalty * chunk_valid_mask).sum()

                                # Only add KL to loss if kl_coef > 0, otherwise just track it
                                if task_kl_coef > 0:
                                    per_token_loss = individual_ppo_reward - task_kl_coef * individual_kl_penalty
                                else:
                                    per_token_loss = individual_ppo_reward  # KL tracked but not penalized
                            else:
                                per_token_loss = individual_ppo_reward
                                chunk_kl = 0.0

                            sum_loss_per_response = (per_token_loss * chunk_valid_mask).sum(dim=-1)
                            if task_use_dr_grpo:
                                chunk_grpo_loss = -sum_loss_per_response.mean()
                            else:
                                count_per_response = chunk_valid_mask.sum(dim=-1)
                                reward_ave_response = sum_loss_per_response / count_per_response.clamp_min(1)
                                chunk_grpo_loss = -reward_ave_response.mean()

                            scaled_loss = chunk_grpo_loss * chunk_weight / grad_accum_steps
                            scaled_loss.backward()

                        grpo_loss_accum += chunk_grpo_loss.detach().item() * chunk_weight
                        kl_loss_accum += chunk_kl.detach().item() if isinstance(chunk_kl, torch.Tensor) else chunk_kl
                        valid_tokens_accum += chunk_valid_mask.sum().item()
                        del out_new, logits_new

                    accum_loss += grpo_loss_accum
                    accum_grpo_loss += grpo_loss_accum
                    accum_kl_loss += kl_loss_accum / max(valid_tokens_accum, 1)
                    accum_reward += batch_rewards.mean().item()

                # Optimizer step
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                if task_scheduler is not None:
                    task_scheduler.step()

                # Sync updated weights to vLLM server
                if master_process and vllm_client is not None:
                    vllm_client.update_model_params(policy_model.module)
                    vllm_client.reset_prefix_cache()
                dist.barrier()  # Wait for weight sync to complete

                # Update counters
                task_forward_passes += forward_passes_per_step
                total_forward_passes_all += forward_passes_per_step

                # Log metrics
                avg_loss = accum_loss / grad_accum_steps
                avg_grpo_loss = accum_grpo_loss / grad_accum_steps
                avg_kl_loss = accum_kl_loss / grad_accum_steps
                avg_reward = accum_reward / grad_accum_steps
                avg_adv_mean = accum_advantage_mean / grad_accum_steps
                avg_adv_std = accum_advantage_std / grad_accum_steps

                metrics = torch.tensor([avg_loss, avg_grpo_loss, avg_kl_loss, avg_reward, avg_adv_mean, avg_adv_std], device=local_rank)
                dist.all_reduce(metrics, op=dist.ReduceOp.AVG)

                if master_process:
                    step_time = time.time() - step_start_time
                    pos, total, epochs_done = dataloader.get_epoch_progress()
                    progress_pct = 100.0 * task_forward_passes / task_total_forward_passes

                    # Get current LR
                    if task_scheduler is not None:
                        current_lr = task_scheduler.get_last_lr()[0]
                    else:
                        current_lr = optimizer.param_groups[0]['lr']

                    # Log to JSONL
                    log_metrics({
                        "type": "train",
                        "cycle": cycle,
                        "task": task.name,
                        "step": local_step,
                        "forward_passes": task_forward_passes,
                        "total_forward_passes": total_forward_passes_all,
                        "progress_pct": progress_pct,
                        "loss": metrics[0].item(),
                        "grpo_loss": metrics[1].item(),
                        "kl_loss": metrics[2].item(),
                        "reward": metrics[3].item(),
                        "advantage_mean": metrics[4].item(),
                        "advantage_std": metrics[5].item(),
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "epoch": epochs_done,
                        "step_time": step_time,
                        "lr": current_lr,
                    })

                    # Log to file
                    with open(log_file, "a") as f:
                        f.write(f'[C{cycle}/{task.name}] Step {local_step} | FP: {task_forward_passes:,}/{task_total_forward_passes:,} ({progress_pct:.1f}%) | '
                                f'Loss: {metrics[0]:.4f} | GRPO: {metrics[1]:.4f} | KL: {metrics[2]:.4f} | Reward: {metrics[3]:.3f} | Grad: {grad_norm:.4f}\n')

                    # Print progress
                    if local_step % debug_print_every == 0:
                        print(f"\n[C{cycle}/{task.name}] Step {local_step} | FP: {task_forward_passes:,}/{task_total_forward_passes:,} ({progress_pct:.1f}%)")
                        print(f"  Loss: {metrics[0]:.4f} | GRPO: {metrics[1]:.4f} | KL: {metrics[2]:.4f} | Reward: {metrics[3]:.3f} | Grad: {grad_norm:.4f} | Time: {step_time:.1f}s")
                        print(f"  Advantages: mean={metrics[4]:.4f}, std={metrics[5]:.4f}")

                        # Debug samples - show 1 prompt with num_samples_to_print K-samples
                        print(f"\n[DEBUG] Sample prompt/responses:")
                        debug_samples = []
                        if len(accum_prompts) > 0:
                            prompt = accum_prompts[0]
                            print(f"\n  [Prompt 0]")
                            print(f"  PROMPT: {prompt}")
                            for k in range(min(debug_num_samples, task_K)):
                                resp_idx = k
                                if resp_idx < len(accum_responses):
                                    resp = accum_responses[resp_idx]
                                    rew = accum_rewards_list[resp_idx] if resp_idx < len(accum_rewards_list) else 0.0
                                    print(f"    [K={k}] reward={rew:.3f}: {resp}")
                                    debug_samples.append({
                                        "prompt_idx": 0,
                                        "k": k,
                                        "reward": rew,
                                        "prompt": prompt,
                                        "response": resp
                                    })

                        # Write debug samples to JSONL
                        with open(debug_samples_file, "a") as f:
                            f.write(json.dumps({
                                "cycle": cycle,
                                "task": task.name,
                                "step": local_step,
                                "forward_passes": task_forward_passes,
                                "samples": debug_samples
                            }) + "\n")

                # Sync all ranks after master-process logging to prevent NCCL timeout
                dist.barrier()

                step_start_time = time.time()
                local_step += 1

            # Checkpoint (save with metadata for batch eval later)
            dist.barrier()
            if master_process:
                has_space, free_gb = check_disk_space(logging_dir, required_gb=20)
                if has_space:
                    ckpt_dir = f"{logging_dir}/c{cycle}_t{task_idx}_{task.name}"
                    policy_model.module.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    # Save metadata for run_all_evals.py
                    metadata = {
                        "cycle": cycle,
                        "task_idx": task_idx,
                        "task_name": task.name,
                        "forward_passes": total_forward_passes_all,
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(os.path.join(ckpt_dir, "metadata.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)

                    print(f"Checkpoint saved: {ckpt_dir}")
                else:
                    print(f"WARNING: Skipping checkpoint save - only {free_gb:.1f} GB free (need 20 GB)")

    # --------------------------------------
    # Training Complete
    # --------------------------------------
    if master_process:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")

        training_end_time = time.time()
        end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_elapsed = training_end_time - training_start_time
        hours, remainder = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\nStart time: {start_timestamp}")
        print(f"End time:   {end_timestamp}")
        print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        print(f"Total forward passes: {total_forward_passes_all:,}")
        print(f"\nCheckpoints saved to: {logging_dir}/")
        print(f"\nTo run batch evaluation:")
        print(
            f"  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "
            f"python -m es_vs_grpo.evaluation.run --folder {logging_dir}/"
        )

        with open(log_file, "a") as f:
            f.write(f"\nStart time: {start_timestamp}\n")
            f.write(f"End time: {end_timestamp}\n")
            f.write(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s\n")
            f.write(f"Total forward passes: {total_forward_passes_all:,}\n")

    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    """Parse the GRPO training command line."""
    parser = argparse.ArgumentParser(
        description="Train the four-task sequential GRPO baseline."
    )
    parser.add_argument("config", help="Path to a GRPO experiment JSON file.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config)
