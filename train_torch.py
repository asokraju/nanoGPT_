"""
Training script for GPT-MoE model with LoRA integration.
This script supports both single GPU and distributed data parallel (DDP) training.

To run on a single GPU:
$ python train.py

To run with DDP on 4 GPUs on 1 node:
$ torchrun --standalone --nproc_per_node=1 train_torch.py

Dependencies:
- torch
- omegaconf
- numpy
- peft (for LoRA)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_torch import GPTConfig, GPT  # Import your GPT-MoE model here

# Import OmegaConf for configuration management
from omegaconf import OmegaConf

# Load configuration from YAML file
config = OmegaConf.load('config_torch.yaml')

# -----------------------------------------------------------------------------
# Setup for DDP and device configuration
# -----------------------------------------------------------------------------

# Check if running with DDP (Distributed Data Parallel)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=config.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # This process will do logging, checkpointing, etc.
    seed_offset = ddp_rank  # Each process gets a different seed
    # Scale down the desired gradient accumulation iterations per process proportionally
    assert config.gradient_accumulation_steps % ddp_world_size == 0
    config.gradient_accumulation_steps //= ddp_world_size
else:
    # If not DDP, we are running on a single GPU and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = config.device

tokens_per_iter = (config.gradient_accumulation_steps * ddp_world_size *
                   config.batch_size * config.block_size)
print(f"Tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cuDNN
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Add new parameters to the configuration
# -----------------------------------------------------------------------------

# Ensure new parameters exist
if not hasattr(config, 'num_checkpoints'):
    config.num_checkpoints = 5  # Default value
if not hasattr(config, 'save_current_ckpt'):
    config.save_current_ckpt = True  # Default value
if not hasattr(config, 'save_lora_only'):
    config.save_lora_only = True  # Default: save only LoRA parameters
if not hasattr(config, 'lora_ckpt_path'):
    config.lora_ckpt_path = os.path.join(config.out_dir, 'lora_ckpt')  # Default path for LoRA parameters
if not hasattr(config, 'merged_model_ckpt_path'):
    config.merged_model_ckpt_path = os.path.join(config.out_dir, 'merged_model.pt')  # Default path for merged model

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

# Define data directories
data_dir = os.path.join('data', config.dataset)

def get_batch(split):
    """
    Fetches a batch of data.

    Args:
        split (str): 'train' or 'val'

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and target tensors
    """
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # Pin arrays x, y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------

# Initialize iteration number and best validation loss
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model arguments
model_args = dict(
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    block_size=config.block_size,
    bias=config.bias,
    vocab_size=None,
    dropout=config.dropout,
    use_moe=config.use_moe,
    num_experts=config.num_experts,
    num_experts_per_tok=config.num_experts_per_tok,
    static_experts=config.static_experts,
    moe_loss=config.moe_loss,
    moe_loss_type=config.moe_loss_type,
    moe_loss_coef=config.moe_loss_coef,
)

# Initialize the model
if config.init_from == 'scratch':
    # Initialize a new model from scratch
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("Defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif config.init_from == 'resume':
    # Resume training from a checkpoint
    print(f"Resuming training from {config.out_dir}")
    ckpt_path = os.path.join(config.out_dir, 'current_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Ensure that model configurations match
    for k in model_args:
        model_args[k] = checkpoint_model_args.get(k, model_args.get(k))
    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Fix state_dict keys if needed
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif config.init_from == 'finetune':
    # Fine-tune your own pre-trained GPT-MoE model with LoRA
    print("Loading your pre-trained GPT-MoE model for fine-tuning with LoRA")
    # Load your saved model checkpoint
    ckpt_path = config.finetune_ckpt_path  # Path to your saved checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Update model_args with values from the checkpoint
    for k in model_args:
        model_args[k] = checkpoint_model_args.get(k, model_args.get(k))
    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Fix state_dict keys if needed
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print("Successfully loaded your pre-trained GPT-MoE model.")

    # Apply LoRA to the model
    print("Applying LoRA to the model")
    # Ensure that the peft library is installed
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError("To use LoRA, please install the 'peft' library.")

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout
        # bias='none',
        # task_type=TaskType.CAUSAL_LM
    )
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    print("LoRA has been applied to the model.")
    # Start iter_num from 0 for fine-tuning
    iter_num = 0
    best_val_loss = 1e9
else:
    raise ValueError(f"Unknown init_from option: {config.init_from}")

# Adjust block size if necessary
if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    model_args['block_size'] = config.block_size

model.to(device)

# Initialize a GradScaler if using float16
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

# Optimizer
if config.init_from == 'finetune':
    # Only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
    )
    print("Optimizer is set to only update LoRA parameters.")
elif config.init_from == 'resume':
    optimizer = model.configure_optimizers(
        config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # Free up memory
else:
    optimizer = model.configure_optimizers(
        config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)

# Compile the model (requires PyTorch 2.0)
if config.compile:
    print("Compiling the model... (this may take some time)")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Evaluation function
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss over the training and validation sets.

    Returns:
        dict: Dictionary containing the mean loss for 'train' and 'val' splits.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, main_loss, aux_loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Learning rate decay scheduler (cosine with warmup)
# -----------------------------------------------------------------------------

def get_lr(it):
    """
    Compute the learning rate at iteration it.

    Args:
        it (int): Current iteration number

    Returns:
        float: Learning rate
    """
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # After decay
    if it > config.lr_decay_iters:
        return config.min_lr
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Ranges from 1 to 0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

if config.wandb_log and master_process:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=OmegaConf.to_container(config))

# -----------------------------------------------------------------------------
# Saving the checkpoints
# -----------------------------------------------------------------------------

# Function to load existing checkpoints from the output directory
def load_existing_checkpoints():
    """
    Load existing checkpoints from the output directory and populate best_checkpoints.
    """
    checkpoint_pattern = re.compile(r'ckpt_iter_(\d+)_val_(\d+\.\d+)\.pt')
    checkpoint_files = [f for f in os.listdir(config.out_dir) if checkpoint_pattern.match(f)]
    for ckpt_file in checkpoint_files:
        match = checkpoint_pattern.match(ckpt_file)
        if match:
            iter_num = int(match.group(1))
            val_loss = float(match.group(2))
            ckpt_path = os.path.join(config.out_dir, ckpt_file)
            checkpoint_info = {
                'val_loss': val_loss,
                'iter_num': iter_num,
                'ckpt_path': ckpt_path,
            }
            best_checkpoints.append(checkpoint_info)
    # Sort the checkpoints based on validation loss (lower is better)
    best_checkpoints.sort(key=lambda x: x['val_loss'])
    # Keep only the top num_checkpoints
    while len(best_checkpoints) > config.num_checkpoints:
        worst_ckpt = best_checkpoints.pop()
        # Optionally, delete the checkpoint file
        if os.path.exists(worst_ckpt['ckpt_path']):
            os.remove(worst_ckpt['ckpt_path'])
            print(f"Removed old checkpoint {worst_ckpt['ckpt_path']} with val_loss {worst_ckpt['val_loss']:.4f}")

# Function to update the list of best checkpoints
def update_best_checkpoints(checkpoint_info):
    """
    Update the list of best checkpoints based on validation loss.

    Args:
        checkpoint_info (dict): A dictionary containing 'val_loss', 'iter_num', and 'ckpt_path'.
    """
    # Add the new checkpoint info
    best_checkpoints.append(checkpoint_info)
    # Sort the checkpoints based on validation loss (lower is better)
    best_checkpoints.sort(key=lambda x: x['val_loss'])
    # Keep only the top num_checkpoints
    if len(best_checkpoints) > config.num_checkpoints:
        # Remove the checkpoint with the highest validation loss
        worst_ckpt = best_checkpoints.pop()
        # Delete the checkpoint file
        if os.path.exists(worst_ckpt['ckpt_path']):
            os.remove(worst_ckpt['ckpt_path'])
            print(f"Removed old checkpoint {worst_ckpt['ckpt_path']} with val_loss {worst_ckpt['val_loss']:.4f}")

# Initialize a list to keep track of best checkpoints
best_checkpoints = []
# Load existing checkpoints if resuming
if config.init_from == 'resume':
    load_existing_checkpoints()

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

X, Y = get_batch('train')  # Fetch the first batch
t0 = time.time()
local_iter_num = 0  # Number of iterations in this process
raw_model = model.module if ddp else model  # Unwrap DDP container if needed
running_mfu = -1.0

while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # In the training loop, during evaluation and checkpoint saving
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_loss = losses['val']
        print(f"Step {iter_num}: Train loss {losses['train']:.4f}, Val loss {val_loss:.4f}")
        if config.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": val_loss,
                "lr": lr,
                "mfu": running_mfu * 100,  # Convert to percentage
            })
        if val_loss < best_val_loss or config.always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                # Checkpoint saving logic
                if config.save_lora_only and config.init_from == 'finetune':
                    # Save only LoRA parameters
                    model.save_pretrained(config.lora_ckpt_path)
                    print(f"Saved LoRA parameters to {config.lora_ckpt_path}")
                else:
                    # Save the full model checkpoint
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': OmegaConf.to_container(config),
                    }
                    # Save the checkpoint with val_loss in the filename
                    ckpt_path = os.path.join(config.out_dir, f'ckpt_iter_{iter_num}_val_{val_loss:.4f}.pt')
                    print(f"Saving checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)
                    # Update the list of best checkpoints
                    checkpoint_info = {
                        'val_loss': val_loss.item(),
                        'iter_num': iter_num,
                        'ckpt_path': ckpt_path,
                    }
                    update_best_checkpoints(checkpoint_info)

        # Save the current model to current_ckpt.pt if save_current_ckpt is True
        if config.save_current_ckpt:
            if config.save_lora_only and config.init_from == 'finetune':
                # Save only LoRA parameters
                model.save_pretrained(config.lora_ckpt_path)
                print(f"Saved current LoRA parameters to {config.lora_ckpt_path}")
            else:
                current_ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': OmegaConf.to_container(config),
                }
                current_ckpt_path = os.path.join(config.out_dir, 'current_ckpt.pt')
                print(f"Saving current checkpoint to {current_ckpt_path}")
                torch.save(current_ckpt, current_ckpt_path)

    if iter_num == 0 and config.eval_only:
        break

    # Forward, backward, and update
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, main_loss, aux_loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps  # Scale loss for gradient accumulation
        # Prefetch next batch
        X, Y = get_batch('train')
        # Backward pass with gradient scaling if using fp16
        scaler.scale(loss).backward()
    # Gradient clipping
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)  # Reset gradients

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config.log_interval == 0 and master_process:
        # Get loss as float (this is a CPU-GPU sync point)
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5:  # Let training loop settle
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        # Gather MoE usage statistics
        if config.use_moe:
            moe_usages = []
            for block in raw_model.transformer.h:
                if hasattr(block.mlp, 'get_usage_percentages'):
                    usage_percentages = block.mlp.get_usage_percentages()
                    usage_str = ", ".join([f"Expert {i}: {usage_percentages[i]:.2f}%" for i in range(len(usage_percentages))])
                    moe_usages.append(f"Block MoE usage: {usage_str}")
                    # Reset usage counts after logging
                    block.mlp.reset_usage_counts()
        else:
            moe_usages = []
        print(f"Iter {iter_num}: Loss {lossf:.4f}, Main Loss {main_loss.item():.4f}, Aux Loss {aux_loss.item():.4f}, "
              f"Time {dt*1000:.2f}ms, MFU {running_mfu*100:.2f}%")
        for usage in moe_usages:
            print(usage)
        if config.wandb_log:
            wandb.log({
                "iter": iter_num,
                "loss": lossf,
                "main_loss": main_loss.item(),
                "aux_loss": aux_loss.item(),
                "mfu": running_mfu * 100,
                "lr": lr,
            })
    iter_num += 1
    local_iter_num += 1

    # Termination conditions
    if iter_num > config.max_iters:
        break

if ddp:
    destroy_process_group()

# -----------------------------------------------------------------------------
# Saving the final model
# -----------------------------------------------------------------------------

if master_process and config.init_from == 'finetune':
    if not config.save_lora_only:
        # Merge LoRA parameters into the base model and save
        print("Merging LoRA parameters into the base model")
        model = model.merge_and_unload()
        # Save the merged model
        torch.save(model.state_dict(), config.merged_model_ckpt_path)
        print(f"Saved merged model to {config.merged_model_ckpt_path}")
