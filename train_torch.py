"""
Training script for GPT model with Mixture of Experts (MoE) integration.
This script supports both single GPU and distributed data parallel (DDP) training.

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=1 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

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

from model_torch import GPTConfig, GPT, MoE

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
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Ensure that model configurations match
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Fix state_dict keys if needed
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    # Handle other initialization methods (e.g., from pretrained GPT-2)
    pass  # Implement as needed

# Adjust block size if necessary
if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    model_args['block_size'] = config.block_size

model.to(device)

# Initialize a GradScaler if using float16
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(
    config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
if config.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # Free up memory

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

    # Evaluate the loss and save checkpoints
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"Step {iter_num}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
        if config.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # Convert to percentage
            })
        if losses['val'] < best_val_loss or config.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': OmegaConf.to_container(config),
                }
                print(f"Saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
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
                if isinstance(block.mlp, MoE):
                    usage_percentages = block.mlp.get_usage_percentages()
                    usage_str = ", ".join([f"Expert {i}: {usage_percentages[i]:.2f}%" for i in range(block.mlp.num_experts)])
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
