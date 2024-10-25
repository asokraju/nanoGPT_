"""
Training script for GPT model with Mixture of Experts (MoE) integration.
This script supports both single GPU and distributed data parallel (DDP) training.
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

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default configuration values for training
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False  # If True, script exits right after the first eval
always_save_checkpoint = True  # If True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

# WandB logging
wandb_log = False  # Disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # Used to simulate larger batch sizes
batch_size = 12  # If gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # For pretraining 0 is good, for finetuning try 0.1+
bias = False  # Do we use bias inside LayerNorm and Linear layers?
# MoE parameters
use_moe = True
num_experts = 4
num_experts_per_tok = 2
moe_loss = True
moe_loss_type = 'variance_penalty'
moe_loss_coef = 0.01

# AdamW optimizer
learning_rate = 6e-4  # Max learning rate
max_iters = 600000  # Total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # Clip gradients at this value, or disable if == 0.0

# Learning rate decay settings
decay_lr = True  # Whether to decay the learning rate
warmup_iters = 2000  # How many steps to warm up for
lr_decay_iters = 600000  # Should be ~= max_iters per Chinchilla
min_lr = 6e-5  # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.

# System
device = 'cuda'  # Examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # Use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
# Override config values from command line or config file
# -----------------------------------------------------------------------------
# (You can add code here to override defaults if needed)

# -----------------------------------------------------------------------------
# Setup for DDP and device configuration
# -----------------------------------------------------------------------------

# Check if running with DDP (Distributed Data Parallel)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # This process will do logging, checkpointing, etc.
    seed_offset = ddp_rank  # Each process gets a different seed
    # Scale down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # If not DDP, we are running on a single GPU and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cuDNN
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

# Define data directories
data_dir = os.path.join('data', dataset)

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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
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
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    use_moe=use_moe,
    num_experts=num_experts,
    num_experts_per_tok=num_experts_per_tok,
    moe_loss=moe_loss,
    moe_loss_type=moe_loss_type,
    moe_loss_coef=moe_loss_coef,
)

# Initialize the model
if init_from == 'scratch':
    # Initialize a new model from scratch
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("Defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # Resume training from a checkpoint
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# Initialize a GradScaler if using float16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # Free up memory

# Compile the model (requires PyTorch 2.0)
if compile:
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
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
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
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # After decay
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Ranges from 1 to 0
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=model_args)

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
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss and save checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"Step {iter_num}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # Convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': model_args,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Forward, backward, and update
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
        # Prefetch next batch
        X, Y = get_batch('train')
        # Backward pass with gradient scaling if using fp16
        scaler.scale(loss).backward()
    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)  # Reset gradients

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # Get loss as float (this is a CPU-GPU sync point)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # Let training loop settle
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"Iter {iter_num}: Loss {lossf:.4f}, Time {dt*1000:.2f}ms, MFU {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # Termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
