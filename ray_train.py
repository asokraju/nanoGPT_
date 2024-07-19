import torch
import ray
import os
import glob
import logging
from ray.train.torch import TorchTrainer
from ray.train import CheckpointConfig, RunConfig, ScalingConfig


def find_highest_checkpoint_folder(base_path):
    """
    Finds the highest numbered checkpoint folder within a TorchTrainer_ directory.
    
    :param base_path: The base directory path where TorchTrainer_ folders are expected.
    :return: The path to the highest numbered checkpoint folder.
    :raises Exception: If no TorchTrainer_ or checkpoint folders are found.
    """
    # Find the TorchTrainer_ folder
    torch_trainer_folders = glob.glob(os.path.join(base_path, "TorchTrainer_*"))
    if not torch_trainer_folders:
        raise Exception("No TorchTrainer_ folders found.")
    
    # Assuming there is only one such folder, or you want the first one if sorted
    torch_trainer_folder = sorted(torch_trainer_folders)[0]
    
    # Find the highest numbered checkpoint folder
    checkpoint_folders = glob.glob(os.path.join(torch_trainer_folder, "checkpoint_*"))
    if not checkpoint_folders:
        raise Exception("No checkpoint folders found.")
    
    # Sort the folders to find the highest number (lexical sort works because of zero padding)
    highest_checkpoint_folder = sorted(checkpoint_folders)[-1]
    
    return highest_checkpoint_folder

def get_batch(data, block_size, batch_size, device):
    import numpy as np
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if 'cuda' in device[:4]:
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, ctx, config, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        data = train_data if split == 'train' else val_data
        for k in range(eval_iters):
            X, Y = get_batch(data, config['block_size'], config['batch_size'], device)
            with ctx:
                logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    import math
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train_func(config):
    from model import GPT, GPTConfig
    import torch
    from contextlib import nullcontext
    import numpy as np
    import time
    import os
    import tempfile
    import math
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group, is_initialized
    from torch.cuda.amp import GradScaler
    # Setup device and dtype
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device_type = 'cuda' if 'cuda' in device.type else 'cpu'
    # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    import logging
    logging.basicConfig(level=logging.INFO)
    # DDP
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        if not is_initialized():
            init_process_group(backend=config["backend"])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config["gradient_accumulation_steps"] % ddp_world_size == 0
        config["gradient_accumulation_steps"] //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = config["gradient_accumulation_steps"] * ddp_world_size * config["batch_size"] * config["block_size"]
    logging.info(f"tokens per iteration will be: {tokens_per_iter:,}")

    # seed:
    torch_seed = config["seed"] + seed_offset
    torch.manual_seed(torch_seed)
    logging.info(f"random seed set to: {torch_seed}")

    # Check and set TF32 for matmul operations
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
        logging.info("TF32 for matrix multiplication is enabled.")
    else:
        logging.warning("TF32 for matrix multiplication is not supported on this version of PyTorch.")

    # Check and set TF32 for cuDNN
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
        logging.info("TF32 for cuDNN is enabled.")
    else:
        logging.warning("TF32 for cuDNN is not supported on this version of PyTorch.")
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    logging.info(f"device: {device}, device type: {device_type}")

    # Determine the appropriate dtype for mixed precision
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = 'bfloat16'
            logging.info("CUDA is available and bfloat16 is supported. Using bfloat16.")
        else:
            dtype = 'float16'
            logging.info("CUDA is available but bfloat16 is not supported. Using float16.")
    else:
        dtype = 'float16'
        logging.info("CUDA is not available. Using float16.")

    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
        }[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    logging.info(f"Using context: {ctx}")
    
    if config["init_from"] == "scratch":
        # Model configuration
        model_args = {k: v for k, v in config.items() if k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'dropout', 'bias', 'vocab_size']}
        # Logging the filtered model_args
        logging.info("Filtered model arguments:")
        for k, v in model_args.items():
            logging.info(f"{k}: {v}")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(device)
        iter_num = 0
        best_val_loss = 1e9
        start_epoch = 0
        start_batch_idx = 0

    if config["init_from"] == "resume":
        logging.info("Resuming the model")
        if not os.path.exists(config["checkpoint_dir"]):
            message = f"Checkpoint directory does not exist: {config['checkpoint_dir']}"
            logging.info(message)
            raise FileNotFoundError(message)
        else:
            torch_save_path = find_highest_checkpoint_folder(config["checkpoint_dir"])
            logging.info(f"Loading checkpoint from: {torch_save_path}")
        ckpt_path = os.path.join(torch_save_path, 'ckpt.pt')
        checkpoint_dict = torch.load(ckpt_path, map_location=device)
        model_args = checkpoint_dict['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        #     model_args[k] = checkpoint_model_args[k]
        # create the model

        logging.info(f"Loading model with checkpoint model arguments:{model_args}")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(device)
        state_dict = checkpoint_dict['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint_dict['iter_num']
        best_val_loss = checkpoint_dict['best_val_loss']
        start_epoch = checkpoint_dict["current_epoch"]
        start_batch_idx = checkpoint_dict["current_batch_idx"]
    
    # Optimizer setup using configure_optimizers method
    betas = (config['beta1'], config['beta2'])
    logging.info(f"Optimizer betas: {betas}")
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], betas, device_type)
    logging.info(f"Configured optimizer: {optimizer}")
    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # Compile for faster training
    if config["compile"]:
        logging.info("Compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0
    # Data loading setup
    data_dir = os.path.join(config['dataset'])
    logging.info(f"loading dataset, {data_dir}")
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    logging.info(f"train dataset loaded {train_data}")
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    logging.info(f"validation dataset loaded {val_data}")


    # Initialize training variables
    eval_interval = config.get('eval_interval', 2000)
    log_interval = config.get('log_interval', 1)
    eval_iters = config.get('eval_iters', 200)
    always_save_checkpoint = config.get('always_save_checkpoint', True)
    grad_clip = config.get('grad_clip', 1.0)
    out_dir = config.get('out_dir', 'out')
    os.makedirs(out_dir, exist_ok=True)
    
    # Training loop
    t0 = time.time()
    num_batches = len(train_data) // (config['block_size'] * config['batch_size'])
    logging.info(f"Total epochs:{config['num_epochs']}")
    logging.info(f"Total batches per epoch:{num_batches}")
    logging.info(f"Total micro steps per batch:{config['gradient_accumulation_steps']}")
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        total_loss = 0
        batch_start = start_batch_idx if epoch == start_epoch else 0
        for batch_idx in range(batch_start, num_batches):
            X, Y = get_batch(train_data, config['block_size'], config['batch_size'], device)
            optimizer.zero_grad()

            for micro_step in range(config['gradient_accumulation_steps']):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == config["gradient_accumulation_steps"] - 1)
                with ctx:
                    logits, loss = model(X, targets=Y)
                    loss = loss / config['gradient_accumulation_steps']
                if (micro_step + 1) < config['gradient_accumulation_steps']:
                    # Prefetch the next batch
                    X, Y = get_batch(train_data, config['block_size'], config['batch_size'], device)
                scaler.scale(loss).backward()
            # Clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

            if iter_num % log_interval == 0 and master_process:
                lossf = total_loss / (batch_idx + 1)
                logging.info(f"Epoch: {epoch}/{config['num_epochs']}, Batch: {batch_idx}/{num_batches}, Loss: {lossf:.4f}")


            # Determine and set the learning rate for this iteration
            lr = get_lr(iter_num, config['warmup_iters'], config['learning_rate'], config['lr_decay_iters'], config['min_lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Evaluate and log losses
            if iter_num % eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data, eval_iters, ctx, config, device)
                logging.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, , time {dt*1000:.2f}ms")
                metrics = {"loss": loss.item(), "epoch": epoch, "batch": batch_idx}
                ray.train.report(metrics=metrics)
                # Conditional checkpointing based on validation loss improvement
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = min(best_val_loss, losses['val'])  # Update best validation loss
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        # rank = ray.train.get_context().get_world_rank()
                        checkpoint_dict = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                            'current_epoch':epoch,
                            "current_batch_idx": batch_idx
                        }
                        # torch.save(checkpoint_dict, os.path.join(temp_checkpoint_dir, f"model-rank={rank}.pt"))
                        torch.save(checkpoint_dict, os.path.join(temp_checkpoint_dir, f"ckpt.pt"))
                        checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                        ray.train.report(metrics=metrics, checkpoint=checkpoint)
                        logging.info(f"saving checkpoint at iteration {iter_num}, epoch {epoch}")


                # Optional: Print metrics on the main process
                if ray.train.get_context().get_world_rank() == 0:
                    logging.info(f"Epoch {epoch}: {metrics}")

            iter_num += 1
    if ddp:
        destroy_process_group()
    return {"loss": total_loss / num_batches, "model_state_dict": model.state_dict()}


def main():
    config = {
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024,
        'dropout': 0.1,
        'bias': False,
        'vocab_size': 50257,  # Update based on your dataset
        'learning_rate': 6e-4,
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        'num_epochs': 10,
        'batch_size': 13,
        'gradient_accumulation_steps': 8,
        'eval_interval': 2000,
        'log_interval': 100,
        'eval_iters': 200,
        'warmup_iters': 2000,
        'lr_decay_iters': 600000,
        'min_lr': 6e-5,
        'dataset': '/home/kosaraju/raytrain_nanogpt/data/openwebtext',
        'compile': True,
        'always_save_checkpoint': True,
        'out_dir': 'out',
        "backend":'nccl',  # 'nccl', 'gloo', what kind of GPU are we using Nvidia for nccl,
        "seed": 1234,
        "init_from": "resume",  # scratch, resume
        "checkpoint_dir": "/home/kosaraju/ray_results/TorchTrainer_2024-07-19_00-03-02",
    }

    run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=2))
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True)

    trainer = TorchTrainer(train_func, scaling_config=scaling_config, train_loop_config=config, run_config=run_config)
    result = trainer.fit()
    print("Training completed with result:", result)

if __name__ == '__main__':
    main()
