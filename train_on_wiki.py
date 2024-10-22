import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, GPT2Tokenizer, default_data_collator
from typing import Optional
import logging
from torch.utils.tensorboard import SummaryWriter
from model_hf import GPTConfig, GPTLMHeadModel, MoEUsageLoggingCallback
import numpy as np
from omegaconf import OmegaConf


# Define a small synthetic dataset for debugging
class SyntheticDataset(Dataset):
    def __init__(self, vocab_size: int, block_size: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.block_size,))
        labels = input_ids.clone()  # For simplicity, labels are the same as input_ids
        return {'input_ids': input_ids, 'labels': labels}

# Define the NumpyMemmapDataset class
class NumpyMemmapDataset(Dataset):
    """
    PyTorch Dataset for loading numpy memmapped arrays efficiently.
    """
    def __init__(self, filename="data/mock_train.bin", block_size=1024, device="cpu", iterate='random', 
                 return_labels=False, eval_data=False, eval_samples=1000):
        """
        Args:
        - filename (str): Path to the binary file.
        - block_size (int): Length of the sequence.
        - device (str): Device to move the tensors ('cpu' or 'cuda').
        - iterate (str): 'random' or 'linear'. If 'random', samples random slices from the dataset.
        - return_labels (bool): Whether to return labels (used for GPT-2 LMHeadModel).
        - eval_data (bool): If True, dataset works in evaluation mode and samples a limited number of sequences.
        - eval_samples (int): Number of samples for evaluation mode (if eval_data=True).
        """
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')  # Load binary file with memmap
        self.max_len = len(self.data)
        self.block_size = block_size
        self.iterate = iterate
        self.device = torch.device(device)
        self.return_labels = return_labels
        self.eval_data = eval_data
        self.eval_samples = eval_samples

    def __len__(self):
        # If evaluation mode, return eval_samples, otherwise return full length
        return self.eval_samples if self.eval_data else self.max_len // self.block_size

    def __getitem__(self, index):
        # Handle random sampling if 'iterate' is set to 'random'
        if self.iterate == 'random':
            index = torch.randint(0, self.max_len // self.block_size, (1,)).item()

        idx = index * self.block_size
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))

        if self.device.type == 'cuda':
            # Pin memory for GPU asynchronous transfer
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)

        # If `return_labels` is true, return both input and target (for GPT training)
        if self.return_labels:
            return {"input_ids": x, "labels": y}
        return x


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MoELogger')


# Load the GPT-2 tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)

# Add special tokens, such as PAD token (if not already present)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Set pad token to eos token

# Load config using OmegaConf
config_file = 'config.yaml'  # Path to your config file
cfg = OmegaConf.load(config_file)


# Prepare the model configuration
model_config = GPTConfig(**cfg.model)
model = GPTLMHeadModel(model_config)
train_dataset = NumpyMemmapDataset(**cfg.dataset.train)
eval_dataset = NumpyMemmapDataset(**cfg.dataset.eval)
training_args = TrainingArguments(**cfg.train.training_args)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# Optional MoE logging callback
moe_logging_callback = MoEUsageLoggingCallback(
    trainer=trainer, logger=logger,**cfg.train.moe_log
)
trainer.add_callback(moe_logging_callback)
# Train the model
trainer.train()

# Save the trained model
trainer.model.save_pretrained(cfg.train.model_save_dir)