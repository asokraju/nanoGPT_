
'''
Call Backs for MoE Utilization
================================

1. **MoEUsageLoggingCallback**
   - **Purpose:** A custom callback for Hugging Face's `Trainer` to log Mixture of Experts (MoE) usage statistics to the console.
   - **Methods:**
     - `__init__(self, logger=None, log_interval=100)`: Initializes the callback with an optional logger and logging interval.
     - `on_step_end(...)`: Called at the end of each training step to log expert usage if the interval is met.
     - `on_epoch_end(...)`: Optionally logs expert usage at the end of each epoch.
     - `_get_moe_layers(self, model) -> list`: Retrieves all MoE layers from the model.

2. **MoEUsageTensorBoardCallback**
    - **Purpose:** A custom callback for Hugging Face's `Trainer` to log Mixture of Experts (MoE) usage statistics to TensorBoard for advanced visualization.
    - **Methods:**
      - `__init__(self, log_interval=100, tb_writer=None)`: Initializes the callback with an optional TensorBoard `SummaryWriter` and logging interval.
      - `on_step_end(...)`: Called at the end of each training step to log expert usage to TensorBoard if the interval is met.
      - `on_train_end(...)`: Closes the TensorBoard writer at the end of training.
      - `_get_moe_layers(self, model) -> list`: Retrieves all MoE layers from the model.
'''

from transformers import TrainerCallback, TrainerState, TrainerControl
import logging
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

class MoEUsageTensorBoardCallback(TrainerCallback):
    """
    A custom callback for Hugging Face's Trainer to log Mixture of Experts (MoE) usage statistics to TensorBoard.
    """
    def __init__(self, log_interval=100, tb_writer=None):
        """
        Initializes the callback.
        
        Args:
            log_interval (int): Number of training steps between logs.
            tb_writer (SummaryWriter, optional): TensorBoard SummaryWriter instance. If None, a default writer is created.
        """
        super().__init__()
        self.log_interval = log_interval
        self.steps_since_last_log = 0
        self.tb_writer = tb_writer or SummaryWriter()
    
    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the end of a training step.
        
        Args:
            args: TrainingArguments instance.
            state (TrainerState): State information.
            control (TrainerControl): Control flags.
            **kwargs: Additional arguments.
        """
        self.steps_since_last_log += 1
        if self.steps_since_last_log >= self.log_interval:
            model = kwargs.get("model")
            if model is not None:
                moe_layers = self._get_moe_layers(model)
                for idx, moe in enumerate(moe_layers):
                    usage_counts = moe.get_usage_counts()
                    total_assignments = moe.total_assignments
                    if total_assignments > 0:
                        usage_percentages = (usage_counts.float() / total_assignments * 100).cpu().numpy()
                    else:
                        usage_percentages = usage_counts.cpu().numpy()
                    
                    for expert_id, usage_pct in enumerate(usage_percentages):
                        tag = f"MoE_Layer_{idx+1}/Expert_{expert_id}_Usage_Percentage"
                        self.tb_writer.add_scalar(tag, usage_pct, state.global_step)
                
                # Optionally, reset counts after logging
                for moe in moe_layers:
                    moe.reset_usage_counts()
            
            self.steps_since_last_log = 0
    
    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the end of training.
        
        Args:
            args: TrainingArguments instance.
            state (TrainerState): State information.
            control (TrainerControl): Control flags.
            **kwargs: Additional arguments.
        """
        self.tb_writer.close()
    
    def _get_moe_layers(self, model) -> list:
        """
        Retrieves all MoE layers from the model.
        
        Args:
            model: The model instance.
        
        Returns:
            list: A list of MoE layer instances.
        """
        moe_layers = []
        for block in model.transformer.h:
            if isinstance(block.mlp, MoE):
                moe_layers.append(block.mlp)
        return moe_layers


class MoEUsageLoggingCallback(TrainerCallback):
    """
    A custom callback for Hugging Face's Trainer to log Mixture of Experts (MoE) usage statistics.
    """
    def __init__(self, logger=None, log_interval=100):
        """
        Initializes the callback.
        
        Args:
            logger (logging.Logger, optional): Logger instance. If None, the root logger is used.
            log_interval (int): Number of training steps between logs.
        """
        super().__init__()
        self.log_interval = log_interval
        self.steps_since_last_log = 0
        self.logger = logger or logging.getLogger(__name__)
    
    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the end of a training step.
        
        Args:
            args: TrainingArguments instance.
            state (TrainerState): State information.
            control (TrainerControl): Control flags.
            **kwargs: Additional arguments.
        """
        self.steps_since_last_log += 1
        if self.steps_since_last_log >= self.log_interval:
            model = kwargs.get("model")
            if model is not None:
                moe_layers = self._get_moe_layers(model)
                for idx, moe in enumerate(moe_layers):
                    usage_counts = moe.get_usage_counts()
                    total_assignments = moe.total_assignments
                    if total_assignments > 0:
                        usage_percentages = (usage_counts.float() / total_assignments * 100).cpu().numpy()
                    else:
                        usage_percentages = usage_counts.cpu().numpy()
                    
                    usage_str = ", ".join([f"Expert {i}: {usage_percentages[i]:.2f}%" for i in range(moe.num_experts)])
                    self.logger.info(f"MoE Layer {idx + 1} Usage: {usage_str}")
                    
                    # Optionally, reset counts after logging
                    moe.reset_usage_counts()
            
            self.steps_since_last_log = 0
    
    def on_epoch_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Optionally, log usage statistics at the end of each epoch.
        
        Args:
            args: TrainingArguments instance.
            state (TrainerState): State information.
            control (TrainerControl): Control flags.
            **kwargs: Additional arguments.
        """
        model = kwargs.get("model")
        if model is not None:
            moe_layers = self._get_moe_layers(model)
            for idx, moe in enumerate(moe_layers):
                usage_counts = moe.get_usage_counts()
                total_assignments = moe.total_assignments
                if total_assignments > 0:
                    usage_percentages = (usage_counts.float() / total_assignments * 100).cpu().numpy()
                else:
                    usage_percentages = usage_counts.cpu().numpy()
                
                usage_str = ", ".join([f"Expert {i}: {usage_percentages[i]:.2f}%" for i in range(moe.num_experts)])
                self.logger.info(f"End of Epoch MoE Layer {idx + 1} Usage: {usage_str}")
                
                # Optionally, reset counts after logging
                moe.reset_usage_counts()

    def _get_moe_layers(self, model) -> list:
        """
        Retrieves all MoE layers from the model.
        
        Args:
            model: The model instance.
        
        Returns:
            list: A list of MoE layer instances.
        """
        moe_layers = []
        for block in model.transformer.h:
            if isinstance(block.mlp, MoE):
                moe_layers.append(block.mlp)
        return moe_layers


# Usage
'''
## Usage with tensorboad
```python
from transformers import Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard SummaryWriter
tb_writer = SummaryWriter(log_dir="./tensorboard_logs")

# Initialize the TensorBoard logging callback
moe_tb_callback = MoEUsageTensorBoardCallback(
    log_interval=100,
    tb_writer=tb_writer
)

# Initialize the Trainer with the TensorBoard callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[moe_tb_callback],  # Add the TensorBoard callback here
)

# Start training
trainer.train()
```

## Usage with logging

Setting Up the Logger
```python
import logging

# Configure the root logger or create a specific logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MoELogger')

```

Initializing the Callback
```python
# Initialize the MoE usage logging callback
moe_logging_callback = MoEUsageLoggingCallback(
    logger=logger,
    log_interval=100  # Log every 100 training steps
)

```

Initializing the Trainer with the Callback
```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,  # Already handled by the callback
    # Other arguments as needed
)

# Assume train_dataset and eval_dataset are predefined
# train_dataset = ...
# eval_dataset = ...

# Initialize the Trainer with the MoE usage logging callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[moe_logging_callback],  # Add the custom callback here
)

# Start training
trainer.train()

```

'''
