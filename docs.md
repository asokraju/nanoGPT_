
GPT Language Model Implementation
================================

This module provides a complete implementation of a GPT (Generative Pre-trained Transformer)
language model, including optional Mixture of Experts (MoE) layers for enhanced capacity.
The model is built using PyTorch and integrates with Hugging Face's Transformers library for
configuration and output handling. Additionally, it includes mechanisms to log and monitor
the usage of experts within MoE layers, facilitating balanced utilization and model diagnostics.

Classes
-------

1. **GPTConfig**
   - **Purpose:** Configuration class to store all configuration parameters for the GPT model.
   - **Attributes:**
     - `block_size` (int): Maximum sequence length.
     - `vocab_size` (int): Size of the vocabulary.
     - `n_layer` (int): Number of transformer layers.
     - `n_head` (int): Number of attention heads.
     - `n_embd` (int): Embedding dimensionality.
     - `dropout` (float): Dropout rate.
     - `bias` (bool): Whether to include bias terms in linear layers.
     - `use_moe` (bool): Whether to use Mixture of Experts.
     - `num_experts` (int): Number of experts in MoE.
     - `num_experts_per_tok` (int): Number of experts to route per token.
     - `moe_loss` (bool): Whether to include MoE auxiliary loss for load balancing.
     - `moe_loss_type` (str): Type of load balancing loss (`'variance_penalty'`, `'entropy_regularization'`, `'diversity_regularization'`).
     - `moe_loss_coef` (float): Coefficient for the auxiliary loss.

2. **LayerNorm**
   - **Purpose:** Custom Layer Normalization with an optional bias.
   - **Methods:**
     - `__init__(self, ndim: int, bias: bool)`: Initializes the LayerNorm layer.
     - `forward(self, input: torch.Tensor) -> torch.Tensor`: Applies layer normalization to the input tensor.

3. **CausalSelfAttention**
   - **Purpose:** Implements the causal (masked) self-attention mechanism, supporting both standard and Flash Attention.
   - **Methods:**
     - `__init__(self, config: GPTConfig)`: Initializes the self-attention layer with configuration parameters.
     - `forward(self, x: torch.Tensor) -> torch.Tensor`: Performs the forward pass of self-attention on the input tensor.

4. **MLP**
   - **Purpose:** Implements the Feed-Forward Network (MLP) component of a Transformer block.
   - **Methods:**
     - `__init__(self, config: GPTConfig)`: Initializes the MLP with two linear layers and activation.
     - `forward(self, x: torch.Tensor) -> torch.Tensor`: Applies the MLP to the input tensor.

5. **MoE**
   - **Purpose:** Implements a Mixture of Experts (MoE) layer to dynamically route inputs to a subset of expert networks.
     - **Additional Features:** Tracks and logs expert usage statistics to ensure balanced utilization.
   - **Methods:**
     - `__init__(self, config: GPTConfig)`: Initializes the MoE layer with multiple experts and a gating mechanism.
     - `forward(self, x: torch.Tensor) -> torch.Tensor`: Routes inputs to experts based on gating scores, aggregates their outputs, and computes auxiliary loss if enabled.
     - `get_auxiliary_loss(self) -> torch.Tensor`: Retrieves the computed auxiliary loss.
     - `get_usage_counts(self) -> torch.Tensor`: Retrieves the current usage counts for each expert.
     - `reset_usage_counts(self)`: Resets the expert usage counts and total assignments, typically called after logging.

6. **Block**
   - **Purpose:** Represents a single Transformer block, comprising LayerNorm, Causal Self-Attention, and an MLP or MoE.
   - **Methods:**
     - `__init__(self, config: GPTConfig)`: Initializes the Transformer block with normalization, attention, and MLP/MoE components.
     - `forward(self, x: torch.Tensor) -> torch.Tensor`: Applies the Transformer block operations with residual connections.

7. **GPT**
   - **Purpose:** The core GPT model class extending Hugging Face's `PreTrainedModel`. It assembles the entire transformer architecture.
   - **Attributes:**
     - `transformer` (nn.ModuleDict): Contains embeddings, dropout, Transformer blocks, and final normalization.
     - `lm_head` (nn.Linear): Language modeling head tied to token embeddings.
   - **Methods:**
     - `__init__(self, config: GPTConfig)`: Initializes the GPT model with embeddings, Transformer blocks, and output head.
     - `get_input_embeddings(self) -> nn.Embedding`: Retrieves the input embeddings.
     - `set_input_embeddings(self, new_embeddings: nn.Embedding)`: Sets new input embeddings.
     - `get_num_params(self, non_embedding: bool = True) -> int`: Returns the total number of model parameters.
     - `_init_weights(self, module: nn.Module)`: Initializes model weights.
     - `forward(...) -> BaseModelOutputWithPastAndCrossAttentions`: Defines the forward pass of the GPT model.
     - `crop_block_size(self, block_size: int)`: Adjusts the model's block size by cropping position embeddings and attention masks.
     - `configure_optimizers(...) -> torch.optim.Optimizer`: Configures the optimizer with appropriate weight decay settings.
     - `estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float`: Estimates Model FLOPs Utilization.
     - `prepare_inputs_for_generation(...) -> dict`: Prepares inputs for generation tasks.

8. **GPTLMHeadModel**
   - **Purpose:** Extends the `GPT` class to include a language modeling head, facilitating tasks like text generation and next-token prediction.
   - **Methods:**
     - `__init__(self, config: GPTConfig)`: Initializes the GPTLMHeadModel, leveraging the base GPT model's components.
     - `get_input_embeddings(self) -> nn.Embedding`: Retrieves the input embeddings.
     - `set_input_embeddings(self, new_embeddings: nn.Embedding)`: Sets new input embeddings.
     - `get_output_embeddings(self) -> nn.Linear`: Retrieves the output embeddings.
     - `set_output_embeddings(self, new_embeddings: nn.Linear)`: Sets new output embeddings and ties weights.
     - `prepare_inputs_for_generation(...) -> dict`: Prepares inputs specifically for generation tasks.
     - `forward(...) -> CausalLMOutputWithCrossAttentions`: Defines the forward pass, computing logits and optionally loss if labels are provided.

9. **MoEUsageLoggingCallback**
   - **Purpose:** A custom callback for Hugging Face's `Trainer` to log Mixture of Experts (MoE) usage statistics to the console.
   - **Methods:**
     - `__init__(self, logger=None, log_interval=100)`: Initializes the callback with an optional logger and logging interval.
     - `on_step_end(...)`: Called at the end of each training step to log expert usage if the interval is met.
     - `on_epoch_end(...)`: Optionally logs expert usage at the end of each epoch.
     - `_get_moe_layers(self, model) -> list`: Retrieves all MoE layers from the model.
    - Usage with logging
        - Setting Up the Logger
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

        - Initializing the Trainer with the Callback
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
10. **MoEUsageTensorBoardCallback**
    - **Purpose:** A custom callback for Hugging Face's `Trainer` to log Mixture of Experts (MoE) usage statistics to TensorBoard for advanced visualization.
    - **Methods:**
      - `__init__(self, log_interval=100, tb_writer=None)`: Initializes the callback with an optional TensorBoard `SummaryWriter` and logging interval.
      - `on_step_end(...)`: Called at the end of each training step to log expert usage to TensorBoard if the interval is met.
      - `on_train_end(...)`: Closes the TensorBoard writer at the end of training.
      - `_get_moe_layers(self, model) -> list`: Retrieves all MoE layers from the model.
    - Usage with tensorboad
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
'''
Notes
-----

- **Mixture of Experts (MoE):** If `use_moe` is set to `True` in the configuration, the model integrates MoE layers to dynamically enhance its capacity by routing inputs to specialized experts.
- **Weight Tying:** The language modeling head (`lm_head`) is tied with the token embeddings (`wte`) to reduce the number of parameters and improve performance.
- **Flash Attention:** If available, the model leverages Flash Attention for efficient computation. Ensure that your PyTorch version supports this feature.
- **Device Compatibility:** The model is designed to work seamlessly on both CPU and CUDA-enabled GPUs. Ensure that tensors and the model are moved to the appropriate device before training or inference.
- **Extensibility:** The modular design allows for easy extension and customization, such as adding more Transformer blocks, experimenting with different activation functions, or integrating additional regularization techniques.

### References
https://github.com/Antlera/nanoGPT-moe
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py