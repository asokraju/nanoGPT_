"""
GPT Language Model Implementation
=================================

This module provides a complete implementation of a GPT (Generative Pre-trained Transformer)
language model, including optional Mixture of Experts (MoE) layers for enhanced capacity.
The model is built using PyTorch and integrates with Hugging Face's Transformers library for
configuration and output handling.

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
   - **Methods:**
     - `__init__(self, config: GPTConfig)`: Initializes the MoE layer with multiple experts and a gating mechanism.
     - `forward(self, x: torch.Tensor) -> torch.Tensor`: Routes inputs to experts based on gating scores and aggregates their outputs.

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

Usage
-----

To utilize the GPT language model, follow these steps:

1. **Configuration:**
   - Instantiate a `GPTConfig` object with desired parameters.
   ```python
   config = GPTConfig(
       block_size=1024,
       vocab_size=50304,
       n_layer=12,
       n_head=12,
       n_embd=768,
       dropout=0.1,
       use_moe=True,
       num_experts=8,
       num_experts_per_tok=2
   )
   ```

2. **Model Initialization:**
   - Create an instance of `GPTLMHeadModel` using the configuration.
   ```python
   model = GPTLMHeadModel(config)
   ```

3. **Forward Pass:**
   - Pass input tensors through the model to obtain outputs.
   ```python
   outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
   ```

4. **Optimization:**
   - Configure the optimizer as needed.
   ```python
   optimizer = model.configure_optimizers(
       weight_decay=0.01,
       learning_rate=5e-5,
       betas=(0.9, 0.999),
       device_type='cuda'
   )
   ```

5. **Generation:**
   - Prepare inputs for text generation tasks.
   ```python
   generation_inputs = model.prepare_inputs_for_generation(input_ids)
   generated_outputs = model.generate(**generation_inputs)
   ```

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
"""

import math
import inspect
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions
)


class GPTConfig(PretrainedConfig):
    """
    GPTConfig is a configuration class to store out configuration for our GPT model.
    It does so by extending the base class 'PretrainedConfig' from Hugging Face.
    """
    model_type = "custom_gpt"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,  # Corrected from n_layers to n_layer
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer  # Ensuring consistency in naming
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok


class LayerNorm(nn.Module):
    """
    LayerNorm with an optional bias.
    PyTorch's built-in LayerNorm does not support disabling the bias, so we implement it manually.
    """
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """
    Implements causal (masked) self-attention mechanism.
    Supports both standard and Flash Attention (if available).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1 / math.sqrt(self.head_dim)

        # Combined projection for query, key, and value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Determine if Flash Attention is available
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Create a causal mask to ensure that each position can only attend to previous positions
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Project inputs to query, key, and value
        qkv = self.c_attn(x)  # Shape: (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # Each shape: (B, T, C)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_head, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_head, T, head_dim)

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Apply output projection and dropout
        y = self.resid_dropout(self.c_proj(y))  # (B, T, C)
        return y


class MLP(nn.Module):
    """
    Implements the Feed-Forward Network (MLP) component of a Transformer block.
    Consists of two linear layers with a GELU activation in between and dropout for regularization.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd  # Typically, the hidden dimension is 4 times the embedding size
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)           # (B, T, 4*C)
        x = self.gelu(x)           # (B, T, 4*C)
        x = self.c_proj(x)         # (B, T, C)
        x = self.dropout(x)        # (B, T, C)
        return x


class MoE(nn.Module):
    """
    Implements a Mixture of Experts (MoE) layer.
    Dynamically routes inputs to a subset of expert networks based on gating scores.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()
        x_flat = x.view(-1, C)  # (B*T, C)

        # Compute gating scores
        scores = self.gate(x_flat)  # (B*T, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)  # (B*T, num_experts_per_tok)

        # Softmax the top-k scores
        topk_weights = F.softmax(topk_scores, dim=-1)  # (B*T, num_experts_per_tok)

        # Flatten the expert indices
        flat_expert_indices = topk_indices.view(-1)  # (B*T*num_experts_per_tok,)

        # Repeat inputs for each expert assignment
        x_repeated = x_flat.unsqueeze(1).repeat(1, self.num_experts_per_tok, 1).view(-1, C)  # (B*T*num_experts_per_tok, C)

        # Initialize output tensor
        y = torch.zeros_like(x_repeated, dtype=torch.float16, device=x.device)  # Using float16 for efficiency

        # Apply each expert to its assigned inputs
        for i, expert in enumerate(self.experts):
            mask = flat_expert_indices == i  # (B*T*num_experts_per_tok,)
            if mask.any():
                y[mask] = expert(x_repeated[mask])

        # Reshape y to (B, T, num_experts_per_tok, C)
        y = y.view(B, T, self.num_experts_per_tok, C)

        # Apply the weights
        topk_weights = topk_weights.unsqueeze(-1)  # (B*T, num_experts_per_tok, 1)
        y = y * topk_weights  # (B, T, num_experts_per_tok, C)

        # Sum the experts' outputs
        y = y.sum(dim=2)  # (B, T, C)

        return y


class Block(nn.Module):
    """
    Represents a single Transformer block, consisting of a LayerNorm, 
    Causal Self-Attention, and a Feed-Forward Network (MLP or MoE).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        
        if config.use_moe:
            print("Using Mixture of Experts (MoE) in MLP")
            self.mlp = MoE(config)
        else:
            print("Using regular MLP")
            self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        # Apply LayerNorm and Self-Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # Apply LayerNorm and MLP (or MoE) with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x


class GPT(PreTrainedModel):
    """
    The main GPT model class extending Hugging Face's PreTrainedModel.
    It consists of token and position embeddings, multiple Transformer blocks,
    and a final LayerNorm. The language modeling head is tied with the token embeddings.
    """
    config_class = GPTConfig

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        assert config.vocab_size is not None, "vocab_size must be defined in the configuration."
        assert config.block_size is not None, "block_size must be defined in the configuration."

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            "wpe": nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks
            "ln_f": LayerNorm(config.n_embd, bias=config.bias),  # Final LayerNorm
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Language modeling head

        # Weight tying between token embeddings and the language modeling head
        self.transformer.wte.weight = self.lm_head.weight  # Tied weights

        # Initialize all weights
        self.apply(self._init_weights)

        # Special scaled initialization for residual projections as per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.transformer.wte = new_embeddings

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Returns the total number of parameters in the model.
        
        Args:
            non_embedding (bool): If True, excludes position embeddings from the count.
        
        Returns:
            int: Total number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        """
        Initializes the weights of the model.
        - Linear layers are initialized with a normal distribution.
        - Embedding layers are initialized with a normal distribution.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        Forward pass for the GPT model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (B, T)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (B, T)
            labels (Optional[torch.Tensor]): Labels for computing the loss
            return_dict (Optional[bool]): Whether to return a dict
            past_key_values (Optional[torch.Tensor]): Past key values for caching
            head_mask (Optional[torch.FloatTensor]): Mask for attention heads
            inputs_embeds (Optional[torch.FloatTensor]): Input embeddings
            encoder_hidden_states (Optional[torch.Tensor]): Encoder hidden states
            encoder_attention_mask (Optional[torch.FloatTensor]): Encoder attention mask
            use_cache (Optional[bool]): Whether to use caching
            output_attentions (Optional[bool]): Whether to output attentions
            output_hidden_states (Optional[bool]): Whether to output hidden states
        
        Returns:
            BaseModelOutputWithPastAndCrossAttentions: Model outputs
        """
        device = input_ids.device
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Generate position IDs
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # Shape: (1, T)

        # Compute token and position embeddings
        tok_emb = self.transformer.wte(input_ids)  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)        # (1, T, C)
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C)

        all_hidden_states = () if output_hidden_states else None

        # Pass through each Transformer block
        for block in self.transformer.h:
            if output_hidden_states:
                all_hidden_states += (x,)
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states += (x,)

        # Prepare output
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=x,
            past_key_values=None,  # To be implemented if caching is desired
            hidden_states=all_hidden_states,
            attentions=None,        # To be implemented if attention outputs are desired
            cross_attentions=None,  # To be implemented if cross attentions are desired
        )

    def crop_block_size(self, block_size: int):
        """
        Adjusts the model's block size by cropping position embeddings and attention masks.
        
        Args:
            block_size (int): The new block size to set.
        """
        assert block_size <= self.config.block_size, "New block size must be less than or equal to the original block size."
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str
    ) -> torch.optim.Optimizer:
        """
        Configures the optimizer with weight decay for specific parameters.
        
        Args:
            weight_decay (float): Weight decay coefficient.
            learning_rate (float): Learning rate.
            betas (Tuple[float, float]): Betas for the Adam optimizer.
            device_type (str): Type of device ('cuda' or 'cpu').
        
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        # Collect parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters for weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [            {"params": decay_params, "weight_decay": weight_decay},            {"params": nodecay_params, "weight_decay": 0.0},        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Number of decayed parameters: {len(decay_params)} with {num_decay_params:,} total parameters.")
        print(f"Number of non-decayed parameters: {len(nodecay_params)} with {num_nodecay_params:,} total parameters.")

        # Check if fused AdamW is available and desired
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimates the Model FLOPs Utilization (MFU) based on the number of forward-backward passes per iteration and time per iteration.
        
        Args:
            fwdbwd_per_iter (int): Number of forward-backward passes per iteration.
            dt (float): Time taken per iteration in seconds.
        
        Returns:
            float: MFU as a ratio of A100 bfloat16 peak FLOPS.
        """
        # Total number of parameters
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size

        # Estimate FLOPs per token
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # FLOPs achieved per second
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPS

        mfu = flops_achieved / flops_promised
        return mfu

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> dict:
        """
        Prepares inputs specifically for generation tasks.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            past_key_values (Optional[torch.Tensor]): Cached past key values.
            inputs_embeds (Optional[torch.FloatTensor]): Input embeddings.
            **kwargs: Additional arguments.
        
        Returns:
            dict: Model inputs for generation.
        """
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        return model_inputs


class GPTLMHeadModel(GPT):
    """
    GPT Language Model Head Model extending the base GPT class.
    It includes the transformer model with embedding layers, multiple transformer blocks,
    and an output layer for language modeling.
    """
    config_class = GPTConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        # The base GPT class already initializes the transformer and lm_head with tied weights

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.transformer.wte = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings
        self.transformer.wte.weight = self.lm_head.weight  # Ensure weight tying

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> dict:
        """
        Prepares inputs specifically for generation tasks.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            past_key_values (Optional[torch.Tensor]): Cached past key values.
            inputs_embeds (Optional[torch.FloatTensor]): Input embeddings.
            **kwargs: Additional arguments.
        
        Returns:
            dict: Model inputs for generation.
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        return model_inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        """
        Forward pass for the GPTLMHeadModel.
        Computes the logits and optionally the loss if labels are provided.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (B, T)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (B, T)
            labels (Optional[torch.Tensor]): Labels for computing the loss
            return_dict (Optional[bool]): Whether to return a dict
            past_key_values (Optional[torch.Tensor]): Past key values for caching
            head_mask (Optional[torch.FloatTensor]): Mask for attention heads
            inputs_embeds (Optional[torch.FloatTensor]): Input embeddings
            encoder_hidden_states (Optional[torch.Tensor]): Encoder hidden states
            encoder_attention_mask (Optional[torch.FloatTensor]): Encoder attention mask
            use_cache (Optional[bool]): Whether to use caching
            output_attentions (Optional[bool]): Whether to output attentions
            output_hidden_states (Optional[bool]): Whether to output hidden states
        
        Returns:
            CausalLMOutputWithCrossAttentions: Model outputs including logits and loss
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # We handle labels separately
            return_dict=True,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state  # (B, T, C)

        if labels is not None:
            # Compute logits
            logits = self.lm_head(hidden_states)  # (B, T, vocab_size)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Compute loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1
            )
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
        else:
            # Return only the logits for the last token
            logits = self.lm_head(hidden_states[:, -1, :]).unsqueeze(1)  # (B, 1, vocab_size)
            return CausalLMOutputWithCrossAttentions(
                loss=None,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
