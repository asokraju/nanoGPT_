"""
GPT Language Model Implementation
================================

This module provides a complete implementation of a GPT (Generative Pre-trained Transformer)
language model, including optional Mixture of Experts (MoE) layers for enhanced capacity.
The model is built using PyTorch and integrates with Hugging Face's Transformers library for
configuration and output handling. Additionally, it includes mechanisms to log and monitor
the usage of experts within MoE layers, facilitating balanced utilization and model diagnostics.

### References
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions
)

from transformers import TrainerCallback, TrainerState, TrainerControl, Trainer, TrainingArguments
import logging
from dataclasses import dataclass
# Import FlashAttention if available
try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    flash_attn_available = True
except ImportError:
    flash_attn_available = False
    print("FlashAttention not available, using default attention.")

class GPTConfig(PretrainedConfig):
    """
    GPTConfig is a configuration class to store our configuration for the GPT model.
    It extends the base class 'PretrainedConfig' from Hugging Face.
    """
    model_type = "custom_gpt"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_loss: bool = True,
        moe_loss_type: str = "entropy_regularization",
        moe_loss_coef: float = 1e-2, # aux loss is scaled at each expert
        aux_loss_weight: float = 1.0, # average aux loss is again multiplied by aux_loss_weight before added to the total loss. Overall loss coeff will be moe_loss_coef*aux_loss_weight
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_loss = moe_loss
        self.moe_loss_type = moe_loss_type
        self.moe_loss_coef = moe_loss_coef
        self.aux_loss_weight = aux_loss_weight
        print(f"WARNING: Overall aux loss coef is {moe_loss_coef*aux_loss_weight}.")

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
    Supports both standard and FlashAttention (if available).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Combined projection for query, key, and value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Check if FlashAttention is available
        self.flash = flash_attn_available
        if not self.flash:
            print("WARNING: FlashAttention is not available. Using default attention mechanism.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Project inputs to query, key, and value
        qkv = self.c_attn(x)  # Shape: (B, T, 3*C)

        # Reshape qkv to (B, T, 3, n_head, head_dim)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)

        if self.flash and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16]:
            # Ensure qkv is contiguous
            qkv = qkv.contiguous()

            # Use FlashAttention
            y = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=True
            )  # Output shape: (B, T, n_head, head_dim)

            # Reshape y to (B, T, C)
            y = y.view(B, T, C)
        else:
            # Fallback to default attention mechanism
            # Split qkv into q, k, v
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Each shape: (B, n_head, T, head_dim)

            # Compute scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)

            # Apply causal mask
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            att = att.masked_fill(causal_mask == 0, float('-inf'))

            # Compute attention probabilities
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Compute attention output
            y = att @ v  # (B, n_head, T, head_dim)

            # Transpose and reshape y to (B, T, C)
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
    Mixture of Experts (MoE) layer implementation for transformer models.

    Args:
        config (GPTConfig): Configuration object containing model parameters.

    Attributes:
        num_experts (int): Total number of experts.
        num_experts_per_tok (int): Number of experts assigned per token.
        experts (nn.ModuleList): List of expert modules (e.g., MLPs).
        gate (nn.Linear): Gating network mapping input embeddings to expert scores.
        moe_loss (bool): Flag to enable or disable auxiliary loss.
        moe_loss_type (str): Type of auxiliary loss to use.
        moe_loss_coef (float): Coefficient for scaling the auxiliary loss.
        expert_usage_counts (torch.Tensor): Counts of expert usage for monitoring.
        total_assignments (int): Total number of expert assignments.
    """

    def __init__(self, config: GPTConfig) -> None:
        """
        Initializes the MoE layer.

        Args:
            config (GPTConfig): Configuration object with model parameters.

        Raises:
            AssertionError: If an invalid moe_loss_type is specified.
        """
        super().__init__()
        self.num_experts: int = config.num_experts
        self.num_experts_per_tok: int = config.num_experts_per_tok
        self.experts: nn.ModuleList = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
        self.gate: nn.Linear = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.moe_loss: bool = config.moe_loss
        self.moe_loss_type: str = config.moe_loss_type
        self.moe_loss_coef: float = config.moe_loss_coef

        # Validate the loss type
        valid_loss_types: List[str] = ["variance_penalty", "entropy_regularization", "diversity_regularization"]
        assert self.moe_loss_type in valid_loss_types, f"Invalid moe_loss_type. Choose from {valid_loss_types}"

        # Initialize expert usage counters for monitoring purposes
        self.register_buffer("expert_usage_counts", torch.zeros(self.num_experts, dtype=torch.long))
        self.total_assignments: int = 0  # Total number of expert assignments

        # Accumulators for auxiliary loss
        self.register_buffer("auxiliary_loss_sum", torch.tensor(0.0))
        self.num_auxiliary_loss_updates = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and auxiliary loss.
        """
        # Get batch size (B), sequence length (T), and embedding dimension (C)
        B, T, C = x.size()

        # Flatten the input to shape (B*T, C)
        x_flat: torch.Tensor = x.view(-1, C)

        # Compute gating scores: shape (B*T, num_experts)
        scores: torch.Tensor = self.gate(x_flat)

        # Compute gating probabilities using softmax over the experts dimension
        gating_probs: torch.Tensor = F.softmax(scores, dim=-1)  # (B*T, num_experts)

        # Select top-k experts per token based on gating scores
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)  # (B*T, num_experts_per_tok)

        # Apply softmax to the top-k scores to get normalized weights
        topk_weights: torch.Tensor = F.softmax(topk_scores, dim=-1).view(-1, self.num_experts_per_tok, 1)  # (B*T, num_experts_per_tok, 1)

        # Flatten the expert indices for indexing
        flat_expert_indices: torch.Tensor = topk_indices.view(-1)  # (B*T*num_experts_per_tok,)

        # Update expert usage counts for monitoring (not used in training)
        with torch.no_grad():
            counts: torch.Tensor = flat_expert_indices.bincount(minlength=self.num_experts)
            self.expert_usage_counts += counts
            self.total_assignments += flat_expert_indices.size(0)

        # Repeat inputs for each expert assignment
        x_repeated: torch.Tensor = x_flat.unsqueeze(1).repeat(1, self.num_experts_per_tok, 1).view(-1, C)  # (B*T*num_experts_per_tok, C)

        # Ensure x_repeated is in the same dtype as y
        x_repeated = x_repeated.to(x.dtype)

        # Initialize the output tensor y
        y = torch.zeros_like(x_repeated, dtype=x.dtype, device=x_repeated.device)

        # Apply each expert to its assigned inputs
        for i, expert in enumerate(self.experts):
            # Create a mask for the current expert
            mask: torch.Tensor = flat_expert_indices == i  # (B*T*num_experts_per_tok,)
            if mask.any():
                # Apply the expert to the inputs where the mask is True
                expert_output = expert(x_repeated[mask])
                y[mask] = expert_output.to(y.dtype)  # Ensure dtype matches

        # Reshape y to (B*T, num_experts_per_tok, C)
        y = y.view(-1, self.num_experts_per_tok, C)

        # Apply the top-k weights to the expert outputs and sum over experts
        y = (y * topk_weights).sum(dim=1)  # (B*T, C)

        # Reshape y back to (B, T, C)
        y = y.view(B, T, C)

        # Initialize auxiliary loss with the correct dtype
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.moe_loss:
            # Compute assigned probabilities after top-k selection
            # Initialize assigned_probs tensor of shape (B*T, num_experts)
            assigned_probs: torch.Tensor = torch.zeros(B * T, self.num_experts, device=x.device, dtype=x.dtype)

            # Flatten topk_weights to shape (B*T, num_experts_per_tok)
            flat_topk_weights: torch.Tensor = topk_weights.squeeze(-1)  # (B*T, num_experts_per_tok)

            # Scatter the top-k weights into the assigned_probs tensor
            assigned_probs.scatter_(dim=1, index=topk_indices, src=flat_topk_weights)

            # Compute expert usage as the mean over all tokens
            expert_usage: torch.Tensor = assigned_probs.mean(dim=0)  # (num_experts,)

            # Compute auxiliary loss based on the specified moe_loss_type
            if self.moe_loss_type == "variance_penalty":
                # Compute variance of expert usage
                mean_usage: torch.Tensor = expert_usage.mean()
                variance: torch.Tensor = torch.mean((expert_usage - mean_usage) ** 2)
                auxiliary_loss += self.moe_loss_coef * variance

            elif self.moe_loss_type == "entropy_regularization":
                # Compute entropy of expert usage
                entropy: torch.Tensor = -torch.sum(expert_usage * torch.log(expert_usage + 1e-10))
                # Compute maximum possible entropy
                max_entropy = torch.log(torch.tensor(float(self.num_experts), dtype=x.dtype, device=x.device))
                # Normalize entropy to range between 0 and 1
                normalized_entropy = entropy / max_entropy
                # To maximize entropy, minimize negative entropy
                auxiliary_loss += self.moe_loss_coef * (1 - normalized_entropy)

            elif self.moe_loss_type == "diversity_regularization":
                # Compute KL divergence between expert usage and uniform distribution
                uniform: torch.Tensor = torch.ones_like(expert_usage) / self.num_experts
                divergence: torch.Tensor = F.kl_div(torch.log(expert_usage + 1e-10), uniform, reduction='batchmean')
                auxiliary_loss += self.moe_loss_coef * divergence

        # Detach auxiliary_loss before accumulation to avoid retaining the computational graph
        detached_aux_loss = auxiliary_loss.detach()
        self.auxiliary_loss_sum += detached_aux_loss
        self.num_auxiliary_loss_updates += 1

        return y, auxiliary_loss

    def get_auxiliary_loss(self) -> float:
        """
        Returns the average auxiliary loss accumulated over forward passes.

        Returns:
            float: The average auxiliary loss.
        """
        if self.num_auxiliary_loss_updates > 0:
            avg_aux_loss = (self.auxiliary_loss_sum / self.num_auxiliary_loss_updates).item()
        else:
            avg_aux_loss = 0.0
        return avg_aux_loss

    def reset_auxiliary_loss(self) -> None:
        """
        Resets the accumulated auxiliary loss and the count.
        """
        self.auxiliary_loss_sum.zero_()
        self.num_auxiliary_loss_updates = 0

    def get_usage_counts(self) -> torch.Tensor:
        """
        Returns the counts of expert usage.

        Returns:
            torch.Tensor: A tensor containing the usage counts of each expert.
        """
        return self.expert_usage_counts

    def reset_usage_counts(self) -> None:
        """
        Resets the expert usage counts and total assignments.
        """
        self.expert_usage_counts.zero_()
        self.total_assignments = 0

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and auxiliary loss.
        """
        # Apply LayerNorm and Self-Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # Apply LayerNorm and MLP (or MoE) with residual connection
        if isinstance(self.mlp, MoE):
            mlp_output, auxiliary_loss = self.mlp(self.ln_2(x))
        else:
            mlp_output = self.mlp(self.ln_2(x))
            auxiliary_loss = torch.tensor(0.0, device=x.device)
        
        x = x + mlp_output
        
        return x, auxiliary_loss

from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

@dataclass
class BaseModelOutputWithAuxiliaryLoss(BaseModelOutputWithPastAndCrossAttentions):
    """
    Extends BaseModelOutputWithPastAndCrossAttentions to include auxiliary loss.
    """
    auxiliary_loss: Optional[torch.FloatTensor] = None

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
        self.lm_head.weight = self.transformer.wte.weight 

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
    ) -> BaseModelOutputWithAuxiliaryLoss:
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
            BaseModelOutputWithAuxiliaryLoss: Model outputs including auxiliary loss
        """
        device = input_ids.device
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Generate position IDs
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # Shape: (1, T)

        # Compute token and position embeddings
        pos_emb = self.transformer.wpe(pos)        # (1, T, C)
        if inputs_embeds is not None:
            x = self.transformer.drop(inputs_embeds + pos_emb)  # (B, T, C)
        else:
            tok_emb = self.transformer.wte(input_ids)  # (B, T, C)
            x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C)

        all_hidden_states = () if output_hidden_states else None

        auxiliary_losses = []
        # Pass through each Transformer block
        for block in self.transformer.h:
            if output_hidden_states:
                all_hidden_states += (x,)
            x, auxiliary_loss = block(x)
            auxiliary_losses.append(auxiliary_loss)

        # Final LayerNorm
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states += (x,)

        # Compute average auxiliary loss
        if len(auxiliary_losses) > 0:
            average_auxiliary_loss = torch.stack(auxiliary_losses).mean()
        else:
            average_auxiliary_loss = torch.tensor(0.0, device=x.device)

        # Prepare output
        return BaseModelOutputWithAuxiliaryLoss(
            last_hidden_state=x,
            past_key_values=None,  # To be implemented if caching is desired
            hidden_states=all_hidden_states,
            attentions=None,        # To be implemented if attention outputs are desired
            cross_attentions=None,  # To be implemented if cross attentions are desired
            auxiliary_loss=average_auxiliary_loss
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

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

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

@dataclass
class CausalLMOutputWithAuxiliaryLoss(CausalLMOutputWithCrossAttentions):
    primary_loss: Optional[torch.FloatTensor] = None
    auxiliary_loss: Optional[torch.FloatTensor] = None


class GPTLMHeadModel(GPT):
    """
    GPT Language Model Head Model extending the base GPT class.
    It includes the transformer model with embedding layers, multiple transformer blocks,
    an optional MoE layer, and an output layer for language modeling.
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
    ) -> CausalLMOutputWithAuxiliaryLoss:
        """
        Forward pass for the GPTLMHeadModel.
        Computes the logits and optionally the loss if labels are provided.
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

        # Compute logits
        logits = self.lm_head(hidden_states)  # (B, T, vocab_size)

        # If we are generating text (no labels provided), just return logits
        if labels is None:
            return CausalLMOutputWithAuxiliaryLoss(
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
                primary_loss=None,
                auxiliary_loss=None,
            )

        # Shift logits and labels for next-token prediction when labels are provided
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute primary loss
        primary_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-1
        )

        # Retrieve average auxiliary loss
        average_auxiliary_loss = outputs.auxiliary_loss

        # Adjust the weighting of auxiliary loss if necessary
        aux_loss_weight = self.config.aux_loss_weight  # We'll add this to GPTConfig

        # Combine losses
        total_loss = primary_loss + aux_loss_weight * average_auxiliary_loss

        return CausalLMOutputWithAuxiliaryLoss(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            primary_loss=primary_loss,
            auxiliary_loss=average_auxiliary_loss,
        )

class MoETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        primary_loss = outputs.primary_loss
        auxiliary_loss = outputs.auxiliary_loss

        # Log the separate losses
        self.log({
            "loss": loss.item(),
            "primary_loss": primary_loss.item(),
            "auxiliary_loss": auxiliary_loss.item()
        })

        if return_outputs:
            return loss, outputs
        else:
            return loss
        
class MoEUsageLoggingCallback(TrainerCallback):
    """
    Callback to log MoE expert usage statistics and auxiliary loss during training.
    """
    def __init__(self, trainer: Trainer, logger: Optional[logging.Logger] = None, log_interval: int = 100, log_dir: str = './logs_tensorboard'):
        super().__init__()
        self.log_interval = log_interval
        self.steps_since_last_log = 0
        self.logger = logger or logging.getLogger(__name__)
        self.trainer = trainer  # Store the trainer instance
        self.writer = SummaryWriter(log_dir)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.steps_since_last_log += 1
        if self.steps_since_last_log >= self.log_interval:
            print("Logging MoE usage stats...")
            model = self.trainer.model  # Access the model directly from the trainer instance
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
                # Log to TensorBoard
                for i in range(moe.num_experts):
                    self.writer.add_scalar(f"MoE_Layer_{idx + 1}/Expert_{i}_Usage", usage_percentages[i], state.global_step)
                # Log average auxiliary loss
                avg_aux_loss = moe.get_auxiliary_loss()
                self.writer.add_scalar(f"MoE_Layer_{idx + 1}/Auxiliary_Loss", avg_aux_loss, state.global_step)
                # Reset accumulators
                moe.reset_usage_counts()
                moe.reset_auxiliary_loss()

            self.steps_since_last_log = 0

    def _get_moe_layers(self, model) -> list:
        moe_layers = []
        for block in model.transformer.h:
            if hasattr(block, 'mlp') and isinstance(block.mlp, MoE):
                moe_layers.append(block.mlp)
        return moe_layers
    
    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()
