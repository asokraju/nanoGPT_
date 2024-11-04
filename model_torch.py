"""
Full definition of a GPT Language Model with Mixture of Experts (MoE) integration.
References:
1) The official GPT-2 TensorFlow implementation released by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) HuggingFace Transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect


class LayerNorm(nn.Module):
    """LayerNorm module with optional bias parameter."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention module."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, and value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Attention parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Flash attention (if available)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality

        # Compute query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)
        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Feed-forward network (MLP) module."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) layer implementation for transformer models.

    Args:
        config (GPTConfig): Configuration object containing model parameters.
            - num_experts (int): Total number of experts (dynamic + static).
            - num_experts_per_tok (int): Number of dynamic experts per token.
            - static_experts (int): Number of static experts.

    If static_experts is not zero, tokens always go through static experts,
    and some of the other experts depending on the routing.
    """
    def __init__(self, config):
        super().__init__()
        self.total_experts = config.num_experts  # Total number of experts (dynamic + static)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.static_experts = config.static_experts  # Number of static experts
        self.num_dynamic_experts = self.total_experts - self.static_experts  # Number of dynamic experts

        # Initialize dynamic experts
        self.dynamic_experts = nn.ModuleList([MLP(config) for _ in range(self.num_dynamic_experts)])
        # Initialize static experts
        if self.static_experts > 0:
            print(f"Using {self.static_experts} static experts in MoE")
            self.static_expert_modules = nn.ModuleList([MLP(config) for _ in range(self.static_experts)])

        # Gating network outputs scores for dynamic experts only
        self.gate = nn.Linear(config.n_embd, self.num_dynamic_experts, bias=False)
        self.moe_loss = config.moe_loss
        self.moe_loss_type = config.moe_loss_type
        self.moe_loss_coef = config.moe_loss_coef

        # Validate the loss type
        valid_loss_types = ["variance_penalty", "entropy_regularization", "diversity_regularization"]
        assert self.moe_loss_type in valid_loss_types, f"Invalid moe_loss_type. Choose from {valid_loss_types}"

        # Initialize expert usage counters for monitoring purposes (dynamic experts only)
        self.register_buffer("expert_usage_counts", torch.zeros(self.num_dynamic_experts, dtype=torch.long))
        self.total_assignments = 0  # Total number of expert assignments to dynamic experts

    def forward(self, x):
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and auxiliary loss.
        """
        B, T, C = x.size()

        # Process static experts if any
        if self.static_experts > 0:
            # Process x through each static expert
            y_static_list = [expert(x) for expert in self.static_expert_modules]
            # Sum the outputs from static experts
            y_static = torch.stack(y_static_list, dim=0).sum(dim=0)  # (B, T, C)
        else:
            y_static = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Flatten the input to shape (B*T, C)
        x_flat = x.view(-1, C)

        # Compute gating scores for dynamic experts: shape (B*T, num_dynamic_experts)
        scores = self.gate(x_flat)

        # Select top-k dynamic experts per token based on gating scores
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)  # (B*T, k)

        # Apply softmax to the top-k scores to get normalized weights
        topk_weights = F.softmax(topk_scores, dim=-1).view(-1, self.num_experts_per_tok, 1)  # (B*T, k, 1)

        # Flatten the expert indices for indexing
        flat_expert_indices = topk_indices.view(-1)  # (B*T*k,)

        # Update expert usage counts for monitoring (dynamic experts only)
        with torch.no_grad():
            counts = flat_expert_indices.bincount(minlength=self.num_dynamic_experts)
            self.expert_usage_counts += counts
            self.total_assignments += flat_expert_indices.size(0)

        # Repeat inputs for each expert assignment
        x_repeated = x_flat.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, C)  # (B*T*k, C)

        # Initialize the output tensor y_dynamic
        y_dynamic = torch.zeros_like(x_repeated, dtype=x.dtype, device=x.device)

        # Apply each dynamic expert to its assigned inputs
        for i, expert in enumerate(self.dynamic_experts):
            # Create a mask for the current expert
            mask = flat_expert_indices == i  # (B*T*k,)
            if mask.any():
                # Apply the expert to the inputs where the mask is True
                expert_output = expert(x_repeated[mask])
                y_dynamic[mask] = expert_output.to(y_dynamic.dtype)  # Ensure dtype matches

        # Reshape y_dynamic to (B*T, num_experts_per_tok, C)
        y_dynamic = y_dynamic.view(-1, self.num_experts_per_tok, C)

        # Apply the top-k weights to the expert outputs and sum over experts
        y_dynamic = (y_dynamic * topk_weights).sum(dim=1)  # (B*T, C)

        # Reshape y_dynamic back to (B, T, C)
        y_dynamic = y_dynamic.view(B, T, C)

        # Combine outputs from static and dynamic experts
        y = y_dynamic + y_static  # (B, T, C)

        # Initialize auxiliary loss
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.moe_loss:
            # Compute assigned probabilities after top-k selection
            assigned_probs = torch.zeros(B * T, self.num_dynamic_experts, device=x.device, dtype=x.dtype)

            # Flatten topk_weights to shape (B*T, num_experts_per_tok)
            flat_topk_weights = topk_weights.squeeze(-1)  # (B*T, k)

            # Scatter the top-k weights into the assigned_probs tensor
            assigned_probs.scatter_(dim=1, index=topk_indices, src=flat_topk_weights)  # (B*T, num_dynamic_experts)

            # Compute expert usage as the mean over all tokens
            expert_usage = assigned_probs.mean(dim=0)  # (num_dynamic_experts,)

            # Compute auxiliary loss based on the specified moe_loss_type
            if self.moe_loss_type == "variance_penalty":
                # Compute variance of expert usage
                mean_usage = expert_usage.mean()
                variance = torch.mean((expert_usage - mean_usage) ** 2)
                auxiliary_loss += self.moe_loss_coef * variance

            elif self.moe_loss_type == "entropy_regularization":
                # Compute entropy of expert usage
                entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-10))
                # Compute maximum possible entropy
                max_entropy = torch.log(torch.tensor(float(self.num_dynamic_experts), dtype=x.dtype, device=x.device))
                # Normalize entropy to range between 0 and 1
                normalized_entropy = entropy / max_entropy
                # To maximize entropy, minimize negative entropy
                auxiliary_loss += self.moe_loss_coef * (1 - normalized_entropy)

            elif self.moe_loss_type == "diversity_regularization":
                # Compute KL divergence between expert usage and uniform distribution
                uniform = torch.ones_like(expert_usage) / self.num_dynamic_experts
                divergence = F.kl_div(torch.log(expert_usage + 1e-10), uniform, reduction='batchmean')
                auxiliary_loss += self.moe_loss_coef * divergence

        return y, auxiliary_loss

    def get_usage_percentages(self):
        """
        Returns the usage percentages of each dynamic expert.

        Returns:
            List[float]: A list containing the usage percentage of each dynamic expert.
        """
        if self.total_assignments > 0:
            usage_percentages = (self.expert_usage_counts.float() / self.total_assignments) * 100
            return usage_percentages.tolist()
        else:
            return [0.0] * self.num_dynamic_experts

    def reset_usage_counts(self):
        """
        Resets the expert usage counts and total assignments.
        """
        self.expert_usage_counts.zero_()
        self.total_assignments = 0


class Block(nn.Module):
    """
    Represents a single Transformer block, consisting of LayerNorm,
    Causal Self-Attention, and a Feed-Forward Network (MLP or MoE).
    """
    def __init__(self, config):
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

    def forward(self, x):
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

@dataclass
class GPTConfig:
    """
    Configuration class for GPT model.

    Args:
        block_size (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        n_layer (int): Number of Transformer blocks.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding dimension.
        dropout (float): Dropout rate.
        bias (bool): Whether to use bias in Linear and LayerNorm layers.
        use_moe (bool): Whether to use Mixture of Experts (MoE) in MLP layers.
        num_experts (int): Number of dynamic experts in MoE.
        num_experts_per_tok (int): Number of experts per token in MoE.
        static_experts (int): Number of static experts in MoE.
        moe_loss (bool): Whether to use auxiliary loss in MoE.
        moe_loss_type (str): Type of auxiliary loss in MoE.
        moe_loss_coef (float): Coefficient for the auxiliary loss.
    """
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # MoE parameters
    use_moe: bool = True  # Whether to use MoE in MLP layers
    num_experts: int = 4  # Number of dynamic experts in MoE
    num_experts_per_tok: int = 2  # Number of experts per token
    static_experts: int = 0  # Number of static experts
    moe_loss: bool = True  # Whether to use auxiliary loss
    moe_loss_type: str = 'variance_penalty'  # Type of auxiliary loss
    moe_loss_coef: float = 0.01  # Coefficient for the auxiliary loss

class GPT(nn.Module):
    """
    GPT Language Model with optional Mixture of Experts (MoE) integration.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Embedding layers
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks
            ln_f = LayerNorm(config.n_embd, bias=config.bias),  # Final LayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled initialization to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.

        Args:
            idx (torch.Tensor): Input indices of shape (B, T)
            targets (torch.Tensor, optional): Target indices for computing loss

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Logits, total loss, main loss, auxiliary loss
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # Position indices

        # Embedding lookup
        tok_emb = self.transformer.wte(idx)  # Token embeddings (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # Position embeddings (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # Combine embeddings

        # Forward through Transformer blocks
        auxiliary_losses = []
        for block in self.transformer.h:
            x, auxiliary_loss = block(x)
            auxiliary_losses.append(auxiliary_loss)

        x = self.transformer.ln_f(x)  # Final LayerNorm

        if targets is not None:
            # Compute logits and main loss
            logits = self.lm_head(x)
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            # Compute mean of the auxiliary losses
            total_auxiliary_loss = torch.stack(auxiliary_losses).mean()
            # Combine main loss and auxiliary loss
            loss = main_loss + total_auxiliary_loss
            return logits, loss, main_loss, total_auxiliary_loss
        else:
            # Inference mode: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            return logits, None, None, None

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
