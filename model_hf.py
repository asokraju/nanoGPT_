"""
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
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import PreTrainedModel

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers import PretrainedConfig


class GPTConfig(PretrainedConfig):
    '''
    GPTConfig is a configuration class to store out configuration for our GPT model.
    It does so by extending the base class 'PretrainedConfig' from Hugging Face.
    '''
    model_type = "custom_gpt"

    def __init__(
        self,
        block_size:int = 1024,
        vocab_size:int = 50304,
        n_layer:int = 12,
        n_head: int = 12,
        n_embd:int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        use_moe: bool = False,
        num_experts:int = 8,
        num_experts_per_tok:int = 2,
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

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

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

    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([MLP(config) for i in range(config.num_experts)])
        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, x):

        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        scores = self.gate(x)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)
        flat_expert_indices = expert_indices.view(-1)
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x, dtype=torch.bfloat16, device=x.device)
        for i, expert in enumerate(self.experts):
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # print(config.use_moe)
        if config.use_moe:
            print("using mixture of experts")
            self.mlp = MoE(config)
        else:
            print("using regular MLP")
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(PreTrainedModel):
    '''

    subclassing PreTrainedModel
    '''
    config_class = GPTConfig
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_input_embeddings(self):
        return self.transformer.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

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
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None,
                labels=None,
                return_dict=None,
                past_key_values: Optional[torch.tensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t,  n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t,  n_embd)
        x =  self.transformer.drop(tok_emb + pos_emb)
        all_hidden_states = () if output_hidden_states else None
        for block in self.transformer.h:
            if output_hidden_states:
                all_hidden_states += (x,)
            x = block(x)
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states +=(x,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=x,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
            cross_attentions=None,
        )

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:, :block_size, :block_size]

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
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def prepare_inputs_for_generation(self, input_ids,
                                      past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]: 1]
        else:
            position_ids = None
        # if `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        return model_inputs

class GPTLMHeadModel(GPT):
    '''
    This Main model class extending PreTrainedModel from Hugging Face
    It includes the transformer model with embedding layers, multiple transformer blocks, and an output layer for language modeling.
    '''
    config_class = GPTConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # self.transformer = GPT(config)
    
    def get_input_embeddings(self):
        return self.transformer.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte  = new_embeddings

    def  get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def prepare_inputs_for_generation(self, input_ids, 
                                      past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]: 1]
        else:
            position_ids = None
        
        # if `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        return model_inputs
    
    def forward(self, input_ids, attention_mask=None,
                labels=None,
                return_dict=None,
                past_key_values: Optional[torch.tensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None):
        
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t,  n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t,  n_embd)
        x =  self.transformer.drop(tok_emb + pos_emb)
        all_hidden_states = () if output_hidden_states else None
        for block in self.transformer.h:
            if output_hidden_states:
                all_hidden_states += (x,)
            x = block(x)
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states +=(x,)

        logits = x
        if labels is not None:
            logits = self.lm_head(logits)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
            return CausalLMOutputWithCrossAttentions(
                hidden_states=all_hidden_states,
                loss=loss,
                logits=logits
            )
        else:
            logits = self.lm_head(logits[:, [-1], :])
            return CausalLMOutputWithCrossAttentions(
                hidden_states=all_hidden_states,
                loss=None,
                logits=logits
            )

    # @torch.no_grad()
    # def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    #     """
    #     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     """
    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         idx_cond = idx if idx.size(
    #             1) <= self.config.block_size else idx[:, -self.config.block_size:]
    #         # forward the model to get the logits for the index in the sequence
    #         logits, _ = self(idx_cond)
    #         # pluck the logits at the final step and scale by desired temperature
    #         logits = logits[:, -1, :] / temperature
    #         # optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float('Inf')
    #         # apply softmax to convert logits to (normalized) probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         # append sampled index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)

    #     return idx
    
    # @torch.no_grad()
    # def generate_2(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    #     """
    #     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     """
    #     print("Initial idx dtype:", idx.dtype)  # Debugging data type of initial indices
        
    #     for _ in range(max_new_tokens):
    #         idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    #         print("idx_cond shape:", idx_cond.shape, "dtype:", idx_cond.dtype)  # Debugging idx_cond

    #         # forward the model to get the logits for the index in the sequence
    #         logits, _ = self(idx_cond)
    #         print("Logits dtype before temperature scaling:", logits.dtype)  # Debugging logits dtype

    #         # Convert logits to float32 for stable operations involving topk and softmax
    #         logits = logits[:, -1, :].float() / temperature
    #         print("Logits dtype after temperature scaling:", logits.dtype)  # Debugging logits dtype

    #         # optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             logits = logits.float()  # Ensuring logits are in float32 for top_k operation
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float('Inf')
    #             print("Logits after top_k application:", logits)  # Debugging logits after top_k

    #         # apply softmax to convert logits to (normalized) probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         print("Probabilities dtype:", probs.dtype)  # Debugging probabilities dtype

    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         print("idx_next dtype:", idx_next.dtype)  # Debugging sampled indices dtype

    #         # append sampled index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)
    #         print("Updated idx dtype:", idx.dtype)  # Debugging updated indices dtype

    #     return idx

        
    # @torch.no_grad()
    # def generate_3(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    #     for _ in range(max_new_tokens):
    #         idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    #         logits, _ = self(idx_cond)
    #         logits = logits[:, -1, :].float() / temperature  # Convert to float32 for temperature scaling
    #         if top_k is not None:
    #             logits = logits.float()  # Ensure logits are in float32 for top_k operation
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float('Inf')
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         idx = torch.cat((idx, idx_next), dim=1)
    #     return idx


