from transformers import PretrainedConfig

class MediclaimGPTConfig(PretrainedConfig):
    '''
    GPTConfig is a configuration class to store out configuration for our GPT model.
    It does so by extending the base class 'PretrainedConfig' from Hugging Face.
    '''
    model_type = "custom_gpt"

    def __init__(
        self,
        block_size:int = 1024,
        vocab_size:int = 50304,
        n_layers:int = 12,
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
        self.n_layers = n_layers
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
