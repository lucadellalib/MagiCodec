# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU
# General Public License version 3.

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers import GPT2Config

from components.gpt import GPTModel, ELEMWISE_WINDOW_MASK

__all__ = ["Roformer"]

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = None
    mlp_extend: float = None
    vocab_size: int = 1024  # defined later by tokenizer
    out_dim: int = 1024  # maybe not same as vocab_size
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-6
    causal: bool = True
    use_window_mask: bool = False
    window_size: list = field(default_factory=lambda: [-1, -1])
    window_type: str = "elemwise"  # elemwise, blockwise
    max_batch_size: int = 32
    max_seq_len: int = 2048
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    sparse: bool = False
    checkpointing: bool = False

    # for codec
    num_res: int = -1
    num_coarse: int = -1
    num_fine: int = -1

    audio_tokens_num: int = 1024
    phone_tokens_num: int = 200

    use_unet_style_skip_connect: bool = False
    use_qk_norm: str = ""
    flashattn_version: str = "2"
    condition_mode: str = None

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class Roformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.layers = self.create_layers(
            params = params
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
        )

    def forward(
        self,
        h: torch.Tensor,
        start_pos: int = 0,
        **kwargs,
    ):
        h = self.forward_layers(
            h,
            start_pos = start_pos,
            **kwargs,
        )
        
        return h

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean = 0.0,
                std = 0.02 / math.sqrt(2 * self.n_layers),
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean = 0.0,
                std = 0.02 / math.sqrt(2 * self.n_layers),
            )

    def create_layers(self, params):
        def get_n_inner_dim(n_embd, multiple_of):
            n_inner = 4 * n_embd
            n_inner = int(2 * n_inner / 3)
            N = multiple_of
            return ((n_inner - 1) // N) * N + N

        def get_compute_capability():
            cc = torch.cuda.get_device_properties(0)
            return cc.major + cc.minor * 0.1
    
        cuda_cc = get_compute_capability()
        backbone_config = GPT2Config(
            n_positions = 0,
            vocab_size = params.vocab_size,
            num_logits = params.out_dim,
            n_embd = params.dim,
            n_head = params.n_heads,
            n_layer = params.n_layers,
            n_kv_heads = getattr(params, "n_kv_heads", None),
            mlp_extend = getattr(params, "mlp_extend", None),
            layer_norm_epsilon = params.norm_eps,
            attn_pdrop = params.attn_pdrop,
            resid_pdrop = params.resid_pdrop,
            embd_pdrop = 0.0,
            n_inner = get_n_inner_dim(params.dim, params.multiple_of),
            activation_function = "swiglu",
            rotary_emb_fraction = 1.0,
            rotary_emb_interleaved = True,
            rotary_emb_compat = "default",
            tie_word_embeddings = False,
            initializer_range = 0.02,
            rms_norm = True,
            qkv_proj_bias = False,
            out_proj_bias = False,
            mlp_fc1_bias = False,
            mlp_fc2_bias = False,
            use_flash_attn = cuda_cc > 7.0,
            fused_bias_fc = cuda_cc > 7.0,
            fused_mlp = False,
            fused_dropout_add_ln = cuda_cc > 7.0,
            residual_in_fp32 = True,
            checkpointing = params.checkpointing,
            causal = getattr(params, "causal", True),
            use_window_mask = getattr(params, "use_window_mask", False),
            use_qk_norm = getattr(params, "use_qk_norm", ""),
            window_size = getattr(params, "window_size", [-1, -1]),
            window_type = getattr(params, "window_type", ELEMWISE_WINDOW_MASK),
            use_unet_style_skip_connect = getattr(params, "use_unet_style_skip_connect", False),
            flashattn_version = getattr(params, "flashattn_version", "2"),
        )
        backbone_config.condition_mode = getattr(params, "condition_mode", None)

        layers = GPTModel(backbone_config)
        del layers.embeddings.word_embeddings

        return layers

    def forward_layers(
        self,
        h,
        start_pos,
        **kwargs,
    ):
        return self.layers(
            inputs_embeds = h,
            **kwargs,
        )


if __name__ == "__main__":
    pass
