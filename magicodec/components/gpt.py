# Copyright (c) 2023, Tri Dao.

import logging
import math
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import GPT2Config

from magicodec.components.block import Block
from magicodec.components.mha import FLASHATTN_VERSIONS, MHA
from magicodec.components.ops.activations import sqrelu_fwd
from magicodec.components.mlp import GatedMlp, Mlp
from importlib.metadata import version


def get_s3a_version():
    return version("s3a")

try:
    from magicodec.components.ops.norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from magicodec.components.ops.norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None

ELEMWISE_WINDOW_MASK = "elemwise"
BLOCKWISE_WINDOW_MASK = "blockwise"
WINDOW_MASK_TYPES = {ELEMWISE_WINDOW_MASK: 0, BLOCKWISE_WINDOW_MASK: 1}

logger = logging.getLogger(__name__)


def create_mixer_cls(
    config,
    cross_attn = False,
    layer_idx = None,
    process_group = None,
    device = None,
    dtype = None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    flashattn_version = getattr(config, "flashattn_version", 2)
    if isinstance(flashattn_version, (int, float)):
        flashattn_version = str(flashattn_version)
    assert flashattn_version in FLASHATTN_VERSIONS

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    n_kv_heads = getattr(config, "n_kv_heads", None)
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim**(-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, "attn_dwconv", False)
    if dwconv:
        assert process_group is None, "TensorParallel MHA does not support dwconv yet"
    qkv_proj_bias = getattr(config, "qkv_proj_bias", True)
    out_proj_bias = getattr(config, "out_proj_bias", True)
    rotary_emb_dim = int(getattr(config, "rotary_emb_fraction", 0.0) * head_dim)
    rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
    rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    rotary_emb_compat = getattr(config, "rotary_emb_compat", "default")
    use_rotary_triton = getattr(config, "use_rotary_triton", False)
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    causal = getattr(config, "causal", True)
    blocksparse = getattr(config, "blocksparse", False)
    blockmask = getattr(config, "blockmask", None)
    grad_checkpointing = getattr(config, "grad_checkpointing", False)
    use_qk_norm = getattr(config, "use_qk_norm", "")

    use_window_mask = getattr(config, "use_window_mask", False)
    window_size = getattr(config, "window_size", [-1, -1])
    window_type = getattr(config, "window_type", ELEMWISE_WINDOW_MASK)
    """
    if flashattn_version == "2.3":
        assert (use_flash_attn and process_group
                is None), "flashattn_2.3 only support use_flash_attn=True and process_group is None"
        assert not (causal and
                    use_window_mask), "don't support causal=True and use_window_mask=True meanwhile"
        if use_window_mask and not causal:
            assert window_type in WINDOW_MASK_TYPES and len(window_size) == 2
        elif not use_window_mask and causal:
            window_size = [-1, 0]
            window_type = ELEMWISE_WINDOW_MASK
        else:
            window_size = [-1, -1]
            window_type = ELEMWISE_WINDOW_MASK
        window_type = WINDOW_MASK_TYPES[window_type]
        if window_size[0] != -1 and window_type == 1:
            s3a_version = get_s3a_version()
            assert s3a_version is not None, "s3a has not been installed"
            major, _, _, minor = [int(n) for n in s3a_version.split(".")]
            assert (
                major == 1 and minor >= 68
            ), f"blockwise window_size_left!=-1, got 's3a=={s3a_version}'(expect' s3a>=1.0.0.68')"
    else:
        window_type = WINDOW_MASK_TYPES[window_type]
        assert (
            not use_window_mask
        ), f"only support use_window_mask=True in flashattn_version=2.3 now, but got {flashattn_version}"
    """
    if blocksparse:
        assert (isinstance(blockmask, List) and len(blockmask) == config.num_hidden_layers)
    if not fused_bias_fc:
        assert process_group is None, "TensorParallel MHA requires fused_bias_fc"
    mha_cls = MHA
    serial_kwargs = ({
        "fused_bias_fc": fused_bias_fc,
        "dwconv": dwconv
    } if process_group is None else {})
    parallel_kwargs = ({
        "process_group": process_group,
        "sequence_parallel": getattr(config, "sequence_parallel", True),
    } if process_group is not None else {})

    mixer_cls = partial(
        mha_cls,
        num_heads=config.num_attention_heads,
        num_heads_kv=n_kv_heads,
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.attn_pdrop,
        softmax_scale=softmax_scale,
        causal=causal,
        cross_attn=cross_attn,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim if not cross_attn else 0,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        rotary_emb_compat=rotary_emb_compat,
        use_rotary_triton=use_rotary_triton,
        use_flash_attn=use_flash_attn,
        checkpointing=grad_checkpointing,
        blocksparse=blocksparse,
        blockmask=blockmask[layer_idx] if blocksparse else None,
        use_qk_norm=use_qk_norm,
        window_size=window_size,
        window_type=window_type,
        version=flashattn_version,
        **serial_kwargs,
        **parallel_kwargs,
        **factory_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    mlp_fc1_bias = getattr(config, "mlp_fc1_bias", True)
    mlp_fc2_bias = getattr(config, "mlp_fc2_bias", True)
    fused_mlp = getattr(config, "fused_mlp", False)
    n_head = getattr(config, "n_head", 0)
    n_kv_heads = getattr(config, "n_kv_heads", None)
    mlp_extend = getattr(config, "mlp_extend", None)
    n_inner = getattr(config, "n_inner", 0)
    if ((n_kv_heads is not None) and (mlp_extend is not None) and (n_head % n_kv_heads == 0)):
        hidden_features = ((int(n_inner * mlp_extend) - 1) // 256) * 256 + 256
    else:
        hidden_features = n_inner
    if fused_mlp:
        assert config.activation_function in [
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
        ]
    fused_dense_sqrelu_dense = getattr(config, "fused_dense_sqrelu_dense", False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == "sqrelu", (
            "fused_dense_sqrelu_dense only "
            "supports approximate activation_function sqrelu")
    assert not (fused_dense_sqrelu_dense and fused_mlp)
    if process_group is not None:
        assert fused_mlp, "Tensor Parallel is only implemented for FusedMLP"
    if not fused_mlp and not fused_dense_sqrelu_dense:
        assert config.activation_function in [
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        if config.activation_function in ["glu", "swiglu", "geglu"]:
            activation = (F.sigmoid if config.activation_function == "glu" else
                          (F.silu if config.activation_function == "swiglu" else F.gelu))
            mlp_cls = partial(
                GatedMlp,
                hidden_features=hidden_features,
                activation=activation,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                multiple_of=getattr(config, "multiple_of", 256),
                **factory_kwargs,
            )
        else:
            if config.activation_function == "relu":
                activation = partial(F.relu, inplace=True)
            elif config.activation_function == "sqrelu":
                activation = sqrelu_fwd
            else:
                approximate = ("tanh" if config.activation_function
                               in ["gelu_new", "gelu_fast", "gelu_approx"] else "none")
                activation = partial(F.gelu, approximate=approximate)
            mlp_cls = partial(
                Mlp,
                hidden_features=hidden_features,
                activation=activation,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                **factory_kwargs,
            )

    return mlp_cls


def create_block(
    config,
    #use_cross_attn,
    layer_idx = None,
    process_group = None,
    device = None,
    dtype = None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    sequence_parallel = getattr(config, "sequence_parallel", True)
    flashattn_version = getattr(config, "flashattn_version", 2)
    if isinstance(flashattn_version, (int, float)):
        flashattn_version = str(flashattn_version)
    assert flashattn_version in FLASHATTN_VERSIONS


    mixer_cls = create_mixer_cls(
        config,
        False,
        layer_idx,
        process_group = process_group,
        **factory_kwargs,
    )

    mlp_cls = create_mlp_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    use_rms_norm = getattr(config, "rms_norm", False)
    norm_cls = partial(
        nn.LayerNorm if not use_rms_norm else RMSNorm,
        eps=config.layer_norm_epsilon,
        **factory_kwargs,
    )
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    resid_dropout1 = (
        config.resid_pdrop
        if layer_idx is None or layer_idx > 0 
        else config.embd_pdrop
    )
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls = norm_cls,
        prenorm = True,
        resid_dropout1 = resid_dropout1,
        resid_dropout2 = config.resid_pdrop,
        fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False),
        residual_in_fp32 = residual_in_fp32,
        version = flashattn_version,
        device = device,
        dtype = dtype,
    )
    block.layer_idx = layer_idx
    return block


class GPTPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # print(f"rescale_prenorm_residual {name}")
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))


class GPT2Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx = None,
        word_embed_proj_dim = None,
        device = None,
        dtype = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, 
                embed_dim, 
                padding_idx = padding_idx, 
                **factory_kwargs,
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                word_embed_proj_dim,
                padding_idx = padding_idx,
                **factory_kwargs,
            )
            self.project_in = nn.Linear(
                word_embed_proj_dim, 
                embed_dim, 
                bias = False, 
                **factory_kwargs,
            )
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, 
                embed_dim, 
                **factory_kwargs,
            )

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        B, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=input_ids.device
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

class GPTModel(GPTPreTrainedModel):
    def __init__(
        self, 
        config: GPT2Config
    ):
        super().__init__(config)
        self.process_group = None
        self.sequence_parallel = getattr(config, "sequence_parallel", True)
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]

        vocab_size = config.vocab_size
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        self.prenorm = True
        use_rms_norm = getattr(config, "rms_norm", False)
        self.parallel_block = False
        
        self.embeddings = GPT2Embeddings(
            config.hidden_size,
            vocab_size,
            config.max_position_embeddings,
        )

        self.layers = nn.ModuleList([
            create_block(
                config,
                layer_idx = i,
            ) for i in range(config.num_hidden_layers)
        ])

        self.num_hidden_layers = config.num_hidden_layers

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln:
            if (not self.parallel_block and dropout_add_layer_norm is None):
                raise ImportError("dropout_layer_norm is not installed")
        
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            self.ln_f = norm_cls(
                config.hidden_size,
                eps = config.layer_norm_epsilon,
            )

        self.apply(
            partial(
                _init_weights,
                n_layer = config.num_hidden_layers,
                initializer_range = config.initializer_range,
                rescale_prenorm_residual = True,
            ))

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds.")

        seqlen = inputs_embeds.size(1)
        hidden_states = inputs_embeds
        if self.embeddings.project_in is not None:
            hidden_states = self.embeddings.project_in(hidden_states)
        if self.embeddings.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen,
                                            dtype=torch.long,
                                            device=inputs_embeds.device)
            position_embeddings = self.embeddings.position_embeddings(position_ids)
            hidden_states += position_embeddings

        residual = None

        for i, layer in enumerate(self.layers):
            layer_outs = layer(
                hidden_states,
                residual = residual,
                mixer_kwargs={},
            )
            
            hidden_states, residual = layer_outs

        if self.prenorm:
            if isinstance(self.ln_f, RMSNorm):
                fused_add_norm_fn = dropout_add_rms_norm
            else:
                fused_add_norm_fn = dropout_add_layer_norm
            
            hidden_states = fused_add_norm_fn(
                hidden_states,
                residual,
                self.ln_f.weight,
                self.ln_f.bias,
                self.drop_f.p if self.training else 0.0,
                self.ln_f.eps,
                prenorm = False,
                residual_in_fp32 = self.residual_in_fp32,
            )
        
        return hidden_states
