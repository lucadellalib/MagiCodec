# Copyright (c) 2022, Tri Dao.
import math
import warnings
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange

from magicodec.components.ops.padding import pad_input, unpad_input
from magicodec.components.ops.norm import RMSNorm

try:
    # flash_attn_2_8
    from magicodec.components.ops.flash_attn_2_8_interface import (
        flash_attn_qkvpacked_func as flash_attn_2_8_qkvpacked_func,
    )
except Exception as e:
    warnings.warn(
        f"Failed to import flash attn related modules with error message {e}")
    flash_attn_2_8_qkvpacked_func = None
    flash_attn_2_8_varlen_qkvpacked_func = None


try:
    from magicodec.components.ops.fused_dense import FusedDense
except Exception as e:
    warnings.warn(
        f"Failed to import flash attn related modules with error message {e}")
    FusedDense = None

try:
    from magicodec.components.ops.rotary import RotaryEmbedding
except Exception as e:
    rank_zero_warn(
        f"Failed to import flash attn related modules with error message {e}")
    RotaryEmbedding = None

try:
    import ft_attention
except Exception as e:
    warnings.warn(
        f"Failed to import flash attn related modules with error message {e}")
    ft_attention = None



class FlashSelfAttentionV2_8(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        window_size=[-1, -1],
        window_type=0,
    ):
        super().__init__()
        assert (flash_attn_2_8_qkvpacked_func
                is not None), "FlashAttention v2.8 is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

        self.window_type = window_type
        if window_type == 1:
            assert (
                window_size[1]
                >= 1), "use blockwise window mask only support [any, x>=1] now"
        self.window_size = window_size

    def forward(
        self,
        qkv,
        causal: Optional[bool],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
        return_attn_probs: bool,
        use_window_mask: bool,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        if use_window_mask:
            if causal:
                window_size = [-1, 0]
                #window_type = 0
            else:
                window_size = self.window_size
                #window_type = self.window_type
        else:
            if causal:
                window_size = [-1, 0]
            else:
                window_size = [-1, -1]
            #window_type = 0

        unpadded = cu_seqlens is not None
        if unpadded:
            raise NotImplementedError
        else:
            return flash_attn_2_8_qkvpacked_func(
                qkv,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                window_size=window_size,
                #window_type=window_type,
                return_attn_probs=return_attn_probs,
            )


class FlashCrossAttentionV2_8(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
            self,
            causal=False,
            softmax_scale=None,
            attention_dropout=0.0,
            window_size=(-1, -1),
            window_type=0,
    ):
        super().__init__()
        #assert (flash_attn_varlen_kvpacked_func
        #        is not None), "FlashAttention is not installed"
        #assert flash_attn_kvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

        self.window_type = window_type
        if window_type == 1:
            assert (
                window_size[1]
                >= 1), "use blockwise window mask only support [any, x>=1] now"
        self.window_size = window_size

    def forward(
        self,
        q,
        kv,
        causal: Optional[bool],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen_k: Optional[int],
        return_attn_probs: bool,
        use_window_mask: bool,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        if use_window_mask:
            window_size = [-1, 0] if causal else self.window_size
            window_type = 0 if causal else self.window_type
        else:
            window_size = [-1, 0] if causal else [-1, -1]
            window_type = 0

        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_2_8_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens,
                cu_seqlens_k,
                max_seqlen,
                max_seqlen_k,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                window_size=window_size,
                window_type=window_type,
                return_attn_probs=return_attn_probs,
            )
        else:
            batch_size = q.shape[0]
            # seqlen_k = kv.shape[1]
            assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
            return flash_attn_2_8_kvpacked_func(
                q,
                kv,
                self.drop.p if self.training else 0.0,
                causal=causal,
                window_size=window_size,
                window_type=window_type,
                softmax_scale=self.softmax_scale,
                return_attn_probs=return_attn_probs,
            )

class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self,
                 causal=False,
                 softmax_scale=None,
                 attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen),
                                      -10000.0,
                                      dtype=scores.dtype,
                                      device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device),
                1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)

        return output


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self,
                 causal=False,
                 softmax_scale=None,
                 attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert (kv.shape[0] == batch_size and kv.shape[3] == q.shape[2]
                and kv.shape[4] == q.shape[3])
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k),
                -10000.0,
                dtype=scores.dtype,
                device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(
                torch.full((seqlen_q, seqlen_k),
                           -10000.0,
                           device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_sequence_len,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        if not inference_params.fused_ft_kernel:
            kv_cache = inference_params.key_value_memory_dict[layer_idx]
        else:
            # For FT, k_cache has shape (b, h, headdim / packsize, s, packsize)
            # where packsize = 4 if fp32, 8 if fp16 or bf16.
            # v_cache has shape (b, h, s, headdim)
            k_cache, v_cache = inference_params.key_value_memory_dict[
                layer_idx]
            kv_cache = None
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.sequence_len_offset
    if inference_params.n_look_feature is not None and not inference_params.last:
        assert (kv.shape[1] == inference_params.n_look_feature
                ), f"{kv.shape[1]} != {inference_params.n_look_feature}"
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= (kv_cache.shape[0]
                         if kv_cache is not None else v_cache.shape[0])
    assert sequence_end <= (kv_cache.shape[1]
                            if kv_cache is not None else v_cache.shape[2])
    # Copy key and values.
    if not inference_params.fused_ft_kernel:
        assert kv_cache is not None
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        if inference_params.n_look_past is None:
            kv = kv_cache[batch_start:batch_end, :sequence_end, ...]
        else:
            kv = kv_cache[
                batch_start:batch_end,
                max(0, sequence_start - inference_params.n_look_past +
                    1):sequence_end,
                ...,
            ]
        return kv
    else:
        assert inference_params.sequence_len_offset == 0
        # FT kernel requires different layouts for the k_cache and v_cache.
        assert kv.dtype in [torch.float16, torch.bfloat16, torch.float32]
        packsize = 4 if kv.dtype == torch.float32 else 8
        if kv_cache is not None:
            kv_cache[batch_start:batch_end, sequence_start:sequence_end,
                     ...] = kv
            k_cache = rearrange(
                kv_cache[:, :, 0],
                "b s h (d packsize) -> b h d s packsize",
                packsize=packsize,
            ).contiguous()
            v_cache = rearrange(kv_cache[:, :, 1],
                                "b s h d -> b h s d").contiguous()
            inference_params.key_value_memory_dict[layer_idx] = (k_cache,
                                                                 v_cache)
        else:
            k_cache[batch_start:batch_end, :, :, :sequence_end, :] = rearrange(
                kv[:, :, 0],
                "b s h (d packsize) -> b h d s packsize",
                packsize=packsize)
            v_cache[batch_start:batch_end, :, :sequence_end, :] = rearrange(
                kv[:, :, 1], "b s h d -> b h s d")
        return kv


FLASHATTN_VERSIONS = ["2.8"]

fa_selfattn_cls = {
    "2.8": FlashSelfAttentionV2_8,
}

fa_crossattn_cls = {
    "2.8": FlashCrossAttentionV2_8,
}


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        rotary_emb_compat="default",
        use_rotary_triton=False,
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        blocksparse=False,
        blockmask=None,
        use_qk_norm="",
        version="2",
        window_size=[-1, -1],  # no mask
        window_type=0,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """

        assert isinstance(version, (str, int, float))
        version = str(version)
        assert version in FLASHATTN_VERSIONS
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.version = version
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert (self.embed_dim %
                num_heads == 0), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (self.num_heads % self.num_heads_kv == 0
                ), "self.num_heads must be divisible by self.num_heads_kv"
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv
        self.k_embed_dim = self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert (
                not cross_attn
            ), "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                compat=rotary_emb_compat,
                use_triton=use_rotary_triton,
                device=device,
            )

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (LinearResidual if not fused_bias_fc else partial(
            FusedDense, return_residual=True))

        inner_attn_cls_args = {
            "causal": causal,
            "softmax_scale": softmax_scale,
            "attention_dropout": dropout,
        }
        inner_cross_attn_cls_args = {
            "causal": causal,
            "softmax_scale": softmax_scale,
            "attention_dropout": dropout,
        }

        self.window_size = window_size
        self.window_type = window_type
        if self.version == "2.8" and use_flash_attn:
            inner_attn_cls_args.update({
                "window_size": self.window_size,
                "window_type": self.window_type
            })
            inner_cross_attn_cls_args.update({
                "window_size": self.window_size,
                "window_type": self.window_type
            })
        if False and blocksparse:
            assert (self.version == "1" and use_flash_attn
                    ), "enable blocksparse only supoort version==1"
            inner_attn_cls = (FlashBlocksparseSelfAttention
                              if use_flash_attn else BlocksparseSelfAttention)
            assert blockmask is not None
            inner_attn_cls_args["blockmask"] = blockmask
        else:
            inner_attn_cls = (fa_selfattn_cls[version]
                              if use_flash_attn else SelfAttention)

        inner_cross_attn_cls = (fa_crossattn_cls[version]
                                if use_flash_attn else CrossAttention)

        if not self.cross_attn:
            if not self.return_residual:
                self.Wqkv = linear_cls(embed_dim,
                                       qkv_dim,
                                       bias=qkv_proj_bias,
                                       **factory_kwargs)
            else:
                self.Wqkv = linear_resid_cls(embed_dim,
                                             qkv_dim,
                                             bias=qkv_proj_bias,
                                             **factory_kwargs)
            if self.dwconv:
                if self.num_heads_kv == self.num_heads:
                    self.dwconv_qkv = nn.Conv1d(qkv_dim,
                                                qkv_dim,
                                                kernel_size=3,
                                                padding=2,
                                                groups=qkv_dim)
                else:
                    self.dwconv_q = nn.Conv1d(embed_dim,
                                              embed_dim,
                                              kernel_size=3,
                                              padding=2,
                                              groups=embed_dim)
                    self.dwconv_kv = nn.Conv1d(kv_dim,
                                               kv_dim,
                                               kernel_size=3,
                                               padding=2,
                                               groups=kv_dim)

        else:
            self.Wq = linear_cls(embed_dim,
                                 embed_dim,
                                 bias=qkv_proj_bias,
                                 **factory_kwargs)
            if not self.return_residual:
                self.Wkv = linear_cls(embed_dim,
                                      kv_dim,
                                      bias=qkv_proj_bias,
                                      **factory_kwargs)
            else:
                self.Wkv = linear_resid_cls(embed_dim,
                                            kv_dim,
                                            bias=qkv_proj_bias,
                                            **factory_kwargs)
            if self.dwconv:
                self.dwconv_q = nn.Conv1d(embed_dim,
                                          embed_dim,
                                          kernel_size=3,
                                          padding=2,
                                          groups=embed_dim)
                self.dwconv_kv = nn.Conv1d(kv_dim,
                                           kv_dim,
                                           kernel_size=3,
                                           padding=2,
                                           groups=kv_dim)
        self.inner_attn = inner_attn_cls(**inner_attn_cls_args)
        self.inner_cross_attn = inner_cross_attn_cls(
            **inner_cross_attn_cls_args)
        self.out_proj = linear_cls(embed_dim,
                                   embed_dim,
                                   bias=out_proj_bias,
                                   **factory_kwargs)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm == "head":
            self.q_norm = RMSNorm(self.head_dim, eps=1e-8)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-8)
        elif use_qk_norm == "channel":
            self.q_norm = RMSNorm(embed_dim, eps=1e-8)
            self.k_norm = RMSNorm(self.k_embed_dim, eps=1e-8)
        elif use_qk_norm != "":
            raise NotImplementedError

    def allocate_inference_cache(self,
                                 batch_size,
                                 max_seqlen,
                                 dtype=None,
                                 fused_ft_kernel=True):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        if not fused_ft_kernel:
            return torch.empty(
                batch_size,
                max_seqlen,
                2,
                self.num_heads_kv,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
        else:
            assert dtype in [torch.float16, torch.bfloat16, torch.float32]
            packsize = 4 if dtype == torch.float32 else 8
            assert self.head_dim % packsize == 0
            k_cache = torch.empty(
                batch_size,
                self.num_heads_kv,
                self.head_dim // packsize,
                max_seqlen,
                packsize,
                dtype=dtype,
                device=device,
            )
            v_cache = torch.empty(
                batch_size,
                self.num_heads_kv,
                max_seqlen,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            return k_cache, v_cache

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, n_kv_heads, head_dim) or (batch_size, 1, 2, n_kv_heads, head_dim)"""
        assert not self.dwconv, "Generation does not support dwconv yet"
        assert (
            self.layer_idx
            is not None), "Generation requires layer_idx in the constructor"
        # modify inference_params for attention window mask
        if (self.window_type == 0 and self.window_size[0] == -1
                and self.window_size[1] in [-1, 0]):
            # non-window-mask (casual or non-casual)
            pass
        else:
            if inference_params.n_look_past is None:
                if self.window_size[0] != -1:
                    inference_params.n_look_past = self.window_size[0]
            if inference_params.n_look_feature is None:
                if self.window_type == 0:
                    inference_params.n_look_feature = self.window_size[1] + 1
                elif self.window_type == 1:
                    inference_params.n_look_feature = self.window_size[1]
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _get_inner_attn_args(
        self,
        qkv,
        causal: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        return_attn_probs=False,
        key_padding_mask=None,
        use_window_mask=False,
    ):
        if self.use_flash_attn:
            input_args = (qkv, causal, cu_seqlens, max_seqlen)
            if self.version in ["2", "2.8"]:
                input_args += (return_attn_probs, )
            if self.version in ["2.8"]:
                input_args += (use_window_mask, )
        else:
            input_args = (qkv, causal, key_padding_mask)
        return input_args

    def _get_inner_cross_attn_args(
        self,
        q,
        kv,
        causal: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        return_attn_probs: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        use_window_mask: bool = False,
    ):
        if self.use_flash_attn:
            input_args = (
                q,
                kv,
                causal,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_k,
                max_seqlen_k,
            )
            if self.version in ["2", "2.8"]:
                input_args += (return_attn_probs, )
            if self.version in ["2.8"]:
                input_args += (use_window_mask, )
        else:
            input_args = (q, kv, causal, key_padding_mask)
        return input_args

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        indices=None,
        mixer_subset=None,
        inference_params=None,
        return_attn_probs=False,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            return_attn_probs: return attn_probs (softmax(mm(qk)/sqrt(d))) default:False
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        is_pad = True
        if cu_seqlens is not None:
            assert max_seqlen is not None
            # assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            # if self.rotary_emb_dim > 0:
            #     # assert indices is not None
            #     # assert key_padding_mask is not None
            # else:
            #     assert key_padding_mask is None
            is_pad = False

        if key_padding_mask is not None:
            if self.rotary_emb_dim == 0:
                assert cu_seqlens is None
                assert max_seqlen is None
                assert not self.use_flash_attn

        if inference_params is not None:
            if self.version == 2.8 and self.use_flash_attn:
                # if self.cross_attn:
                #     raise NotImplementedError("no support crossattn generation in flashattn 2.8 now")
                if not (self.window_type == 0 and self.window_size[0] <= -1
                        and self.window_size[1] <= 0):
                    raise NotImplementedError(
                        "no support causal or no_causal&no_mask generation in flashattn 2.8 now"
                    )
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv

        kwargs = ({
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            **kwargs
        } if self.use_flash_attn else {
            "key_padding_mask": key_padding_mask,
            **kwargs
        })

        if return_attn_probs:
            assert (self.use_flash_attn and self.version == 2
                    ), "only support return_attn_probs=True used by FlashAttn2"
            assert self.training, "only support return_attn_probs=True in training now"

        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(
                    self.dwconv_qkv(rearrange(qkv,
                                              "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
            if self.use_qk_norm == "channel":
                if qkv.ndim == 2:
                    qkv[:, :self.embed_dim] = self.q_norm(
                        qkv[:, :self.embed_dim])
                    qkv[:,
                        self.embed_dim:int(self.embed_dim * 2)] = self.k_norm(
                            qkv[:, self.embed_dim:int(self.embed_dim * 2)])
                elif qkv.ndim == 3:
                    qkv[:, :, :self.embed_dim] = self.q_norm(
                        qkv[:, :, :self.embed_dim])
                    qkv[:, :,
                        self.embed_dim:int(self.embed_dim * 2)] = self.k_norm(
                            qkv[:, :, self.embed_dim:int(self.embed_dim * 2)])
            qkv = rearrange(qkv,
                            "... (three h d) -> ... three h d",
                            three=3,
                            d=self.head_dim)
            if self.use_qk_norm == "head":
                if qkv.ndim == 4:
                    qkv[:, 0] = self.q_norm(qkv[:, 0])
                    qkv[:, 1] = self.k_norm(qkv[:, 1])
                elif qkv.ndim == 5:
                    qkv[:, :, 0] = self.q_norm(qkv[:, :, 0])
                    qkv[:, :, 1] = self.k_norm(qkv[:, :, 1])

            if inference_params is None:
                if self.rotary_emb_dim > 0:
                    if key_padding_mask is not None and indices is not None:
                        # print("rotary mha path0")
                        if not is_pad:
                            qkv = pad_input(qkv, indices,
                                            cu_seqlens.shape[0] - 1,
                                            max_seqlen)
                        qkv = self.rotary_emb(qkv, None, 0, None, None, None,
                                              None, None)
                        if not is_pad:
                            qkv, _, _, _ = unpad_input(qkv, key_padding_mask)
                    else:
                        # print("rotary mha path1")
                        qkv = self.rotary_emb(qkv, None, 0, None, cu_seqlens,
                                              None, max_seqlen, None)
                input_args = self._get_inner_attn_args(
                    qkv,
                    causal=kwargs.get("causal", None),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    return_attn_probs=return_attn_probs,
                    key_padding_mask=key_padding_mask,
                    use_window_mask=True,
                )

                if not self.checkpointing:
                    attn_outs = self.inner_attn(*input_args)
                    if not isinstance(attn_outs, tuple) and not isinstance(attn_outs, list):
                        attn_outs = (attn_outs, )
                else:
                    attn_outs = torch.utils.checkpoint.checkpoint(
                        self.inner_attn, *input_args)

                if return_attn_probs:
                    assert (
                        len(attn_outs) == 4
                    ), f"Expect 4 but got {len(attn_outs)} when return_attn_probs=True in attention"
                    context, lse, score_cummax, dmask = attn_outs
                else:
                    context = attn_outs[0]
            else:
                if (not inference_params.fused_ft_kernel
                    ) or inference_params.sequence_len_offset == 0:
                    # TODO: support varlen
                    assert is_pad
                    if self.rotary_emb_dim > 0:
                        qkv = self.rotary_emb(
                            qkv,
                            None,
                            inference_params.sequence_len_offset,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    q = qkv[:, :, 0]
                    kv = self._update_kv_cache(qkv[:, :, 1:], inference_params)
                    # If we're processing the prompt, causal=None (use self.causal).
                    # If we're decoding, then causal=False.
                    causal = (None if inference_params.sequence_len_offset == 0
                              else False)
                    input_args = self._get_inner_cross_attn_args(
                        q, kv, causal=causal, use_window_mask=False)
                    context = self.inner_cross_attn(*input_args)

                    if isinstance(context, (tuple, list)):
                        assert len(context) == 1
                        context = context[0]
                else:
                    assert inference_params.fused_ft_kernel
                    assert ft_attention is not None
                    batch_start = inference_params.batch_size_offset
                    batch_end = batch_start + qkv.shape[0]
                    k_cache, v_cache = inference_params.key_value_memory_dict[
                        self.layer_idx]
                    lengths_per_sample = (
                        inference_params.
                        lengths_per_sample[batch_start:batch_end]
                        if inference_params.lengths_per_sample is not None else
                        None)
                    rotary_emb_base = (self.rotary_emb.base
                                       if self.rotary_emb_dim > 0 else 0)
                    context = ft_attention.single_query_attention(
                        *rearrange(
                            qkv, "b 1 three h d -> b three h d").unbind(dim=1),
                        k_cache[batch_start:batch_end],
                        v_cache[batch_start:batch_end],
                        lengths_per_sample,
                        None,  # rotary_cos_
                        None,  # rotary_sin_
                        None,  # nnz_head_idx
                        inference_params.sequence_len_offset,
                        self.rotary_emb_dim,
                        rotary_emb_base,
                        # neox_rotary_style
                        ((not self.rotary_emb.interleaved)
                         if self.rotary_emb_dim > 0 else True),
                    )
                    context = rearrange(context, "b h d -> b 1 h d")
        else:
            if self.cross_attn:  # mha cross branch
                if not self.return_residual:
                    q = self.Wq(x if mixer_subset is None else x[:,
                                                                 mixer_subset])
                    kv = self.Wkv(x_kv if x_kv is not None else x)
                else:
                    if x_kv is not None:
                        kv, x_kv = self.Wkv(x_kv)
                    else:
                        kv, x = self.Wkv(x)
                    q = self.Wq(x if mixer_subset is None else x[:,
                                                                 mixer_subset])
                q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
                kv = rearrange(kv,
                               "... (two h d) -> ... two h d",
                               two=2,
                               d=self.head_dim)
                if self.dwconv:
                    q = rearrange(
                        self.dwconv_q(rearrange(q,
                                                "b s d -> b d s"))[..., :-2],
                        "b d s -> b s d",
                    ).contiguous()
                    kv = rearrange(
                        self.dwconv_kv(rearrange(kv,
                                                 "b s d -> b d s"))[..., :-2],
                        "b d s -> b s d",
                    ).contiguous()
                if inference_params is None:
                    input_args = self._get_inner_cross_attn_args(
                        q,
                        kv,
                        causal=kwargs.get("causal", None),
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        cu_seqlens_k=kwargs.get("cu_seqlens_k", None),
                        max_seqlen_k=kwargs.get("max_seqlen_k", None),
                        return_attn_probs=return_attn_probs,
                        key_padding_mask=key_padding_mask,
                        use_window_mask=True,
                    )
                    if not self.checkpointing:
                        attn_outs = self.inner_cross_attn(*input_args)
                    else:
                        attn_outs = torch.utils.checkpoint.checkpoint(
                            self.inner_cross_attn, *input_args)

                    if return_attn_probs:
                        assert (
                            len(attn_outs) == 4
                        ), f"Expect 4 but got {len(attn_outs)} when return_attn_probs=True in attention"
                        context, lse, score_cummax, dmask = attn_outs
                    else:
                        context = attn_outs
                        if isinstance(context, (tuple, list)):
                            assert len(context) == 1
                            context = attn_outs[0]
                else:
                    kv = self._update_kv_cache(kv)
                    input_args = self._get_inner_cross_attn_args(
                        q, kv, causal=False, use_window_mask=False)
                    context = self.inner_cross_attn(*input_args)
                    if isinstance(context, (tuple, list)):
                        assert len(context) == 1
                        context = context[0]
            else:  # gqa branch
                assert self.num_heads_kv != self.num_heads
                if not self.return_residual:
                    qkv = self.Wqkv(x)
                else:
                    qkv, x = self.Wqkv(x)
                q = qkv[..., :self.num_heads * self.head_dim]
                kv = qkv[..., self.num_heads * self.head_dim:]
                if self.dwconv:
                    q = rearrange(
                        self.dwconv_q(rearrange(q,
                                                "b s d -> b d s"))[..., :-2],
                        "b d s -> b s d",
                    ).contiguous()
                    kv = rearrange(
                        self.dwconv_kv(rearrange(kv,
                                                 "b s d -> b d s"))[..., :-2],
                        "b d s -> b s d",
                    ).contiguous()
                if self.use_qk_norm == "channel":
                    if kv.ndim == 2:
                        q = self.q_norm(q)
                        kv[:, :self.k_embed_dim] = self.k_norm(
                            kv[:, :self.k_embed_dim])
                    elif kv.ndim == 3:
                        q = self.q_norm(q)
                        kv[:, :, :self.k_embed_dim] = self.k_norm(
                            kv[:, :, :self.k_embed_dim])
                q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
                kv = rearrange(kv,
                               "... (two h d) -> ... two h d",
                               two=2,
                               d=self.head_dim)
                if self.use_qk_norm == "head":
                    if kv.ndim == 4:
                        q = self.q_norm(q)
                        kv[:, 1] = self.k_norm(kv[:, 1])
                    elif kv.ndim == 5:
                        q = self.q_norm(q)
                        kv[:, :, 1] = self.k_norm(kv[:, :, 1])
                if inference_params is None:
                    if self.rotary_emb_dim > 0:
                        if key_padding_mask is not None and indices is not None:
                            # print("rotary gqa path0")
                            if not is_pad:
                                q = pad_input(q, indices,
                                              cu_seqlens.shape[0] - 1,
                                              max_seqlen)
                                kv = pad_input(kv, indices,
                                               cu_seqlens.shape[0] - 1,
                                               max_seqlen)
                            q, kv = self.rotary_emb(q, kv, 0, 0, None, None,
                                                    None, None)
                            if not is_pad:
                                q, _, _, _ = unpad_input(q, key_padding_mask)
                                kv, _, _, _ = unpad_input(kv, key_padding_mask)

                        else:
                            # print("rotary gqa path1")
                            q, kv = self.rotary_emb(
                                q,
                                kv,
                                0,
                                0,
                                cu_seqlens,
                                cu_seqlens,
                                max_seqlen,
                                max_seqlen,
                            )
                    input_args = self._get_inner_cross_attn_args(
                        q,
                        kv,
                        causal=kwargs.get("causal", None),
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_k=max_seqlen,
                        return_attn_probs=return_attn_probs,
                        key_padding_mask=key_padding_mask,
                        use_window_mask=True,
                    )
                    if not self.checkpointing:
                        attn_outs = self.inner_cross_attn(*input_args)
                    else:
                        attn_outs = torch.utils.checkpoint.checkpoint(
                            self.inner_cross_attn, *input_args)

                    if return_attn_probs:
                        assert (
                            len(attn_outs) == 4
                        ), f"Expect 4 but got {len(attn_outs)} when return_attn_probs=True in attention"
                        context, lse, score_cummax, dmask = attn_outs
                    else:
                        context = attn_outs[0]
                else:
                    if self.rotary_emb_dim > 0:
                        assert is_pad
                        q, kv = self.rotary_emb(
                            q,
                            kv,
                            inference_params.sequence_len_offset,
                            inference_params.sequence_len_offset,
                            None,
                            None,
                            None,
                            None,
                        )
                    kv = self._update_kv_cache(kv, inference_params)
                    # If we're processing the prompt, causal=None (use self.causal).
                    # If we're decoding, then causal=False.
                    causal = (None if inference_params.sequence_len_offset == 0
                              else False)
                    input_args = self._get_inner_cross_attn_args(
                        q, kv, causal=causal, use_window_mask=False)
                    context = self.inner_cross_attn(*input_args)

                    if isinstance(context, (tuple, list)):
                        assert len(context) == 1
                        context = context[0]

        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        outputs = (out, )
        if self.return_residual:
            outputs += (x, )
        if return_attn_probs:
            outputs += ((lse, score_cummax, dmask), )
        return outputs
