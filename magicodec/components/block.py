# Copyright (c) 2022, Tri Dao.

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from magicodec.components.mha import (
    FLASHATTN_VERSIONS,
)

try:
    from magicodec.components.ops.norm import DropoutAddLayerNorm, dropout_add_layer_norm
except ImportError:
    DropoutAddLayerNorm, dropout_add_layer_norm = None, None

try:
    from magicodec.components.ops.norm import DropoutAddRMSNorm, RMSNorm, dropout_add_rms_norm
except ImportError:
    DropoutAddLayerNorm, RMSNorm, dropout_add_rms_norm = None, None, None


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x = x * (1 + scale.squeeze(0)) + shift.squeeze(0)
    return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls=None,
        mlp_cls=None,
        norm_cls=nn.LayerNorm,
        dropout_cls=nn.Dropout,
        prenorm=True,
        resid_dropout1=0.0,
        resid_dropout2=0.0,
        fused_dropout_add_ln=False,
        return_residual=False,
        residual_in_fp32=False,
        version="2",
        device=None,
        dtype=None,
    ):
        """
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.

        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        assert isinstance(version, (int, float, str))
        version = str(version)
        assert version in FLASHATTN_VERSIONS
        super().__init__()
        self.version = version
        self.prenorm = prenorm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, "residual_in_fp32 is only compatible with prenorm=True"
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)

        self.drop_path1 = None

        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = None
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert (dropout_add_layer_norm is not None), "dropout_layer_norm is not installed"
            assert (dropout_add_rms_norm is not None), "dropout_layer_norm is not installed"
            assert isinstance(self.norm1, (nn.LayerNorm, RMSNorm)) and isinstance(
                self.dropout1, nn.Dropout), (type(self.norm1), type(self.dropout1))

            if isinstance(self.norm1, RMSNorm):
                self.norm1 = DropoutAddRMSNorm(
                    dim,
                    prenorm=self.prenorm,
                    p=resid_dropout1,
                    eps=self.norm1.eps,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    dtype=dtype,
                )
            else:
                self.norm1 = DropoutAddLayerNorm(
                    dim,
                    prenorm=self.prenorm,
                    p=resid_dropout1,
                    eps=self.norm1.eps,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    dtype=dtype,
                )

            if isinstance(self.norm2, RMSNorm):
                self.norm2 = DropoutAddRMSNorm(
                    dim,
                    prenorm=self.prenorm,
                    p=resid_dropout2,
                    eps=self.norm2.eps,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    dtype=dtype,
                )
            else:
                self.norm2 = DropoutAddLayerNorm(
                    dim,
                    prenorm=self.prenorm,
                    p=resid_dropout2,
                    eps=self.norm2.eps,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    dtype=dtype,
                )
        self.use_fused_mha = False
        self.use_fused_mlp = False


    def forward(
        self,
        hidden_states: Tensor,
        #cond_embs: Optional[Tensor] = None,
        residual: Optional[Tensor] = None,
        mixer_subset=None,
        mixer_kwargs=None,
        condition: Optional[Tensor] = None,
        return_attn_probs=False,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        return_attn_probs = False
        fused_add_norm_fn = (dropout_add_rms_norm if RMSNorm and isinstance(self.norm1, RMSNorm)
                             else dropout_add_layer_norm)
        if self.prenorm:
            if not  (self.use_fused_mha and mixer_subset is None and not return_attn_probs and \
                (mixer_kwargs is None or "inference_params" not in mixer_kwargs)):
                if not self.fused_dropout_add_ln:
                    if self.drop_path1 is not None:
                        dropped = self.drop_path1(self.dropout1(hidden_states))
                    else:
                        dropped = self.dropout1(hidden_states)
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    # zxb
                    rowscale1 = None
                    hidden_states, residual = self.norm1(hidden_states, residual)

                if mixer_kwargs is None:
                    mixer_kwargs = {}
                if mixer_subset is not None:
                    mixer_kwargs["mixer_subset"] = mixer_subset
                assert not self.mixer.return_residual and not self.return_residual

                # self atten
                mixer_out = self.mixer(
                    hidden_states,
                    return_attn_probs = False,
                    **mixer_kwargs
                )
                assert len(mixer_out) == 1
                hidden_states = mixer_out[0]

                if mixer_subset is not None:
                    residual = residual[:, mixer_subset]

            ####################################################################################
            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    if self.drop_path2 is not None:
                        dropped = self.drop_path2(self.dropout2(hidden_states))
                    else:
                        dropped = self.dropout2(hidden_states)
                    residual = ((dropped + residual) if residual is not None else dropped)
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2 is not None:
                        if self.drop_path2.p == 0 or not self.training:
                            rowscale2 = None
                        else:
                            rowscale2 = self.drop_path2(
                                torch.ones(
                                    hidden_states.shape[:-1],
                                    device=hidden_states.device,
                                    dtype=hidden_states.dtype,
                                ))
                    else:
                        rowscale2 = None
                    hidden_states, residual = self.norm2(hidden_states, residual)
                hidden_states = self.mlp(hidden_states)
            block_outs = (hidden_states, residual)
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states,
                return_attn_probs=return_attn_probs,
                **(mixer_kwargs if mixer_kwargs is not None else {}),
            )
            assert self.return_residual and self.mixer.return_residual
            # if self.return_residual:  # mixer out is actually a pair here
            #     mixer_out, hidden_states = mixer_out
            if return_attn_probs:
                assert len(mixer_out) == 3
                (mixer_out, hidden_states, attn_porbs) = (
                    mixer_out  # attn_probs: tuple(lse,score_cummax,dmask)
                )
            else:
                mixer_out, hidden_states = mixer_out

            if not self.fused_dropout_add_ln:
                if self.drop_path1 is not None:
                    hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out)) +
                                                hidden_states).to(dtype=self.norm1.weight.dtype))
                else:
                    hidden_states = self.norm1((self.dropout1(mixer_out) +
                                                hidden_states).to(dtype=self.norm1.weight.dtype))
            else:
                if self.drop_path1 is not None:
                    if self.drop_path1.p == 0 or not self.training:
                        rowscale1 = None
                    else:
                        rowscale1 = self.drop_path1(
                            torch.ones(
                                mixer_out.shape[:-1],
                                device=mixer_out.device,
                                dtype=mixer_out.dtype,
                            ))
                else:
                    rowscale1 = None
                hidden_states = fused_add_norm_fn(
                    mixer_out,
                    hidden_states,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.dropout1.p if self.training else 0.0,
                    self.norm1.eps,
                    rowscale=rowscale1,
                    prenorm=False,
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    if self.drop_path2 is not None:
                        hidden_states = self.norm2(
                            (self.drop_path2(self.dropout2(mlp_out)) +
                             hidden_states).to(dtype=self.norm2.weight.dtype))
                    else:
                        hidden_states = self.norm2(
                            (self.dropout2(mlp_out) +
                             hidden_states).to(dtype=self.norm2.weight.dtype))
                else:
                    if self.drop_path2 is not None:
                        if self.drop_path2.p == 0 or not self.training:
                            rowscale2 = None
                        else:
                            rowscale2 = self.drop_path2(
                                torch.ones(
                                    mlp_out.shape[:-1],
                                    device=mlp_out.device,
                                    dtype=mlp_out.dtype,
                                ))
                    else:
                        rowscale2 = None
                    hidden_states = fused_add_norm_fn(
                        mlp_out,
                        hidden_states,
                        self.norm2.weight,
                        self.norm2.bias,
                        self.dropout2.p if self.training else 0.0,
                        self.norm2.eps,
                        rowscale=rowscale2,
                        prenorm=False,
                    )
            block_outs = (hidden_states,)
        return block_outs