# Copyright (c) 2022, Tri Dao.
# Adapted from https://github.com/NVIDIA/apex/blob/master/apex/contrib/layer_norm/layer_norm.py

import torch
import dropout_layer_norm

def _dropout_add_layer_norm_forward(
    x0,
    residual,
    gamma,
    beta,
    rowscale,
    colscale,
    dropout_p,
    epsilon,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    """Assume that arguments are contiguous"""
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat,
        residualmat,
        gamma,
        beta,
        rowscale,
        colscale,
        None,
        None,
        dropout_p,
        epsilon,
        1.0,
        0,
        None,
        residual_in_fp32,
        is_rms_norm,
    )
    # dmask is None if dropout_p == 0.0
    # xmat is None if dropout_p == 0.0 and residual is None and residual_dtype != input_dtype
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def _dropout_add_layer_norm_backward(
    dz,
    dx,
    x,
    x0,
    dmask,
    mu,
    rsigma,
    gamma,
    rowscale,
    colscale,
    dropout_p,
    has_residual,
    is_rms_norm=False,
):
    """Assume that arguments are contiguous
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + residual was not returned in the fwd).
    x0 must not be None if we have colscale.
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    if colscale is not None:
        assert x0 is not None, "x0 is required to compute the gradient of colscale"
    (dx0mat, dresidualmat, dgamma, dbeta, _, _, *rest) = (
        dropout_layer_norm.dropout_add_ln_bwd(
            dzmat,
            dxmat,
            xmat,
            x0mat,
            dmask,
            mu,
            rsigma,
            gamma,
            rowscale,
            colscale,
            None,
            None,
            dropout_p,
            1.0,
            0,
            has_residual,
            is_rms_norm,
        )
    )
    # dresidualmat is None if not has_residual
    if colscale is None:
        return dx0mat, dresidualmat, dgamma, dbeta
    else:
        dcolscale = rest[0]
        return dx0mat, dresidualmat, dgamma, dbeta, dcolscale


class DropoutAddLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0,
        residual,
        gamma,
        beta,
        rowscale,
        colscale,
        dropout_p,
        epsilon,
        residual_in_fp32=False,
        prenorm=False,
        is_rms_norm=False,
        return_dmask=False,
    ):
        x0 = x0.contiguous()
        residual = residual.contiguous() if residual is not None else None
        gamma = gamma.contiguous()
        beta = beta.contiguous() if beta is not None else None
        rowscale = rowscale.contiguous() if rowscale is not None else None
        colscale = colscale.contiguous() if colscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(
            x0,
            residual,
            gamma,
            beta,
            rowscale,
            colscale,
            dropout_p,
            epsilon,
            residual_in_fp32,
            is_rms_norm,
        )
        # Only need to save x0 if we need to compute gradient wrt colscale
        x0_saved = x0 if colscale is not None else None
        ctx.save_for_backward(
            xmat.view(x0.shape), x0_saved, dmask, gamma, mu, rsigma, rowscale, colscale
        )
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta is not None
        if not return_dmask:
            return (
                zmat.view(x0.shape)
                if not prenorm
                else (zmat.view(x0.shape), xmat.view(x0.shape))
            )
        else:
            dmask = (
                dmask.view(x0.shape)
                if dropout_p > 0.0
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            ctx.mark_non_differentiable(dmask)
            return (
                (zmat.view(x0.shape), dmask)
                if not prenorm
                else (zmat.view(x0.shape), xmat.view(x0.shape), dmask)
            )

    @staticmethod
    def backward(ctx, dz, *args):
        # assert dz.is_contiguous()
        dz = dz.contiguous()  # this happens!
        dx = args[0].contiguous() if ctx.prenorm else None
        x, x0, dmask, gamma, mu, rsigma, rowscale, colscale = ctx.saved_tensors
        # x0 is None if colscale is None
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dresidualmat, dgamma, dbeta, *rest = _dropout_add_layer_norm_backward(
            dz,
            dx,
            x,
            x0,
            dmask,
            mu,
            rsigma,
            gamma,
            rowscale,
            colscale,
            dropout_p,
            has_residual,
            ctx.is_rms_norm,
        )
        dx0 = dx0mat.view(x.shape)
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return (
            dx0,
            dresidual,
            dgamma,
            dbeta if ctx.has_beta else None,
            None,
            dcolscale,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dropout_add_layer_norm(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormFn.apply(
        x0,
        residual,
        weight,
        bias,
        rowscale,
        layerscale,
        dropout_p,
        epsilon,
        residual_in_fp32,
        prenorm,
        False,
        return_dropout_mask,
    )

class DropoutAddLayerNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        prenorm=False,
        p=0.0,
        eps=1e-5,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x0, residual=None):
        return dropout_add_layer_norm(
            x0,
            residual,
            self.weight,
            self.bias,
            self.p if self.training else 0.0,
            self.eps,
            prenorm=self.prenorm,
            residual_in_fp32=self.residual_in_fp32,
        )


# -----------------------

def rms_norm(x, weight, epsilon):
    return DropoutAddLayerNormFn.apply(
        x, None, weight, None, None, None, 0.0, epsilon, False, False, True
    )

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def dropout_add_rms_norm(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormFn.apply(
        x0,
        residual,
        weight,
        bias,
        rowscale,
        layerscale,
        dropout_p,
        epsilon,
        residual_in_fp32,
        prenorm,
        True,
        return_dropout_mask,
    )


class DropoutAddRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        prenorm=False,
        p=0.0,
        eps=1e-5,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x0, residual=None):
        return dropout_add_rms_norm(
            x0,
            residual,
            self.weight,
            None,
            self.p if self.training else 0.0,
            self.eps,
            prenorm=self.prenorm,
            residual_in_fp32=self.residual_in_fp32,
        )
