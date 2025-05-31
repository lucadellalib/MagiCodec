# Copyright (c) 2023, Tri Dao.
# Inspired by https://github.com/NVIDIA/apex/blob/master/apex/fused_dense/fused_dense.py
# We make it work with pytorch amp and with bfloat16.
# The TensorParallel linear modules are inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/layers.py
from functools import partial
from typing import Optional

# import fused_dense_cuda  # from apex
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup

from components.ops.activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd
from components.utils.distributed import (
    all_gather_raw,
    all_reduce,
    all_reduce_raw,
    reduce_scatter,
    reduce_scatter_raw,
)


class FusedDenseFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        weight,
        bias,
        return_residual=False,
        process_group=None,
        sequence_parallel=True,
    ):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = (
                bias.to(dtype=torch.get_autocast_gpu_dtype())
                if bias is not None
                else None
            )
        weight = weight.contiguous()
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            else:
                total_x = x
        else:
            (weight,) = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, weight.t())
            else:
                grad_input = torch.addmm(
                    grad_input.reshape(batch_dim, grad_input.shape[-1]),
                    grad_output,
                    weight,
                )
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
                grad_input, handle_grad_input = reduce_fn(
                    grad_input, process_group, async_op=True
                )
        else:
            grad_input = None
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if process_group is not None and sequence_parallel:
                handle_x.wait()
            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                total_x.reshape(batch_dim, total_x.shape[-1]),
                grad_output,
                ctx.needs_input_grad[2],
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None


def fused_dense_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    if (
        x.is_cuda
        and weight.is_cuda
        and (bias is None or bias.is_cuda)
        and dtype_eligible
    ):
        return FusedDenseFunc.apply(
            x, weight, bias, return_residual, process_group, sequence_parallel
        )
    else:
        assert process_group is None
        out = F.linear(x, weight, bias)
        return out if not return_residual else (out, x)


class FusedDense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        return_residual: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.return_residual = return_residual

    def forward(self, x, process_group=None):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul.
        """
        return fused_dense_func(
            x,
            self.weight,
            self.bias,
            return_residual=self.return_residual,
            process_group=process_group,
        )
