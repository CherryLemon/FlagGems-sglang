# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hy4-preview numerical op: logical active-head BF16 sparse MLA only.

No device query at import. No Torch compute fallback. The plugin owns OFF and
native fallback routing; this module owns numerical metadata/allocations/JIT.
"""

import math

SPLITS, BK, BH, BDV = 32, 64, 16, 128


def metadata(q, kv, indices, sm_scale, active_heads, sink=None, lengths=None):
    # Current actual path always supplies both. Keep absent variants native
    # until independently qualified; they need no extra speculative compile.
    if sink is None or lengths is None:
        return None
    try:
        sq = q.shape[0]
        dev = q.device
        if not (
            type(sq) is int
            and sq in (1, 4)
            and type(active_heads) is int
            and active_heads == 4
            and dev.type == "cuda"
            and tuple(q.shape) == (sq, 64, 576)
            and tuple(q.stride()) == (36864, 576, 1)
            and str(q.dtype) == "torch.bfloat16"
            and kv.ndim == 3
            and kv.shape[0] > 0
            and tuple(kv.shape[1:]) == (1, 576)
            and tuple(kv.stride()) == (576, 576, 1)
            and str(kv.dtype) == "torch.bfloat16"
            and kv.device == dev
            and tuple(indices.shape) == (sq, 1, 2048)
            and tuple(indices.stride()) == (2048, 2048, 1)
            and str(indices.dtype) == "torch.int32"
            and indices.device == dev
            and type(sm_scale) in (int, float)
            and math.isfinite(sm_scale)
            and sm_scale > 0
        ):
            return None
        for tensor, shape, dtype, strides in (
            (sink, (64,), "torch.float32", (1,)),
            (lengths, (sq,), "torch.int32", (1,)),
        ):
            if tensor is not None and not (
                tuple(tensor.shape) == shape
                and str(tensor.dtype) == dtype
                and tensor.device == dev
                and tuple(tensor.stride()) == strides
            ):
                return None
        import torch

        if (
            torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or torch.cuda.get_device_capability(dev) != (9, 0)
        ):
            return None
        for tensor in (q, kv, indices, sink, lengths):
            if (
                tensor.requires_grad
                or tensor.storage_offset() != 0
                or tensor.is_conj()
                or tensor.is_neg()
            ):
                return None
        return sq, active_heads, kv.shape[0], dev
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return None


def workspace(q, active_heads):
    """Caller may prepare these empty buffers once and retain through replay."""
    sq = q.shape[0]
    if not (
        type(sq) is int
        and sq in (1, 4)
        and type(active_heads) is int
        and active_heads == 4
    ):
        raise NotImplementedError(
            "prototype workspace is limited to SQ1/SQ4 with four active heads"
        )
    import torch

    return (
        torch.empty(
            (sq, SPLITS, active_heads, 512),
            device=q.device,
            dtype=torch.float32,
        ),
        torch.empty(
            (sq, SPLITS, active_heads), device=q.device, dtype=torch.float32
        ),
        torch.empty(
            (sq, SPLITS, active_heads), device=q.device, dtype=torch.float32
        ),
    )


def sparse_mla_active_split(
    q,
    kv,
    indices,
    sm_scale,
    *,
    active_heads,
    sink=None,
    lengths=None,
    partials=None,
):
    spec = metadata(q, kv, indices, sm_scale, active_heads, sink, lengths)
    if spec is None:
        raise NotImplementedError(
            "unsupported active-head sparse MLA metadata"
        )
    sq, heads, skv, dev = spec
    import torch

    from ._sparse_mla_active_split_kernels import (
        _active_head_sparse_mla_merge,
        _active_head_sparse_mla_partials,
    )

    if partials is None:
        partials = workspace(q, heads)
    po, pm, pl = partials
    expected = (
        (sq, SPLITS, heads, 512),
        (sq, SPLITS, heads),
        (sq, SPLITS, heads),
    )
    for value, shape in zip(partials, expected):
        if (
            tuple(value.shape) != shape
            or str(value.dtype) != "torch.float32"
            or value.device != dev
            or not value.is_contiguous()
        ):
            raise NotImplementedError("partial workspace metadata changed")
    # Only metadata/storage identity checks, never device value reads/copies.
    pointers = [v.untyped_storage().data_ptr() for v in partials]
    read = [
        v.untyped_storage().data_ptr()
        for v in (q, kv, indices, sink, lengths)
        if v is not None
    ]
    if len(set(pointers)) != 3 or any(v in read for v in pointers):
        raise NotImplementedError(
            "partial workspace must have independent owners"
        )
    out = torch.empty((sq, heads, 512), device=dev, dtype=q.dtype)
    maximum = torch.empty((sq, heads), device=dev, dtype=torch.float32)
    lse = torch.empty((sq, heads), device=dev, dtype=torch.float32)
    _active_head_sparse_mla_partials[(sq * SPLITS,)](
        q,
        kv,
        indices,
        lengths,
        po,
        pm,
        pl,
        skv,
        float(sm_scale),
        q.stride(0),
        kv.stride(0),
        heads,
        2048,
        lengths is not None,
        SPLITS,
        BK,
        BH,
        num_warps=8,
        num_stages=1,
    )
    _active_head_sparse_mla_merge[(sq * heads, 512 // BDV)](
        po,
        pm,
        pl,
        sink,
        out,
        maximum,
        lse,
        heads,
        sink is not None,
        SPLITS,
        BDV,
        num_warps=4,
        num_stages=1,
    )
    return out, maximum, lse


__all__ = ["sparse_mla_active_split"]
