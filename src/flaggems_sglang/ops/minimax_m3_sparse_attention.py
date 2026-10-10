# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang M3 selected-block attention backed by FlagAttention.

The framework retains its indexer and metadata construction. K/V token-slot
buffers are split into page views without copying. Device kernels and decode
workspace policy are shared with the vLLM adapter through FlagAttention.
"""

__all__ = [
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "minimax_m3_sparse_decode_workspace",
]


def _cache_views(k_cache, v_cache, page_size):
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or page_size <= 0
    ):
        raise ValueError("page_size must be a positive integer")
    if k_cache.ndim != 3 or tuple(k_cache.shape) != tuple(v_cache.shape):
        raise ValueError(
            "K/V must have matching [slots, kv_heads, head_dim] shapes"
        )
    if k_cache.shape[0] % page_size:
        raise ValueError("KV slot count must be divisible by page_size")
    pages = k_cache.shape[0] // page_size
    return (
        k_cache.unflatten(0, (pages, page_size)),
        v_cache.unflatten(0, (pages, page_size)),
    )


def _features(block_size_q, block_size_k, sink):
    if block_size_q != 1 or block_size_k != 128:
        raise NotImplementedError(
            "Only query blocks of 1 and sparse blocks of 128 are supported"
        )
    if sink is not None:
        raise NotImplementedError("Attention sinks are not supported")


def minimax_m3_sparse_attn(
    q,
    k_cache,
    v_cache,
    topk_idx,
    page_table,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    max_seqlen_q,
    sm_scale=None,
    output=None,
    *,
    page_size=1,
    block_size_q=1,
    block_size_k=128,
    sink=None,
    q_scale=None,
    k_scale=None,
    v_scale=None,
):
    """Return causal GQA prefill output using the framework's selected blocks.

    topk_idx is [kv_heads, total_q, topk] with logical 128-token block IDs.
    Packed verification can use explicit cu_seqlens/prefix_lens through this API;
    arbitrary tree/custom masks are unsupported.
    """
    _features(block_size_q, block_size_k, sink)
    k, v = _cache_views(k_cache, v_cache, page_size)
    from flag_attn import minimax_m3_sparse_attn_paged

    return minimax_m3_sparse_attn_paged(
        q,
        k,
        v,
        topk_idx,
        page_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_seqlen_q,
        q.shape[-1] ** -0.5 if sm_scale is None else sm_scale,
        output,
        page_size=page_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def minimax_m3_sparse_attn_decode(
    q,
    k_cache,
    v_cache,
    topk_idx,
    page_table,
    seq_lens,
    sm_scale=None,
    output=None,
    decode_query_len=1,
    *,
    page_size=1,
    block_size_k=128,
    sink=None,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    workspace=None,
):
    """Return uniform multi-token causal decode output.

    Reuse a preallocated workspace across sequential launches only. The API
    leaves the indexer and request/Graph metadata in the framework.
    """
    _features(1, block_size_k, sink)
    k, v = _cache_views(k_cache, v_cache, page_size)
    from flag_attn import minimax_m3_sparse_attn_decode_paged

    return minimax_m3_sparse_attn_decode_paged(
        q,
        k,
        v,
        topk_idx,
        page_table,
        seq_lens,
        q.shape[-1] ** -0.5 if sm_scale is None else sm_scale,
        output,
        decode_query_len,
        page_size=page_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace=workspace,
    )


def minimax_m3_sparse_decode_workspace(q, num_kv_heads, topk):
    """Allocate shared-library decode scratch before capture."""
    from flag_attn import minimax_m3_sparse_decode_workspace as allocate

    return allocate(q, num_kv_heads, topk)
