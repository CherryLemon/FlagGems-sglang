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

import torch


def reference(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids,
):
    # ``num_experts`` counts the trailing "filtered expert" slot, which never
    # receives tokens and therefore gets no blocks; only 0..num_experts-2 do.
    num_routed = num_experts - 1
    flat = topk_ids.flatten()
    numel = flat.numel()
    sorted_ids = torch.full_like(sorted_token_ids, numel)
    eids = expert_ids.clone()

    offset = 0
    for e in range(num_routed):
        idx = (flat == e).nonzero().flatten()
        n = idx.numel()
        aligned = ((n + block_size - 1) // block_size) * block_size
        if n:
            sorted_ids[offset : offset + n] = idx.to(sorted_ids.dtype)
        nblocks = aligned // block_size
        if nblocks:
            beg = offset // block_size
            eids[beg : beg + nblocks] = e
        offset += aligned

    npost = torch.full_like(num_tokens_post_pad, offset)
    return sorted_ids, eids, npost
