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


def reference(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    bs = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == bs
    cumsum = torch.cumsum(extend_seq_lens, dim=0)
    start = torch.zeros_like(cumsum)
    start[1:] = cumsum[:-1]

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    for i in range(bs):
        s = int(extend_seq_lens[i])
        p = int(extend_prefix_lens[i]) if has_prefix else 0
        beg = int(start[i])
        positions[beg : beg + s] = torch.arange(
            p, p + s, dtype=torch.int64, device=extend_seq_lens.device
        )
    return positions, start.to(torch.int32)
