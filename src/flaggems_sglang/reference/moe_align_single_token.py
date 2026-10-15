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


def reference(topk_ids, block_size):
    topk = topk_ids.shape[1]
    numel = topk_ids.numel()
    device = topk_ids.device
    sorted_ids = torch.full(
        (topk * block_size,), numel, dtype=torch.int32, device=device
    )
    experts = torch.sort(topk_ids.flatten()).values.to(torch.int32)
    order = torch.argsort(topk_ids.flatten())
    for slot in range(topk):
        sorted_ids[slot * block_size] = int(order[slot])
    num_post = torch.full(
        (1,), topk * block_size, dtype=torch.int32, device=device
    )
    return sorted_ids, experts, num_post
