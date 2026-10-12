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
import torch.nn.functional as F


def reference(
    router_logits,
    input_ids,
    tid2eid,
    num_fused_shared_experts,
    routed_scaling_factor,
    scoring_func,
):
    assert scoring_func == "sqrtsoftplus"
    num_tokens, num_routed = router_logits.shape

    expert_ids = tid2eid[input_ids.long()].long()
    logits = torch.gather(router_logits.float(), 1, expert_ids)
    w = torch.sqrt(F.softplus(logits))
    w = w / w.sum(dim=-1, keepdim=True)

    shared_w = torch.full(
        (num_tokens, num_fused_shared_experts),
        1.0 / routed_scaling_factor,
        dtype=torch.float32,
        device=router_logits.device,
    )
    shared_ids = (
        num_routed
        + torch.arange(num_fused_shared_experts, device=router_logits.device)
    ).expand(num_tokens, -1)

    weights = torch.cat([w, shared_w], dim=-1).float()
    ids = torch.cat(
        [expert_ids.to(torch.int32), shared_ids.to(torch.int32)], dim=-1
    )
    return weights.contiguous(), ids.contiguous()
