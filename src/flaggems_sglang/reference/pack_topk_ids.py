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


def reference(topk_ids, topk_weights):
    # The low 16 bits are the weight's bf16 bit pattern, not a rounded
    # fixed-point value: the bf16 truncation is part of the contract.
    weight_bits = (
        topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        & 0xFFFF
    )
    return (topk_ids.to(torch.int32) << 16) | weight_bits
