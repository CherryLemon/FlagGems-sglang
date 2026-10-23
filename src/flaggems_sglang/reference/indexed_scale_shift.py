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


def reference(x, shift, scale, indices):
    idx = indices.long()
    sh = shift[idx].float()
    sc = scale[idx].float()
    # The two explicit bf16 rounds reproduce the eager kernel's boundaries and
    # are part of the contract.
    one_plus = (1.0 + sc).to(torch.bfloat16).float()
    scaled = (x.float() * one_plus).to(torch.bfloat16).float()
    return (scaled + sh).to(x.dtype)
