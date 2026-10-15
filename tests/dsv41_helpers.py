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

"""Independent numerical references and deterministic synthetic fixtures."""

import torch


def fp8_inputs(m, n, k, seed=41):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, device="cuda", generator=g) * 0.25).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn(n, k, device="cuda", generator=g) * 0.25).to(
        torch.float8_e4m3fn
    )
    sa = torch.exp2(
        torch.randint(-3, 3, (m, k // 32), device="cuda", generator=g).float()
    )
    sb = torch.exp2(
        torch.randint(
            -3, 3, ((n + 31) // 32, k // 32), device="cuda", generator=g
        ).float()
    )
    return a, b, sa, sb


def fp8_reference(a, b, sa, sb, output_dtype=torch.bfloat16, **_):
    m, k = a.shape
    n = b.shape[0]
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for i in range(k // 32):
        sl = slice(32 * i, 32 * (i + 1))
        dot = a[:, sl].float() @ b[:, sl].float().T
        out += (
            dot * sa[:, i, None] * sb[:, i].repeat_interleave(32)[:n][None, :]
        )
    return out.to(output_dtype)


def fp4_inputs(b, width, ratio=1, seed=52, page=256):
    g = torch.Generator(device="cuda").manual_seed(seed)
    groups = (b + 5) // 6
    capacity = max(page, ((width * ratio + page - 1) // page) * page)
    payload = torch.randint(
        0, 256, (capacity, 64), device="cuda", dtype=torch.uint8, generator=g
    )
    scales = torch.randint(
        125, 129, (capacity, 4), device="cuda", dtype=torch.uint8, generator=g
    )
    table = torch.cat(
        (payload.reshape(-1, page * 64), scales.reshape(-1, page * 4)), 1
    )
    q = torch.randn((b, 32, 128), device="cuda", generator=g).bfloat16()
    weights = (torch.rand((b, 32), device="cuda", generator=g) / 32).bfloat16()
    mapping = torch.arange(capacity, dtype=torch.int32, device="cuda").repeat(
        groups, 1
    )
    req = torch.arange(b, device="cuda", dtype=torch.int64) // 6
    lens = torch.full((b,), width, dtype=torch.int64, device="cuda")
    return q, weights, mapping, req, lens, table, page, ratio, width


def fp4_reference(q, weights, mapping, req, lens, table, page, ratio, width):
    payload = table[:, : page * 64].reshape(-1, 64)
    scales = table[:, page * 64 :].reshape(-1, 4)
    codes = torch.stack((payload & 15, payload >> 4), -1).reshape(-1, 128)
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=q.device
    )
    values = lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    s = torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32), scales.int() - 127
    )
    s = torch.where(scales == 255, torch.nan, s)
    keys = (values * s.repeat_interleave(32, 1)).bfloat16()
    result = []
    for row in range(q.shape[0]):
        # The map stores full-token slots; the compressed pool divides by ratio.
        logical = torch.arange(width, device=q.device) * ratio
        slots = mapping[req[row], logical].long() // ratio
        dot = (q[row].float() @ keys[slots.clamp_min(0)].float().T).bfloat16()
        score = (
            (dot.relu().float() * weights[row, :, None].float())
            .bfloat16()
            .float()
            .sum(0)
            .bfloat16()
            .float()
        )
        valid = (torch.arange(width, device=q.device) < lens[row]) & (
            slots >= 0
        )
        result.append(torch.where(valid, score, -torch.inf))
    return torch.stack(result)


def fp4_serving_graph_inputs(b, ratio, seed=52):
    """640K Graph capacity, 128K visible input, distinct per-request KV pages."""
    page = 64
    full_capacity = 655627
    visible_full = 131080
    groups = (b + 5) // 6
    compressed_per_group = (
        (visible_full + ratio * page - 1) // (ratio * page)
    ) * page
    slots = groups * compressed_per_group
    g = torch.Generator(device="cuda").manual_seed(seed)
    payload = torch.randint(
        0, 256, (slots, 64), device="cuda", dtype=torch.uint8, generator=g
    )
    scales = torch.randint(
        125, 129, (slots, 4), device="cuda", dtype=torch.uint8, generator=g
    )
    table = torch.cat(
        (payload.reshape(-1, page * 64), scales.reshape(-1, page * 4)), 1
    )
    q = torch.randn((b, 32, 128), device="cuda", generator=g).bfloat16()
    weights = (torch.rand((b, 32), device="cuda", generator=g) / 32).bfloat16()
    mapping = torch.zeros(
        (groups, full_capacity), dtype=torch.int32, device="cuda"
    )
    for row in range(groups):
        mapping[row, :visible_full] = (
            torch.arange(visible_full, device="cuda", dtype=torch.int32)
            + row * compressed_per_group * ratio
        )
    req = torch.arange(b, device="cuda", dtype=torch.int64) // 6
    lens = torch.full(
        (b,), visible_full // ratio, dtype=torch.int64, device="cuda"
    )
    return (
        q,
        weights,
        mapping,
        req,
        lens,
        table,
        page,
        ratio,
        full_capacity // ratio,
    )


def mhc_inputs(m, k=20480, seed=71):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), device="cuda", generator=g).bfloat16()
    fn = (torch.randn((24, k), device="cuda", generator=g) * 0.01).float()
    scale = torch.tensor([0.7, 0.8, 0.9], device="cuda")
    base = torch.randn((24,), device="cuda", generator=g) * 0.1
    return x, fn, scale, base


def assert_exact(a, b):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a.shape == b.shape and a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), "visible storage bytes differ"


def changed_graph(fn, args, refresh, reference, repeats=3):
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(*args)
    for seed in range(repeats):
        refresh(args, seed)
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(out, reference(*args))
