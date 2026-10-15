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

"""Adapter lifecycle and scheduler device ordering, with no GPU execution."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def fresh():
    path = Path(__file__).parents[1] / "src/flaggems_sglang_plugin/__init__.py"
    spec = importlib.util.spec_from_file_location(
        "isolated_dsv41_plugin", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disabled_entry_does_not_import_gpu_or_sglang(monkeypatch):
    monkeypatch.delenv("FLAGGEMS_SGLANG_DSV41", raising=False)
    p = fresh()
    before = set(sys.modules)
    p.load_plugin()
    assert set(sys.modules) == before
    assert p._configure_original is None


def test_device_is_set_before_operator_install(monkeypatch):
    p = fresh()
    events = []
    scheduler = types.ModuleType("sglang.srt.managers.scheduler")
    scheduler.configure_scheduler_process = (
        lambda *a, **k: events.append("configure") or 3
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        set_device=lambda device: events.append(("device", device))
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.managers.scheduler", scheduler
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("FLAGGEMS_SGLANG_DSV41", "1")
    monkeypatch.setattr(p, "install", lambda: events.append("install"))
    p.load_plugin()
    assert scheduler.configure_scheduler_process(None, 5) == 3
    assert events == ["configure", ("device", 5), "install"]
    wrapped = scheduler.configure_scheduler_process
    p.load_plugin()
    assert scheduler.configure_scheduler_process is wrapped


def test_uninstall_preserves_subsequent_owner(monkeypatch):
    p = fresh()
    module = types.SimpleNamespace()
    old, new, other = (lambda: 1), (lambda: 2), (lambda: 3)
    module.a = new
    module.b = other
    handle = p.Installation(
        [(module, "a", old, new), (module, "b", old, new)], {}
    )
    p._handle = handle
    handle.uninstall()
    assert module.a is old and module.b is other and p._handle is None
    handle.uninstall()


def test_unknown_source_fails_before_any_rebind(monkeypatch, tmp_path):
    p = fresh()
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        get_device_capability=lambda: (9, 0)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    source = tmp_path / "op.py"
    source.write_text("unreviewed source")
    sentinel = lambda: None
    module = types.SimpleNamespace(__file__=str(source), sentinel=sentinel)
    monkeypatch.setattr(p.importlib, "import_module", lambda _: module)
    with pytest.raises(RuntimeError, match="unreviewed SGLang"):
        p.install()
    assert module.sentinel is sentinel and p._handle is None
