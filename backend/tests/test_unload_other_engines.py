"""
Tests for "unload other TTS engines before loading one".

Every TTS backend used to stay resident once loaded, so switching engines on
an 8 GB GPU stacked their weights (VoxCPM2 ~5.6 GB + ZONOS3 Mini ~2.6 GB) and
OOMed. With the setting on, ``load_engine_model`` and the Models-page load
path unload every other loaded TTS engine first, and wait for in-flight
generations on other engines instead of unloading them underneath.

These use fake backends injected into the backend cache, so no model weights
are touched.

Usage:
    python -m pytest backend/tests/test_unload_other_engines.py -v
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import backends
from backend.backends import (
    get_model_config,
    get_model_load_func,
    load_engine_model,
    tts_engine_in_use,
)


class FakeBackend:
    """Stand-in TTS backend that records load/unload calls."""

    def __init__(self, name: str, events: list, model_size: str | None = None):
        self.name = name
        self.events = events
        self.loaded = False
        self.model_size = model_size

    def is_loaded(self) -> bool:
        return self.loaded

    async def load_model(self, model_size: str | None = None) -> None:
        await asyncio.sleep(0)
        self.loaded = True
        if model_size is not None:
            self.model_size = model_size
        self.events.append(f"load:{self.name}")

    load_model_async = load_model

    def unload_model(self) -> None:
        self.loaded = False
        self.events.append(f"unload:{self.name}")


@pytest.fixture
def fake_engines(monkeypatch):
    """Install fake kokoro / zonos3 / omnivoice / voxcpm backends."""
    events: list[str] = []
    fakes = {
        "kokoro": FakeBackend("kokoro", events),
        "zonos3": FakeBackend("zonos3", events),
        "omnivoice": FakeBackend("omnivoice", events),
        "voxcpm": FakeBackend("voxcpm", events, model_size="2B"),
    }
    monkeypatch.setattr(backends, "_tts_backends", dict(fakes))
    monkeypatch.setattr(backends, "_engine_use_counts", {})
    monkeypatch.setattr(backends, "_unload_other_engines", True)
    from backend.utils import cache

    monkeypatch.setattr(cache, "clear_voice_prompt_memory_cache", lambda: None)
    return fakes, events


def _loaded(fakes) -> set[str]:
    return {name for name, b in fakes.items() if b.is_loaded()}


async def test_load_unloads_other_engines_when_enabled(fake_engines):
    fakes, events = fake_engines

    await load_engine_model("voxcpm", "2B")
    await load_engine_model("zonos3")
    assert _loaded(fakes) == {"zonos3"}

    await load_engine_model("omnivoice")
    await load_engine_model("kokoro")
    assert _loaded(fakes) == {"kokoro"}
    assert events == [
        "load:voxcpm",
        "unload:voxcpm",
        "load:zonos3",
        "unload:zonos3",
        "load:omnivoice",
        "unload:omnivoice",
        "load:kokoro",
    ]


async def test_load_keeps_other_engines_when_disabled(fake_engines, monkeypatch):
    fakes, events = fake_engines
    monkeypatch.setattr(backends, "_unload_other_engines", False)

    await load_engine_model("voxcpm", "2B")
    await load_engine_model("zonos3")
    assert _loaded(fakes) == {"voxcpm", "zonos3"}
    assert not any(e.startswith("unload:") for e in events)


async def test_reloading_same_engine_does_not_unload_it(fake_engines):
    fakes, events = fake_engines

    await load_engine_model("kokoro")
    await load_engine_model("kokoro")
    assert _loaded(fakes) == {"kokoro"}
    assert "unload:kokoro" not in events


async def test_models_page_load_path_unloads_other_engines(fake_engines):
    """``get_model_load_func`` (Models page load + download) goes through the same switch."""
    fakes, events = fake_engines

    await load_engine_model("voxcpm", "2B")
    result = get_model_load_func(get_model_config("kokoro"))()
    assert asyncio.iscoroutine(result)
    await result
    assert _loaded(fakes) == {"kokoro"}

    await get_model_load_func(get_model_config("voxcpm-0.5b"))()
    assert _loaded(fakes) == {"voxcpm"}
    assert fakes["voxcpm"].model_size == "0.5B"


async def test_does_not_instantiate_unloaded_engines(fake_engines, monkeypatch):
    """Unloading others must only touch backends that already exist."""
    fakes, _events = fake_engines
    created: list[str] = []
    real_factory = backends.get_tts_backend_for_engine

    def tracking_factory(engine):
        if engine not in backends._tts_backends:
            created.append(engine)
        return real_factory(engine)

    monkeypatch.setattr(backends, "get_tts_backend_for_engine", tracking_factory)
    await load_engine_model("kokoro")
    assert created == []


async def test_switch_waits_for_generation_on_other_engine(fake_engines):
    """A load for engine B must not unload engine A while A is mid-generation."""
    fakes, events = fake_engines
    generation_may_finish = asyncio.Event()
    generation_started = asyncio.Event()

    async def generate_with_voxcpm():
        async with tts_engine_in_use("voxcpm"):
            await load_engine_model("voxcpm", "2B")
            generation_started.set()
            await generation_may_finish.wait()
            # Still loaded at the end of "inference".
            assert fakes["voxcpm"].is_loaded()
            events.append("generated:voxcpm")

    gen_task = asyncio.create_task(generate_with_voxcpm())
    await generation_started.wait()

    switch_task = asyncio.create_task(load_engine_model("zonos3"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert not switch_task.done(), "switch must wait for the in-flight generation"
    assert fakes["voxcpm"].is_loaded()
    assert not fakes["zonos3"].is_loaded()

    generation_may_finish.set()
    await asyncio.wait_for(asyncio.gather(gen_task, switch_task), timeout=2)

    assert _loaded(fakes) == {"zonos3"}
    assert events.index("generated:voxcpm") < events.index("unload:voxcpm")
    assert events.index("unload:voxcpm") < events.index("load:zonos3")
    assert backends.get_tts_engines_in_use() == {}


async def test_same_engine_requests_run_concurrently(fake_engines):
    """Two generations on the same engine must not block each other."""
    fakes, _events = fake_engines
    both_inside = asyncio.Event()
    inside = 0

    async def generate():
        nonlocal inside
        async with tts_engine_in_use("kokoro"):
            await load_engine_model("kokoro")
            inside += 1
            if inside == 2:
                both_inside.set()
            await asyncio.wait_for(both_inside.wait(), timeout=2)

    await asyncio.wait_for(asyncio.gather(generate(), generate()), timeout=2)
    assert _loaded(fakes) == {"kokoro"}


async def test_nested_hold_is_reentrant(fake_engines):
    """A generation that holds its engine can call load_engine_model for it."""
    fakes, _events = fake_engines

    async with tts_engine_in_use("omnivoice"):
        await asyncio.wait_for(load_engine_model("omnivoice"), timeout=2)
        assert backends.get_tts_engines_in_use() == {"omnivoice": 1}
    assert backends.get_tts_engines_in_use() == {}
    assert _loaded(fakes) == {"omnivoice"}


async def test_failed_load_releases_hold(fake_engines):
    """A load that raises must not leave the engine marked in use."""
    fakes, _events = fake_engines

    async def boom(model_size=None):
        raise RuntimeError("CUDA out of memory")

    fakes["zonos3"].load_model = boom
    with pytest.raises(RuntimeError):
        await load_engine_model("zonos3")
    assert backends.get_tts_engines_in_use() == {}

    # Another engine can still load afterwards.
    await asyncio.wait_for(load_engine_model("kokoro"), timeout=2)
    assert _loaded(fakes) == {"kokoro"}


async def test_run_generation_holds_engine(fake_engines, monkeypatch):
    """The queued generation path holds its engine from load through inference."""
    from backend.services import generation as generation_svc
    from backend.services import history, profiles
    from backend.utils import chunked_tts

    fakes, events = fake_engines
    seen_in_use: list[dict] = []

    async def fake_status(*args, **kwargs):
        return None

    async def fake_voice_prompt(*args, **kwargs):
        return {}

    async def fake_generate_chunked(tts_model, text, voice_prompt, **kwargs):
        seen_in_use.append(backends.get_tts_engines_in_use())
        raise RuntimeError("stop after inference")

    class FakeTaskManager:
        def complete_generation(self, generation_id):
            pass

    class FakeDB:
        def close(self):
            pass

    monkeypatch.setattr(history, "update_generation_status", fake_status)
    monkeypatch.setattr(profiles, "create_voice_prompt_for_profile", fake_voice_prompt)
    monkeypatch.setattr(chunked_tts, "generate_chunked", fake_generate_chunked)
    monkeypatch.setattr(generation_svc, "get_task_manager", lambda: FakeTaskManager())
    monkeypatch.setattr(generation_svc, "get_db", lambda: iter([FakeDB()]))
    monkeypatch.setattr(generation_svc, "release_generation_memory", lambda m: None)
    monkeypatch.setattr(generation_svc, "_notify_speak_end", lambda *a, **k: None)

    await load_engine_model("voxcpm", "2B")
    await generation_svc.run_generation(
        generation_id="g1",
        profile_id="p1",
        text="hi",
        language="en",
        engine="kokoro",
        model_size="default",
        seed=None,
        mode="generate",
    )
    assert seen_in_use == [{"kokoro": 1}]
    assert backends.get_tts_engines_in_use() == {}
    assert _loaded(fakes) == {"kokoro"}


def test_default_is_on_for_small_cuda_gpus(monkeypatch):
    import types

    def fake_torch(available: bool, total_gb: float):
        props = types.SimpleNamespace(total_memory=int(total_gb * 1024**3))
        cuda = types.SimpleNamespace(
            is_available=lambda: available,
            get_device_properties=lambda idx: props,
        )
        return types.SimpleNamespace(cuda=cuda)

    cases = [
        (True, 8, True),  # RTX 4060 Laptop
        (True, 11.6, True),  # nominal 12 GB card
        (True, 16, False),
        (True, 24, False),
        (False, 0, False),  # CPU / MLX
    ]
    for available, total_gb, expected in cases:
        monkeypatch.setitem(sys.modules, "torch", fake_torch(available, total_gb))
        monkeypatch.setattr(backends, "_default_unload_other_engines", None)
        monkeypatch.setattr(backends, "_unload_other_engines", None)
        assert backends.default_unload_other_engines() is expected, (available, total_gb)
        assert backends.should_unload_other_engines() is expected

    # An explicit preference wins over the auto default.
    monkeypatch.setattr(backends, "_unload_other_engines", False)
    monkeypatch.setattr(backends, "_default_unload_other_engines", True)
    assert backends.should_unload_other_engines() is False


def test_setting_persists_and_syncs_backend_state(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend import models
    from backend.database.models import Base
    from backend.services import settings as settings_service

    monkeypatch.setattr(backends, "_default_unload_other_engines", True)
    monkeypatch.setattr(backends, "_unload_other_engines", None)

    engine = create_engine(f"sqlite:///{tmp_path / 'settings.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        row = settings_service.get_generation_settings(db)
        assert row.unload_other_engines is None
        # NULL resolves to the auto default in the API response.
        assert models.GenerationSettingsResponse.model_validate(row).unload_other_engines is True

        row = settings_service.update_generation_settings(db, {"unload_other_engines": False})
        assert backends.should_unload_other_engines() is False
        assert models.GenerationSettingsResponse.model_validate(row).unload_other_engines is False

        # Startup bootstrap re-seeds the runtime state from the DB.
        backends.set_unload_other_engines(None)
        settings_service.bootstrap_generation_backend_config(db)
        assert backends.should_unload_other_engines() is False

        # Explicit null goes back to auto.
        settings_service.update_generation_settings(db, {"unload_other_engines": None})
        assert backends.should_unload_other_engines() is True
    finally:
        db.close()
