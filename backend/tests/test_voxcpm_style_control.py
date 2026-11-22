"""
VoxCPM2 style prompts must not be spoken aloud.

VoxCPM2 takes a style as a "(style)" prefix on the text, but only in
reference-only cloning. In continuation mode (prompt audio + its transcript)
the model treats the prefix as words to say, so a "cheerful" take started with
the voice saying "cheerful". Upstream's CLI rejects --control with
--prompt-text for the same reason. With a style, the backend must clone from
the reference clip alone.

Usage:
    python -m pytest backend/tests/test_voxcpm_style_control.py -v
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.backends.voxcpm_backend import VoxCPMBackend


class FakeVoxCPM:
    def __init__(self):
        self.calls = []
        self.tts_model = types.SimpleNamespace(sample_rate=48000)

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return np.zeros(480, dtype=np.float32)


def _backend(model_size: str) -> tuple[VoxCPMBackend, FakeVoxCPM]:
    backend = VoxCPMBackend()
    fake = FakeVoxCPM()
    backend.model = fake
    backend.model_size = model_size
    backend._device = "cpu"
    return backend, fake


PROMPT = {"ref_audio": "/tmp/ref.wav", "ref_text": "Hello, this is my voice."}


async def test_v2_without_style_uses_continuation_prompt():
    backend, fake = _backend("2B")
    await backend.generate("Good morning.", PROMPT)
    call = fake.calls[-1]
    assert call["text"] == "Good morning."
    assert call["reference_wav_path"] == "/tmp/ref.wav"
    assert call["prompt_wav_path"] == "/tmp/ref.wav"
    assert call["prompt_text"] == "Hello, this is my voice."


@pytest.mark.parametrize("instruct", ["warm, calm", "(warm, calm)", "  warm, calm  "])
async def test_v2_style_uses_reference_only_mode(instruct):
    backend, fake = _backend("2B")
    await backend.generate("Good morning.", PROMPT, instruct=instruct)
    call = fake.calls[-1]
    assert call["text"] == "(warm, calm)Good morning."
    assert call["reference_wav_path"] == "/tmp/ref.wav"
    assert "prompt_wav_path" not in call, "style + continuation prompt makes VoxCPM2 speak the style"
    assert "prompt_text" not in call


async def test_v2_blank_style_is_ignored():
    backend, fake = _backend("2B")
    await backend.generate("Good morning.", PROMPT, instruct="  ()  ")
    call = fake.calls[-1]
    assert call["text"] == "Good morning."
    assert call["prompt_text"] == "Hello, this is my voice."


async def test_05b_ignores_style_and_keeps_continuation():
    backend, fake = _backend("0.5B")
    await backend.generate("Good morning.", PROMPT, instruct="warm")
    call = fake.calls[-1]
    assert call["text"] == "Good morning."
    assert call["prompt_text"] == "Hello, this is my voice."
    assert "reference_wav_path" not in call
