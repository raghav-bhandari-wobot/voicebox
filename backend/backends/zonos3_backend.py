"""
ZONOS3-Mini backend implementation.

Wraps Zyphra/ZONOS3-Mini through its plain-transformers port,
``Zyphra/ZONOS3-Mini-hf`` (remote code). ~100M active parameters (800M total
MoE backbone), 44.1 kHz output, Apache-2.0 weights, MIT runtime code.

Why the -hf port and not Zyphra's native server: the native CUDA runtime
(``zonos3-sgl``) depends on a Hopper-tested kernel stack (flashinfer, tilelang,
sgl_kernel, sonic-moe) pinned to specific wheels. The -hf port only needs
torch + transformers >= 4.57 + soundfile, which Voicebox already ships, and runs
on consumer GPUs (verified on an RTX 4060 Laptop, ~2.6 GB peak VRAM in bf16).
It drops the native server's steering controls (emotion/pace sliders, accents)
and streaming decode.

Voices:
  - Preset: the 27 release voices bundled in the repo (voices.json +
    voices.safetensors). The two Telugu voices are omitted because Telugu is
    not in Voicebox's language enum.
  - Cloned: zero-shot from reference audio via the 192-d speaker embedding.
    No transcript needed. The embedding is cached per sample.

Optional ``zyphra-textnorm`` reads numbers, dates and currencies aloud the way
the model was trained; without it the model only collapses line breaks.
"""

import asyncio
import gc
import logging
import threading
from typing import Optional, Tuple

import numpy as np

from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    model_load_progress,
)
from ..utils.cache import cache_voice_prompt, get_cache_key, get_cached_voice_prompt

logger = logging.getLogger(__name__)

ZONOS3_HF_REPO = "Zyphra/ZONOS3-Mini-hf"
# Pinned because the repo ships remote code (trust_remote_code=True): an
# unpinned load would execute whatever the repo's main branch holds at the time.
# Bump deliberately after reviewing modeling_zonos3.py.
ZONOS3_HF_REVISION = "e39d30df560125d9c434acd5cb0e83e4c2d64e0d"
ZONOS3_SAMPLE_RATE = 44100

ZONOS3_DEFAULT_VOICE = "Savannah"

# (voice_id, display_name, gender, voicebox_language). voice_id is the name the
# model's generate(voice=...) expects.
ZONOS3_VOICES = [
    ("Savannah", "Savannah (American)", "female", "en"),
    ("Wyatt", "Wyatt (American)", "male", "en"),
    ("Imogen", "Imogen (British)", "female", "en"),
    ("Rupert", "Rupert (British)", "male", "en"),
    ("Cillian", "Cillian (Irish)", "male", "en"),
    ("Jett", "Jett (Energetic)", "male", "en"),
    ("Thaddeus", "Thaddeus (Announcer)", "male", "en"),
    ("Sakura", "Sakura (Anime)", "female", "en"),
    ("Morgana", "Morgana (Witch)", "female", "en"),
    ("Willow", "Willow (ASMR)", "female", "en"),
    ("Gaspard", "Gaspard", "male", "fr"),
    ("Liesel", "Liesel", "female", "de"),
    ("Wolfgang", "Wolfgang", "male", "de"),
    ("Ananya", "Ananya", "female", "hi"),
    ("Arjun", "Arjun", "male", "hi"),
    ("Ginevra", "Ginevra", "female", "it"),
    ("Lorenzo", "Lorenzo", "male", "it"),
    ("Seoyeon", "Seoyeon", "female", "ko"),
    ("Minjun", "Minjun", "male", "ko"),
    ("Meilin", "Meilin (Energetic)", "female", "zh"),
    ("Haoran", "Haoran (Energetic)", "male", "zh"),
    ("Leonor", "Leonor", "female", "pt"),
    ("Thiago", "Thiago", "male", "pt"),
    ("Ximena", "Ximena", "female", "es"),
    ("Rodrigo", "Rodrigo", "male", "es"),
]

# Voicebox language code -> ZONOS3 language code (config.generation.languages).
# Codes not listed pass through unchanged (de, es, it, ja, ko, ru, ...).
# Swahili is the one Voicebox language ZONOS3 does not cover.
LANGUAGE_CODE_MAP = {
    "en": "en_us",
    "zh": "cmn",
    "fr": "fr_fr",
    "pt": "pt_br",
}

ZONOS3_LANGUAGES = [
    "en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "he", "ar",
    "da", "el", "fi", "hi", "ms", "nl", "no", "pl", "sv", "tr",
]

_SPEAKER_PROMPT_FORMAT_VERSION = 1


class Zonos3Backend:
    """ZONOS3-Mini backend: preset voices + zero-shot cloning."""

    def __init__(self):
        self.model = None
        self.model_size = "default"  # single checkpoint
        self._device: Optional[str] = None
        # threading.Lock (not asyncio): the load runs in a to_thread worker,
        # and concurrent from_pretrained() calls corrupt each other's weights.
        self._load_lock = threading.Lock()

    def _get_device(self) -> str:
        return get_torch_device(allow_xpu=True)

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        return ZONOS3_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(ZONOS3_HF_REPO)

    async def load_model(self, model_size: str = "default") -> None:
        if self.model is not None:
            return
        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        with self._load_lock:
            if self.model is not None:
                return

            is_cached = self._is_model_cached()
            with model_load_progress("zonos3-mini", is_cached):
                import torch
                from transformers import AutoModel

                device = self.device
                # bf16 LM on accelerators (the remote code keeps the codec
                # decoder and speaker encoder in fp32 itself); fp32 on CPU,
                # which is also the model card's "reference numerics" mode.
                dtype = torch.bfloat16 if device in ("cuda", "xpu") else torch.float32
                logger.info("Loading ZONOS3-Mini on %s (%s)...", device, dtype)

                model = AutoModel.from_pretrained(
                    ZONOS3_HF_REPO,
                    revision=ZONOS3_HF_REVISION,
                    trust_remote_code=True,
                    dtype=dtype,
                )
                model = model.to(device).eval()

            self.model = model
            logger.info("ZONOS3-Mini loaded successfully")

    def unload_model(self) -> None:
        if self.model is not None:
            device = self.device
            del self.model
            self.model = None
            gc.collect()
            empty_device_cache(device)
            logger.info("ZONOS3-Mini unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """Embed a reference clip into ZONOS3's 192-d speaker embedding.

        The transcript is not used: ZONOS3 clones from the embedding alone.
        Cached as a plain dict of CPU tensors so the weights_only cache loader
        accepts it.
        """
        cache_key = ("zonos3_" + get_cache_key(audio_path, "")) if use_cache else None
        if cache_key:
            cached = get_cached_voice_prompt(cache_key)
            if (
                isinstance(cached, dict)
                and "zonos3_speaker_embedding" in cached
                and cached.get("format_version") == _SPEAKER_PROMPT_FORMAT_VERSION
            ):
                return cached, True

        await self.load_model()

        def _embed_sync() -> dict:
            import soundfile as sf

            wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            # soundfile gives [samples, channels]; the model wants [channels, samples].
            embedding = self.model.embed_speaker(wav.T, sr)
            return {
                "format_version": _SPEAKER_PROMPT_FORMAT_VERSION,
                "zonos3_speaker_embedding": embedding.detach().float().cpu(),
            }

        encoded = await asyncio.to_thread(_embed_sync)
        if cache_key:
            cache_voice_prompt(cache_key, encoded)
        return encoded, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=ZONOS3_SAMPLE_RATE
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech. ``instruct`` is not supported by the -hf port (ignored)."""
        await self.load_model()

        zonos_language = LANGUAGE_CODE_MAP.get(language, language)
        if voice_prompt.get("voice_type") == "preset":
            voice_kwargs = {"voice": voice_prompt.get("preset_voice_id") or ZONOS3_DEFAULT_VOICE}
        elif "zonos3_speaker_embedding" in (voice_prompt or {}):
            version = voice_prompt.get("format_version")
            if version != _SPEAKER_PROMPT_FORMAT_VERSION:
                raise ValueError(f"Unsupported ZONOS3 voice prompt format version: {version}")
            voice_kwargs = {"speaker_embedding": voice_prompt["zonos3_speaker_embedding"]}
        else:
            voice_kwargs = {"voice": ZONOS3_DEFAULT_VOICE}

        def _generate_sync() -> Tuple[np.ndarray, int]:
            import torch

            with torch.inference_mode():
                out = self.model.generate(
                    text,
                    language=zonos_language,
                    seed=seed,
                    **voice_kwargs,
                )
            wav = out.waveform
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().float().cpu().numpy()
            return np.asarray(wav, dtype=np.float32).reshape(-1), int(out.sampling_rate)

        return await asyncio.to_thread(_generate_sync)
