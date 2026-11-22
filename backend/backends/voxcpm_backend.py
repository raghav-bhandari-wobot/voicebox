"""
VoxCPM backend implementation.

Wraps OpenBMB's tokenizer-free VoxCPM TTS through the ``voxcpm`` package, with
two model sizes behind one engine:

  - "2B"   -> openbmb/VoxCPM2       30 languages, 48 kHz output, cloning from a
                                    reference clip (transcript optional), and
                                    style control via a parenthesised prefix,
                                    e.g. "(cheerful, slightly faster)Hello".
  - "0.5B" -> openbmb/VoxCPM-0.5B   Chinese + English, 16 kHz output, cloning by
                                    prompt continuation (needs the transcript).

Both are Apache-2.0. (The 0.5B model card additionally asks that it not be used
in production/commercial settings.)

Integration notes:
  - ``voxcpm`` is installed ``--no-deps``: its requirements pull in gradio,
    funasr, modelscope and datasets for the web demo, denoiser and fine-tuning,
    none of which the inference path below touches.
  - The ZipEnhancer denoiser (a separate ModelScope model) is never loaded
    (``load_denoiser=False``), and ``torch.compile`` is off (``optimize=False``)
    to skip the slow warm-up compile.
  - The 0.5B model reads prompt audio with ``torchaudio.load``, which needs
    ``torchcodec`` on torchaudio >= 2.9. We swap in a soundfile reader for that
    module only (see ``_patch_torchaudio_load``).
  - ``generate`` has no seed argument, so the torch RNG is seeded directly.
  - The model is built directly on the GPU in bf16 to avoid upstream's ~9 GB
    float32 CPU staging copy (see ``_load_model_sync``).
  - VRAM (RTX 4060, measured): VoxCPM2 ~5.6 GB peak, 0.5B ~2 GB.
"""

import asyncio
import contextlib
import gc
import logging
from typing import Optional, Tuple

import numpy as np

from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    manual_seed,
    model_load_progress,
)

logger = logging.getLogger(__name__)

VOXCPM_MODEL_REPOS = {
    "2B": "openbmb/VoxCPM2",
    "0.5B": "openbmb/VoxCPM-0.5B",
}
# Must match the ModelConfig.model_name entries so download progress lines up.
VOXCPM_MODEL_NAMES = {
    "2B": "voxcpm-2b",
    "0.5B": "voxcpm-0.5b",
}
VOXCPM_DEFAULT_SIZE = "2B"

VOXCPM2_LANGUAGES = [
    "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "he", "ar",
    "da", "el", "fi", "hi", "ms", "nl", "no", "pl", "sv", "sw", "tr",
]
VOXCPM_05B_LANGUAGES = ["en", "zh"]

# Release defaults from the model cards.
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 10

# Combined multi-sample prompts are written at this rate before being handed
# back as a reference path; VoxCPM resamples prompt audio itself.
COMBINE_SAMPLE_RATE = 24000


@contextlib.contextmanager
def _default_dtype(dtype):
    """Temporarily set torch's default floating dtype."""
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _patch_torchaudio_load() -> None:
    """Point voxcpm.model.voxcpm's ``torchaudio.load`` at a soundfile reader.

    Scoped to that one module through a proxy, so the global ``torchaudio``
    stays untouched for every other engine.
    """
    import sys

    module = sys.modules.get("voxcpm.model.voxcpm")
    if module is None or getattr(module.torchaudio, "_voicebox_proxy", False):
        return

    import soundfile as sf
    import torch
    import torchaudio as real_torchaudio

    def _sf_load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    class _TorchaudioProxy:
        _voicebox_proxy = True
        load = staticmethod(_sf_load)

        def __getattr__(self, name):
            return getattr(real_torchaudio, name)

    module.torchaudio = _TorchaudioProxy()


class VoxCPMBackend:
    """VoxCPM2 / VoxCPM-0.5B backend for zero-shot voice cloning."""

    def __init__(self):
        self.model = None
        self.model_size = VOXCPM_DEFAULT_SIZE
        self._device: Optional[str] = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        return get_torch_device()

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        return VOXCPM_MODEL_REPOS.get(model_size, VOXCPM_MODEL_REPOS[VOXCPM_DEFAULT_SIZE])

    def _is_model_cached(self, model_size: str = VOXCPM_DEFAULT_SIZE) -> bool:
        # Main weights: model.safetensors (2B) / pytorch_model.bin (0.5B). Not
        # ".pth": audiovae.pth is small and lands first, so it would report a
        # half-finished download as cached.
        return is_model_cached(self._get_model_path(model_size))

    @property
    def sample_rate(self) -> int:
        return int(self.model.tts_model.sample_rate)

    async def load_model(self, model_size: Optional[str] = None) -> None:
        model_size = model_size if model_size in VOXCPM_MODEL_REPOS else VOXCPM_DEFAULT_SIZE
        if self.model is not None and self.model_size == model_size:
            return
        async with self._model_load_lock:
            if self.model is not None and self.model_size == model_size:
                return
            if self.model is not None:
                self.unload_model()
            await asyncio.to_thread(self._load_model_sync, model_size)

    def _load_model_sync(self, model_size: str) -> None:
        repo = self._get_model_path(model_size)
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(VOXCPM_MODEL_NAMES[model_size], is_cached):
            import torch
            from voxcpm import VoxCPM

            _patch_torchaudio_load()

            device = self.device
            logger.info("Loading %s on %s...", repo, device)
            # Upstream from_local() builds the model in float32 on the CPU, then
            # casts to bf16 and moves it: ~9 GB of RAM for VoxCPM2, which OOMs a
            # typical 8 GB Docker/WSL VM. Building it straight on the
            # accelerator in bf16 keeps peak RAM at roughly the checkpoint size
            # (~5 GB measured) and the weights still load from the file as
            # before. The audio VAE is cast back to fp32 by from_local itself.
            build_ctx = (
                (torch.device(device), _default_dtype(torch.bfloat16))
                if device in ("cuda", "xpu")
                else ()
            )
            with contextlib.ExitStack() as stack:
                for ctx in build_ctx:
                    stack.enter_context(ctx)
                model = VoxCPM.from_pretrained(
                    repo,
                    load_denoiser=False,
                    optimize=False,
                    device=device,
                )

        self.model = model
        self.model_size = model_size
        logger.info("%s loaded successfully (%d Hz)", repo, self.sample_rate)

    def unload_model(self) -> None:
        if self.model is not None:
            device = self.device
            del self.model
            self.model = None
            gc.collect()
            empty_device_cache(device)
            logger.info("VoxCPM unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """Deferred-path prompt: VoxCPM builds its prompt cache from the file."""
        return {"ref_audio": str(audio_path), "ref_text": reference_text or ""}, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=COMBINE_SAMPLE_RATE
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech.

        ``language`` is unused: VoxCPM detects it from the text.
        ``instruct`` (VoxCPM2 only) is sent as the model's parenthesised style
        prefix, e.g. "cheerful, slightly faster".
        """
        await self.load_model(self.model_size)

        is_v2 = self.model_size == "2B"
        ref_audio = (voice_prompt or {}).get("ref_audio")
        ref_text = ((voice_prompt or {}).get("ref_text") or "").strip()

        kwargs = {
            "cfg_value": CFG_VALUE,
            "inference_timesteps": INFERENCE_TIMESTEPS,
            "normalize": False,
            "denoise": False,
            "retry_badcase": True,
        }
        if ref_audio:
            if is_v2:
                # Reference clip alone clones the timbre; adding the clip +
                # transcript as a continuation prompt is the model card's
                # highest-similarity mode.
                kwargs["reference_wav_path"] = ref_audio
                if ref_text:
                    kwargs["prompt_wav_path"] = ref_audio
                    kwargs["prompt_text"] = ref_text
            else:
                if not ref_text:
                    raise ValueError(
                        "VoxCPM 0.5B clones by prompt continuation and needs the "
                        "reference text of the voice sample."
                    )
                kwargs["prompt_wav_path"] = ref_audio
                kwargs["prompt_text"] = ref_text

        target_text = text
        if is_v2 and instruct and instruct.strip():
            style = instruct.strip().strip("()")
            target_text = f"({style}){text}"

        def _generate_sync() -> Tuple[np.ndarray, int]:
            import torch

            if seed is not None:
                manual_seed(seed, self.device)
            with torch.inference_mode():
                wav = self.model.generate(text=target_text, **kwargs)
            return np.asarray(wav, dtype=np.float32).reshape(-1), self.sample_rate

        return await asyncio.to_thread(_generate_sync)
