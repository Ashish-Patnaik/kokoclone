"""
core/chunked_convert.py
-----------------------
VRAM-aware chunked voice conversion using the Kanade model.

On CUDA devices, the source waveform is split into overlapping chunks so that
peak activation memory stays within a configurable fraction of total VRAM
(default 50%).  On CPU the waveform is still chunked to respect the model's
RoPE sequence-length limit.

RoPE ceiling (why chunks must be small)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Kanade ``mel_decoder`` Transformer processes mel-spectrogram frames of the
source chunk.  Its RoPE positional embeddings are precomputed for
``_ROPE_MAX_FRAMES = 1024`` positions.  The mel frame count for a window of
``W`` samples is ``W // hop_length + 1``.  Keeping that ≤ 1024 requires:

    W  ≤  (1024 − 1) × hop_length  =  1023 × 256  =  261,888 samples  ≈ 10.9 s

Each chunk window includes a 0.5 s overlap on both sides for boundary
smoothing, so the *chunk* itself must be:

    chunk  ≤  261,888 − 2 × (0.5 s × sample_rate)  ≈  9.9 s

A 10 % safety margin is applied, giving ``_ROPE_SAFE_CHUNK_FACTOR ≈ 8.9 s``
worth of source audio per chunk.

Overlap / boundary handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each chunk includes a short overlap window on both sides.  After the
voice-conversion forward pass, the overlap frames are trimmed from the mel
output before the pieces are concatenated.  The final assembled mel is vocoded
in a single pass.
"""

from __future__ import annotations

import time
import torch
from kanade_tokenizer import vocode


# Empirical constant: ~10 seconds of audio fit in 1 GB of VRAM budget for the
# Kanade-12.5hz model.  Adjust downward if you observe OOM errors.
_SECONDS_PER_GB: float = 10.0

# Overlap window on each side of a chunk (seconds).
_OVERLAP_SECONDS: float = 0.5

# --------------------------------------------------------------------------
# RoPE safety ceiling — derived from the mel_decoder Transformer
# --------------------------------------------------------------------------
# mel_decoder seqlen = audio_length // hop_length + 1 (center-padding mel).
# Its RoPE freqs_cis is precomputed for _ROPE_MAX_FRAMES positions.
# hop_length comes directly from KanadeModelConfig (hop_length = 256).
_ROPE_MAX_FRAMES: int = 1024   # precomputed RoPE window (freqs_cis.shape[0])
_MEL_HOP_LENGTH: int = 256     # KanadeModelConfig.hop_length
_ROPE_SAFETY_MARGIN: float = 0.75

# Output mel frame rate — kept for reference only; NOT used for overlap trimming.
# Mel frames used internally are at sample_rate / hop_length (93.75 fps), not 12.5 fps.
_MEL_FPS: float = 12.5


def chunked_voice_conversion(
    kanade,
    vocoder_model,
    source_wav: torch.Tensor,
    ref_wav: torch.Tensor,
    sample_rate: int,
    vram_fraction: float = 0.9,
) -> torch.Tensor:
    """Convert *source_wav* to the reference voice in VRAM-safe chunks.

    Parameters
    ----------
    kanade:
        A loaded ``KanadeModel`` instance (already on the target device).
    vocoder_model:
        The vocoder loaded via ``load_vocoder`` (already on the target device).
    source_wav:
        Source waveform tensor of shape ``[T]`` or ``[1, T]``, on the same
        device as *kanade*.
    ref_wav:
        Reference waveform tensor of shape ``[T]`` or ``[1, T]``, on the same
        device as *kanade*.
    sample_rate:
        Audio sample rate in Hz (taken from ``kanade.config.sample_rate``).
    vram_fraction:
        Fraction of total VRAM to target per chunk.  Default ``0.5`` → 50 %.

    Returns
    -------
    torch.Tensor
        Converted waveform as a 1-D CPU float32 tensor.
    """
    device: torch.device = source_wav.device
    n_samples: int = source_wav.shape[-1]
    _start = time.perf_counter()

    # ── 1. Determine chunk size ──────────────────────────────────────────────
    # The mel_decoder RoPE ceiling limits the total window (chunk + overlaps).
    # Max window in samples: (ROPE_MAX_FRAMES - 1) * MEL_HOP_LENGTH
    # Subtract both overlap sides, then apply a safety margin.
    overlap_samples = int(_OVERLAP_SECONDS * sample_rate)
    rope_max_window = (_ROPE_MAX_FRAMES - 1) * _MEL_HOP_LENGTH  # 261,888 samples ≈ 10.9 s
    rope_safe_chunk = int((rope_max_window - 2 * overlap_samples) * _ROPE_SAFETY_MARGIN)
    rope_safe_seconds = rope_safe_chunk / sample_rate

    if device.type == "cuda":
        total_vram_bytes = torch.cuda.get_device_properties(device).total_memory
        budget_bytes = total_vram_bytes * vram_fraction
        budget_gb = budget_bytes / (1024 ** 3)

        vram_chunk_samples = int(max(5.0, budget_gb * _SECONDS_PER_GB) * sample_rate)

        # Take the smaller of VRAM-based and RoPE-safe limits.
        chunk_samples = min(vram_chunk_samples, rope_safe_chunk)
        chunk_seconds = chunk_samples / sample_rate

        print(
            f"[chunked_convert] VRAM budget: {budget_gb:.2f} GB "
            f"({vram_fraction*100:.0f}% of {total_vram_bytes / (1024**3):.2f} GB) "
            f"→ chunk size: {chunk_seconds:.1f}s / {chunk_samples:,} samples "
            f"(RoPE ceiling: {rope_safe_seconds:.1f}s)"
        )
    else:
        # CPU: no VRAM limit, but still respect the RoPE ceiling for quality.
        chunk_samples = rope_safe_chunk

    # ── 2. Short-circuit when the whole file fits in one chunk ───────────────
    if n_samples <= chunk_samples:
        with torch.inference_mode():
            mel = kanade.voice_conversion(
                source_waveform=source_wav, reference_waveform=ref_wav
            )
            wav = vocode(vocoder_model, mel.unsqueeze(0))
        elapsed = time.perf_counter() - _start
        print(f"[chunked_convert] Completed in {elapsed:.1f}s")
        return wav.squeeze().cpu()

    # ── 3. Chunked processing with overlap ──────────────────────────────────
    # Mel frames corresponding to the overlap window.
    # The mel output is at sample_rate / hop_length = 93.75 fps, NOT _MEL_FPS.
    overlap_frames = overlap_samples // _MEL_HOP_LENGTH  # 12000 // 256 = 46

    mel_parts: list[torch.Tensor] = []
    pos = 0

    while pos < n_samples:
        # Extend the window on both sides by overlap_samples so the model has
        # context at each boundary.
        win_start = max(0, pos - overlap_samples)
        win_end   = min(n_samples, pos + chunk_samples + overlap_samples)

        chunk = source_wav[..., win_start:win_end]

        with torch.inference_mode():
            mel_chunk: torch.Tensor = kanade.voice_conversion(
                source_waveform=chunk, reference_waveform=ref_wav
            )

        # Move to CPU immediately so the GPU buffer is freed before the next chunk.
        mel_chunk = mel_chunk.cpu()

        # Trim overlap frames that were only there for context.
        left_trim  = 0 if pos == 0 else overlap_frames
        right_trim = mel_chunk.shape[-1] if win_end >= n_samples else mel_chunk.shape[-1] - overlap_frames

        mel_parts.append(mel_chunk[..., left_trim:right_trim])

        pos += chunk_samples

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── 4. Assemble full mel and vocode in one pass ──────────────────────────
    full_mel = torch.cat(mel_parts, dim=-1).to(device)

    with torch.inference_mode():
        wav = vocode(vocoder_model, full_mel.unsqueeze(0))

    elapsed = time.perf_counter() - _start
    print(f"[chunked_convert] Completed in {elapsed:.1f}s")
    return wav.squeeze().cpu()
