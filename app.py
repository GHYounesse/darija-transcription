from dotenv import load_dotenv
load_dotenv()
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import logging
import time
import re
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
import torch
from peft import PeftModel
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
    GenerationConfig,
)
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import tempfile
from pyannote.audio import Pipeline as DiarizationPipeline

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("darija_transcribe")

app = FastAPI(
    title="Darija Transcription & Diarization API",
    description="Convert Darija audio to diarized transcript with speaker labels",
    version="2.2",
)

# ── Enums ──────────────────────────────────────────────────────────────────
class ExportFormat(str, Enum):
    json = "json"
    vtt  = "vtt"
    srt  = "srt"
    txt  = "txt"

# ── Model loading ──────────────────────────────────────────────────────────
BASE_MODEL = "openai/whisper-large-v3-turbo"
LORA_MODEL = "anaszil/whisper-large-v3-turbo-darija"

dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
device = 0 if torch.cuda.is_available() else "cpu"

logger.info(f"CUDA disponible : {torch.cuda.is_available()} — device : {device}, dtype : {dtype}")

_t0 = time.perf_counter()
logger.info(f"Chargement du modèle de base Whisper : {BASE_MODEL}")
base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype)

# ── Anti-hallucination generation config ──────────────────────────────────
# These settings are the primary fix for the "اه اه اه" / "ها ها ها" loops
# seen in the original output.
base.generation_config = GenerationConfig.from_pretrained(BASE_MODEL)
base.generation_config.no_repeat_ngram_size   = 5     # block 5-gram repeats
base.generation_config.repetition_penalty     = 1.2   # penalise token repetition
base.generation_config.condition_on_prev_tokens = False  # don't feed previous chunk context
base.generation_config.compression_ratio_threshold = 2.0 # discard suspiciously repetitive chunks
base.generation_config.logprob_threshold      = -1.0  # discard low-confidence chunks
base.generation_config.no_speech_threshold    = 0.6   # mark silence aggressively
logger.info(f"Modèle de base chargé en {time.perf_counter() - _t0:.1f}s")

_t0 = time.perf_counter()
logger.info(f"Chargement de l'adaptateur LoRA Darija : {LORA_MODEL}")
model = PeftModel.from_pretrained(base, LORA_MODEL)
logger.info(f"Adaptateur LoRA chargé en {time.perf_counter() - _t0:.1f}s")

processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="Arabic", task="transcribe")
logger.info("Processor Whisper chargé (langue=Arabic, task=transcribe)")

asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    stride_length_s=5,
    return_timestamps=True,
    device=device,
    generate_kwargs={
        "no_repeat_ngram_size":       5,
        "repetition_penalty":         1.2,
        "condition_on_prev_tokens":   False,
        "compression_ratio_threshold": 2.0,
        "logprob_threshold":          -1.0,
        "no_speech_threshold":        0.6,
    },
)
logger.info("Pipeline ASR prêt")

_t0 = time.perf_counter()
logger.info("Chargement du pipeline de diarisation pyannote/speaker-diarization-3.1")
diarizer = DiarizationPipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HF_TOKEN"),
)
logger.info(f"Pipeline de diarisation chargé en {time.perf_counter() - _t0:.1f}s")

logger.info("✓ Tous les modèles sont chargés — le serveur est prêt à recevoir des requêtes")

# ── Schemas ────────────────────────────────────────────────────────────────
class TranscribeRequest(BaseModel):
    audio_url: str = Field(..., description="URL of audio file (HTTP/HTTPS)")
    language: str = Field(default="ar", description="Language code (ar for Arabic)")
    num_speakers: int = Field(default=2, ge=1, le=10, description="Expected number of speakers")
    merge_same_speaker: bool = Field(default=True, description="Merge consecutive same-speaker segments")

    @validator("audio_url")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("audio_url must start with http:// or https://")
        return v

class Utterance(BaseModel):
    start: float = Field(..., description="Start timestamp in seconds")
    end: float   = Field(..., description="End timestamp in seconds")
    text: str    = Field(..., description="Transcribed text in Darija")
    speaker: str = Field(..., description="Speaker label (SPEAKER_00, SPEAKER_01, …)")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")

class TranscribeResponse(BaseModel):
    utterances: List[Utterance]
    total_duration: float
    num_speakers: int
    language: str
    processing_time_seconds: float

# ── Utility helpers ────────────────────────────────────────────────────────

# Regex compiled once — matches strings that are ≥ 80 % identical repeated
# short tokens (e.g. "اه اه اه اه" or "ها ها ها ها").
_REPEAT_RE = re.compile(r"^(\S{1,6}\s*){4,}$")

def _is_hallucinated(text: str) -> bool:
    """
    Return True when a chunk looks like a Whisper hallucination loop.
    Heuristic: if the text has ≥ 8 tokens and more than 70 % of them are the
    same token, the chunk is almost certainly a loop artefact.
    """
    tokens = text.strip().split()
    if len(tokens) < 8:
        return False
    most_common_count = max(tokens.count(t) for t in set(tokens))
    return most_common_count / len(tokens) > 0.70


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_audio_with_retry(url: str, timeout: int = 60) -> bytes:
    """Download audio from URL with retry logic."""
    logger.info(f"Downloading audio from {url}")
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise


def _get_audio_duration(path: str) -> float:
    """
    Return audio duration in seconds using ffprobe (reads container metadata only,
    no decoding). Falls back to soundfile, then 0.0 if both are unavailable.
    ffprobe is preferred because soundfile can misread duration from Google Drive
    downloads that lack a proper WAV header.
    """
    # Primary: ffprobe — accurate even for malformed/headerless files
    try:
        import subprocess, json as _json
        probe = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0:
            data = _json.loads(probe.stdout)
            duration = float(data["format"]["duration"])
            logger.info(f"Durée audio (ffprobe) : {duration:.2f}s")
            return duration
    except Exception as e:
        logger.warning(f"ffprobe duration failed: {e} — falling back to soundfile")

    # Fallback: soundfile
    try:
        import soundfile as sf
        info = sf.info(path)
        logger.info(f"Durée audio (soundfile) : {info.duration:.2f}s")
        return info.duration
    except Exception as e:
        logger.warning(f"soundfile duration failed: {e} — duration unknown")
        return 0.0


def assign_speakers_improved(chunks: list, audio_path: str, num_speakers: int) -> list:
    """
    Merge Whisper chunks with pyannote diarization using overlap-based matching.
    """
    logger.info(f"Lancement de la diarisation (num_speakers={num_speakers})")
    _t0 = time.perf_counter()

    try:
        diarization = diarizer(audio_path, num_speakers=num_speakers)
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        for i, chunk in enumerate(chunks):
            chunk["speaker"] = f"SPEAKER_{i % num_speakers:02d}"
        return chunks

    logger.info(f"Diarisation terminée en {time.perf_counter() - _t0:.1f}s")

    # Build a flat sorted list of (start, end, label) for fast iteration
    turns = [
        (turn.start, turn.end, label)
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]

    unmatched = 0
    for chunk in chunks:
        t0 = chunk["timestamp"][0] or 0.0
        t1 = chunk["timestamp"][1]
        if t1 is None or t1 <= t0:
            chunk["speaker"] = "SPEAKER_00"
            unmatched += 1
            continue

        speaker_overlaps: dict[str, float] = {}
        for turn_start, turn_end, label in turns:
            if turn_start > t1:
                break  # turns are sorted; no point continuing
            overlap = max(0.0, min(t1, turn_end) - max(t0, turn_start))
            if overlap > 0:
                speaker_overlaps[label] = speaker_overlaps.get(label, 0.0) + overlap

        if speaker_overlaps:
            chunk["speaker"] = max(speaker_overlaps, key=speaker_overlaps.get)
        else:
            chunk["speaker"] = "SPEAKER_00"
            unmatched += 1

    if unmatched:
        logger.warning(
            f"{unmatched}/{len(chunks)} segments had no speaker match "
            "(assigned default SPEAKER_00)"
        )

    logger.info(f"Speaker assignment completed in {time.perf_counter() - _t0:.1f}s")
    return chunks


def merge_consecutive_speakers(utterances: List[Utterance]) -> List[Utterance]:
    """Merge consecutive utterances from the same speaker."""
    if not utterances:
        return []

    merged = [utterances[0].copy()]
    for utt in utterances[1:]:
        if utt.speaker == merged[-1].speaker:
            merged[-1].end  = utt.end
            merged[-1].text = merged[-1].text.strip() + " " + utt.text.strip()
            if merged[-1].confidence is not None and utt.confidence is not None:
                merged[-1].confidence = (merged[-1].confidence + utt.confidence) / 2
        else:
            merged.append(utt.copy())

    return merged


# ── Export helpers ─────────────────────────────────────────────────────────

def _fmt_vtt(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def _fmt_srt(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def export_to_vtt(utterances: List[Utterance]) -> str:
    lines = ["WEBVTT", ""]
    for utt in utterances:
        lines += [f"{_fmt_vtt(utt.start)} --> {_fmt_vtt(utt.end)}", utt.speaker, utt.text, ""]
    return "\n".join(lines)

def export_to_srt(utterances: List[Utterance]) -> str:
    lines = []
    for i, utt in enumerate(utterances, 1):
        lines += [str(i), f"{_fmt_srt(utt.start)} --> {_fmt_srt(utt.end)}", utt.speaker, utt.text, ""]
    return "\n".join(lines)

def export_to_txt(utterances: List[Utterance]) -> str:
    return "\n".join(f"[{utt.speaker}] {utt.text}" for utt in utterances)

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    """
    Transcribe and diarize Darija audio from URL.
    Returns utterances with speaker labels, timestamps, and text.
    """
    request_start = time.perf_counter()
    logger.info(
        f"Nouvelle requête /transcribe — "
        f"audio_url={req.audio_url}, "
        f"num_speakers={req.num_speakers}, "
        f"merge_same_speaker={req.merge_same_speaker}"
    )

    tmp_path = None
    try:
        # ── Download ────────────────────────────────────────────────────────
        _t0 = time.perf_counter()
        try:
            audio_data = download_audio_with_retry(req.audio_url)
        except Exception as e:
            logger.error(f"Failed to download audio after retries: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to download audio: {e}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        audio_duration = _get_audio_duration(tmp_path)
        logger.info(
            f"Audio téléchargé ({len(audio_data) / 1024:.1f} Ko, "
            f"durée estimée={audio_duration:.1f}s) en {time.perf_counter() - _t0:.1f}s"
        )

        # ── ASR ─────────────────────────────────────────────────────────────
        _t0 = time.perf_counter()
        logger.info("Lancement de la transcription ASR")
        result = asr(tmp_path)
        chunks = result.get("chunks", [])
        logger.info(
            f"Transcription ASR terminée en {time.perf_counter() - _t0:.1f}s "
            f"— {len(chunks)} segments bruts"
        )

        if not chunks:
            raise HTTPException(status_code=422, detail="No speech detected in audio file")

        # ── Post-process chunks ─────────────────────────────────────────────
        # 1. Drop empty text
        chunks = [c for c in chunks if c.get("text", "").strip()]

        # 2. Fix missing end timestamps using the next chunk's start or audio duration
        for i, chunk in enumerate(chunks):
            if chunk["timestamp"][1] is None or chunk["timestamp"][1] == 0:
                next_start = chunks[i + 1]["timestamp"][0] if i + 1 < len(chunks) else None
                fixed_end  = next_start or audio_duration or (chunk["timestamp"][0] + 5.0)
                chunk["timestamp"] = (chunk["timestamp"][0], fixed_end)
                logger.debug(f"Fixed end timestamp for chunk {i}: → {fixed_end:.2f}s")

        # 3. Filter hallucination loops — log every chunk before/after for diagnostics
        logger.info(f"Chunks avant filtre hallucination : {len(chunks)}")
        for i, c in enumerate(chunks):
            ts = c.get("timestamp", (None, None))
            preview = c.get("text", "")[:80].replace("\n", " ")
            logger.info(f"  chunk[{i}] ts={ts} | {preview!r}")

        before = len(chunks)
        chunks = [c for c in chunks if not _is_hallucinated(c["text"])]
        dropped = before - len(chunks)
        if dropped:
            logger.warning(f"Dropped {dropped} hallucinated chunk(s)")
        logger.info(f"Chunks après filtre hallucination : {len(chunks)}")

        if not chunks:
            raise HTTPException(status_code=422, detail="No speech detected in audio file")

        # ── Diarisation ─────────────────────────────────────────────────────
        chunks = assign_speakers_improved(chunks, tmp_path, req.num_speakers)

        # ── Build utterances ────────────────────────────────────────────────
        utterances = [
            Utterance(
                start=chunk["timestamp"][0],
                end=chunk["timestamp"][1],
                text=chunk["text"].strip(),
                speaker=chunk["speaker"],
                confidence=chunk.get("confidence"),
            )
            for chunk in chunks
        ]

        # ── Optional merge ───────────────────────────────────────────────────
        if req.merge_same_speaker:
            utterances = merge_consecutive_speakers(utterances)
            logger.info(f"Merged to {len(utterances)} utterances")

        # Use the larger of (max utterance end) and (known audio duration) so we
        # never report a total_duration shorter than the actual file.
        utt_end_max = max((u.end for u in utterances), default=0.0)
        total_duration = max(utt_end_max, audio_duration)

        response = TranscribeResponse(
            utterances=utterances,
            total_duration=total_duration,
            num_speakers=req.num_speakers,
            language=req.language,
            processing_time_seconds=time.perf_counter() - request_start,
        )

        logger.info(
            f"✓ Requête terminée avec succès — {len(utterances)} énoncés retournés "
            f"en {response.processing_time_seconds:.1f}s"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erreur inattendue : {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"Temporary file cleaned up: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.post("/transcribe_with_export")
def transcribe_with_export(
    req: TranscribeRequest,
    format: ExportFormat = Query(ExportFormat.json, description="Export format"),
):
    """Transcribe and diarize Darija audio, returning in specified format (json/vtt/srt/txt)."""
    result = transcribe(req)
    if format == ExportFormat.json:
        return result
    if format == ExportFormat.vtt:
        return {"format": "vtt", "content": export_to_vtt(result.utterances)}
    if format == ExportFormat.srt:
        return {"format": "srt", "content": export_to_srt(result.utterances)}
    return {"format": "txt", "content": export_to_txt(result.utterances)}


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "models_loaded": True,
    }


@app.get("/")
def root():
    return {
        "name": "Darija Transcription & Diarization API",
        "version": "2.1",
        "endpoints": {
            "POST /transcribe":             "Transcribe and diarize Darija audio",
            "POST /transcribe_with_export": "Transcribe and export in VTT/SRT/TXT format",
            "GET  /health":                 "Health check",
        },
        "docs": "/docs",
    }


# ── Run server ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")