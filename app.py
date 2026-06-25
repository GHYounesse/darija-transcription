from dotenv import load_dotenv
load_dotenv()
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import logging
import time
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
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
    version="2.0"
)

# ── Enums ──────────────────────────────────────────────────────────────────
class ExportFormat(str, Enum):
    json = "json"
    vtt = "vtt"
    srt = "srt"
    txt = "txt"

# ── Model loading ──────────────────────────────────────────────────────────
BASE_MODEL = "openai/whisper-large-v3-turbo"
LORA_MODEL = "anaszil/whisper-large-v3-turbo-darija"

dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
device = 0 if torch.cuda.is_available() else "cpu"

logger.info(f"CUDA disponible : {torch.cuda.is_available()} — device utilisé : {device}, dtype : {dtype}")

_t0 = time.perf_counter()
logger.info(f"Chargement du modèle de base Whisper : {BASE_MODEL}")
base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype)
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
)
logger.info("Pipeline ASR prêt")

_t0 = time.perf_counter()
logger.info("Chargement du pipeline de diarisation pyannote/speaker-diarization-3.1")
diarizer = DiarizationPipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HF_TOKEN")
)
logger.info(f"Pipeline de diarisation chargé en {time.perf_counter() - _t0:.1f}s")

logger.info("✓ Tous les modèles sont chargés — le serveur est prêt à recevoir des requêtes")

# ── Schemas ────────────────────────────────────────────────────────────────
class TranscribeRequest(BaseModel):
    audio_url: str = Field(..., description="URL of audio file (HTTP/HTTPS)")
    language: str = Field(default="ar", description="Language code (ar for Arabic)")
    num_speakers: int = Field(
        default=2, 
        ge=1, 
        le=10,
        description="Expected number of speakers (1-10)"
    )
    merge_same_speaker: bool = Field(
        default=True,
        description="Merge consecutive segments from same speaker"
    )

    @validator('audio_url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("audio_url must start with http:// or https://")
        return v

    @validator('num_speakers')
    def validate_speakers(cls, v):
        if v < 1 or v > 10:
            raise ValueError("num_speakers must be between 1 and 10")
        return v

class Utterance(BaseModel):
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Transcribed text in Darija")
    speaker: str = Field(..., description="Speaker label (SPEAKER_00, SPEAKER_01, etc.)")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")

class TranscribeResponse(BaseModel):
    utterances: List[Utterance]
    total_duration: float
    num_speakers: int
    language: str
    processing_time_seconds: float

# ── Utility Functions ──────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def download_audio_with_retry(url: str, timeout: int = 60) -> bytes:
    """Download audio from URL with retry logic."""
    try:
        logger.info(f"Downloading audio from {url}")
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise

def assign_speakers_improved(chunks: list, audio_path: str, num_speakers: int) -> list:
    """
    Merge Whisper chunks with pyannote diarization using overlap-based matching.
    More robust than midpoint matching.
    """
    logger.info(f"Lancement de la diarisation (num_speakers={num_speakers})")
    _t0 = time.perf_counter()
    
    try:
        diarization = diarizer(audio_path, num_speakers=num_speakers)
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        # Fallback: assign default speakers
        for i, chunk in enumerate(chunks):
            chunk["speaker"] = f"SPEAKER_{i % num_speakers:02d}"
        return chunks
    
    logger.info(f"Diarisation terminée en {time.perf_counter() - _t0:.1f}s")

    unmatched = 0
    for chunk in chunks:
        t0 = chunk["timestamp"][0] or 0.0
        t1 = chunk["timestamp"][1] or 0.0
        
        # Skip invalid timestamps
        if t1 <= t0:
            chunk["speaker"] = "SPEAKER_00"
            unmatched += 1
            continue
        
        # Find all overlapping speaker segments
        speaker_overlaps = {}
        for turn, _, label in diarization.itertracks(yield_label=True):
            # Calculate overlap between chunk [t0, t1] and turn [turn.start, turn.end]
            overlap_start = max(t0, turn.start)
            overlap_end = min(t1, turn.end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                speaker_overlaps[label] = speaker_overlaps.get(label, 0) + overlap_duration
        
        # Assign speaker with maximum overlap
        if speaker_overlaps:
            chunk["speaker"] = max(speaker_overlaps, key=speaker_overlaps.get)
        else:
            chunk["speaker"] = "SPEAKER_00"
            unmatched += 1

    if unmatched:
        logger.warning(
            f"{unmatched}/{len(chunks)} segments had no speaker match "
            f"(assigned default SPEAKER_00)"
        )

    return chunks

def merge_consecutive_speakers(utterances: List[Utterance]) -> List[Utterance]:
    """Merge consecutive utterances from the same speaker."""
    if not utterances:
        return []
    
    merged = [utterances[0]]
    for utt in utterances[1:]:
        if utt.speaker == merged[-1].speaker:
            # Merge: extend end time and concatenate text
            merged[-1].end = utt.end
            merged[-1].text = merged[-1].text.strip() + " " + utt.text.strip()
            # Average confidence if both exist
            if merged[-1].confidence and utt.confidence:
                merged[-1].confidence = (merged[-1].confidence + utt.confidence) / 2
        else:
            merged.append(utt)
    
    return merged

def export_to_vtt(utterances: List[Utterance]) -> str:
    """Export utterances to WebVTT format."""
    vtt = "WEBVTT\n\n"
    for utt in utterances:
        start = f"{int(utt.start//3600):02d}:{int((utt.start%3600)//60):02d}:{utt.start%60:06.3f}"
        end = f"{int(utt.end//3600):02d}:{int((utt.end%3600)//60):02d}:{utt.end%60:06.3f}"
        vtt += f"{start} --> {end}\n"
        vtt += f"{utt.speaker}\n{utt.text}\n\n"
    return vtt

def export_to_srt(utterances: List[Utterance]) -> str:
    """Export utterances to SubRip format."""
    srt = ""
    for i, utt in enumerate(utterances, 1):
        start = f"{int(utt.start//3600):02d}:{int((utt.start%3600)//60):02d}:{int(utt.start%60):02d},{int((utt.start%1)*1000):03d}"
        end = f"{int(utt.end//3600):02d}:{int((utt.end%3600)//60):02d}:{int(utt.end%60):02d},{int((utt.end%1)*1000):03d}"
        srt += f"{i}\n{start} --> {end}\n{utt.speaker}\n{utt.text}\n\n"
    return srt

def export_to_txt(utterances: List[Utterance]) -> str:
    """Export utterances to plain text format."""
    txt = ""
    for utt in utterances:
        txt += f"[{utt.speaker}] {utt.text}\n"
    return txt

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
        # ── Download audio ──────────────────────────────────────────────────
        _t0 = time.perf_counter()
        try:
            audio_data = download_audio_with_retry(req.audio_url)
        except Exception as e:
            logger.error(f"Failed to download audio after retries: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download audio: {str(e)}"
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        logger.info(
            f"Audio téléchargé ({len(audio_data) / 1024:.1f} Ko) "
            f"en {time.perf_counter() - _t0:.1f}s"
        )

        # ── Transcription ──────────────────────────────────────────────────
        _t0 = time.perf_counter()
        logger.info("Lancement de la transcription ASR")
        result = asr(tmp_path)
        chunks = result.get("chunks", [])
        logger.info(
            f"Transcription ASR terminée en {time.perf_counter() - _t0:.1f}s "
            f"— {len(chunks)} segments bruts"
        )

        if not chunks:
            logger.warning("Aucun segment retourné par le modèle ASR")
            raise HTTPException(
                status_code=422,
                detail="No speech detected in audio file"
            )

        # Filter out empty chunks
        chunks = [c for c in chunks if c.get("text", "").strip()]
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in audio file"
            )

        # ── Diarization ─────────────────────────────────────────────────
        _t0 = time.perf_counter()
        chunks = assign_speakers_improved(chunks, tmp_path, req.num_speakers)
        logger.info(f"Speaker assignment completed in {time.perf_counter() - _t0:.1f}s")

        # ── Build utterances ────────────────────────────────────────────
        utterances = [
            Utterance(
                start=chunk["timestamp"][0] or 0.0,
                end=chunk["timestamp"][1] or 0.0,
                text=chunk["text"].strip(),
                speaker=chunk["speaker"],
                confidence=chunk.get("confidence")
            )
            for chunk in chunks
        ]

        # ── Optional: merge consecutive same-speaker segments ────────────
        if req.merge_same_speaker:
            utterances = merge_consecutive_speakers(utterances)
            logger.info(f"Merged to {len(utterances)} utterances")

        # Calculate total duration
        total_duration = max([u.end for u in utterances], default=0.0)

        response = TranscribeResponse(
            utterances=utterances,
            total_duration=total_duration,
            num_speakers=req.num_speakers,
            language=req.language,
            processing_time_seconds=time.perf_counter() - request_start
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
        raise HTTPException(
            status_code=500,
            detail=f"Internal processing error: {str(e)}"
        )
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
    format: ExportFormat = Query(ExportFormat.json, description="Export format")
):
    """
    Transcribe and diarize Darija audio, returning in specified format.
    
    Formats: json, vtt, srt, txt
    """
    result = transcribe(req)
    
    if format == ExportFormat.json:
        return result
    elif format == ExportFormat.vtt:
        return {
            "format": "vtt",
            "content": export_to_vtt(result.utterances)
        }
    elif format == ExportFormat.srt:
        return {
            "format": "srt",
            "content": export_to_srt(result.utterances)
        }
    elif format == ExportFormat.txt:
        return {
            "format": "txt",
            "content": export_to_txt(result.utterances)
        }

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "models_loaded": True
    }

@app.get("/")
def root():
    """API documentation."""
    return {
        "name": "Darija Transcription & Diarization API",
        "version": "2.0",
        "endpoints": {
            "POST /transcribe": "Transcribe and diarize Darija audio",
            "POST /transcribe_with_export": "Transcribe and export in VTT/SRT/TXT format",
            "GET /health": "Health check",
        },
        "docs": "/docs"
    }

# ── Run server ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Use port 7860 for HF Spaces, fallback to 8000 locally
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")