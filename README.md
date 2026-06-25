---
title: Darija Transcription & Diarization API
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: t4-small
license: apache-2.0
---
# Darija Transcription & Diarization API

A FastAPI-based service for transcribing and diarizing Moroccan Darija audio using OpenAI Whisper with LoRA fine-tuning and pyannote speaker diarization.

## Features

✨ **Speech-to-Text**: Transcribes Darija audio using Whisper + LoRA fine-tuned model  
👥 **Speaker Diarization**: Identifies and labels different speakers  
📤 **Multiple Export Formats**: JSON, VTT, SRT, TXT  
🚀 **GPU-Accelerated**: Optimized for CUDA devices  
🔄 **Retry Logic**: Automatic retries for robust downloads  

## Deploy to Hugging Face Spaces 🤗

### Step 1: Prepare Your GitHub Repository

1. Create a new GitHub repository (or use an existing one)
2. Clone it locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/darija-transcription.git
   cd darija-transcription
   ```

3. Add these files to your repo:
   - `app.py` ✓
   - `requirements.txt` ✓
   - `Dockerfile` ✓
   - `.gitignore` ✓
   - `README.md` (this file)

4. Commit and push:
   ```bash
   git add .
   git commit -m "Initial commit: Darija transcription API"
   git push origin main
   ```

### Step 2: Create a Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `darija-transcription` (or your preferred name)
   - **License**: Apache 2.0 (or your choice)
   - **Space SDK**: Docker
   - **Space hardware**: **GPU T4** (free tier: ~20 hrs/week)
   - **Private or Public**: Choose your preference

4. Click **"Create Space"**

### Step 3: Connect Your Repository

1. In your newly created Space, click **"Settings"** (⚙️ icon)
2. Under **"Linked Repository"**, paste your GitHub repo URL:
   ```
   https://github.com/YOUR_USERNAME/darija-transcription
   ```
3. Click **"Link repository"**

The Space will automatically build and deploy from your GitHub repo!

### Step 4: Add Environment Variables (Important!)

1. Go to **Settings** → **"Repository secrets"**
2. Add `HF_TOKEN`:
   - Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Make sure it has **"repo"** permissions
   - Copy and paste it as `HF_TOKEN` secret

This allows the pyannote diarization model to download (it requires authentication).

## Local Development

### Prerequisites

- Python 3.10+
- CUDA 12.1 (or CPU-only mode)
- ~20 GB disk space for models

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/darija-transcription.git
   cd darija-transcription
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` file:
   ```bash
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

   Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

5. Run the app:
   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:8000`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Transcribe Audio

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.wav",
    "num_speakers": 2,
    "merge_same_speaker": true
  }'
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_url` | string | required | URL to audio file (HTTP/HTTPS) |
| `language` | string | "ar" | Language code (ar for Arabic/Darija) |
| `num_speakers` | int | 2 | Expected number of speakers (1-10) |
| `merge_same_speaker` | bool | true | Merge consecutive segments from same speaker |

**Response:**

```json
{
  "utterances": [
    {
      "start": 0.5,
      "end": 2.3,
      "text": "السلام عليكم ورحمة الله وبركاته",
      "speaker": "SPEAKER_00",
      "confidence": 0.95
    },
    {
      "start": 2.8,
      "end": 5.1,
      "text": "عليكم السلام ورحمة الله وبركاته",
      "speaker": "SPEAKER_01",
      "confidence": 0.92
    }
  ],
  "total_duration": 5.1,
  "num_speakers": 2,
  "language": "ar",
  "processing_time_seconds": 12.4
}
```

### Export to Different Formats

```bash
# Export to VTT (WebVTT subtitles)
curl -X POST "http://localhost:8000/transcribe_with_export?format=vtt" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.wav"
  }' | jq -r '.content'
# Export to SRT (SubRip subtitles)
curl -X POST "http://localhost:8000/transcribe_with_export?format=srt" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.wav"
  }' | jq -r '.content'
# Export to TXT (plain text)
curl -X POST "http://localhost:8000/transcribe_with_export?format=txt" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.wav"
  }' | jq -r '.content'
```

## Interactive Documentation

Once the API is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

You can test endpoints directly from the browser!

## Performance Notes

### First Deployment
- Model loading takes ~3-5 minutes on first startup
- Subsequent requests are much faster

### Free Tier Limitations (HF Spaces)
- GPU available: ~20 hours/week (T4)
- Space sleeps after 48 hours of inactivity
- Cold starts take longer

### Optimize Performance
To reduce model size (trades accuracy for speed):

```python
# In app.py, change to:
BASE_MODEL = "openai/whisper-base"  # 150MB instead of 3GB
```

## Troubleshooting

### "HF_TOKEN not found"
- Make sure you added `HF_TOKEN` to Space secrets
- Token needs "repo" access level at minimum

### "CUDA out of memory"
- Reduce chunk size in app.py:
  ```python
  chunk_length_s=15,  # Reduce from 30
  ```

### Models not downloading
- Check your HF token is valid: `huggingface-cli login`
- Ensure you have enough disk space

### Slow first request
- Normal! Models are loading from cache on first startup
- Takes 3-5 minutes; subsequent requests are faster

## Project Structure

```
darija-transcription/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration for HF Spaces
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Models Used

- **ASR**: OpenAI Whisper Large V3 Turbo + LoRA fine-tune for Darija
- **Diarization**: pyannote/speaker-diarization-3.1
- **Hardware**: CUDA GPU (T4 on HF Spaces)

## License

Apache 2.0 — Feel free to use, modify, and distribute!

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API logs in Space runtime
3. Open an issue on GitHub

---

**Built with ❤️ for Darija speakers**