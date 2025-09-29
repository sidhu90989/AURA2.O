from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

router = APIRouter(prefix="/voice", tags=["voice"])

class STTResponse(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    voice: str | None = None

@router.post("/stt", response_model=STTResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    # Placeholder: integrate Whisper
    return STTResponse(text=f"[transcribed:{audio.filename}]")

@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    # Placeholder: integrate ElevenLabs/Coqui
    return {"audio_url": "data:audio/wav;base64,PLACEHOLDER", "voice": req.voice or "default"}
