from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

try:
    import vision_processor  # type: ignore  # existing module placeholder
except Exception:  # pragma: no cover
    vision_processor = None

router = APIRouter(prefix="/vision", tags=["vision"])

class VisionIdentifyResponse(BaseModel):
    status: str
    detail: str | None = None

@router.post('/identify', response_model=VisionIdentifyResponse)
async def identify(image: UploadFile = File(...)):
    # Placeholder: just acknowledge upload; later integrate real face/object recognition
    name = getattr(image, 'filename', 'unknown')
    if not vision_processor:
        return VisionIdentifyResponse(status='unavailable', detail=f'received {name}')
    return VisionIdentifyResponse(status='ok', detail=f'processed {name}')
