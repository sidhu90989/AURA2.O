from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import get_settings
from .routers import memory as memory_router
from .routers import nlp as nlp_router
from .routers import voice as voice_router
from .routers import system as system_router
from .routers import security as security_router
from .routers import graph as graph_router
from .routers import vision as vision_router

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.api_version)

# CORS
origins = [o.strip() for o in settings.allowed_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(memory_router.router)
app.include_router(nlp_router.router)
app.include_router(voice_router.router)
app.include_router(system_router.router)
app.include_router(security_router.router)
app.include_router(graph_router.router)
app.include_router(vision_router.router)

@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name, "env": settings.environment}

@app.get("/")
async def root():
    return {"message": "AURA2.O backend online", "version": settings.api_version}
