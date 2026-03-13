import io
import os
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

APP_BUILD = os.getenv("RENDER_GIT_COMMIT", "local")

# -------------------------------------------------------------------
# Security / deployment settings (env-driven)
# -------------------------------------------------------------------

# CORS: comma-separated list of allowed origins, e.g.
# "https://your-frontend.com,https://www.your-frontend.com"
_cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
CORS_ALLOW_ORIGINS: List[str] = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
if not CORS_ALLOW_ORIGINS:
    # Safe-ish dev default. Set CORS_ALLOW_ORIGINS explicitly in production.
    CORS_ALLOW_ORIGINS = ["http://localhost:3000", "http://localhost:8501"]

CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

# Admin key to protect sensitive routes (set this in Render env vars)
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()

# Upload protection
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5MB default
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(20_000_000)))       # 20 MP default
ALLOWED_IMAGE_CONTENT_TYPES = {
    ct.strip().lower()
    for ct in os.getenv(
        "ALLOWED_IMAGE_CONTENT_TYPES",
        "image/jpeg,image/png,image/webp",
    ).split(",")
    if ct.strip()
}

# Basic per-IP rate limit (best-effort, in-memory; use a gateway/WAF for stronger control)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_rate_state: Dict[str, Dict[str, float]] = {}
# -------------------------------------------------------------------
# Model configuration & versioning
# -------------------------------------------------------------------

IMG_SIZE = (224, 224)
CLASS_NAMES = ["fmd", "healthy", "lumpy skin", "mastitis"]

# Map of version -> model file path (relative to project root)
MODEL_PATHS: Dict[str, str] = {
    "v1": "cattle_disease_model.h5",
    # later:
    # "v2": "models/cattle_disease_model_v2.h5",
}

# Default version (can also be set via env var on Render)
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Cache of loaded models to avoid reloading on every request
_loaded_models: Dict[str, tf.keras.Model] = {}


def get_model(version: str) -> tf.keras.Model:
    if version not in MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model version '{version}'. Available: {list(MODEL_PATHS.keys())}",
        )
    if version in _loaded_models:
        return _loaded_models[version]

    path = MODEL_PATHS[version]
    if not os.path.exists(path):
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found for version '{version}' at '{path}'",
        )

    model = tf.keras.models.load_model(path)
    _loaded_models[version] = model
    return model


# Preload default model at startup
_active_version = DEFAULT_MODEL_VERSION
get_model(_active_version)

DISEASE_INFO = {
    "fmd": {
        "full_name":    "Foot and Mouth Disease",
        "severity":     "Urgent",
        "requires_vet": True,
        "what_you_see": "Blisters on the mouth, tongue, and hooves. The animal may be drooling heavily or struggling to walk.",
        "what_to_do":   "Separate this animal from the rest of the herd right away. Do not move animals off the farm. Call your vet or the nearest livestock office today — FMD spreads very fast.",
        "urgency_msg":  "Act today. Every hour matters.",
    },
    "healthy": {
        "full_name":    "No Disease Detected",
        "severity":     "All Clear",
        "requires_vet": False,
        "what_you_see": "The animal shows no visible signs of disease.",
        "what_to_do":   "Your animal looks healthy. Keep up regular check-ups, ensure clean water and feed, and stay on your vaccination schedule.",
        "urgency_msg":  "Continue routine care.",
    },
    "lumpy skin": {
        "full_name":    "Lumpy Skin Disease",
        "severity":     "Urgent",
        "requires_vet": True,
        "what_you_see": "Round raised lumps or nodules appearing across the skin. The animal may have a fever and reduced milk output.",
        "what_to_do":   "Separate the animal from the herd immediately. LSD spreads through insect bites — treat the whole herd with insect repellent. Contact your vet to vaccinate at-risk animals.",
        "urgency_msg":  "Isolate today. Protect the herd.",
    },
    "mastitis": {
        "full_name":    "Mastitis",
        "severity":     "Needs Attention",
        "requires_vet": True,
        "what_you_see": "The udder looks swollen or feels warm and painful. Milk may appear watery, lumpy, or discoloured.",
        "what_to_do":   "Contact your vet for antibiotic treatment. Milk the affected quarters separately and discard that milk. Wash hands and equipment between animals.",
        "urgency_msg":  "Book a vet visit soon.",
    },
}

# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------

app = FastAPI(title="Cattle Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = (request.client.host if request.client else "unknown").strip()
    now = time.time()
    window = 60.0

    state = _rate_state.get(client_ip)
    if not state or now >= state["reset_at"]:
        state = {"count": 0.0, "reset_at": now + window}
        _rate_state[client_ip] = state

    state["count"] += 1.0
    if RATE_LIMIT_PER_MINUTE > 0 and state["count"] > RATE_LIMIT_PER_MINUTE:
        retry_after = max(1, int(state["reset_at"] - now))
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

async def read_upload_limited(file: UploadFile) -> bytes:
    if file.content_type:
        ct = file.content_type.lower().strip()
        if ct not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type '{file.content_type}'. Allowed: {sorted(ALLOWED_IMAGE_CONTENT_TYPES)}",
            )

    buf = bytearray()
    chunk_size = 1024 * 1024  # 1MB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed is {MAX_UPLOAD_BYTES} bytes.",
            )
    return bytes(buf)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image resolution too large ({w}x{h}).",
        )

    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)  # 0–255, matches training
    return np.expand_dims(arr, axis=0)


def predict_single(image_bytes: bytes, version: str) -> dict:
    model = get_model(version)
    x = preprocess_image(image_bytes)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    disease = CLASS_NAMES[idx]

    # Confidence as whole-number percentage, e.g. 95
    confidence = int(round(float(preds[idx]) * 100.0))

    info = DISEASE_INFO[disease]

    return {
        "model_version": version,
        "predicted_class": disease,
        "confidence": confidence,  # whole-number percent
        "probabilities": {
            CLASS_NAMES[i]: float(p) for i, p in enumerate(preds)
        },
        "info": {
            "full_name": info["full_name"],
            "what_you_see": info["what_you_see"],
            "what_to_do": info["what_to_do"],
            "urgency": info["urgency_msg"],
            "severity": info["severity"],
            "requires_vet": info["requires_vet"],
        },
    }

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Cattle Disease Detection API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "models": "/models",
        },
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_model_version": _active_version,
        "available_versions": list(MODEL_PATHS.keys()),
         "build": APP_BUILD,
         "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
    }

@app.get("/models")
def list_models():
    return {
        "active_model_version": _active_version,
        "available_versions": MODEL_PATHS,
    }

@app.post("/models/set-active")
def set_active_model(
    version: str,
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
):
    if not ADMIN_API_KEY:
        # Hide the endpoint if not configured
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_admin_key or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    global _active_version
    if version not in MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model version '{version}'. Available: {list(MODEL_PATHS.keys())}",
        )
    # ensure it loads successfully before switching
    get_model(version)
    _active_version = version
    return {"active_model_version": _active_version}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    version: str = Query(default=None, description="Optional model version to use"),
):
    """
    Single-image prediction.

    - Upload one image file as 'file' (multipart/form-data).
    - Optionally pass ?version=v1/v2 to override the active model.
    """
    image_bytes = await read_upload_limited(file)
    v = version or _active_version
    result = predict_single(image_bytes, v)
    return result