"""FastAPI service for AgriGuard AI predictions.

Two endpoints only. The AI layer is a separate service on purpose: Member 1
owns the ESP32, Member 3 owns the backend/dashboard, and this process does
neither. It receives readings, returns a recommendation, and keeps no state.

    POST /v1/predict/irrigation   JSON      -> irrigation recommendation
    POST /v1/predict/crop-health  multipart  -> Healthy / Possible Disease
    GET  /v1/health               service + model status
"""

import logging
import os

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import IRRIGATION_MODEL_PATH
from .contracts import CropHealthResponse, IrrigationRequest, IrrigationResponse
from .cropdata import API_LABELS
from .crophealth import InvalidImageError, ModelNotAvailableError, crop_health_model
from .imageutils import MAX_UPLOAD_BYTES, load_image_rgb
from .irrigation import irrigation_model
from .security import SecurityMiddleware, build_security_options
from .version import __version__

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agriguard.api")

app = FastAPI(
    title="AgriGuard AI",
    version=__version__,
    description=(
        "Advisory AI layer for AgriGuard. Recommendations only - the service "
        "never switches a pump and never diagnoses a specific crop disease."
    ),
)


#: Comma-separated allow-list for the browser-based dashboard, read once at
#: import. See `configure_cors` for why this is opt-in.
CORS_ORIGINS_ENV = "AGRIGUARD_CORS_ORIGINS"


def parse_cors_origins(raw: str) -> list:
    """Split a comma-separated origin allow-list, dropping blanks.

    Split on commas and strip each entry so a value copied out of a shell with a
    trailing space still works. Blank entries are dropped rather than kept as "",
    because CORSMiddleware treats "" as a real origin and answers requests from
    a null origin, which is exactly the thing an allow-list is meant to exclude.
    """
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def configure_cors(application: FastAPI, origins) -> FastAPI:
    """Attach the CORS middleware only when an allow-list is configured.

    CORS is opt-in because the two deployment shapes want opposite defaults. If
    the dashboard is served from the same origin (a reverse proxy in front of
    both), no header is needed and adding one only widens what a browser will
    send. If the dashboard is hosted separately, the browser blocks the
    preflight until these origins are listed, and there is no safe implicit
    default - "*" would make the prediction endpoints callable from any page the
    user happens to have open, so it is never used.

    Returns the application so this reads as a decorator-free call in a test.
    """
    origins = list(origins or [])
    if not origins:
        return application
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # No cookies or Authorization headers: the service is token-free, so
        # allowing credentials would only widen the preflight for no benefit.
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        # A recommendation is not worth a 24 h browser cache.
        max_age=600,
    )
    return application


configure_cors(app, parse_cors_origins(os.environ.get(CORS_ORIGINS_ENV, "")))

#: API key and rate limiting, both off unless AGRIGUARD_API_KEY /
#: AGRIGUARD_RATE_LIMIT are set. Added last so it is the outermost middleware:
#: a request is authenticated before it is rate limited, so an unauthenticated
#: flood cannot spend a legitimate client's budget. See agriguard/security.py
#: for what these guards do and do not defend against.
_security_options = build_security_options()
if _security_options is not None:
    app.add_middleware(SecurityMiddleware, **_security_options)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Return a clean 422 with field names instead of a raw traceback."""
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        problems.append({"field": location or "body", "message": error.get("msg", "invalid")})
    return JSONResponse(
        status_code=422,
        content={"detail": "request validation failed", "errors": problems},
    )


@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    """Log the traceback server-side, return an opaque 500 to the caller."""
    logger.exception("unhandled error while serving %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal AI service error"},
    )


@app.get("/")
def root():
    return {
        "service": "AgriGuard AI",
        "version": __version__,
        "advisory_only": True,
        "endpoints": [
            "POST /v1/predict/irrigation",
            "POST /v1/predict/crop-health",
            "GET /v1/health",
        ],
        "docs": "/docs",
    }


@app.get("/v1/health")
def health():
    return {
        "status": "ok" if irrigation_model.is_trained and crop_health_model.is_trained else "degraded",
        "version": __version__,
        "advisory_only": True,
        "models": {
            "irrigation": irrigation_model.describe(),
            "crop_health": crop_health_model.describe(),
        },
    }


@app.post("/v1/predict/irrigation", response_model=IrrigationResponse)
def predict_irrigation(req: IrrigationRequest):
    if not irrigation_model.is_trained and irrigation_model.load_error:
        logger.warning("irrigation model not loaded: %s", irrigation_model.load_error)
    return IrrigationResponse(
        version=__version__,
        prediction=irrigation_model.predict(req),
    )


@app.post("/v1/predict/crop-health", response_model=CropHealthResponse)
async def predict_crop_health(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="no image file was provided")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content type '{image.content_type}'; send image/*",
        )

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        # Decode first so a broken upload is reported as 422 even when the
        # weights are missing; only then ask the model.
        image = load_image_rgb(data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=f"invalid image: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("could not decode the uploaded image")
        raise HTTPException(
            status_code=422, detail=f"invalid image: {type(exc).__name__}"
        )

    if not crop_health_model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="crop-health model unavailable: "
            + (crop_health_model.load_error or "weights are not installed"),
        )

    try:
        prediction = crop_health_model.predict(image)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=f"crop-health model unavailable: {exc}")
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=f"invalid image: {exc}")
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to the caller
        logger.exception("crop-health inference failed")
        raise HTTPException(
            status_code=500, detail=f"crop-health inference error: {type(exc).__name__}"
        )

    return CropHealthResponse(version=__version__, prediction=prediction)


@app.get("/v1/classes/crop-health")
def crop_health_classes():
    """Machine-readable class list so the dashboard can build its legend."""
    return {
        "model": "crop-health",
        "version": __version__,
        "classes": [
            {"class": key, "label": label, "dashboard_hint": hint}
            for (key, label), hint in zip(
                API_LABELS.items(),
                ("green card", "amber/red 'inspect plant' alert"),
            )
        ],
        "note": (
            "Possible Disease is a screening flag, not a diagnosis. The model "
            "does not identify which disease is present."
        ),
        "irrigation_model_file": IRRIGATION_MODEL_PATH.name,
    }
