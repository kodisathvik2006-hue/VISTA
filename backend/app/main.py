from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import get_settings
from app.exceptions import AppError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurovision")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.uploads_path
    settings.frames_path
    settings.jobs_path
    settings.evidence_path
    logger.info(
        "Neurovision API ready (uploads=%s, frames=%s, jobs=%s, device=%s)",
        settings.uploads_path,
        settings.frames_path,
        settings.jobs_path,
        settings.resolved_device,
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.app_version,
        description="Video ingestion, OpenCV metadata, YOLO neural detection, and ByteTrack tracking.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
            body = {"error": exc.detail}
        else:
            body = {
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                }
            }
        return JSONResponse(status_code=exc.status_code, content=body)

    @application.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request did not match the expected schema.",
                    "details": exc.errors(),
                }
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected server error occurred while processing the request.",
                }
            },
        )

    application.include_router(router, prefix="/api")
    return application


app = create_app()
