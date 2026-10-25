"""Versioned FastAPI application for student performance model and analytics."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..logging_config import configure_file_logging
from .schemas import (
    AnalyticsSummaryResponse,
    DistributionResponse,
    ErrorDetail,
    ErrorResponse,
    FeatureImportanceResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelMetricsResponse,
    PredictionRequest,
    PredictionResponse,
)
from .service import ApiServiceError, BackendService
from .settings import ApiSettings

LOGGER = logging.getLogger(__name__)


class RequestSizeLimitMiddleware:
    """Reject oversized HTTP bodies without logging or retaining their contents."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_content_length = headers.get(b"content-length")
        if raw_content_length is not None:
            try:
                too_large = int(raw_content_length) > self.max_bytes
            except ValueError:
                too_large = True
            if too_large:
                await self._send_too_large(scope, send)
                return

        received_bytes = 0
        response_started = False

        async def track_response(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def track_request() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, track_request, track_response)
        except _RequestTooLarge:
            if not response_started:
                await self._send_too_large(scope, send)

    async def _send_too_large(self, scope: dict[str, Any], send: Any) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "request_too_large",
                    "message": "Request body exceeds the configured size limit.",
                }
            },
        )
        await response(scope, _empty_receive, send)


class _RequestTooLarge(Exception):
    """Internal control flow for rejecting a request body above the configured cap."""


async def _empty_receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def create_app(
    settings: ApiSettings | None = None,
    *,
    backend_service: BackendService | None = None,
) -> FastAPI:
    """Create an independently configurable API application."""
    resolved_settings = settings or ApiSettings.from_environment()
    service = backend_service or BackendService(resolved_settings)
    router = APIRouter(prefix="/api/v1")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_file_logging("backend.log")
        service.initialize_model()
        yield

    app = FastAPI(
        title="EduPredict Student Performance API",
        description=(
            "Versioned API for measured student performance analytics and "
            "saved-model score estimates."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.backend_service = service
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=resolved_settings.max_request_bytes,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.exception_handler(ApiServiceError)
    async def handle_api_service_error(
        _: Request, exc: ApiServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(code=exc.code, message=exc.message)
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "field": ".".join(
                    str(part)
                    for part in (
                        error["loc"][1:]
                        if error["loc"] and error["loc"][0] == "body"
                        else error["loc"]
                    )
                    if str(part) not in {"int", "float", "str", "none"}
                ),
                "issue": error["type"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": (
                        "Request validation failed. Check required fields, field "
                        "types, finite numeric values, and allowed keys."
                    ),
                    "details": details,
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        messages = {
            404: ("not_found", "The requested API route was not found."),
            405: ("method_not_allowed", "The HTTP method is not allowed for this route."),
        }
        code, message = messages.get(
            exc.status_code,
            ("http_error", "The request could not be completed."),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(code=code, message=message)
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        LOGGER.error(
            "Unexpected API request failure",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected server error occurred.",
                }
            },
        )

    @router.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> dict[str, Any]:
        return service.health()

    @router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
    def model_info() -> dict[str, Any]:
        return service.model_info()

    @router.get(
        "/analytics/summary",
        response_model=AnalyticsSummaryResponse,
        tags=["analytics"],
    )
    def analytics_summary() -> dict[str, Any]:
        return service.analytics_summary()

    @router.get(
        "/analytics/distributions",
        response_model=DistributionResponse,
        tags=["analytics"],
    )
    def analytics_distribution(
        feature: str = Query(min_length=1, max_length=64),
    ) -> dict[str, Any]:
        return service.distribution(feature)

    @router.get(
        "/models/metrics",
        response_model=ModelMetricsResponse,
        tags=["model"],
    )
    def model_metrics() -> dict[str, Any]:
        return service.model_metrics()

    @router.get(
        "/models/feature-importance",
        response_model=FeatureImportanceResponse,
        tags=["model"],
    )
    def model_feature_importance() -> dict[str, Any]:
        return service.feature_importance()

    @router.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["prediction"],
    )
    def predict(request: PredictionRequest) -> dict[str, Any]:
        return service.predict(request)

    app.include_router(router)
    return app


app = create_app()
