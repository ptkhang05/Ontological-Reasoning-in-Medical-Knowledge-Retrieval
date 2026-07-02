from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response

from clinical_nlp.btc import BtcEntity, response_to_btc_entities
from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    APIError,
    APIErrorBody,
    ErrorDetail,
)


def create_app(pipeline: ClinicalPipeline | None = None) -> FastAPI:
    app = FastAPI(
        title="Clinical Concept Normalization Prototype API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.pipeline = pipeline or ClinicalPipeline()

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request
        details = [
            ErrorDetail(
                loc=list(error.get("loc", ())),
                msg=str(error.get("msg", "Invalid input")),
                type=str(error.get("type", "validation_error")),
            )
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=APIError(
                error=APIErrorBody(
                    code="VALIDATION_ERROR",
                    message="Invalid request.",
                    details=details,
                )
            ).model_dump(by_alias=True),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIError(
                error=APIErrorBody(
                    code="INTERNAL_ERROR",
                    message="Internal server error.",
                    details=None,
                )
            ).model_dump(by_alias=True),
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post(
        "/v1/analyze",
        response_model=AnalyzeResponse,
        status_code=status.HTTP_200_OK,
        response_model_by_alias=True,
    )
    async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        service: ClinicalPipeline = app.state.pipeline
        return service.analyze(request)

    @app.post(
        "/v1/analyze/btc",
        response_model=list[BtcEntity],
        status_code=status.HTTP_200_OK,
    )
    async def analyze_btc(request: AnalyzeRequest) -> list[BtcEntity]:
        service: ClinicalPipeline = app.state.pipeline
        response = service.analyze(request)
        return response_to_btc_entities(response, request.text)

    return app


app = create_app()
