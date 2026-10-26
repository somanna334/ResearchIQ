import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api import (
    analysis,
    ask,
    auth,
    chunks,
    documents,
    embeddings,
    evaluation,
    hybrid,
    search,
)
from app.config import get_settings
from app.database import init_db
from app.middleware import (
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    configure_logging,
)
from app.ratelimit import llm_rate_limit

logger = logging.getLogger("researchiq")


class HealthResponse(BaseModel):
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    app = FastAPI(title=settings.app_name, version="0.9.1", lifespan=lifespan)

    # Middleware added LAST is outermost. Order here, outside in:
    #   request context (ID + access log) -> security headers -> CORS -> routes
    # so even CORS preflight answers and 429 errors carry an ID and the headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        # Without this, browser code cannot read these response headers.
        expose_headers=["Retry-After", "X-Request-ID"],
    )
    app.add_middleware(
        SecurityHeadersMiddleware, hsts=settings.environment == "production"
    )
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(chunks.router)
    app.include_router(search.router)
    app.include_router(embeddings.router)
    app.include_router(hybrid.router)
    # Routes that call the LLM share a per-user limit.
    app.include_router(ask.router, dependencies=[Depends(llm_rate_limit)])
    app.include_router(analysis.router, dependencies=[Depends(llm_rate_limit)])
    app.include_router(evaluation.router)

    logger.info("App created (environment=%s)", settings.environment)
    return app


app = create_app()
