"""
TrendZap Intelligence API

FastAPI service for ML model inference.
"""

import hashlib
import json
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trendzap_intelligence import (
    ViralityPredictor,
    EngagementForecaster,
    AnomalyDetector,
    TrendDetector,
    AIAnalyzer,
    settings,
)

# Redis client (shared across requests)
_redis: aioredis.Redis | None = None
AI_CACHE_TTL = 300  # seconds — cache AI analysis results for 5 min


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    try:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        await _redis.ping()
    except Exception:
        # Redis is optional — service still works without it
        _redis = None
    yield
    if _redis:
        await _redis.aclose()


def _cache_key(prefix: str, data: dict) -> str:
    payload = json.dumps(data, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"tz:intel:{prefix}:{digest}"


async def _get_cached(key: str) -> Any | None:
    if _redis is None:
        return None
    try:
        raw = await _redis.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def _set_cached(key: str, value: Any) -> None:
    if _redis is None:
        return
    try:
        await _redis.setex(key, AI_CACHE_TTL, json.dumps(value))
    except Exception:
        pass



logger = logging.getLogger("trendzap.intelligence")


def internal_error(exc: Exception, context: str) -> HTTPException:
    """
    Log an exception server-side and return an opaque reference to the caller.

    Handlers previously returned `str(e)` directly, which leaks absolute file paths,
    library internals and — for connection errors — host details.
    """
    error_id = uuid.uuid4().hex[:12]
    logger.exception("[%s] %s failed (error_id=%s)", context, context, error_id)
    return HTTPException(
        status_code=500,
        detail=f"Internal error processing this request (reference: {error_id})",
    )


app = FastAPI(
    title="TrendZap Intelligence API",
    description="ML models for social media virality prediction, powered by Groq AI",
    version="0.1.0",
    lifespan=lifespan,
)

# `allow_origins=["*"]` with `allow_credentials=True` is a combination browsers reject
# outright, so the previous config neither achieved its intent nor restricted anything.
# Restrict to configured origins and drop credentials, which this API does not use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """
    Shared-secret guard for the LLM-backed endpoints.

    These forward caller-supplied text to a paid Groq account. With no auth and no
    rate limiting, anyone who found the service URL could spend the project's LLM
    budget indefinitely. Fails CLOSED when INTELLIGENCE_API_KEY is unset.
    """
    expected = settings.api_key
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Service is not configured for authenticated requests",
        )

    provided = x_api_key
    if provided is None and authorization and authorization.startswith("Bearer "):
        provided = authorization[len("Bearer "):]

    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

virality_predictor = ViralityPredictor()
engagement_forecaster = EngagementForecaster()
anomaly_detector = AnomalyDetector()
trend_detector = TrendDetector()
ai_analyzer = AIAnalyzer()


class ViralityRequest(BaseModel):
    """Request body for virality prediction."""
    
    platform: str = Field(..., description="Social platform")
    post_url: str = Field(..., description="URL of the post")
    post_text: str = Field("", description="Post text content")
    follower_count: int = Field(0, description="Creator's follower count")
    initial_likes: int = Field(0, description="Current like count")
    initial_retweets: int = Field(0, description="Current retweet/share count")
    threshold: int = Field(100000, description="Virality threshold")
    metric: str = Field("likes", description="Metric to predict")


class ViralityResponse(BaseModel):
    """Response for virality prediction."""

    probability: float
    #: "model" when trained weights are loaded, "heuristic" otherwise. Clients must
    #: not present a heuristic estimate as a model prediction.
    method: str
    #: Spread of the estimate, NOT model certainty. Previously mislabelled "confidence".
    dispersion: float
    threshold: int
    likely_outcome: str


class EngagementRequest(BaseModel):
    """Request body for engagement forecast."""
    
    platform: str
    current_engagement: int
    time_elapsed_hours: float
    time_remaining_hours: float
    follower_count: int = 0
    metric: str = "likes"


class EngagementResponse(BaseModel):
    """Response for engagement forecast."""

    predicted_value: int
    #: Bounds are a fixed +/-20% band around the point estimate, NOT a fitted interval.
    #: The model previously reported `confidence_interval: 0.95` alongside them, which
    #: gave a flat multiplier a statistical label it had not earned.
    lower_bound: int
    upper_bound: int
    growth_rate: float
    #: "model" when trained weights are loaded, "heuristic" otherwise.
    method: str


class AnomalyRequest(BaseModel):
    """Request body for anomaly detection."""
    
    engagement_velocity: float
    engagement_count: int
    follower_count: int
    new_account_ratio: float = 0.0
    single_region_ratio: float = 0.0


class AnomalyResponse(BaseModel):
    """Response for anomaly detection."""
    
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str | None
    signals: list[str]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_ok = False
    if _redis:
        try:
            await _redis.ping()
            redis_ok = True
        except Exception:
            pass
    return {
        "status": "healthy",
        "version": "0.1.0",
        "ai_provider": settings.ai_provider,
        "ai_model": settings.ai_model,
        "redis": "connected" if redis_ok else "unavailable",
    }


@app.post("/api/v1/predict/virality", response_model=ViralityResponse)
async def predict_virality(request: ViralityRequest):
    """Predict viral probability for a social media post."""
    try:
        result = virality_predictor.predict({
            "platform": request.platform,
            "post_text": request.post_text,
            "follower_count": request.follower_count,
            "initial_likes": request.initial_likes,
            "initial_retweets": request.initial_retweets,
        })
        
        return ViralityResponse(
            probability=result.probability,
            method=result.method,
            dispersion=result.dispersion,
            threshold=request.threshold,
            likely_outcome="OVER" if result.probability > 0.5 else "UNDER",
        )
    except Exception as e:
        raise internal_error(e, "predict/virality")


@app.post("/api/v1/predict/engagement", response_model=EngagementResponse)
async def predict_engagement(request: EngagementRequest):
    """Forecast final engagement count."""
    try:
        result = engagement_forecaster.predict({
            "platform": request.platform,
            "current_engagement": request.current_engagement,
            "time_elapsed_hours": request.time_elapsed_hours,
            "time_remaining_hours": request.time_remaining_hours,
            "follower_count": request.follower_count,
            "metric": request.metric,
        })
        
        return EngagementResponse(
            predicted_value=result.predicted_value,
            lower_bound=result.lower_bound,
            upper_bound=result.upper_bound,
            growth_rate=result.growth_rate,
            method="model" if engagement_forecaster.model is not None else "heuristic",
        )
    except Exception as e:
        raise internal_error(e, "predict/engagement")


@app.post("/api/v1/detect/anomaly", response_model=AnomalyResponse)
async def detect_anomaly(request: AnomalyRequest):
    """Detect artificial engagement patterns."""
    try:
        result = anomaly_detector.detect({
            "engagement_velocity": request.engagement_velocity,
            "engagement_count": request.engagement_count,
            "follower_count": request.follower_count,
            "new_account_ratio": request.new_account_ratio,
            "single_region_ratio": request.single_region_ratio,
        })
        
        return AnomalyResponse(
            is_anomaly=result.is_anomaly,
            anomaly_score=result.anomaly_score,
            anomaly_type=result.anomaly_type,
            signals=result.signals,
        )
    except Exception as e:
        raise internal_error(e, "detect/anomaly")


@app.get("/api/v1/trends")
async def get_trends():
    """Get current trending topics (placeholder)."""
    return {
        "trends": [],
        "message": "Feed data to detect trends",
    }


# ---------------------------------------------------------------------------
# AI-Powered Endpoints (Groq LLM)
# ---------------------------------------------------------------------------


class AIPostAnalysisRequest(BaseModel):
    """Request body for AI-powered post analysis."""

    platform: str = Field(..., description="Social platform")
    post_text: str = Field(..., description="Post text content")
    follower_count: int = Field(0, description="Creator's follower count")
    current_likes: int = Field(0, description="Current like count")
    current_shares: int = Field(0, description="Current share count")


class AITrendAnalysisRequest(BaseModel):
    """Request body for AI-powered trend analysis."""

    topic: str = Field(..., description="Trend topic")
    keywords: list[str] = Field(default_factory=list, description="Related keywords")
    volume: int = Field(0, description="Number of posts")
    velocity: float = Field(0.0, description="Posts per hour")
    platform: str = Field("cross-platform", description="Platform")


class AIAnomalyExplainRequest(BaseModel):
    """Request body for AI anomaly explanation."""

    is_anomaly: bool = Field(..., description="Whether an anomaly was detected")
    anomaly_score: float = Field(0.0, description="Anomaly score")
    anomaly_type: str | None = Field(None, description="Type of anomaly")
    signals: list[str] = Field(default_factory=list, description="Detection signals")
    engagement_velocity: float = Field(0.0, description="Engagement velocity")
    engagement_count: int = Field(0, description="Total engagements")
    follower_count: int = Field(0, description="Follower count")


@app.post("/api/v1/ai/analyze-post", dependencies=[Depends(require_api_key)])
async def ai_analyze_post(request: AIPostAnalysisRequest):
    """Use Groq AI to analyze a social media post and provide actionable insights."""
    try:
        key = _cache_key("post", request.model_dump())
        cached = await _get_cached(key)
        if cached:
            return {"provider": settings.ai_provider, "model": settings.ai_model, "analysis": cached, "cached": True}
        result = await ai_analyzer.analyze_post(request.model_dump())
        await _set_cached(key, result)
        return {"provider": settings.ai_provider, "model": settings.ai_model, "analysis": result, "cached": False}
    except Exception as e:
        raise internal_error(e, "ai/analyze-post")


@app.post("/api/v1/ai/analyze-trend", dependencies=[Depends(require_api_key)])
async def ai_analyze_trend(request: AITrendAnalysisRequest):
    """Use Groq AI to provide deeper insights on a detected trend."""
    try:
        key = _cache_key("trend", request.model_dump())
        cached = await _get_cached(key)
        if cached:
            return {"provider": settings.ai_provider, "model": settings.ai_model, "analysis": cached, "cached": True}
        result = await ai_analyzer.analyze_trend(request.model_dump())
        await _set_cached(key, result)
        return {"provider": settings.ai_provider, "model": settings.ai_model, "analysis": result, "cached": False}
    except Exception as e:
        raise internal_error(e, "ai/analyze-trend")


@app.post("/api/v1/ai/explain-anomaly", dependencies=[Depends(require_api_key)])
async def ai_explain_anomaly(request: AIAnomalyExplainRequest):
    """Use Groq AI to explain a detected anomaly in human-readable terms."""
    try:
        key = _cache_key("anomaly", request.model_dump())
        cached = await _get_cached(key)
        if cached:
            return {"provider": settings.ai_provider, "model": settings.ai_model, "analysis": cached, "cached": True}
        result = await ai_analyzer.explain_anomaly(request.model_dump())
        await _set_cached(key, result)
        return {"provider": settings.ai_provider, "model": settings.ai_model, "analysis": result, "cached": False}
    except Exception as e:
        raise internal_error(e, "ai/explain-anomaly")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
