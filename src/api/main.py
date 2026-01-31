"""
TrendZap Intelligence API

FastAPI service for ML model inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trendzap_intelligence import (
    ViralityPredictor,
    EngagementForecaster,
    AnomalyDetector,
    TrendDetector,
)

app = FastAPI(
    title="TrendZap Intelligence API",
    description="ML models for social media virality prediction",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

virality_predictor = ViralityPredictor()
engagement_forecaster = EngagementForecaster()
anomaly_detector = AnomalyDetector()
trend_detector = TrendDetector()


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
    confidence: float
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
    lower_bound: int
    upper_bound: int
    growth_rate: float


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
    return {"status": "healthy", "version": "0.1.0"}


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
            confidence=result.confidence,
            threshold=request.threshold,
            likely_outcome="OVER" if result.probability > 0.5 else "UNDER",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trends")
async def get_trends():
    """Get current trending topics (placeholder)."""
    return {
        "trends": [],
        "message": "Feed data to detect trends",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
