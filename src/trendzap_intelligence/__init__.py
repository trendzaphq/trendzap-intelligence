"""
TrendZap Intelligence

AI/ML models for predicting social media virality.
"""

from .models.virality_predictor import ViralityPredictor
from .models.engagement_forecaster import EngagementForecaster
from .models.anomaly_detector import AnomalyDetector
from .models.trend_detector import TrendDetector
from .ai_analyzer import AIAnalyzer
from .config import settings, get_ai_client

__version__ = "0.1.0"
__all__ = [
    "ViralityPredictor",
    "EngagementForecaster",
    "AnomalyDetector",
    "TrendDetector",
    "AIAnalyzer",
    "settings",
    "get_ai_client",
]
