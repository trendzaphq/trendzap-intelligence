"""
Engagement Forecaster Model

Predicts final engagement count given current state.
Uses XGBoost for gradient boosting regression.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


@dataclass
class EngagementForecast:
    """Result of an engagement forecast."""
    
    predicted_value: int
    lower_bound: int
    upper_bound: int
    confidence_interval: float
    growth_rate: float


class EngagementForecaster:
    """
    Predicts final engagement metrics using XGBoost.
    
    Features:
    - Current engagement count
    - Time elapsed / time remaining
    - Engagement velocity (rate of change)
    - Platform-specific patterns
    - Historical patterns
    """
    
    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []
    
    def _extract_features(self, data: dict[str, Any]) -> np.ndarray:
        """Extract numerical features from input data."""
        current = data.get("current_engagement", 0)
        elapsed = data.get("time_elapsed_hours", 1)
        remaining = data.get("time_remaining_hours", 23)
        total_time = elapsed + remaining
        
        velocity = current / max(elapsed, 0.1)
        
        features = [
            np.log1p(current),
            elapsed / total_time,
            remaining / total_time,
            np.log1p(velocity),
            velocity / max(current, 1) if current > 0 else 0,
            np.log1p(data.get("follower_count", 0)),
            data.get("hour_of_day", 12) / 24.0,
            data.get("day_of_week", 0) / 7.0,
            1.0 if data.get("platform") == "twitter" else 0.0,
            1.0 if data.get("platform") == "tiktok" else 0.0,
            1.0 if data.get("platform") == "instagram" else 0.0,
            1.0 if data.get("platform") == "youtube" else 0.0,
            1.0 if data.get("metric") == "likes" else 0.0,
            1.0 if data.get("metric") == "views" else 0.0,
            1.0 if data.get("metric") == "comments" else 0.0,
            np.log1p(data.get("historical_avg", current)),
            data.get("historical_viral_rate", 0.1),
        ]
        
        self.feature_names = [
            "log_current", "time_elapsed_ratio", "time_remaining_ratio",
            "log_velocity", "acceleration", "log_followers",
            "hour_of_day", "day_of_week",
            "is_twitter", "is_tiktok", "is_instagram", "is_youtube",
            "is_likes", "is_views", "is_comments",
            "log_historical_avg", "historical_viral_rate",
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
    ):
        """
        Train the forecaster model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,) - final engagement counts
            validation_split: Fraction for validation
        """
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train, np.log1p(y_train),
            eval_set=[(X_val, np.log1p(y_val))],
            verbose=False,
        )
    
    def predict(self, data: dict[str, Any]) -> EngagementForecast:
        """
        Predict final engagement from current state.
        
        Args:
            data: Dictionary containing:
                - current_engagement: int
                - time_elapsed_hours: float
                - time_remaining_hours: float
                - platform: str
                - metric: str
                - follower_count: int (optional)
        
        Returns:
            EngagementForecast with prediction and bounds
        """
        features = self._extract_features(data)
        
        if self.model is None:
            predicted_log = self._simple_forecast(data)
        else:
            predicted_log = self.model.predict(features)[0]
        
        predicted = int(np.expm1(predicted_log))
        
        std_factor = 0.2
        lower = int(predicted * (1 - std_factor))
        upper = int(predicted * (1 + std_factor))
        
        current = data.get("current_engagement", 0)
        growth_rate = (predicted - current) / max(current, 1)
        
        return EngagementForecast(
            predicted_value=predicted,
            lower_bound=lower,
            upper_bound=upper,
            confidence_interval=0.95,
            growth_rate=growth_rate,
        )
    
    def _simple_forecast(self, data: dict[str, Any]) -> float:
        """Simple heuristic forecast when no model is loaded."""
        current = data.get("current_engagement", 0)
        elapsed = data.get("time_elapsed_hours", 1)
        remaining = data.get("time_remaining_hours", 23)
        
        velocity = current / max(elapsed, 0.1)
        decay_factor = 0.7
        estimated_additional = velocity * remaining * decay_factor
        
        return np.log1p(current + estimated_additional)
    
    @classmethod
    def load(cls, path: str | Path) -> "EngagementForecaster":
        """Load a pre-trained model from disk."""
        forecaster = cls()
        forecaster.model = xgb.XGBRegressor()
        forecaster.model.load_model(path)
        return forecaster
    
    def save(self, path: str | Path):
        """Save model to disk."""
        if self.model is not None:
            self.model.save_model(path)
