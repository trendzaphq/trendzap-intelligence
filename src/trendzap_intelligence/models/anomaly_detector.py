"""
Anomaly Detector Model

Detects artificial/bot engagement patterns.
Uses Isolation Forest for unsupervised anomaly detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str | None
    confidence: float
    signals: list[str]


class AnomalyDetector:
    """
    Detects artificial engagement using Isolation Forest.
    
    Anomaly Signals:
    - Unusual engagement velocity spikes
    - Low account age / high engagement ratio
    - Geographic clustering anomalies
    - Timing pattern anomalies
    - Engagement-to-follower ratio outliers
    """
    
    ANOMALY_TYPES = [
        "bot_swarm",
        "coordinated_campaign",
        "engagement_farm",
        "velocity_spike",
        "ratio_anomaly",
    ]
    
    THRESHOLDS = {
        "velocity_spike": 10.0,
        "engagement_follower_ratio": 0.5,
        "new_account_ratio": 0.8,
        "single_region_ratio": 0.9,
    }
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.is_fitted = False
    
    def _extract_features(self, data: dict[str, Any]) -> np.ndarray:
        """Extract features for anomaly detection."""
        features = [
            data.get("engagement_velocity", 0),
            data.get("velocity_acceleration", 0),
            np.log1p(data.get("engagement_count", 0)),
            np.log1p(data.get("follower_count", 1)),
            data.get("engagement_count", 0) / max(data.get("follower_count", 1), 1),
            data.get("new_account_ratio", 0),
            data.get("single_region_ratio", 0),
            data.get("burst_count", 0),
            data.get("engagement_variance", 0),
            data.get("hour_entropy", 1.0),
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _apply_rules(self, data: dict[str, Any]) -> list[str]:
        """Apply rule-based anomaly detection."""
        signals = []
        
        velocity = data.get("engagement_velocity", 0)
        if velocity > self.THRESHOLDS["velocity_spike"]:
            signals.append(f"Velocity spike detected: {velocity:.1f}x normal")
        
        engagement = data.get("engagement_count", 0)
        followers = data.get("follower_count", 1)
        ratio = engagement / max(followers, 1)
        if ratio > self.THRESHOLDS["engagement_follower_ratio"]:
            signals.append(f"High engagement/follower ratio: {ratio:.2f}")
        
        new_ratio = data.get("new_account_ratio", 0)
        if new_ratio > self.THRESHOLDS["new_account_ratio"]:
            signals.append(f"High new account ratio: {new_ratio:.0%}")
        
        region_ratio = data.get("single_region_ratio", 0)
        if region_ratio > self.THRESHOLDS["single_region_ratio"]:
            signals.append(f"Geographic clustering: {region_ratio:.0%} from single region")
        
        return signals
    
    def _classify_anomaly(self, signals: list[str]) -> str | None:
        """Classify the type of anomaly based on signals."""
        if not signals:
            return None
        
        signal_text = " ".join(signals).lower()
        
        if "new account" in signal_text and "velocity" in signal_text:
            return "bot_swarm"
        elif "geographic" in signal_text:
            return "coordinated_campaign"
        elif "velocity spike" in signal_text:
            return "velocity_spike"
        elif "engagement/follower" in signal_text:
            return "engagement_farm"
        else:
            return "ratio_anomaly"
    
    def train(self, X: np.ndarray):
        """
        Train the anomaly detector on normal engagement data.
        
        Args:
            X: Feature matrix of normal engagement patterns
        """
        self.model.fit(X)
        self.is_fitted = True
    
    def detect(self, data: dict[str, Any]) -> AnomalyResult:
        """
        Detect anomalies in engagement data.
        
        Args:
            data: Dictionary containing:
                - engagement_velocity: float (engagements per minute)
                - engagement_count: int
                - follower_count: int
                - new_account_ratio: float (0-1)
                - single_region_ratio: float (0-1)
                - burst_count: int
        
        Returns:
            AnomalyResult with detection details
        """
        rule_signals = self._apply_rules(data)
        
        features = self._extract_features(data)
        
        if self.is_fitted:
            ml_score = -self.model.score_samples(features)[0]
            ml_anomaly = self.model.predict(features)[0] == -1
        else:
            ml_score = 0.0
            ml_anomaly = False
        
        rule_score = min(len(rule_signals) / 3, 1.0)
        
        combined_score = 0.6 * rule_score + 0.4 * min(ml_score / 0.5, 1.0)
        is_anomaly = combined_score > 0.5 or len(rule_signals) >= 2
        
        anomaly_type = self._classify_anomaly(rule_signals) if is_anomaly else None
        
        confidence = abs(combined_score - 0.5) * 2
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=combined_score,
            anomaly_type=anomaly_type,
            confidence=confidence,
            signals=rule_signals,
        )
    
    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        """Load a pre-trained model from disk."""
        detector = cls()
        detector.model = joblib.load(path)
        detector.is_fitted = True
        return detector
    
    def save(self, path: str | Path):
        """Save model to disk."""
        joblib.dump(self.model, path)
