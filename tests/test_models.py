"""Tests for TrendZap Intelligence models."""

import pytest
from trendzap_intelligence.models import (
    ViralityPredictor,
    EngagementForecaster,
    AnomalyDetector,
)


class TestViralityPredictor:
    """Tests for ViralityPredictor."""
    
    def test_predict_returns_valid_probability(self):
        """Prediction should return probability between 0 and 1."""
        predictor = ViralityPredictor()
        
        result = predictor.predict({
            "platform": "twitter",
            "post_text": "This is a test post",
            "follower_count": 10000,
            "initial_likes": 100,
            "initial_retweets": 20,
        })
        
        assert 0 <= result.probability <= 1
        assert 0 <= result.confidence <= 1
    
    def test_predict_high_engagement_higher_probability(self):
        """Higher initial engagement should generally mean higher probability."""
        predictor = ViralityPredictor()
        
        low_result = predictor.predict({
            "platform": "twitter",
            "post_text": "Test post",
            "follower_count": 1000,
            "initial_likes": 10,
        })
        
        high_result = predictor.predict({
            "platform": "twitter",
            "post_text": "Test post",
            "follower_count": 1000000,
            "initial_likes": 50000,
        })
        
        assert high_result.probability >= low_result.probability


class TestEngagementForecaster:
    """Tests for EngagementForecaster."""
    
    def test_predict_returns_positive_value(self):
        """Forecast should return positive engagement value."""
        forecaster = EngagementForecaster()
        
        result = forecaster.predict({
            "platform": "twitter",
            "current_engagement": 1000,
            "time_elapsed_hours": 6,
            "time_remaining_hours": 18,
        })
        
        assert result.predicted_value > 0
        assert result.lower_bound <= result.predicted_value <= result.upper_bound
    
    def test_predict_more_time_higher_value(self):
        """More time remaining should allow for higher predictions."""
        forecaster = EngagementForecaster()
        
        short_result = forecaster.predict({
            "platform": "twitter",
            "current_engagement": 1000,
            "time_elapsed_hours": 23,
            "time_remaining_hours": 1,
        })
        
        long_result = forecaster.predict({
            "platform": "twitter",
            "current_engagement": 1000,
            "time_elapsed_hours": 1,
            "time_remaining_hours": 23,
        })
        
        assert long_result.predicted_value >= short_result.predicted_value


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""
    
    def test_normal_engagement_not_anomaly(self):
        """Normal engagement patterns should not be flagged."""
        detector = AnomalyDetector()
        
        result = detector.detect({
            "engagement_velocity": 1.0,
            "engagement_count": 500,
            "follower_count": 10000,
            "new_account_ratio": 0.1,
            "single_region_ratio": 0.3,
        })
        
        assert not result.is_anomaly
    
    def test_high_velocity_flagged(self):
        """Extremely high velocity should be flagged."""
        detector = AnomalyDetector()
        
        result = detector.detect({
            "engagement_velocity": 100.0,
            "engagement_count": 10000,
            "follower_count": 1000,
            "new_account_ratio": 0.9,
            "single_region_ratio": 0.95,
        })
        
        assert result.is_anomaly
        assert len(result.signals) > 0
