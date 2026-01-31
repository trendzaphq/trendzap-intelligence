"""
Trend Detector Model

Identifies emerging trends using clustering and time-series analysis.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Trend:
    """A detected trend."""
    
    topic: str
    keywords: list[str]
    score: float
    velocity: float
    volume: int
    sentiment: float
    platform: str | None


@dataclass
class TrendReport:
    """Collection of detected trends."""
    
    trends: list[Trend]
    timestamp: str
    period_hours: int


class TrendDetector:
    """
    Detects emerging trends using clustering.
    
    Approach:
    1. Collect recent post texts
    2. TF-IDF vectorization
    3. DBSCAN clustering
    4. Rank clusters by growth velocity
    """
    
    def __init__(
        self,
        min_cluster_size: int = 5,
        eps: float = 0.3,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.clusterer = DBSCAN(
            eps=eps,
            min_samples=min_cluster_size,
            metric="cosine",
        )
    
    def _extract_keywords(
        self,
        texts: list[str],
        indices: np.ndarray,
        n_keywords: int = 5,
    ) -> list[str]:
        """Extract top keywords from a cluster."""
        cluster_texts = [texts[i] for i in indices]
        combined = " ".join(cluster_texts)
        
        words = combined.lower().split()
        word_freq: dict[str, int] = {}
        for word in words:
            if len(word) > 3 and not word.startswith(("http", "@", "#")):
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        return [w[0] for w in sorted_words[:n_keywords]]
    
    def detect(
        self,
        posts: list[dict[str, Any]],
        period_hours: int = 24,
    ) -> TrendReport:
        """
        Detect trends from a collection of posts.
        
        Args:
            posts: List of post dictionaries with 'text', 'engagement', 'timestamp'
            period_hours: Time window for trend detection
        
        Returns:
            TrendReport with detected trends
        """
        if len(posts) < 10:
            return TrendReport(
                trends=[],
                timestamp=str(np.datetime64("now")),
                period_hours=period_hours,
            )
        
        texts = [p.get("text", "") for p in posts]
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        labels = self.clusterer.fit_predict(tfidf_matrix)
        
        unique_labels = set(labels) - {-1}
        
        trends = []
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_posts = [posts[i] for i in cluster_indices]
            
            keywords = self._extract_keywords(texts, cluster_indices)
            
            total_engagement = sum(p.get("engagement", 0) for p in cluster_posts)
            avg_engagement = total_engagement / len(cluster_posts)
            
            velocity = len(cluster_posts) / period_hours
            
            score = np.log1p(total_engagement) * velocity * len(cluster_posts) / 100
            
            trend = Trend(
                topic=keywords[0] if keywords else "Unknown",
                keywords=keywords,
                score=float(score),
                velocity=float(velocity),
                volume=len(cluster_posts),
                sentiment=0.0,
                platform=cluster_posts[0].get("platform"),
            )
            trends.append(trend)
        
        trends.sort(key=lambda t: -t.score)
        
        return TrendReport(
            trends=trends[:20],
            timestamp=str(np.datetime64("now")),
            period_hours=period_hours,
        )
