"""
Virality Predictor Model

Predicts the probability of a social media post going viral.
Uses LSTM with attention mechanism for sequence modeling.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class ViralityPrediction:
    """Result of a virality prediction."""

    probability: float
    #: How the number was produced: "model" (trained weights) or "heuristic".
    #: Callers must surface this — a heuristic estimate is not a model prediction.
    method: str
    #: Spread of the estimate. Named honestly: this is NOT model certainty.
    #: The previous field computed abs(p - 0.5) * 2, i.e. distance from the midpoint,
    #: and reported it as "confidence" — an untrained net outputting 0.95 scored 0.9.
    dispersion: float
    features_importance: dict[str, float] | None
    threshold_estimates: dict[int, float]


class AttentionLayer(nn.Module):
    """Self-attention layer for sequence processing."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class ViralityPredictor(nn.Module):
    """
    Predicts viral probability using LSTM + Attention.

    Architecture:
    - Text embeddings from pre-trained BERT
    - Numerical features (followers, engagement velocity, etc.)
    - LSTM with attention for temporal patterns
    - Final dense layers for classification
    """

    # Hidden size of TEXT_ENCODER_NAME. all-MiniLM-L6-v2 is 384-dimensional; the
    # constructor previously defaulted to 768, so text_projection was built as
    # Linear(768, …) and every predict() call raised a shape error on the matmul.
    TEXT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_EMBEDDING_DIM = 384

    def __init__(
        self,
        text_embedding_dim: int | None = None,
        numerical_features: int = 15,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        text_embedding_dim = text_embedding_dim or self.TEXT_EMBEDDING_DIM

        self.text_projection = nn.Linear(text_embedding_dim, hidden_size)
        self.numerical_projection = nn.Linear(numerical_features, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.attention = AttentionLayer(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.tokenizer = None
        self.text_encoder = None

        # True only after load() restores trained weights. The module is otherwise
        # randomly initialised, and a random network's output is not a prediction.
        self.is_trained = False

    def _load_text_encoder(self):
        """Load the pre-trained sentence encoder used for text features."""
        if self.text_encoder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.TEXT_ENCODER_NAME)
            self.text_encoder = AutoModel.from_pretrained(self.TEXT_ENCODER_NAME)
            self.text_encoder.eval()

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text using pre-trained model."""
        self._load_text_encoder()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def forward(
        self,
        text_embeddings: torch.Tensor,
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            text_embeddings: (batch_size, text_embedding_dim)
            numerical_features: (batch_size, num_features)

        Returns:
            Viral probability (batch_size, 1)
        """
        text_proj = self.text_projection(text_embeddings)
        num_proj = self.numerical_projection(numerical_features)

        combined = torch.cat([text_proj, num_proj], dim=-1)
        combined = combined.unsqueeze(1)

        lstm_out, _ = self.lstm(combined)
        context, attention_weights = self.attention(lstm_out)

        probability = self.classifier(context)

        return probability

    def predict(self, features: dict[str, Any]) -> ViralityPrediction:
        """
        Estimate the probability that a post goes viral.

        If trained weights have been loaded, runs the network. Otherwise falls back to
        an explicit, documented heuristic and labels the result `method="heuristic"`.

        This module previously ran the randomly-initialised network unconditionally and
        returned its output as a `probability` with a `confidence` — on a platform where
        users bet money on the answer. There is no checkpoint in the repository and no
        training script, so that number was never meaningful.
        """
        if not self.is_trained:
            return self._heuristic_predict(features)

        self.eval()

        text_emb = self.encode_text([features.get("post_text", "")])
        numerical = self._numerical_features(features)

        with torch.no_grad():
            probability = self.forward(text_emb, numerical).item()

        return ViralityPrediction(
            probability=probability,
            method="model",
            dispersion=abs(probability - 0.5) * 2,
            features_importance=None,
            threshold_estimates=self._threshold_estimates(probability),
        )

    def _heuristic_predict(self, features: dict[str, Any]) -> ViralityPrediction:
        """
        Transparent fallback used when no trained weights are loaded.

        Engagement rate relative to follower count is the strongest widely-agreed
        signal available without a trained model, so the estimate is built from that
        and squashed into (0, 1). It is a rule of thumb, not a learned prediction, and
        is reported as such via `method`.
        """
        followers = max(int(features.get("follower_count", 0) or 0), 1)
        likes = int(features.get("initial_likes", 0) or 0)
        shares = int(features.get("initial_retweets", 0) or 0)

        # Shares travel further than likes, so weight them more heavily.
        engagement = likes + shares * 3
        engagement_rate = engagement / followers

        # ~2% engagement maps to roughly even odds; the log keeps large accounts from
        # saturating the estimate.
        score = math.log1p(engagement_rate / 0.02)
        probability = 1 / (1 + math.exp(-score))
        probability = min(max(probability, 0.01), 0.99)

        return ViralityPrediction(
            probability=probability,
            method="heuristic",
            dispersion=abs(probability - 0.5) * 2,
            # Feature attributions require a trained model. This used to return five
            # hardcoded constants that never varied with the input.
            features_importance=None,
            threshold_estimates=self._threshold_estimates(probability),
        )

    def _numerical_features(self, features: dict[str, Any]) -> torch.Tensor:
        text = features.get("post_text", "")
        platform = features.get("platform")
        return torch.tensor([[
            np.log1p(features.get("follower_count", 0)),
            np.log1p(features.get("initial_likes", 0)),
            np.log1p(features.get("initial_retweets", 0)),
            features.get("post_hour", 12) / 24.0,
            features.get("day_of_week", 0) / 7.0,
            1.0 if platform in ("twitter", "x") else 0.0,
            1.0 if platform == "tiktok" else 0.0,
            1.0 if platform == "instagram" else 0.0,
            1.0 if platform == "youtube" else 0.0,
            len(text) / 280.0,
            text.count("#") / 10.0,
            text.count("@") / 10.0,
            1.0 if "http" in text else 0.0,
            1.0 if any(e in text for e in ["\U0001F525", "\U0001F680", "\U0001F4AF"]) else 0.0,
            features.get("account_age_days", 365) / 3650.0,
        ]], dtype=torch.float32)

    @staticmethod
    def _threshold_estimates(probability: float) -> dict[int, float]:
        """Rough scaling of the estimate across common thresholds."""
        return {
            10_000: min(1.0, probability * 1.2),
            100_000: probability,
            1_000_000: probability * 0.7,
        }

    @classmethod
    def load(cls, path: str | Path) -> "ViralityPredictor":
        """Load a pre-trained model from disk."""
        model = cls()
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.is_trained = True
        return model

    def save(self, path: str | Path):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
