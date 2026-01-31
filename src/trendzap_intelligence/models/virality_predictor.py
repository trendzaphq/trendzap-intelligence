"""
Virality Predictor Model

Predicts the probability of a social media post going viral.
Uses LSTM with attention mechanism for sequence modeling.
"""

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
    confidence: float
    features_importance: dict[str, float]
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
    
    def __init__(
        self,
        text_embedding_dim: int = 768,
        numerical_features: int = 15,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
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
    
    def _load_text_encoder(self):
        """Load pre-trained BERT model for text encoding."""
        if self.text_encoder is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = AutoModel.from_pretrained(model_name)
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
        Make a prediction from raw features.
        
        Args:
            features: Dictionary containing:
                - platform: str
                - post_text: str
                - follower_count: int
                - initial_likes: int
                - initial_retweets: int
                - post_hour: int
                - day_of_week: int
        
        Returns:
            ViralityPrediction with probability and metadata
        """
        self.eval()
        
        text_emb = self.encode_text([features.get("post_text", "")])
        
        numerical = torch.tensor([[
            np.log1p(features.get("follower_count", 0)),
            np.log1p(features.get("initial_likes", 0)),
            np.log1p(features.get("initial_retweets", 0)),
            features.get("post_hour", 12) / 24.0,
            features.get("day_of_week", 0) / 7.0,
            1.0 if features.get("platform") == "twitter" else 0.0,
            1.0 if features.get("platform") == "tiktok" else 0.0,
            1.0 if features.get("platform") == "instagram" else 0.0,
            1.0 if features.get("platform") == "youtube" else 0.0,
            len(features.get("post_text", "")) / 280.0,
            features.get("post_text", "").count("#") / 10.0,
            features.get("post_text", "").count("@") / 10.0,
            1.0 if "http" in features.get("post_text", "") else 0.0,
            1.0 if any(e in features.get("post_text", "") for e in ["🔥", "🚀", "💯"]) else 0.0,
            features.get("account_age_days", 365) / 3650.0,
        ]], dtype=torch.float32)
        
        with torch.no_grad():
            probability = self.forward(text_emb, numerical).item()
        
        confidence = abs(probability - 0.5) * 2
        
        return ViralityPrediction(
            probability=probability,
            confidence=confidence,
            features_importance={
                "follower_count": 0.25,
                "initial_engagement": 0.30,
                "post_content": 0.20,
                "timing": 0.15,
                "platform": 0.10,
            },
            threshold_estimates={
                10000: probability * 1.2,
                100000: probability,
                1000000: probability * 0.7,
            },
        )
    
    @classmethod
    def load(cls, path: str | Path) -> "ViralityPredictor":
        """Load a pre-trained model from disk."""
        model = cls()
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def save(self, path: str | Path):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
