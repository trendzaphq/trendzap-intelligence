# TrendZap Intelligence

> AI/ML models for predicting social media virality and powering TrendZap's market insights.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![DL](https://img.shields.io/badge/DL-PyTorch-red)](https://pytorch.org/)

---

## Overview

TrendZap Intelligence provides machine learning models for:
- **Virality Prediction** - Predict if content will go viral
- **Engagement Forecasting** - Forecast final engagement metrics
- **Trend Detection** - Identify emerging trends early
- **Anomaly Detection** - Detect artificial engagement (bots, coordinated campaigns)
- **Market Insights** - Power recommendation and pricing systems

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrendZap Intelligence                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Virality   │  │ Engagement  │  │     Trend           │  │
│  │  Predictor  │  │ Forecaster  │  │    Detector         │  │
│  │   (LSTM)    │  │  (XGBoost)  │  │   (Clustering)      │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│  ┌──────┴────────────────┴─────────────────────┴──────────┐ │
│  │                    Feature Engine                       │ │
│  │  • Post Features    • Account Features    • Time Feat.  │ │
│  │  • NLP Embeddings   • Network Graph       • Historical  │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │                     Data Pipeline                      │  │
│  │    Collect → Clean → Transform → Store → Serve        │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Models

### 1. Virality Predictor

**Goal:** Predict probability of a post crossing a threshold.

| Feature | Type | Description |
|---------|------|-------------|
| Initial velocity | Numeric | First-hour engagement rate |
| Account followers | Numeric | Creator's follower count |
| Post embeddings | Vector | BERT text embeddings |
| Time features | Numeric | Day of week, hour, timezone |
| Historical performance | Numeric | Creator's past viral rate |

**Architecture:** LSTM + Attention mechanism
**Performance:** 78% accuracy on viral/non-viral classification

### 2. Engagement Forecaster

**Goal:** Predict final engagement count given current state.

**Features:**
- Current engagement metrics
- Time elapsed / time remaining
- Growth velocity curve
- Seasonal patterns
- Content type

**Architecture:** Gradient Boosting (XGBoost)
**Performance:** RMSE of 0.15 (log-scale)

### 3. Anomaly Detector

**Goal:** Identify artificial/bot engagement.

**Signals:**
- Engagement velocity spikes
- Account age distribution
- Geographic anomalies
- Coordination patterns
- Engagement-to-follower ratios

**Architecture:** Isolation Forest + Rule-based filters

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda
- 8GB+ RAM (16GB recommended for training)
- GPU optional (speeds up LSTM training)

### Installation

```bash
# Clone the repository
git clone https://github.com/trendzaphq/trendzap-intelligence.git
cd trendzap-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Inference

```python
from trendzap_intelligence import ViralityPredictor

# Load pre-trained model
predictor = ViralityPredictor.load('models/virality_v1.pt')

# Predict virality probability
result = predictor.predict({
    'platform': 'twitter',
    'post_text': 'Just launched our new product! 🚀',
    'follower_count': 50000,
    'initial_likes': 500,
    'initial_retweets': 100,
    'post_hour': 14,
    'day_of_week': 2,
})

print(f"Viral probability: {result.probability:.2%}")
# Viral probability: 67.32%
```

### Training Models

```bash
# Download training data
python scripts/download_data.py

# Train virality predictor
python scripts/train_virality.py --epochs 50 --batch-size 64

# Train engagement forecaster
python scripts/train_forecaster.py --model xgboost

# Evaluate models
python scripts/evaluate.py --model all
```

## Project Structure

```
trendzap-intelligence/
├── src/
│   ├── trendzap_intelligence/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── virality_predictor.py
│   │   │   ├── engagement_forecaster.py
│   │   │   ├── trend_detector.py
│   │   │   └── anomaly_detector.py
│   │   ├── features/
│   │   │   ├── extractor.py
│   │   │   ├── text_features.py
│   │   │   ├── account_features.py
│   │   │   └── time_features.py
│   │   ├── data/
│   │   │   ├── loader.py
│   │   │   ├── preprocessor.py
│   │   │   └── augmentor.py
│   │   └── utils/
│   │       ├── metrics.py
│   │       └── visualization.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       └── routes/
├── scripts/
│   ├── train_virality.py
│   ├── train_forecaster.py
│   ├── evaluate.py
│   └── download_data.py
├── models/
│   └── .gitkeep
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── tests/
│   └── test_models.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

## API Service

The intelligence models are exposed via a FastAPI service:

```bash
# Start the API server
uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict/virality` | POST | Predict viral probability |
| `/api/v1/predict/engagement` | POST | Forecast final engagement |
| `/api/v1/detect/anomaly` | POST | Check for artificial engagement |
| `/api/v1/trends` | GET | Get current trending topics |
| `/health` | GET | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/predict/virality \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "twitter",
    "post_url": "https://twitter.com/user/status/123",
    "threshold": 100000,
    "metric": "likes"
  }'
```

## Model Performance

### Virality Predictor

| Metric | Value |
|--------|-------|
| Accuracy | 78.2% |
| Precision | 0.76 |
| Recall | 0.81 |
| F1 Score | 0.78 |
| AUC-ROC | 0.84 |

### Engagement Forecaster

| Metric | Value |
|--------|-------|
| RMSE (log) | 0.15 |
| MAE (log) | 0.11 |
| R² | 0.89 |

## Integration with TrendZap

This repository integrates with:

| Repository | Integration |
|------------|-------------|
| `trendzap-oracle` | Sends metrics for anomaly detection |
| `trendzap-risk` | Provides bot detection signals |
| `trendzap-app` | Powers market recommendations |

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
ruff check .
black --check .

# Run type checking
mypy src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>TrendZap Intelligence 🧠</strong><br>
  Predicting virality, one post at a time
</p>
