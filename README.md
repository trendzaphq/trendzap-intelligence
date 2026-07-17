# TrendZap Intelligence

> The AI/ML signal engine behind [TrendZap](https://trendzap.xyz) — a decentralized prediction market for social media virality. Bettors take OVER/UNDER positions on whether posts and trends hit engagement thresholds; this service produces the probability signals, engagement forecasts, and bot-detection that keep those markets priced fairly and resolved honestly.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%C2%B7%20Llama%203.3%2070B-f55036)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live platform:** [app.trendzap.xyz](https://app.trendzap.xyz) · **Docs:** [docs.trendzap.xyz](https://docs.trendzap.xyz)

---

## What it does

In a prediction market for virality, three questions decide everything:

1. **"Will this post go viral?"** — needed to seed market odds (`/predict/virality` returns a probability and an OVER/UNDER call against a threshold)
2. **"Where will engagement end up?"** — needed for market creation and pricing (`/predict/engagement` forecasts final counts with confidence bounds)
3. **"Is this engagement real?"** — needed for fair resolution; bought bots must not settle a market (`/detect/anomaly` flags bot swarms, engagement farms, and coordinated campaigns)

On top of those signal endpoints, a **Groq-powered LLM layer** (Llama 3.3 70B via the OpenAI-compatible SDK) turns raw numbers into human-readable analysis: why a post is likely to move, how long a trend will last, and plain-language explanations of detected anomalies.

## Architecture

```
                        ┌──────────────────────────────┐
   TrendZap app ──────► │      FastAPI  (src/api)      │ ◄────── trendzap-oracle / risk
                        └──────────────┬───────────────┘
                     ┌─────────────────┼──────────────────┐
                     ▼                 ▼                  ▼
          ┌──────────────────┐  ┌─────────────┐  ┌────────────────┐
          │   Model classes  │  │  AIAnalyzer │  │  Redis cache   │
          │ virality · engmt │  │ Groq LLM    │  │ 5-min TTL,     │
          │ anomaly · trends │  │ (async)     │  │ degrades       │
          └──────────────────┘  └─────────────┘  │ gracefully     │
                                                 └────────────────┘
```

- **Stateless FastAPI service**, containerized with Docker, deployed on **Railway** (health-checked, auto-restarting — see `railway.toml`).
- **Redis caching** on all LLM endpoints (SHA-256 of the request payload as key, 5-minute TTL). Redis is optional: if it's unreachable the service silently skips the cache instead of failing requests.
- **Async LLM client** so slow upstream calls never block the event loop.

## Models

| Model | Approach | Status |
| ----- | -------- | ------ |
| `ViralityPredictor` | LSTM + attention over MiniLM sentence embeddings + 15 engineered numerical features (log-scaled engagement, platform one-hots, timing, content signals) | Architecture implemented end-to-end; head not yet trained on production data |
| `EngagementForecaster` | XGBoost regressor over growth-curve features | Serves a heuristic growth-curve forecast today; XGBoost train/predict path is wired and ready for data |
| `AnomalyDetector` | Isolation Forest + rule-based signal filters (velocity spikes, new-account ratios, geo clustering, engagement/follower outliers) | Rule-based detection active in production; Isolation Forest path ready for training |
| `TrendDetector` | TF-IDF vectorization + DBSCAN clustering, ranked by growth velocity | Fully functional (unsupervised — no training required) |

> **Honest status:** the platform is pre-training-data. Models that need supervised training ship as complete architectures with deterministic heuristic fallbacks, and the production insights users see today come from those heuristics plus the Groq analysis layer. Collecting labeled engagement outcomes from live markets — and training on them — is the next milestone (see [Roadmap](#roadmap)).

## API

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/health` | GET | Health check (reports AI provider, model, Redis status) |
| `/api/v1/predict/virality` | POST | Viral probability + OVER/UNDER call vs. a threshold |
| `/api/v1/predict/engagement` | POST | Final engagement forecast with bounds |
| `/api/v1/detect/anomaly` | POST | Artificial-engagement detection with named signals |
| `/api/v1/trends` | GET | Detected trending topics |
| `/api/v1/ai/analyze-post` | POST | LLM analysis: strengths, audience, optimization suggestions |
| `/api/v1/ai/analyze-trend` | POST | LLM analysis: trend drivers, longevity, opportunities/risks |
| `/api/v1/ai/explain-anomaly` | POST | LLM plain-language explanation of a detected anomaly |

### Example

```bash
curl -X POST http://localhost:8000/api/v1/predict/virality \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "twitter",
    "post_url": "https://twitter.com/user/status/123",
    "post_text": "Just launched our new product! 🚀",
    "follower_count": 50000,
    "initial_likes": 500,
    "initial_retweets": 100,
    "threshold": 100000,
    "metric": "likes"
  }'
```

```bash
curl -X POST http://localhost:8000/api/v1/ai/analyze-post \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "tiktok",
    "post_text": "POV: your side project just hit the front page",
    "follower_count": 12000,
    "current_likes": 3400,
    "current_shares": 800
  }'
```

## Quick start

```bash
git clone https://github.com/trendzaphq/trendzap-intelligence.git
cd trendzap-intelligence

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env   # add your GROQ_API_KEY

uvicorn src.api.main:app --reload --port 8000
```

Requires Python 3.11+. Redis is optional (`REDIS_URL` in `.env`) — the service runs without it.

### Docker

```bash
docker build -t trendzap-intelligence .
docker run -p 8000:8000 --env-file .env trendzap-intelligence
```

The image binds to Railway's injected `$PORT` in production and falls back to 8000 locally.

## Project structure

```
trendzap-intelligence/
├── src/
│   ├── trendzap_intelligence/
│   │   ├── __init__.py
│   │   ├── config.py              # Settings + Groq/OpenAI client factories (sync & async)
│   │   ├── ai_analyzer.py         # Groq LLM analysis (posts, trends, anomaly explanations)
│   │   └── models/
│   │       ├── virality_predictor.py
│   │       ├── engagement_forecaster.py
│   │       ├── anomaly_detector.py
│   │       └── trend_detector.py
│   └── api/
│       └── main.py                # FastAPI app: endpoints, schemas, Redis caching
├── tests/
│   └── test_models.py
├── Dockerfile
├── railway.toml                   # Railway deploy config (healthcheck, restart policy)
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Role in the TrendZap platform

| Repository | Relationship |
| ---------- | ------------ |
| `trendzap_app` | Frontend consumes the AI analysis endpoints for market insights |
| `trendzap-oracle` | Engagement metrics feed anomaly detection before market resolution |
| `trendzap-risk` | Bot-detection signals inform the risk engine |
| `trendzap-contracts` | Markets settled on-chain (Avalanche) resolve against oracle data this service helps validate |

## Roadmap

- [ ] Data pipeline: collect labeled engagement outcomes from resolved markets
- [ ] Training scripts + evaluation harness for the virality and forecasting models
- [ ] Model persistence and versioning (currently `save`/`load` exist on each model class, no registry)
- [ ] Feature engineering module shared across models

## Development

```bash
pip install -r requirements-dev.txt

pytest tests/          # tests
ruff check .           # lint
black --check .        # format
mypy src/              # types
```

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <strong>TrendZap Intelligence 🧠</strong><br>
  The signal engine for social prediction markets
</p>
