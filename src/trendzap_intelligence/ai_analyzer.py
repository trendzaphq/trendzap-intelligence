"""
AI Analysis Service

Uses Groq (via OpenAI-compatible SDK) for LLM-powered social media analysis.
Provides AI-driven insights on top of the existing ML models.
"""

import json
import structlog
from typing import Any

from .config import get_ai_client, settings

logger = structlog.get_logger(__name__)

# System prompt for the TrendZap AI analyst
SYSTEM_PROMPT = """You are TrendZap AI, an expert social media analyst. You analyze social media
posts, engagement data, and trends to provide actionable insights.

Your capabilities:
- Analyze why content goes viral
- Predict engagement patterns
- Identify fake/bot engagement signals
- Suggest content optimization strategies
- Analyze trending topics and their trajectory

Always respond with structured, data-driven insights. Be concise and actionable."""


class AIAnalyzer:
    """
    AI-powered social media analyzer using Groq LLM.

    Uses the OpenAI-compatible SDK to communicate with Groq's API,
    providing LLM-based analysis capabilities on top of ML model predictions.
    """

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy-initialize the AI client."""
        if self._client is None:
            self._client = get_ai_client()
        return self._client

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return settings.ai_model

    async def analyze_post(self, post_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze a social media post using the LLM for deeper insights.

        Args:
            post_data: Dictionary with post details (text, platform, metrics, etc.)

        Returns:
            Dictionary with AI-generated analysis
        """
        prompt = f"""Analyze this social media post and provide insights:

Platform: {post_data.get('platform', 'unknown')}
Post Text: {post_data.get('post_text', 'N/A')}
Follower Count: {post_data.get('follower_count', 'N/A')}
Current Likes: {post_data.get('current_likes', 'N/A')}
Current Shares: {post_data.get('current_shares', 'N/A')}

Provide your analysis as JSON with these fields:
- "virality_assessment": brief assessment of viral potential (string)
- "content_strengths": list of what makes this post engaging
- "improvement_suggestions": list of suggestions to improve reach
- "predicted_audience": description of likely audience
- "best_posting_times": list of optimal posting windows
- "hashtag_suggestions": list of recommended hashtags"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.choices[0].message.content

            # Try to parse as JSON, fall back to raw text
            try:
                # Handle cases where LLM wraps JSON in markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return json.loads(content)
            except (json.JSONDecodeError, IndexError):
                return {"analysis": content}

        except Exception as e:
            logger.error("ai_analysis_failed", error=str(e))
            return {"error": f"AI analysis failed: {str(e)}"}

    async def analyze_trend(self, trend_data: dict[str, Any]) -> dict[str, Any]:
        """
        Use LLM to provide deeper insights on a detected trend.

        Args:
            trend_data: Dictionary with trend details (topic, keywords, metrics)

        Returns:
            Dictionary with AI-generated trend insights
        """
        prompt = f"""Analyze this social media trend:

Topic: {trend_data.get('topic', 'unknown')}
Keywords: {', '.join(trend_data.get('keywords', []))}
Volume: {trend_data.get('volume', 'N/A')} posts
Velocity: {trend_data.get('velocity', 'N/A')} posts/hour
Platform: {trend_data.get('platform', 'cross-platform')}

Provide your analysis as JSON with these fields:
- "trend_summary": brief summary of what's driving this trend (string)
- "longevity_prediction": how long this trend is likely to last (string)
- "audience_segments": list of audience segments engaging with this trend
- "brand_opportunities": list of ways brands could engage with this trend
- "risk_factors": list of potential risks in engaging with this trend"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.choices[0].message.content

            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return json.loads(content)
            except (json.JSONDecodeError, IndexError):
                return {"analysis": content}

        except Exception as e:
            logger.error("ai_trend_analysis_failed", error=str(e))
            return {"error": f"AI trend analysis failed: {str(e)}"}

    async def explain_anomaly(self, anomaly_data: dict[str, Any]) -> dict[str, Any]:
        """
        Use LLM to explain detected anomalies in human-readable terms.

        Args:
            anomaly_data: Dictionary with anomaly detection results

        Returns:
            Dictionary with AI-generated explanation
        """
        prompt = f"""Explain this social media engagement anomaly to a non-technical user:

Is Anomaly: {anomaly_data.get('is_anomaly', False)}
Anomaly Score: {anomaly_data.get('anomaly_score', 0):.2f}
Anomaly Type: {anomaly_data.get('anomaly_type', 'none')}
Signals Detected: {', '.join(anomaly_data.get('signals', []))}

Engagement Velocity: {anomaly_data.get('engagement_velocity', 'N/A')}
Engagement Count: {anomaly_data.get('engagement_count', 'N/A')}
Follower Count: {anomaly_data.get('follower_count', 'N/A')}

Provide your analysis as JSON with these fields:
- "explanation": plain-language explanation of what was detected (string)
- "severity": "low", "medium", or "high" (string)
- "likely_cause": most probable cause of the anomaly (string)
- "recommended_actions": list of actions the user should take
- "confidence_note": note about the confidence level of this detection (string)"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )

            content = response.choices[0].message.content

            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return json.loads(content)
            except (json.JSONDecodeError, IndexError):
                return {"analysis": content}

        except Exception as e:
            logger.error("ai_anomaly_explanation_failed", error=str(e))
            return {"error": f"AI anomaly explanation failed: {str(e)}"}
