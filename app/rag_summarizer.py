from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from app.article_extractor import extract_article_text
from app.storage import write_json
import httpx
import re


ROOT = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")


SYSTEM_PROMPT = """
You are the summarization component of FocusFeed.

Summarize the supplied article evidence for a personalized
information digest.

Rules:
1. Use only facts present in the supplied evidence.
2. Do not invent details or background information.
3. If the evidence is insufficient, explicitly say so.
4. Keep the summary concise and factual.
5. Explain briefly why the item may matter to the user.
6. Do not create URLs or citations.
7. Return valid JSON only.
8. Preserve dates, prices, measurements and numbers exactly as written.
9. Never describe a personalization score in the summary.
10. Evidence marked "RSS description only" is insufficient for detailed claims.

Required JSON format:
{
  "headline": "short factual headline",
  "summary": "two or three concise sentences",
  "key_points": ["point one", "point two"],
  "why_relevant": "one concise sentence",
  "evidence_sufficient": true
}
"""


def _interest_phrase(category: str) -> str:
    cleaned = re.sub(r"\s*&\s*", " and ", category.strip()).lower()
    return f"Matches your interest in {cleaned}." if cleaned else "Matches your saved interests."


def _feed_preview(article: dict[str, Any]) -> str:
    summary = " ".join(str(article.get("summary", "")).split())
    if summary:
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        preview = " ".join(sentences[:2]).strip()
        if len(preview) > 360:
            preview = preview[:357].rsplit(" ", 1)[0] + "…"
        return preview
    return "The publisher did not provide a preview. Open the original article to read the full report."


def _load_digest(data_dir: Path | None = None) -> dict[str, Any]:
    path = (data_dir or ROOT / "data") / "digest.json"
    if not path.exists():
        raise FileNotFoundError(
            "Run the ranking pipeline before summarization"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _build_evidence(
    article: dict[str, Any],
) -> str:
    full_text = article.get("full_text", "").strip()
    rss_summary = article.get("summary", "").strip()

    if full_text:
        evidence = full_text
        evidence_type = "full article text"
    else:
        evidence = rss_summary
        evidence_type = "RSS description only"

    return f"""
ARTICLE EVIDENCE

Title:
{article.get("title", "")}

Source:
{article.get("source", "")}

Published:
{article.get("published_at", "Unknown")}

Evidence type:
{evidence_type}

Retrieved text:
{evidence}
""".strip()


def _parse_model_json(content: str) -> dict[str, Any]:
    clean = content.strip()

    if clean.startswith("```"):
        clean = clean.removeprefix("```json")
        clean = clean.removeprefix("```")
        clean = clean.removesuffix("```")
        clean = clean.strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        return {
            "headline": "Summary unavailable",
            "summary": clean,
            "key_points": [],
            "why_relevant": "",
            "evidence_sufficient": False,
        }

    return {
        "headline": str(
            result.get("headline", "Untitled")
        ),
        "summary": str(
            result.get("summary", "")
        ),
        "key_points": [
            str(point)
            for point in result.get("key_points", [])
        ][:3],
        "why_relevant": str(
            result.get("why_relevant", "")
        ),
        "evidence_sufficient": bool(
            result.get("evidence_sufficient", False)
        ),
    }


def _summarize_article(
    client: httpx.Client,
    article: dict[str, Any],
) -> dict[str, Any]:
    article = extract_article_text(article)
    full_text = article.get("full_text", "").strip()
    if len(full_text) < 500:
        title = article.get("title", "Untitled")
        return {
            "headline": title,
            "summary": _feed_preview(article),
            "key_points": [],
            "why_relevant": _interest_phrase(str(article.get("retrieval_category", ""))),
            "evidence_sufficient": False,
            "evidence_type": article.get("evidence_type", "rss_summary"),
            "evidence_length": len(article.get("summary", "")),
            "citation": {"source": article.get("source", "Unknown"), "title": title,
                         "url": article.get("url", ""), "published_at": article.get("published_at")},
            "article_id": article.get("id"), "final_score": article.get("final_score", 0),
            "semantic_score": article.get("semantic_score", 0),
            "retrieval_query": article.get("retrieval_query", ""),
            "retrieval_category": article.get("retrieval_category", ""),
            "related_report_count": article.get("related_report_count", 0),
            "image_url": article.get("image_url", ""),
            "video_url": article.get("video_url", ""),
        }

    response = client.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": _build_evidence(article),
                },
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 300,
            },
        },
    )

    response.raise_for_status()

    response_data = response.json()
    content = response_data["message"]["content"]

    generated = _parse_model_json(content)
    full_text = article.get("full_text", "")
    rss_summary = article.get("summary", "")

    generated["evidence_type"] = article.get(
    "evidence_type",
    "unknown",)

    generated["evidence_length"] = len(
    full_text or rss_summary)

    # Do not trust the model to determine grounding.
    generated["evidence_sufficient"] = (
    len(full_text) >= 500)

    # Citations come from retrieved data, not the LLM.
    generated["citation"] = {
        "source": article.get("source", "Unknown"),
        "title": article.get("title", ""),
        "url": article.get("url", ""),
        "published_at": article.get("published_at"),
    }

    generated["article_id"] = article.get("id")
    generated["final_score"] = article.get(
        "final_score",
        0,
    )
    generated["semantic_score"] = article.get(
        "semantic_score",
        0,
    )
    generated["related_report_count"] = article.get(
        "related_report_count",
        0,
    )
    generated["retrieval_query"] = article.get("retrieval_query", "")
    generated["retrieval_category"] = article.get("retrieval_category", "")
    generated["why_relevant"] = _interest_phrase(str(article.get("retrieval_category", "")))
    generated["image_url"] = article.get("image_url", "")
    generated["video_url"] = article.get("video_url", "")

    return generated


def generate_briefing(
    maximum_items: int = 8,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    data_dir = data_dir or ROOT / "data"
    digest = _load_digest(data_dir)
    # Full-text extraction happens inside _summarize_article. Do not discard
    # direct-source or Hacker News items merely because their feeds omit a description.
    articles = digest.get("articles", [])[:maximum_items]

    if not articles:
        raise ValueError(
            "No ranked articles are available for summarization"
        )

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    try:
        with httpx.Client(timeout=180) as client:
            health_response = client.get(
                f"{OLLAMA_BASE_URL}/api/tags"
            )
            health_response.raise_for_status()

            for article in articles:
                try:
                    summaries.append(
                        _summarize_article(
                            client,
                            article,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "article_id": str(
                                article.get("id", "")
                            ),
                            "title": str(
                                article.get("title", "")
                            ),
                            "error": str(exc),
                        }
                    )
    except httpx.ConnectError as exc:
        raise ConnectionError(
            "Could not connect to Ollama. "
            "Open the Ollama application and try again."
        ) from exc

    result = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": OLLAMA_MODEL,
        "retrieved_article_count": len(articles),
        "summary_count": len(summaries),
        "failure_count": len(failures),
        "grounding_policy": (
            "Full-text items are summarized by the local model. Items with only "
            "feed evidence use a deterministic fallback. Citations come directly "
            "from retrieved metadata."
        ),
        "items": summaries,
        "failures": failures,
    }

    write_json(data_dir / "briefing.json", result)

    return result
