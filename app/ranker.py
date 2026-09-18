from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.feedback import feedback_preferences
from app.semantic_profiler import TOPIC_DESCRIPTIONS, get_model
from app.storage import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent


def _required(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not value:
        raise FileNotFoundError(f"Required file not found: {path.name}")
    return value


def _profile_text(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    for interest in profile.get("interests", []):
        name = str(interest.get("name", ""))
        weight = float(interest.get("weight", 0))
        parts.extend([TOPIC_DESCRIPTIONS.get(name, name)] * max(1, round(weight * 4)))
    parts.extend(str(item.get("name", "")).replace("_", " ") for item in profile.get("detailed_interests", [])[:25])
    return " ".join(parts)


def _article_text(article: dict[str, Any]) -> str:
    return " ".join(str(article.get(key, "")) for key in ("title", "summary", "retrieval_category", "retrieval_query"))


def _recency(value: str | None) -> float:
    if not value:
        return 0.2
    try:
        published = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        hours = max(0, (datetime.now(timezone.utc) - published).total_seconds() / 3600)
        return math.exp(-hours / 96)
    except (TypeError, ValueError):
        return 0.2


def _lexical(profile_text: str, texts: list[str]) -> np.ndarray:
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([profile_text, *texts])
        return (matrix[1:] @ matrix[0].T).toarray().ravel()
    except ValueError:
        return np.zeros(len(texts))


def _deduplicate(ranked: list[dict[str, Any]], vectors: np.ndarray, threshold: float = 0.86) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for index, article in enumerate(ranked):
        duplicate_index = next((i for i in selected_indices if float(np.dot(vectors[index], vectors[i])) >= threshold), None)
        if duplicate_index is not None:
            next(item for item in selected if item["id"] == ranked[duplicate_index]["id"])["related_report_count"] += 1
            continue
        article["related_report_count"] = 0
        selected.append(article)
        selected_indices.append(index)
    return selected


def rank_articles(maximum_results: int = 30, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or ROOT / "data"
    profile = _required(data_dir / "profile.json")
    articles = _required(data_dir / "articles.json").get("articles", [])
    if not articles:
        raise ValueError("No collected articles are available")
    profile_text = _profile_text(profile)
    texts = [_article_text(article) for article in articles]
    embeddings = get_model().encode([profile_text, *texts], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    dense = embeddings[1:] @ embeddings[0]
    lexical = _lexical(profile_text, texts)
    query_feedback, source_feedback = feedback_preferences(data_dir)
    ranked: list[dict[str, Any]] = []

    for index, article in enumerate(articles):
        feedback = max(-0.15, min(0.15,
            query_feedback.get(str(article.get("retrieval_query", "")), 0.0)
            + source_feedback.get(str(article.get("source", "")), 0.0)))
        recency = _recency(article.get("published_at"))
        popularity = min(1.0, math.log1p(int(article.get("popularity", 0))) / math.log1p(500))
        # Prefer direct publisher URLs from GDELT over aggregator redirects,
        # while keeping relevance as the dominant ranking signal.
        collection_bonus = {"gdelt": 0.04, "hacker_news": 0.02}.get(str(article.get("collection", "")), 0.0)
        score = 0.60 * max(0.0, float(dense[index])) + 0.20 * max(0.0, float(lexical[index])) + 0.15 * recency + 0.05 * popularity + feedback + collection_bonus
        item = dict(article)
        item.update({"semantic_score": round(float(dense[index]), 4), "lexical_score": round(float(lexical[index]), 4),
                     "recency_score": round(recency, 4), "feedback_score": round(feedback, 4),
                     "final_score": round(score, 4), "_index": index})
        ranked.append(item)

    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    sorted_vectors = np.asarray([embeddings[1:][item["_index"]] for item in ranked])
    for item in ranked:
        item.pop("_index")
    deduplicated = _deduplicate(ranked, sorted_vectors)

    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for article in deduplicated:
        source = str(article.get("source", "Unknown"))
        category = str(article.get("retrieval_category", "Other"))
        if source == "Hacker News" and source_counts[source] >= 2:
            continue
        if category_counts[category] >= 3:
            continue
        selected.append(article)
        source_counts[source] += 1
        category_counts[category] += 1
        if len(selected) >= maximum_results:
            break

    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "candidate_count": len(articles),
              "deduplicated_count": len(deduplicated), "result_count": len(selected),
              "ranking_formula": "0.60 dense + 0.20 TF-IDF + 0.15 recency + 0.05 popularity + feedback + direct-source bonus",
              "articles": selected}
    write_json(data_dir / "digest.json", result)
    return result
