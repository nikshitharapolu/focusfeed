from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

TOPIC_DESCRIPTIONS = {
    "AI & technology": (
        "Artificial intelligence, machine learning, LLMs, RAG, "
        "programming, software engineering, technology and developer tools"
    ),
    "Jobs & careers": (
        "Job searching, resumes, interviews, recruiters, internships, "
        "career advice and professional development"
    ),
    "Anime & manga": (
        "Anime series, manga, Japanese animation, anime characters, "
        "episode discussions and anime recommendations"
    ),
    "Movies & television": (
        "Movies, television shows, streaming series, actors, film reviews "
        "and entertainment recommendations"
    ),
    "Music": (
        "Music, songs, singers, albums, concerts, musicians and playlists"
    ),
    "Fitness & wellness": (
        "Fitness, gym workouts, strength training, exercise, running, "
        "cardio, mobility, yoga and physical wellness"
    ),
    "Healthy meals & nutrition": (
        "Healthy recipes, high-protein meals, nutrition, meal preparation, "
        "calories, macros and healthy snacks"
    ),
    "Cricket": (
        "Indian cricket, IPL, international cricket matches, cricket "
        "players, BCCI and Team India"
    ),
    "News & current affairs": (
        "Important news, government policies, current affairs, major "
        "incidents and local news from the United States and India"
    ),
    "Fashion & lifestyle": (
        "Fashion, outfits, clothing, skincare, beauty, haircare and "
        "lifestyle inspiration"
    ),
    "Food & restaurants": (
        "Food, restaurants, cafes, desserts, cooking and restaurant "
        "recommendations"
    ),
    "Viral trends": (
        "Internet trends, viral challenges, memes, popular products and "
        "new things becoming popular online"
    ),
    "Deals & shopping": (
        "Shopping deals, discounts, sales, coupons, product recommendations "
        "and affordable purchases"
    ),
}

GENERIC_HASHTAGS = {
    "fyp",
    "explore",
    "explorepage",
    "viral",
    "reels",
    "instagram",
    "instagood",
    "trending",
    "trend",
    "tiktok",
    "facebook",
    "follow",
    "like",
    "reelitfeelit",
}


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def _clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _prepare_documents(texts: list[str]) -> list[str]:
    documents = []
    seen = set()

    ignored_labels = {
        "caption",
        "name",
        "username",
        "url",
        "owner",
        "hashtags",
        "saved posts",
        "posts you've saved",
    }

    for text in texts:
        clean = _clean_text(text)

        if clean.lower() in ignored_labels:
            continue

        # Short fragments such as isolated field labels and usernames
        # are not useful for semantic classification.
        if len(clean) < 40:
            continue

        normalized = clean.lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        documents.append(clean)

    # Prevent extremely large exports from exhausting memory.
    return documents[:5000]


def _extract_hashtags(documents: list[str]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()

    for document in documents:
        hashtags = re.findall(
            r"(?<!\w)#([A-Za-z0-9_]{2,50})",
            document.lower(),
        )

        for hashtag in hashtags:
            if hashtag not in GENERIC_HASHTAGS:
                counts[hashtag] += 1

    return [
        {
            "name": hashtag,
            "count": count,
        }
        for hashtag, count in counts.most_common(30)
    ]


def build_semantic_profile(texts: list[str]) -> dict[str, Any]:
    documents = _prepare_documents(texts)

    if not documents:
        return {
            "model": MODEL_NAME,
            "documents_analyzed": 0,
            "interests": [],
            "detailed_interests": [],
        }

    model = get_model()

    categories = list(TOPIC_DESCRIPTIONS)
    descriptions = [
        TOPIC_DESCRIPTIONS[category]
        for category in categories
    ]

    document_embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    category_embeddings = model.encode(
        descriptions,
        normalize_embeddings=True,
    )

    # Embeddings are normalized, so matrix multiplication produces
    # cosine-similarity scores.
    similarities = np.matmul(
        document_embeddings,
        category_embeddings.T,
    )

    threshold = 0.27
    category_scores: dict[str, list[tuple[float, str]]] = {
        category: []
        for category in categories
    }

    for document_index, document in enumerate(documents):
        scores = similarities[document_index]
        ranked_indices = np.argsort(scores)[::-1]

        best_index = int(ranked_indices[0])
        best_score = float(scores[best_index])

        if best_score >= threshold:
            category_scores[categories[best_index]].append(
                (best_score, document)
            )

        # A post may belong to a second topic, but require stronger
        # evidence to avoid excessive multi-label classification.
        second_index = int(ranked_indices[1])
        second_score = float(scores[second_index])

        if (
            second_score >= 0.34
            and best_score - second_score <= 0.08
        ):
            category_scores[categories[second_index]].append(
                (second_score, document)
            )

    unnormalized: dict[str, float] = {}

    for category, matches in category_scores.items():
        if not matches:
            continue

        average_similarity = sum(
            score for score, _ in matches
        ) / len(matches)

        # Logarithmic frequency prevents one frequently occurring topic
        # from making every other genuine interest appear insignificant.
        unnormalized[category] = (
            average_similarity * math.log1p(len(matches))
        )

    maximum = max(unnormalized.values(), default=1.0)

    interests = []

    for category, raw_score in sorted(
        unnormalized.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        matches = sorted(
            category_scores[category],
            key=lambda item: item[0],
            reverse=True,
        )

        interests.append(
            {
                "name": category,
                "weight": round(raw_score / maximum, 3),
                "matched_posts": len(matches),
                "average_confidence": round(
                    sum(score for score, _ in matches)
                    / len(matches),
                    3,
                ),
                "examples": [
                    document[:180]
                    for _, document in matches[:5]
                ],
            }
        )

    return {
        "model": MODEL_NAME,
        "documents_analyzed": len(documents),
        "interests": interests,
        "detailed_interests": _extract_hashtags(documents),
    }