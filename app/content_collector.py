from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

from app.storage import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
HN_API = "https://hacker-news.firebaseio.com/v0"
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

CATEGORY_QUERIES = {
    "Fitness & wellness": "fitness workouts strength training wellness",
    "Healthy meals & nutrition": "healthy high protein meals nutrition recipes",
    "Anime & manga": "anime manga news recommendations upcoming releases",
    "AI & technology": "artificial intelligence machine learning LLM technology",
    "Jobs & careers": "software engineering AI jobs careers interview hiring",
    "News & current affairs": "important US India news current affairs",
    "Cricket": "Indian cricket Team India IPL news",
    "Movies & television": "movies television streaming upcoming releases",
    "Music": "music artists albums concerts releases",
    "Fashion & lifestyle": "fashion skincare beauty lifestyle trends",
    "Food & restaurants": "food restaurants recipes new food trends",
    "Viral trends": "viral internet trends popular products",
    "Deals & shopping": "shopping deals discounts product recommendations",
}


def _clean_html(value: str) -> str:
    return " ".join(BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True).split())


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _is_recent(value: str | None, maximum_age_days: int = 14) -> bool:
    if not value:
        return False
    try:
        published = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return published >= datetime.now(timezone.utc) - timedelta(days=maximum_age_days)
    except (TypeError, ValueError):
        return False


def _article_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _valid_image(url: str) -> str:
    """Reject tracking pixels and generic Google News artwork."""
    lowered = (url or "").lower()
    blocked = ("gstatic.com", "google.com/images", "favicon", "icon")
    return "" if any(value in lowered for value in blocked) else url


def _gdelt_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def load_profile(data_dir: Path | None = None) -> dict[str, Any]:
    profile = read_json((data_dir or ROOT / "data") / "profile.json")
    if not profile:
        raise FileNotFoundError("Create an interest profile before collecting content")
    return profile


def build_queries(profile: dict[str, Any], maximum: int = 10) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    interests = sorted(profile.get("interests", []), key=lambda item: item.get("weight", 0), reverse=True)
    for interest in interests:
        category = str(interest.get("name", ""))
        if float(interest.get("weight", 0)) < 0.25 or category not in CATEGORY_QUERIES:
            continue
        queries.append({"category": category, "query": CATEGORY_QUERIES[category]})
        if len(queries) >= maximum:
            break
    details = {str(item.get("name", "")).lower() for item in profile.get("detailed_interests", [])}
    if {"chicago", "thingstodoinchicago"} & details:
        queries.append({"category": "Chicago", "query": "Chicago local news events restaurants deals"})
    if {"nyc", "newyork", "thingstodoinnyc"} & details:
        queries.append({"category": "New York City", "query": "New York City events restaurants deals"})
    return queries[:maximum]


def collect_google_news(queries: list[dict[str, str]], articles_per_query: int = 12) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    with httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": "FocusFeed/1.0"}) as client:
        for spec in queries:
            response = client.get(GOOGLE_NEWS_URL.format(query=quote_plus(f"{spec['query']} when:14d")))
            response.raise_for_status()
            for entry in feedparser.parse(response.content).entries[:articles_per_query]:
                url = entry.get("link", "")
                title = _clean_html(entry.get("title", ""))
                published = _parse_date(entry.get("published"))
                if not url or not title or not _is_recent(published):
                    continue
                media = getattr(entry, "media_content", []) or getattr(entry, "media_thumbnail", [])
                image_url = media[0].get("url", "") if media else ""
                if not image_url:
                    image = BeautifulSoup(entry.get("summary", ""), "html.parser").find("img")
                    image_url = image.get("src", "") if image else ""
                image_url = _valid_image(image_url)
                articles.append({
                    "id": _article_id(url), "title": title,
                    "summary": _clean_html(entry.get("summary", "")), "url": url,
                    "source": entry.get("source", {}).get("title", "Google News"),
                    "published_at": published, "retrieval_query": spec["query"],
                    "retrieval_category": spec["category"], "collection": "google_news",
                    "image_url": image_url,
                })
    return articles


def collect_gdelt(queries: list[dict[str, str]], articles_per_query: int = 8) -> list[dict[str, Any]]:
    """Collect direct publisher links and social images from global news coverage."""
    articles: list[dict[str, Any]] = []
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "FocusFeed/2.1"}) as client:
        for spec in queries:
            try:
                response = client.get(GDELT_API, params={
                    "query": spec["category"].replace(" & ", " OR "), "mode": "ArtList", "format": "json",
                    "maxrecords": articles_per_query, "timespan": "14d", "sort": "DateDesc",
                })
                response.raise_for_status()
                rows = response.json().get("articles", [])
            except (httpx.HTTPError, ValueError, TypeError):
                continue
            for row in rows:
                url, title = str(row.get("url", "")), _clean_html(str(row.get("title", "")))
                published = _gdelt_date(row.get("seendate"))
                if not url or not title or not _is_recent(published):
                    continue
                domain = str(row.get("domain") or urlparse(url).netloc).removeprefix("www.")
                articles.append({
                    "id": _article_id(url), "title": title, "summary": "", "url": url,
                    "source": domain, "published_at": published,
                    "retrieval_query": spec["query"], "retrieval_category": spec["category"],
                    "collection": "gdelt", "image_url": _valid_image(str(row.get("socialimage", ""))),
                })
    return articles


def collect_hacker_news(maximum: int = 20) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    with httpx.Client(timeout=20) as client:
        ids_response = client.get(f"{HN_API}/topstories.json")
        ids_response.raise_for_status()
        for story_id in ids_response.json()[:maximum]:
            response = client.get(f"{HN_API}/item/{story_id}.json")
            item = response.json() if response.status_code == 200 else None
            if not item or item.get("type") != "story" or not item.get("title"):
                continue
            url = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
            published = datetime.fromtimestamp(item["time"], tz=timezone.utc).isoformat() if item.get("time") else None
            articles.append({
                "id": _article_id(url), "title": item["title"].strip(), "summary": "", "url": url,
                "source": "Hacker News", "published_at": published,
                "retrieval_query": CATEGORY_QUERIES["AI & technology"],
                "retrieval_category": "AI & technology", "collection": "hacker_news",
                "popularity": item.get("score", 0),
            })
    return articles


def _deduplicate(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    titles: set[str] = set()
    for article in articles:
        title = " ".join(article["title"].lower().split())
        if article["id"] not in unique and title not in titles:
            unique[article["id"]] = article
            titles.add(title)
    return list(unique.values())


def collect_content(data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or ROOT / "data"
    queries = build_queries(load_profile(data_dir))
    # GDELT supplies direct, image-rich publisher results. Google News adds
    # breadth, while Hacker News adds community-ranked technology coverage.
    articles = _deduplicate(
        collect_gdelt(queries)
        + collect_google_news(queries, articles_per_query=6)
        + collect_hacker_news(maximum=15)
    )
    result = {"collected_at": datetime.now(timezone.utc).isoformat(), "queries": queries,
              "article_count": len(articles), "articles": articles}
    write_json(data_dir / "articles.json", result)
    return result
