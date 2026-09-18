from __future__ import annotations

from typing import Any

from trafilatura import extract, fetch_url
from bs4 import BeautifulSoup


MINIMUM_TEXT_LENGTH = 500
MAXIMUM_TEXT_LENGTH = 10000


def extract_article_text(
    article: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(article)
    url = article.get("url", "")

    if not url:
        enriched["full_text"] = ""
        enriched["evidence_type"] = "missing"
        return enriched

    try:
        downloaded = fetch_url(url)

        if not downloaded:
            enriched["full_text"] = ""
            enriched["evidence_type"] = "rss_summary"
            return enriched

        soup = BeautifulSoup(downloaded, "html.parser")
        image = soup.select_one('meta[property="og:image"], meta[name="twitter:image"]')
        if image and image.get("content") and not enriched.get("image_url"):
            enriched["image_url"] = image["content"]
        video = soup.select_one('meta[property="og:video:url"], meta[property="og:video"]')
        if video and video.get("content"):
            enriched["video_url"] = video["content"]

        text = extract(
            downloaded,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        if text and len(text.strip()) >= MINIMUM_TEXT_LENGTH:
            enriched["full_text"] = text[
                :MAXIMUM_TEXT_LENGTH
            ]
            enriched["evidence_type"] = "full_article"
            return enriched

    except Exception as exc:
        enriched["extraction_error"] = str(exc)

    enriched["full_text"] = ""
    enriched["evidence_type"] = "rss_summary"
    return enriched
