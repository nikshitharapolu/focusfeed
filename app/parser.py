from __future__ import annotations
from app.semantic_profiler import build_semantic_profile
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any, Iterable


CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "AI & technology": {
        "ai",
        "artificial intelligence",
        "machine learning",
        "llm",
        "rag",
        "python",
        "software",
        "developer",
        "coding",
        "openai",
        "tech",
    },
    "Jobs & careers": {
        "job",
        "jobs",
        "career",
        "resume",
        "interview",
        "recruiter",
        "internship",
        "leetcode",
        "hiring",
        "engineer",
    },
    "Anime & manga": {
        "anime",
        "manga",
        "one piece",
        "naruto",
        "crunchyroll",
        "otaku",
    },
    "Movies & television": {
        "movie",
        "movies",
        "cinema",
        "film",
        "netflix",
        "series",
        "tv",
    },
    "Music": {
        "music",
        "song",
        "songs",
        "singer",
        "album",
        "spotify",
        "concert",
    },
    "Cricket": {
    "cricket",
    "ipl",
    "bcci",
    "team india",
    "cricket match",
    "kohli",
    "rohit sharma",
    "world cup",
},
    "News & current affairs": {
        "news",
        "breaking",
        "india",
        "usa",
        "united states",
        "hyderabad",
        "policy",
        "election",
        "government",
    },
    "Fashion & lifestyle": {
        "fashion",
        "style",
        "outfit",
        "beauty",
        "skincare",
        "lifestyle",
    },
    "Food & trends": {
        "food",
        "recipe",
        "restaurant",
        "viral",
        "trend",
        "dessert",
        "cafe",
    },
    "Deals & shopping": {
        "deal",
        "deals",
        "sale",
        "discount",
        "shopping",
        "amazon",
        "coupon",
    },
    "Fitness & wellness": {
    "fitness",
    "workout",
    "exercise",
    "gym",
    "strength training",
    "cardio",
    "running",
    "abs",
    "bodybuilding",
    "pilates",
    "yoga",
    "mobility",
    "fat loss",
    "weight loss",
},
"Healthy meals & nutrition": {
    "high protein",
    "protein",
    "healthy meal",
    "healthy recipe",
    "meal prep",
    "nutrition",
    "calories",
    "macros",
    "cottage cheese",
    "chickpea",
    "low calorie",
    "healthy snack",
},
}

SIGNAL_HINTS = {
    "like": "likes",
    "saved": "saved",
    "save": "saved",
    "search": "searches",
    "comment": "comments",
    "follow": "following",
}

SIGNAL_WEIGHTS = {
    "saved": 3.0,
    "comments": 2.5,
    "likes": 2.0,
    "searches": 1.7,
    "following": 1.5,
    "other": 1.0,
}


@dataclass(frozen=True)
class Evidence:
    text: str
    signal: str
    source: str


class _VisibleTextParser(HTMLParser):
    """Extract visible text without retaining Instagram HTML markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in {"script", "style", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            clean = re.sub(r"\s+", " ", data).strip()
            if clean:
                self.parts.append(clean)


def _html_strings(payload: bytes) -> Iterable[str]:
    parser = _VisibleTextParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    yield from parser.parts


def _signal_for_path(path: str) -> str:
    lowered = path.lower()

    for hint, signal in SIGNAL_HINTS.items():
        if hint in lowered:
            return signal

    return "other"


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        clean = re.sub(r"\s+", " ", value).strip()

        if clean:
            yield clean

    elif isinstance(value, dict):
        for key, item in value.items():
            if key.lower() not in {
                "timestamp",
                "href",
                "media_list_data",
            }:
                yield from _walk_strings(item)

    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _documents(
    filename: str,
    payload: bytes,
) -> Iterable[tuple[str, str, Any]]:
    lower_filename = filename.lower()

    if lower_filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                suffix = path.suffix.lower()

                if member.is_dir():
                    continue

                if suffix not in {".json", ".html", ".htm"}:
                    continue

                content = archive.read(member)

                if suffix == ".json":
                    try:
                        document = json.loads(content)
                        yield member.filename, "json", document
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                else:
                    yield member.filename, "html", content

    elif lower_filename.endswith(".json"):
        yield filename, "json", json.loads(payload)

    elif lower_filename.endswith((".html", ".htm")):
        yield filename, "html", payload

    else:
        raise ValueError(
            "Upload an Instagram .zip, .json, or .html file"
        )

def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

def parse_export(filename: str, payload: bytes) -> dict[str, Any]:
    evidence: list[Evidence] = []
    semantic_texts: list[str] = []
    files_scanned = 0
    signals: Counter[str] = Counter()

    for source, document_type, document in _documents(
        filename,
        payload,
    ):
        files_scanned += 1
        signal = _signal_for_path(source)

        if document_type == "json":
            strings = _walk_strings(document)
        else:
            strings = _html_strings(document)

        for text in strings:
            text = text.strip()

            if not text:
                continue

            if len(text) > 500:
                text = text[:500]

            semantic_texts.append(text)

            evidence.append(
                Evidence(
                    text=text,
                    signal=signal,
                    source=source,
                )
            )

            signals[signal] += 1

    if files_scanned == 0:
        raise ValueError(
            "No JSON or HTML activity files were found "
            "in the uploaded file"
        )

    if not semantic_texts:
        raise ValueError(
            "The uploaded files did not contain usable text"
        )

    semantic_profile = build_semantic_profile(
        semantic_texts
    )

    return {
        "files_scanned": files_scanned,
        "text_signals_found": len(evidence),
        "signal_counts": dict(signals),
        "semantic_model": semantic_profile["model"],
        "documents_analyzed": semantic_profile[
            "documents_analyzed"
        ],
        "interests": semantic_profile["interests"],
        "detailed_interests": semantic_profile[
            "detailed_interests"
        ],
        "warning": (
            "Interest weights were inferred automatically "
            "using semantic embeddings. Manual weight "
            "adjustment is not required."
        ),
    }