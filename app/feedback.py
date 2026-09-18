from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel
from app.storage import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent
class FeedbackRequest(BaseModel):
    article_id: str
    value: Literal["useful", "more_like_this", "not_interested"]
    title: str = ""
    retrieval_query: str = ""
    source: str = ""

def load_feedback(data_dir: Path | None = None) -> list[dict[str, Any]]:
    return read_json((data_dir or ROOT / "data") / "feedback.json", []) or []

def save_feedback(request: FeedbackRequest, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or ROOT / "data"
    rows = load_feedback(data_dir)
    row = {**request.model_dump(), "created_at": datetime.now(timezone.utc).isoformat()}
    rows.append(row)
    write_json(data_dir / "feedback.json", rows[-1000:])
    return row

def feedback_preferences(data_dir: Path | None = None) -> tuple[dict[str, float], dict[str, float]]:
    values = {"more_like_this": 0.10, "useful": 0.05, "not_interested": -0.12}
    queries: dict[str, float] = {}
    sources: dict[str, float] = {}
    for row in load_feedback(data_dir):
        score = values.get(str(row.get("value")), 0.0)
        query, source = str(row.get("retrieval_query", "")), str(row.get("source", ""))
        if query: queries[query] = queries.get(query, 0.0) + score
        if source: sources[source] = sources.get(source, 0.0) + score / 2
    return queries, sources
