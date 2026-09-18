from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from app.feedback import load_feedback
from app.storage import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent
def compute_metrics(data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or ROOT / "data"
    collected = read_json(data_dir / "articles.json", {}) or {}
    digest = read_json(data_dir / "digest.json", {}) or {}
    briefing = read_json(data_dir / "briefing.json", {}) or {}
    feedback = load_feedback(data_dir); items = briefing.get("items", [])
    candidates = int(digest.get("candidate_count", 0)); deduped = int(digest.get("deduplicated_count", candidates))
    evidence = Counter(item.get("evidence_type", "unknown") for item in items)
    labels = Counter(row.get("value", "unknown") for row in feedback)
    positive = labels["useful"] + labels["more_like_this"]; rated = positive + labels["not_interested"]
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "collected_articles": collected.get("article_count", 0),
              "ranked_candidates": candidates, "duplicate_reduction_rate": round((candidates-deduped)/candidates, 4) if candidates else 0,
              "briefing_items": len(items), "source_diversity": len({i.get("citation",{}).get("source") for i in items}),
              "full_text_grounding_rate": round(evidence["full_article"]/len(items),4) if items else 0,
              "feedback_count": len(feedback), "feedback_acceptance_rate": round(positive/rated,4) if rated else None,
              "feedback_breakdown": dict(labels)}
    write_json(data_dir / "metrics.json", result); return result
