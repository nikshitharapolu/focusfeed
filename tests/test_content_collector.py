from datetime import datetime, timedelta, timezone

from app.content_collector import _is_recent, build_queries


def test_recent_date_filter():
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    assert _is_recent(recent)
    assert not _is_recent(old)


def test_queries_keep_categories():
    profile = {"interests": [{"name": "Anime & manga", "weight": 0.8}], "detailed_interests": []}
    queries = build_queries(profile)
    assert queries[0]["category"] == "Anime & manga"
