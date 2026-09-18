import io
import json
import zipfile

import app.parser as parser


def make_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "your_instagram_activity/likes/liked_posts.json",
            json.dumps({"likes_media_likes": [{"title": "AI coding and Python news"}]}),
        )
        archive.writestr(
            "your_instagram_activity/saved/saved_posts.json",
            json.dumps({"saved_saved_media": [{"title": "One Piece anime"}]}),
        )
    return buffer.getvalue()


def fake_profile(texts):
    joined = " ".join(texts).lower()
    interests = []
    if "python" in joined:
        interests.append({"name": "AI & technology", "weight": 1.0})
    if "anime" in joined:
        interests.append({"name": "Anime & manga", "weight": 0.8})
    return {"model": "test-model", "documents_analyzed": len(texts), "interests": interests, "detailed_interests": []}


def test_parse_zip_builds_weighted_profile(monkeypatch):
    monkeypatch.setattr(parser, "build_semantic_profile", fake_profile)
    result = parser.parse_export("instagram.zip", make_zip())
    names = {item["name"] for item in result["interests"]}
    assert result["files_scanned"] == 2
    assert "AI & technology" in names
    assert "Anime & manga" in names
    assert result["signal_counts"] == {"likes": 1, "saved": 1}


def test_rejects_unknown_file_type():
    try:
        parser.parse_export("archive.txt", b"hello")
    except ValueError as error:
        assert ".html" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_html_export(monkeypatch):
    monkeypatch.setattr(parser, "build_semantic_profile", fake_profile)
    result = parser.parse_export("saved_posts.html", b"<html><body><p>Anime recommendations and reviews</p></body></html>")
    assert result["files_scanned"] == 1
    assert result["semantic_model"] == "test-model"
    assert result["interests"][0]["name"] == "Anime & manga"
