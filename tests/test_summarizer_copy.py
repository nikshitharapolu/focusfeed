from app.rag_summarizer import _feed_preview, _interest_phrase


def test_interest_phrase_is_natural():
    assert _interest_phrase("Fitness & wellness") == "Matches your interest in fitness and wellness."


def test_feed_preview_uses_available_description():
    article = {"summary": "A useful first sentence. A helpful second sentence. Extra detail."}
    assert _feed_preview(article) == "A useful first sentence. A helpful second sentence."
