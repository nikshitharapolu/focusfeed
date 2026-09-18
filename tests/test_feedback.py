from app.feedback import FeedbackRequest


def test_feedback_schema():
    request = FeedbackRequest(article_id="abc", value="useful")
    assert request.value == "useful"
