import os

from redis import Redis
from rq import Queue


def refresh_queue():
    return Queue("focusfeed", connection=Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        socket_connect_timeout=3,
    ))


def enqueue_refresh(user_id):
    return refresh_queue().enqueue(
        "app.main.run_pipeline", user_id, job_timeout=3600,
        result_ttl=3600, ttl=3600, failure_ttl=86400,
    )
