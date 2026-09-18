from __future__ import annotations
import os
from collections.abc import Callable
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

def create_scheduler(refresh: Callable[[], object]) -> BackgroundScheduler:
    timezone = os.getenv("FOCUSFEED_TIMEZONE", "America/New_York")
    scheduler = BackgroundScheduler(timezone=timezone, daemon=True)
    for hour in (8, 13, 19):
        scheduler.add_job(refresh, CronTrigger(hour=hour, minute=0, timezone=timezone), id=f"focusfeed-{hour}",
                          replace_existing=True, max_instances=1, coalesce=True, misfire_grace_time=3600)
    return scheduler

def scheduler_enabled() -> bool:
    return os.getenv("FOCUSFEED_SCHEDULER", "false").lower() == "true"
