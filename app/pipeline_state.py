from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator
from app.database import read_state, write_state
from app.storage import write_json, read_json

STATE_KEYS = ("profile", "articles", "digest", "briefing", "feedback", "metrics")

@contextmanager
def state_workspace(user_id: str, load: tuple[str, ...] = STATE_KEYS) -> Iterator[Path]:
    with TemporaryDirectory(prefix="focusfeed-") as temporary:
        path = Path(temporary)
        for key in load:
            value = read_state(user_id, key)
            if value is not None: write_json(path / f"{key}.json", value)
        yield path

def persist_workspace(user_id: str, path: Path, keys: tuple[str, ...]) -> None:
    for key in keys:
        value = read_json(path / f"{key}.json")
        if value is not None: write_state(user_id, key, value)
