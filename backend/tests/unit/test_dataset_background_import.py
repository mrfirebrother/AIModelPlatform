from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend.app.api.routes.datasets import DatasetCreate, create_dataset


class _Query:
    def filter(self, *_args):
        return self

    def first(self):
        return None


class _FakeDb:
    def __init__(self) -> None:
        self.events: list[str] = []

    def query(self, *_args):
        return _Query()

    def add(self, _value) -> None:
        self.events.append("add")

    def flush(self) -> None:
        self.events.append("flush")

    def commit(self) -> None:
        self.events.append("commit")


class _FakeBackgroundTasks:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def add_task(self, *_args) -> None:
        self.events.append("schedule")


def test_dataset_is_committed_before_snapshot_background_task_is_scheduled() -> None:
    db = _FakeDb()
    background_tasks = _FakeBackgroundTasks(db.events)
    settings = SimpleNamespace(dataset_dir=Path("/data/datasets"))

    create_dataset(
        DatasetCreate(
            name="async-order-test",
            source_path="/data/datasets/example.zip",
        ),
        background_tasks,
        db,
        "test-api-key",
        settings,
    )

    assert db.events.index("commit") < db.events.index("schedule")
