# Test Fixtures

This directory contains deterministic test fixtures for the AI Model Platform.

## `build_fixtures.py`

Factory module that creates in-memory database records for integration and
end-to-end tests.  All UUIDs are derived from fixed seeds so that tests are
reproducible across runs.

### Fixture Types

| Fixture | Description |
|---------|-------------|
| `RootModelFixture` | A root `.pt` model node with label schema, artifact path, and hash |
| `FirstGenDatasetFixture` | A first-generation YOLO dataset with train/val/test splits |
| `AppendCategoryDatasetFixture` | A dataset that inherits a parent schema and adds a new class |
| `InvalidDatasetFixture` | A dataset with structural errors (missing inherited class coverage) |
| `FailedModelLoadFixture` | A candidate model node that simulates load failure |
| `InsufficientResourceFixture` | GPU resource config where available memory is below reserved |
| `ExpiredLeaseFixture` | A training attempt whose lease has expired |
| `ConcurrentSwitchFixture` | Two releases competing for the same binding (race condition) |
| `ServiceRestartFixture` | A serving runtime instance from a previous worker process |

### Usage

```python
from backend.tests.fixtures.build_fixtures import FixtureBuilder

def test_example(session):
    builder = FixtureBuilder(session)
    root = builder.root_model()
    ds = builder.first_gen_dataset(label_schema=root.label_schema)
    task = training_service.create_task(
        session,
        parent_model_node_id=root.model_node.id,
        dataset_snapshot_id=ds.snapshot.id,
        ...
    )
```

### Determinism

UUIDs are generated via `hashlib.sha256` on a fixed seed string, so the same
input always produces the same UUID.  This allows tests to reference specific
fixture objects by ID without relying on auto-increment or runtime state.
