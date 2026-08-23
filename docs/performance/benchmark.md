# Performance Benchmark Report

> Generated: 2026-08-23 | Phase 3 Task 4

---

## Overview

Performance benchmarks for the AI Model Platform, covering inference latency, throughput, training task overhead, and checkpoint recovery. All tests are marked `slow` and can be skipped with `-m "not slow"`.

---

## Inference Performance

### Single-Image Latency

| Metric | Target | Result |
|--------|--------|--------|
| P50 latency | < 200ms | To be measured |
| P95 latency | < 500ms | To be measured |
| Latency std dev | < 100ms | To be measured |

**Test:** `TestSingleImageLatency::test_single_inference_latency_under_threshold`

Measures 10 consecutive single-image inference calls through `InferenceService.run_inference()` and records P50/P95 latencies.

**Test:** `TestSingleImageLatency::test_single_inference_latency_consistency`

Runs 20 iterations and asserts standard deviation < 100ms, ensuring consistent response times.

### Batch Throughput

| Batch Size | Expected Behavior |
|-----------|-------------------|
| 1 image | Baseline throughput |
| 5 images | >= 1x baseline per-image |
| 10 images | >= 0.5x baseline per-image |
| 20 images | Stable throughput |

**Test:** `TestBatchThroughput::test_batch_sequential_throughput`

Measures `InferenceService.run_batch_inference()` across batch sizes 1, 5, 10, 20. Verifies that throughput (images/second) scales at least proportionally to batch size.

**Test:** `TestBatchThroughput::test_batch_throughput_scales`

Compares 5-image vs 10-image batch latency. Asserts the ratio is < 4x (i.e., doubling batch size should not quadruple latency).

### Concurrent Inference

| Workload | Config |
|----------|--------|
| Single requests | 10 requests, 4 workers |
| Batch requests | 3 concurrent batches of 5 |

**Test:** `TestConcurrentInference::test_concurrent_single_requests`

Runs 10 inference requests in parallel using `ThreadPoolExecutor(max_workers=4)`. P95 latency must stay < 1000ms.

**Test:** `TestConcurrentInference::test_concurrent_batch_requests`

Runs 3 concurrent batch-inference calls (each with 5 items). All 15 items must succeed.

### Model Switch Overhead

| Test | Description |
|------|-------------|
| Two-model switch | Alternate between yolov8n and yolov8s |
| Rapid switching | 10 alternating calls across 2 models |

**Test:** `TestModelSwitchOverhead::test_model_switch_adds_overhead`

Runs 5 inferences on model A, then 5 on model B. Asserts both average under 500ms.

**Test:** `TestModelSwitchOverhead::test_rapid_model_switching`

Alternates between 2 models across 10 calls. P95 latency must stay < 500ms.

---

## Training Performance

### Dataset Size Scaling

| Samples | Expected Behavior |
|---------|-------------------|
| 50 | Baseline creation time |
| 200 | < 5x baseline |
| 1000 | < 10x baseline |

**Test:** `TestDatasetSizeScaling::test_dataset_size_overhead_growth`

Creates training tasks for 50, 200, and 1000 sample datasets. Asserts task creation overhead does not grow faster than 10x.

### Model Size Scaling

| Model | Parameters |
|-------|-----------|
| yolov8n | ~3.2M |
| yolov8s | ~11.2M |
| yolov8m | ~25.9M |

**Test:** `TestModelSizeScaling::test_model_creation_overhead_consistent`

Creates tasks for yolov8n, yolov8s, yolov8m. Asserts task creation overhead for yolov8m is < 5x yolov8n.

### Checkpoint Recovery

| Operation | Target |
|-----------|--------|
| Task completion (checkpoint save) | < 200ms |
| Failure + recovery | < 200ms per cycle |
| 5x recovery cycles | avg < 200ms/cycle |

**Test:** `TestCheckpointRecovery::test_attempt_completion_time`

Measures time to call `training_service.complete_attempt()` and produce a model artifact.

**Test:** `TestCheckpointRecovery::test_attempt_failure_recovery_time`

Simulates OOM failure, then recovers by starting a new attempt. Measures both failure and recovery latency.

**Test:** `TestCheckpointRecovery::test_multiple_recovery_cycles`

Runs 5 fail+recover cycles. Asserts average recovery time < 200ms.

### Concurrent Task Creation

| Tasks | Expected |
|-------|----------|
| 10 tasks | < 100ms total |

**Test:** `TestConcurrentTaskCreation::test_concurrent_task_creation`

Creates 10 training tasks sequentially. Reports per-task overhead.

---

## Running Benchmarks

```bash
# Run all performance tests
python -m pytest backend/tests/performance/ -m slow -v

# Run only inference benchmarks
python -m pytest backend/tests/performance/test_inference_perf.py -m slow -v -s

# Run only training benchmarks
python -m pytest backend/tests/performance/test_training_perf.py -m slow -v -s

# Skip slow tests (run only fast tests)
python -m pytest backend/tests/ -m "not slow" -v
```

### Output

Tests print benchmark metrics to stdout when run with `-s`. Example output:

```
[Perf] Single-image latency: p50=12.34ms p95=18.90ms
[Perf] Batch throughput: batch_size=10: 123.4ms, throughput=81.0 img/s
[Perf] Concurrent (4 workers, 10 requests): p50=45.67ms p95=89.01ms
[Perf] Model switch: model_a avg=12.34ms model_b avg=14.56ms
[Perf] Checkpoint completion: 15.23ms
[Perf] Failure+recovery: fail=8.45ms recover=12.67ms
```

---

## Environment

- **OS:** Windows 10/11
- **Python:** 3.11
- **Database:** SQLite (in-memory for tests)
- **GPU:** NVIDIA (when available, for actual training)
- **Test framework:** pytest with `slow` marker

---

## Notes

- All performance tests use in-memory SQLite, so results reflect service-layer overhead only.
- Actual GPU inference/training latency depends on hardware and is not measured in these tests.
- Tests are designed to be deterministic and CI-friendly.
- Thresholds are generous to avoid flaky failures on different machines.
