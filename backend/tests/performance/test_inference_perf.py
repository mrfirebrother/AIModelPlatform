from __future__ import annotations

import base64
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.models import (
    Base,
    BindingRelease,
    ModelBinding,
    ModelNode,
    RuntimeInstance,
)
from backend.app.schemas.inference import InferenceItem
from backend.app.services.inference_service import InferenceService


def _make_session(thread_safe: bool = False) -> Session:
    if thread_safe:
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine("sqlite+pysqlite:///:memory:")
    event.listen(engine, "connect", lambda c, _: c.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    return Session(engine)


def _make_test_image_base64(size_kb: int = 1) -> str:
    png_header = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    payload = png_header * max(1, size_kb)
    return base64.b64encode(payload).decode("utf-8")


def _seed_binding(session: Session, *, model_family: str = "yolo") -> tuple:
    binding = ModelBinding(external_ref=f"perf-ref-{uuid4().hex[:8]}", status="bound")
    model = ModelNode(
        task_type="object_detection",
        model_family=model_family,
        artifact_path=f"/data/models/{model_family}.pt",
        artifact_hash=f"sha256:{uuid4().hex[:16]}",
        status="approved",
    )
    session.add(binding)
    session.add(model)
    session.flush()

    release = BindingRelease(
        binding_id=binding.id,
        revision_no=1,
        model_node_id=model.id,
        inference_config_json={"confidence": 0.5},
        inference_config_hash="sha256:perf_config",
        release_type="normal",
        status="active",
    )
    session.add(release)
    session.flush()
    binding.current_release_id = release.id
    session.flush()

    instance = RuntimeInstance(
        binding_id=binding.id,
        release_id=release.id,
        model_node_id=model.id,
        config_hash="sha256:perf_config",
        generation=1,
        fencing_token=1,
        status="serving",
        gpu_device="0",
        reserved_memory_mb=4096,
    )
    session.add(instance)
    session.flush()
    binding.current_runtime_instance_id = instance.id
    session.flush()

    return binding, release, model, instance


pytestmark = [pytest.mark.slow]


class TestSingleImageLatency:
    def test_single_inference_latency_under_threshold(self) -> None:
        session = _make_session()
        binding, release, model, instance = _seed_binding(session)
        service = InferenceService(session)
        image_b64 = _make_test_image_base64()

        latencies: list[float] = []
        for _ in range(10):
            start = time.perf_counter()
            resp = service.run_inference(
                binding_id=binding.id,
                image_base64=image_b64,
                image_format="png",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert resp.bindingId == binding.id
            assert resp.modelNodeId == model.id

        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"\n[Perf] Single-image latency: p50={p50:.2f}ms p95={p95:.2f}ms")
        assert p95 < 500, f"P95 latency {p95:.2f}ms exceeds 500ms threshold"

    def test_single_inference_latency_consistency(self) -> None:
        session = _make_session()
        binding, _, _, _ = _seed_binding(session)
        service = InferenceService(session)
        image_b64 = _make_test_image_base64()

        latencies: list[float] = []
        for _ in range(20):
            start = time.perf_counter()
            service.run_inference(binding_id=binding.id, image_base64=image_b64, image_format="png")
            latencies.append((time.perf_counter() - start) * 1000)

        std_dev = statistics.stdev(latencies)
        print(f"\n[Perf] Latency consistency: mean={statistics.mean(latencies):.2f}ms std={std_dev:.2f}ms")
        assert std_dev < 100, f"Latency std dev {std_dev:.2f}ms too high (inconsistent)"


class TestBatchThroughput:
    def test_batch_sequential_throughput(self) -> None:
        session = _make_session()
        binding, _, _, _ = _seed_binding(session)
        service = InferenceService(session)
        image_b64 = _make_test_image_base64()

        batch_sizes = [1, 5, 10, 20]
        throughputs: list[tuple[int, float]] = []

        for size in batch_sizes:
            items = [
                InferenceItem(
                    modelBindingId=binding.id,
                    input={"image_base64": image_b64, "image_format": "png"},
                )
                for _ in range(size)
            ]
            start = time.perf_counter()
            resp = service.run_batch_inference(items)
            elapsed = time.perf_counter() - start
            throughput = size / elapsed if elapsed > 0 else 0
            throughputs.append((size, throughput))
            assert resp.succeeded == size
            print(f"[Perf] Batch size={size}: {elapsed*1000:.1f}ms, throughput={throughput:.1f} img/s")

        print(f"\n[Perf] Batch throughput summary: {throughputs}")
        assert throughputs[-1][1] >= throughputs[0][1] * 0.5, "Batch throughput degraded unexpectedly"

    def test_batch_throughput_scales(self) -> None:
        session = _make_session()
        binding, _, _, _ = _seed_binding(session)
        service = InferenceService(session)
        image_b64 = _make_test_image_base64()

        items_5 = [
            InferenceItem(
                modelBindingId=binding.id,
                input={"image_base64": image_b64, "image_format": "png"},
            )
            for _ in range(5)
        ]
        start = time.perf_counter()
        resp_5 = service.run_batch_inference(items_5)
        time_5 = time.perf_counter() - start

        items_10 = [
            InferenceItem(
                modelBindingId=binding.id,
                input={"image_base64": image_b64, "image_format": "png"},
            )
            for _ in range(10)
        ]
        start = time.perf_counter()
        resp_10 = service.run_batch_inference(items_10)
        time_10 = time.perf_counter() - start

        assert resp_5.succeeded == 5
        assert resp_10.succeeded == 10
        ratio = time_10 / time_5 if time_5 > 0 else float("inf")
        print(f"\n[Perf] Batch scaling: 5 imgs={time_5*1000:.1f}ms 10 imgs={time_10*1000:.1f}ms ratio={ratio:.2f}")
        assert ratio < 4.0, f"Batch scaling ratio {ratio:.2f} too high (should be <4x for 2x data)"


class TestConcurrentInference:
    def test_concurrent_single_requests(self) -> None:
        session = _make_session(thread_safe=True)
        binding, _, _, _ = _seed_binding(session)
        image_b64 = _make_test_image_base64()
        num_requests = 10
        num_workers = 4

        def _infer(idx: int) -> float:
            start = time.perf_counter()
            svc = InferenceService(session)
            resp = svc.run_inference(
                binding_id=binding.id,
                image_base64=image_b64,
                image_format="png",
            )
            assert resp.bindingId == binding.id
            return (time.perf_counter() - start) * 1000

        latencies: list[float] = []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_infer, i): i for i in range(num_requests)}
            for future in as_completed(futures):
                latencies.append(future.result())

        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(
            f"\n[Perf] Concurrent ({num_workers} workers, {num_requests} requests): "
            f"p50={statistics.median(latencies):.2f}ms p95={p95:.2f}ms "
            f"total={sum(latencies):.2f}ms"
        )
        assert p95 < 1000, f"Concurrent P95 {p95:.2f}ms exceeds 1000ms"

    def test_concurrent_batch_requests(self) -> None:
        session = _make_session(thread_safe=True)
        binding, _, _, _ = _seed_binding(session)
        image_b64 = _make_test_image_base64()
        num_concurrent = 3
        batch_size = 5

        def _batch_infer() -> tuple[int, float]:
            items = [
                InferenceItem(
                    modelBindingId=binding.id,
                    input={"image_base64": image_b64, "image_format": "png"},
                )
                for _ in range(batch_size)
            ]
            start = time.perf_counter()
            svc = InferenceService(session)
            resp = svc.run_batch_inference(items)
            return resp.succeeded, (time.perf_counter() - start) * 1000

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [pool.submit(_batch_infer) for _ in range(num_concurrent)]
            results = [f.result() for f in futures]

        total_succeeded = sum(s for s, _ in results)
        max_latency = max(t for _, t in results)
        print(
            f"\n[Perf] Concurrent batches ({num_concurrent}x{batch_size}): "
            f"total_succeeded={total_succeeded} max_latency={max_latency:.1f}ms"
        )
        assert total_succeeded == num_concurrent * batch_size


class TestModelSwitchOverhead:
    def test_model_switch_adds_overhead(self) -> None:
        session = _make_session()
        binding_a, _, model_a, _ = _seed_binding(session, model_family="yolov8n")
        binding_b, _, model_b, _ = _seed_binding(session, model_family="yolov8s")
        service = InferenceService(session)
        image_b64 = _make_test_image_base64()

        latencies_a: list[float] = []
        for _ in range(5):
            start = time.perf_counter()
            service.run_inference(binding_id=binding_a.id, image_base64=image_b64, image_format="png")
            latencies_a.append((time.perf_counter() - start) * 1000)

        latencies_b: list[float] = []
        for _ in range(5):
            start = time.perf_counter()
            service.run_inference(binding_id=binding_b.id, image_base64=image_b64, image_format="png")
            latencies_b.append((time.perf_counter() - start) * 1000)

        avg_a = statistics.mean(latencies_a)
        avg_b = statistics.mean(latencies_b)
        switch_total = sum(latencies_b)

        print(
            f"\n[Perf] Model switch: model_a avg={avg_a:.2f}ms model_b avg={avg_b:.2f}ms "
            f"switch_total={switch_total:.2f}ms"
        )
        assert avg_a < 500 and avg_b < 500, "Model inference latency too high"
        assert len(latencies_a) == len(latencies_b)

    def test_rapid_model_switching(self) -> None:
        session = _make_session()
        binding_a, _, _, _ = _seed_binding(session, model_family="yolo_a")
        binding_b, _, _, _ = _seed_binding(session, model_family="yolo_b")
        service = InferenceService(session)
        image_b64 = _make_test_image_base64()

        latencies: list[float] = []
        for i in range(10):
            bid = binding_a.id if i % 2 == 0 else binding_b.id
            start = time.perf_counter()
            resp = service.run_inference(binding_id=bid, image_base64=image_b64, image_format="png")
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            assert resp.bindingId == bid

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"\n[Perf] Rapid model switching: avg={avg:.2f}ms p95={p95:.2f}ms")
        assert p95 < 500, f"Rapid switch P95 {p95:.2f}ms too high"
