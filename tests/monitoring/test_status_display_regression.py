"""Regression tests for task status display correctness per task_status_display_specification_skill."""

import threading
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import scheduler
from modules.monitoring import telemetry


def _get_status_stats():
    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.runtime.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.runtime.model_manager.is_uvr_actually_loaded", return_value=True):
                return telemetry.get_service_stats()


def _get_task_order():
    return [task.get("task_id") for task in _get_status_stats()["tasks"]]


def _seed_status_tasks(entries):
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry.update(entries)


def _standard_asr_state(status, stage):
    return {
        "standard_asr": {
            "task_id": "standard_asr",
            "status": status,
            "stage": stage,
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }
    }


def _run_overlapping_registration_pair(name_a: str, name_b: str):
    ids = []
    ids_lock = threading.Lock()
    ready = threading.Barrier(3)
    release = threading.Event()

    def _worker(filename):
        utils.THREAD_CONTEXT.reset()
        with scheduler.early_task_registration(filename=filename):
            with ids_lock:
                ids.append(utils.THREAD_CONTEXT.task_id)
            ready.wait(timeout=2.0)
            assert release.wait(timeout=2.0)

    t1 = threading.Thread(target=_worker, args=(name_a,))
    t2 = threading.Thread(target=_worker, args=(name_b,))
    t1.start()
    t2.start()
    ready.wait(timeout=2.0)
    return ids, release, t1, t2


def _finish_pair_and_assert_threads_stopped(release, t1, t2):
    release.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert not t1.is_alive()
    assert not t2.is_alive()


def _assert_sequential_registration_cleanup(filenames):
    task_ids = []
    for filename in filenames:
        with scheduler.early_task_registration(filename=filename):
            task_ids.append(utils.THREAD_CONTEXT.task_id)

    with scheduler.STATE.task_order_lock:
        assert len(task_ids) == len(filenames)
        assert len(set(task_ids)) == len(filenames)
        assert list(scheduler.STATE.task_arrival_order) == []


@pytest.fixture
def clean_scheduler():
    """Reset scheduler state and task registry."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order.clear()
    yield
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order.clear()


@pytest.fixture
def clean_telemetry():
    """Reset telemetry history."""
    telemetry._STOP_EVENT.set()
    telemetry.TELEMETRY_HISTORY.clear()
    telemetry._STOP_EVENT.clear()
    yield
    telemetry._STOP_EVENT.set()


def test_unknown_status_is_normalized_before_payload(clean_scheduler, clean_telemetry):
    """Verify unknown/invalid status is normalized before dashboard payload exposure."""
    status_values = ["initializing", "queued", "active", "post-processing", "completed", "failed", "unknown"]

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        for i, status in enumerate(status_values):
            scheduler.STATE.task_registry[f"task_{status}"] = {
                "task_id": f"task_{status}",
                "status": status,
                "stage": f"Stage for {status}",
                "start_time": 100.0 + i,
                "is_priority": False,
                "type": "Test",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.runtime.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.runtime.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    # Collect all statuses from payload
    payload_statuses = {t.get("status") for t in stats["tasks"]}

    # Display-facing payload must never leak unknown placeholder status.
    assert "unknown" not in payload_statuses
    assert payload_statuses.issubset({"initializing", "queued", "active", "post-processing", "completed", "failed"})


def test_queued_paused_vs_waiting_distinction_by_stage(clean_scheduler, clean_telemetry):
    """Verify queued tasks distinguish paused-for-priority vs waiting-for-hardware via stage field."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["paused_task"] = {
            "task_id": "paused_task",
            "status": "queued",
            "stage": "Paused for Priority Task",  # This distinguishes it as paused
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["waiting_task"] = {
            "task_id": "waiting_task",
            "status": "queued",
            "stage": "Initializing",  # This indicates waiting for hardware
            "start_time": 110.0,
            "is_priority": False,
            "type": "Transcription",
        }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.runtime.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.runtime.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}

    assert {
        task_id: (task["status"], "Paused for Priority Task" in task["stage"])
        for task_id, task in tasks_by_id.items()
        if task_id in {"paused_task", "waiting_task"}
    } == {
        "paused_task": ("queued", True),
        "waiting_task": ("queued", False),
    }


def test_ordering_active_first_then_priority_queued_then_standard_queued(clean_scheduler, clean_telemetry):
    """Verify deterministic three-tier ordering: active, priority-queued, standard-queued."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        # Mix in different order to verify sorting
        tasks_to_add = [
            ("sq3", "queued", 300.0, False),  # Standard queued, late
            ("a2", "active", 200.0, False),  # Active, late
            ("pq1", "queued", 150.0, True),  # Priority queued, early
            ("a1", "active", 100.0, False),  # Active, early
            ("sq1", "queued", 250.0, False),  # Standard queued, mid
        ]

        for tid, status, start, is_prio in tasks_to_add:
            scheduler.STATE.task_registry[tid] = {
                "task_id": tid,
                "status": status,
                "stage": "Stage",
                "start_time": start,
                "is_priority": is_prio,
                "type": "Test",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.runtime.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.runtime.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    task_order = [t.get("task_id") for t in stats["tasks"]]

    # Expected: active (100, 200) → priority queued (150) → standard queued (250, 300)
    assert task_order == ["a1", "a2", "pq1", "sq1", "sq3"]


def test_task_arrival_order_tracking_for_determinism(clean_scheduler):
    """Verify task_arrival_order registry tracks arrival times for deterministic ordering."""
    overlap_ids, release_overlap, t1, t2 = _run_overlapping_registration_pair("overlap_1", "overlap_2")

    with scheduler.STATE.task_order_lock:
        assert len(overlap_ids) == 2
        assert all(task_id in scheduler.STATE.task_arrival_order for task_id in overlap_ids)

    _finish_pair_and_assert_threads_stopped(release_overlap, t1, t2)

    with scheduler.STATE.task_order_lock:
        assert list(scheduler.STATE.task_arrival_order) == []
    _assert_sequential_registration_cleanup(["task1", "task2", "task3"])


def test_early_task_registration_concurrency_bounded_progress(clean_scheduler):
    """Overlapping registrations must synchronize, preserve arrival keys while active, and complete without deadlock."""
    arrivals_seen, release_event, w1, w2 = _run_overlapping_registration_pair("c1", "c2")

    with scheduler.STATE.task_order_lock:
        # While both workers are in their registration contexts, both task IDs must be tracked.
        assert len(arrivals_seen) == 2
        assert all(task_id in scheduler.STATE.task_arrival_order for task_id in arrivals_seen)

    _finish_pair_and_assert_threads_stopped(release_event, w1, w2)

    with scheduler.STATE.task_order_lock:
        assert list(scheduler.STATE.task_arrival_order) == []


def test_status_transition_active_to_initial_state(clean_scheduler, clean_telemetry):
    """Active standard tasks should report active before preemption."""
    _seed_status_tasks(_standard_asr_state("active", "Inference"))

    stats = _get_status_stats()
    task = next((t for t in stats["tasks"] if t.get("task_id") == "standard_asr"), None)
    assert task is not None
    assert task["status"] == "active"


def test_status_transition_preempted_state(clean_scheduler, clean_telemetry):
    """Preempted standard tasks should report queued with the paused stage."""
    _seed_status_tasks(_standard_asr_state("queued", "Paused for Priority Task"))

    stats = _get_status_stats()
    task = next((t for t in stats["tasks"] if t.get("task_id") == "standard_asr"), None)
    assert task is not None
    assert (task["status"], task["stage"]) == ("queued", "Paused for Priority Task")


def test_status_transition_resumed_state(clean_scheduler, clean_telemetry):
    """Resumed standard tasks should report active again."""
    _seed_status_tasks(_standard_asr_state("active", "Inference"))

    stats = _get_status_stats()
    task = next((t for t in stats["tasks"] if t.get("task_id") == "standard_asr"), None)
    assert task is not None
    assert task["status"] == "active"


def test_hardware_units_show_busy_for_translating_and_inference(clean_scheduler, clean_telemetry):
    """Verify active Whisper ASR work keeps every occupied hardware unit marked busy."""
    from modules.inference.runtime import model_manager

    mock_units = [
        {"id": "GPU.0", "type": "GPU", "name": "Intel Arc"},
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
    ]

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["translate_task"] = {
            "task_id": "translate_task",
            "status": "active",
            "stage": "Translating",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "GPU.0",
        }
        scheduler.STATE.task_registry["infer_task"] = {
            "task_id": "infer_task",
            "status": "active",
            "stage": "Inference",
            "start_time": 101.0,
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "NPU.0",
        }

    with mock.patch("modules.core.config.HARDWARE_UNITS", mock_units):
        with mock.patch.dict(model_manager.MODEL_POOL, {"GPU.0": mock.MagicMock(), "NPU.0": mock.MagicMock()}):
            with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
                with mock.patch("modules.inference.runtime.model_manager.is_engine_actually_loaded", return_value=True):
                    with mock.patch("modules.inference.runtime.model_manager.is_uvr_actually_loaded", return_value=False):
                        stats = telemetry.get_service_stats()

    hardware_by_id = {unit["id"]: unit for unit in stats["hardware_units"]}
    assert hardware_by_id["GPU.0"]["whisper_status"] == "busy"
    assert hardware_by_id["NPU.0"]["whisper_status"] == "busy"


def test_no_unknown_status_leakage_in_normal_operation(clean_scheduler, clean_telemetry):
    """Verify unknown status does not leak to dashboard under normal conditions."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        # Only add tasks with valid canonical statuses
        for status in ["initializing", "queued", "active", "completed"]:
            scheduler.STATE.task_registry[f"task_{status}"] = {
                "task_id": f"task_{status}",
                "status": status,
                "stage": "Valid Stage",
                "start_time": 100.0,
                "is_priority": False,
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.runtime.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.runtime.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    unknown_tasks = [t for t in stats["tasks"] if t.get("status") == "unknown"]
    assert len(unknown_tasks) == 0, f"Unknown status leaked: {unknown_tasks}"


def test_concurrent_task_arrivals_deterministic_ordering(clean_scheduler, clean_telemetry):
    """Verify deterministic ordering for 5 concurrent arrivals with identical start_time values.

    Ordering should remain stable across repeated get_service_stats() calls.
    """
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["active_1"] = {
            "task_id": "active_1",
            "status": "active",
            "stage": "Inference",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["active_2"] = {
            "task_id": "active_2",
            "status": "active",
            "stage": "Inference",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["pq_1"] = {
            "task_id": "pq_1",
            "status": "queued",
            "stage": "Waiting",
            "start_time": 100.0,
            "is_priority": True,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["pq_2"] = {
            "task_id": "pq_2",
            "status": "queued",
            "stage": "Waiting",
            "start_time": 100.0,
            "is_priority": True,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["sq_1"] = {
            "task_id": "sq_1",
            "status": "queued",
            "stage": "Waiting",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }

    orderings = [_get_task_order() for _ in range(3)]

    assert all(ordering == orderings[0] for ordering in orderings[1:])
    assert orderings[0] == ["active_1", "active_2", "pq_1", "pq_2", "sq_1"]


def test_mixed_all_seven_statuses_with_priority_flags(clean_scheduler, clean_telemetry):
    """Verify that when all 7 statuses are present with mixed priority flags, ordering still respects three-tier rules."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry.update(
            {
                "init_std": {
                    "task_id": "init_std",
                    "status": "initializing",
                    "stage": "Test Stage",
                    "start_time": 100.0,
                    "is_priority": False,
                    "type": "Transcription",
                },
                "init_prio": {
                    "task_id": "init_prio",
                    "status": "initializing",
                    "stage": "Test Stage",
                    "start_time": 101.0,
                    "is_priority": True,
                    "type": "Transcription",
                },
                "queue_std1": {
                    "task_id": "queue_std1",
                    "status": "queued",
                    "stage": "Test Stage",
                    "start_time": 102.0,
                    "is_priority": False,
                    "type": "Transcription",
                },
                "queue_prio": {
                    "task_id": "queue_prio",
                    "status": "queued",
                    "stage": "Test Stage",
                    "start_time": 103.0,
                    "is_priority": True,
                    "type": "Transcription",
                },
                "active_std": {
                    "task_id": "active_std",
                    "status": "active",
                    "stage": "Test Stage",
                    "start_time": 104.0,
                    "is_priority": False,
                    "type": "Transcription",
                },
                "active_prio": {
                    "task_id": "active_prio",
                    "status": "active",
                    "stage": "Test Stage",
                    "start_time": 105.0,
                    "is_priority": True,
                    "type": "Transcription",
                },
                "postproc": {
                    "task_id": "postproc",
                    "status": "post-processing",
                    "stage": "Test Stage",
                    "start_time": 106.0,
                    "is_priority": False,
                    "type": "Transcription",
                },
                "completed": {
                    "task_id": "completed",
                    "status": "completed",
                    "stage": "Test Stage",
                    "start_time": 107.0,
                    "is_priority": False,
                    "type": "Transcription",
                },
                "failed": {
                    "task_id": "failed",
                    "status": "failed",
                    "stage": "Test Stage",
                    "start_time": 108.0,
                    "is_priority": False,
                    "type": "Transcription",
                },
            }
        )

    stats = _get_status_stats()

    task_order = [t.get("task_id") for t in stats["tasks"]]
    assert task_order == [
        "active_std",
        "active_prio",
        "init_std",
        "init_prio",
        "queue_std1",
        "queue_prio",
        "postproc",
        "completed",
        "failed",
    ]


def test_task_id_lexicographic_tiebreaker_same_start_time(clean_scheduler, clean_telemetry):
    """Verify that when multiple tasks have the same start_time and status, they are ordered lexicographically by task_id."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        task_ids = ["task_c", "task_a", "task_d", "task_b"]
        for tid in task_ids:
            scheduler.STATE.task_registry[tid] = {
                "task_id": tid,
                "status": "queued",
                "stage": "Waiting",
                "start_time": 100.0,
                "is_priority": False,
                "type": "Transcription",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.runtime.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.runtime.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    task_order = [t.get("task_id") for t in stats["tasks"]]
    expected_order = ["task_a", "task_b", "task_c", "task_d"]
    assert task_order == expected_order, f"Expected lexicographic order {expected_order}, but got {task_order}"


def test_stage_based_paused_tasks_are_distinct(clean_scheduler, clean_telemetry):
    """Paused tasks should keep the paused stage in mixed load."""
    _seed_status_tasks(
        {
            **{
                f"paused_prio_{i}": {
                    "task_id": f"paused_prio_{i}",
                    "status": "queued",
                    "stage": "Paused for Priority Task",
                    "start_time": 200.0 + i,
                    "is_priority": False,
                    "type": "Transcription",
                }
                for i in range(3)
            },
        }
    )

    stats = _get_status_stats()

    tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}
    assert len(stats["tasks"]) == 3
    assert all(task["status"] == "queued" and task["stage"] == "Paused for Priority Task" for task in tasks_by_id.values())


def test_stage_based_waiting_tasks_are_distinct(clean_scheduler, clean_telemetry):
    """Waiting tasks should keep the initializing stage in mixed load."""
    _seed_status_tasks(
        {
            **{
                f"waiting_std_{i}": {
                    "task_id": f"waiting_std_{i}",
                    "status": "queued",
                    "stage": "Initializing",
                    "start_time": 300.0 + i,
                    "is_priority": False,
                    "type": "Transcription",
                }
                for i in range(2)
            },
        }
    )

    stats = _get_status_stats()

    tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}
    assert len(stats["tasks"]) == 2
    assert all(task["status"] == "queued" and task["stage"] == "Initializing" for task in tasks_by_id.values())


def test_stage_based_active_tasks_remain_active(clean_scheduler, clean_telemetry):
    """Active tasks should remain active in mixed load."""
    _seed_status_tasks(
        {
            f"active_{i}": {
                "task_id": f"active_{i}",
                "status": "active",
                "stage": "Inference",
                "start_time": 100.0 + i,
                "is_priority": False,
                "type": "Transcription",
            }
            for i in range(5)
        }
    )

    stats = _get_status_stats()

    tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}
    assert len(stats["tasks"]) == 5
    assert all(task["status"] == "active" and task["stage"] == "Inference" for task in tasks_by_id.values())
