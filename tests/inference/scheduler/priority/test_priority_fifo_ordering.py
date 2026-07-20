"""FIFO task ordering verification tests.

Tests that verify:
1. Tasks are processed in arrival order (FIFO)
2. Detect-language tasks can preempt ASR, but respect FIFO among themselves
3. No task skips ahead in the queue
"""

import threading
import time
from contextlib import contextmanager
from functools import partial
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import concurrency, model_manager

_HW_PATCHER = None


def _setup_units(hw_list):
    """Reset scheduler and model/preprocessor pools for a test hardware layout."""
    global _HW_PATCHER
    if _HW_PATCHER is not None:
        _HW_PATCHER.stop()

    _HW_PATCHER = mock.patch("modules.core.config.HARDWARE_UNITS", hw_list)
    _HW_PATCHER.start()
    scheduler.STATE = scheduler.SchedulerState()

    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    for unit in hw_list:
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()


@pytest.fixture(autouse=True)
def _cleanup_hw_patcher():
    """Ensure HARDWARE_UNITS patch does not leak between tests."""
    global _HW_PATCHER
    yield
    if _HW_PATCHER is not None:
        _HW_PATCHER.stop()
        _HW_PATCHER = None


def _assert_events_in_order(events, expected_pairs):
    """Assert each event pair appears in the requested order."""
    for earlier, later in expected_pairs:
        assert events.index(earlier) < events.index(later)


def _start_fifo_tasks(task_count, worker, registration_timeout=3.0):
    """Start FIFO workers and wait for their registration checkpoints."""
    registration_events = [threading.Event() for _ in range(task_count)]
    proceed_events = [threading.Event() for _ in range(task_count)]
    threads = []

    for task_num in range(task_count):
        thread = threading.Thread(target=worker, args=(task_num, registration_events, proceed_events))
        thread.start()
        threads.append(thread)
        registration_events[task_num].wait(timeout=registration_timeout)

    return threads, proceed_events


def _release_fifo_tasks(threads, proceed_events, proceed_delay=0.005, join_timeout=10.0):
    """Release FIFO workers in order and wait for them to finish."""
    for event in proceed_events:
        event.set()
        time.sleep(proceed_delay)

    for thread in threads:
        thread.join(timeout=join_timeout)


@contextmanager
def _auto_confirm_priority_waits():
    """Keep priority pause confirmations flowing during concurrency tests."""
    stop_event = threading.Event()

    def run_auto_confirm():
        while not stop_event.is_set():
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
            for u_sync in scheduler.STATE.unit_sync.values():
                if u_sync["pause_requested"].is_set() and not u_sync["pause_confirmed"].is_set():
                    u_sync["pause_confirmed"].set()
            time.sleep(0.01)

    confirm_thread = threading.Thread(target=run_auto_confirm, daemon=True)
    confirm_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        confirm_thread.join(timeout=2.0)


def _run_fifo_asr_task(task_num, registration_events, proceed_events, *, acquisition_order, lock):
    """Run a standard ASR FIFO worker."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            registration_events[task_num].set()
            proceed_events[task_num].wait()
            with model_manager.model_lock_ctx(priority=False) as (_, _unit_id):
                with lock:
                    acquisition_order.append(f"asr_{task_num}_acquired")
    finally:
        model_manager.decrement_active_session()


def _run_fifo_priority_task(task_num, registration_events, proceed_events, *, events, acquisition_order, lock, delay=0.03):
    """Run a priority FIFO worker."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with lock:
            events.append(f"prio_{task_num}_start")

        with model_manager.early_task_registration(is_priority=True) as task_id:
            with lock:
                events.append(f"prio_{task_num}_registered")
                acquisition_order.append(task_id)
            registration_events[task_num].set()
            proceed_events[task_num].wait()

            model_manager.wait_for_priority()
            with lock:
                events.append(f"prio_{task_num}_waited")

            with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                with lock:
                    events.append(f"prio_{task_num}_unit_{unit_id}")
                time.sleep(delay)
                with lock:
                    events.append(f"prio_{task_num}_done")
    finally:
        model_manager.decrement_active_session()


def _run_simple_asr_task(events, lock, task_num=None, hold_time=0.02):
    """Run a minimal ASR task used by the simpler ordering tests."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        suffix = "" if task_num is None else f"_{task_num}"
        with lock:
            events.append(f"asr{suffix}_start")

        with model_manager.early_task_registration(is_priority=False):
            with lock:
                events.append(f"asr{suffix}_registered")

            with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                with lock:
                    events.append(f"asr{suffix}_acquired_{unit_id}")
                time.sleep(hold_time)
                with lock:
                    events.append(f"asr{suffix}_done")
    finally:
        model_manager.decrement_active_session()


def _run_simple_priority_task(events, lock, task_num=None, hold_time=0.01):
    """Run a minimal priority task used by the simpler ordering tests."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        suffix = "" if task_num is None else f"_{task_num}"
        with lock:
            events.append(f"prio{suffix}_start")

        with model_manager.early_task_registration(is_priority=True):
            with lock:
                events.append(f"prio{suffix}_registered")

            model_manager.wait_for_priority()
            with lock:
                events.append(f"prio{suffix}_waited")

            with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                with lock:
                    events.append(f"prio{suffix}_acquired_{unit_id}")
                time.sleep(hold_time)
                with lock:
                    events.append(f"prio{suffix}_done")
    finally:
        model_manager.decrement_active_session()


def _run_task_pair(first_worker, second_worker, events, lock, first_kwargs=None, second_kwargs=None, start_delay=0.02):
    """Run two worker threads with a fixed start order and return both threads."""
    first_thread = threading.Thread(target=first_worker, args=(events, lock), kwargs=first_kwargs or {})
    first_thread.start()
    time.sleep(start_delay)

    second_thread = threading.Thread(target=second_worker, args=(events, lock), kwargs=second_kwargs or {})
    second_thread.start()

    first_thread.join(timeout=10.0)
    second_thread.join(timeout=10.0)
    return first_thread, second_thread


def _assert_detect_language_fifo_events(events):
    """Assert the detect-language FIFO batch completed in order."""
    assert all(f"prio_{idx}_registered" in events for idx in range(3))
    assert all(f"prio_{idx}_waited" in events for idx in range(3))
    assert all(any(e.startswith(f"prio_{idx}_unit_") for e in events) for idx in range(3))
    assert all(f"prio_{idx}_done" in events for idx in range(3))

    _assert_events_in_order(
        events,
        [
            ("prio_0_registered", "prio_1_registered"),
            ("prio_1_registered", "prio_2_registered"),
            ("prio_0_waited", "prio_1_waited"),
            ("prio_1_waited", "prio_2_waited"),
        ],
    )


def _assert_detect_language_preempts_asr_events(events):
    """Assert the mixed ASR/priority pair completed and preserved FIFO ordering."""
    assert {"asr_start", "prio_start", "asr_registered", "prio_registered", "prio_waited"}.issubset(events)
    assert any(event.startswith("prio_acquired_") for event in events)
    _assert_events_in_order(
        events,
        [
            ("asr_start", "prio_start"),
            ("asr_registered", "prio_registered"),
            ("prio_waited", next(event for event in events if event.startswith("prio_acquired_"))),
        ],
    )


def _assert_numbered_task_pair_events(events, prefix, *, enforce_done_order=True):
    """Assert a numbered two-task FIFO pair completed in order."""
    acquired_0 = _find_event_by_prefix(events, f"{prefix}_0_acquired")
    acquired_1 = _find_event_by_prefix(events, f"{prefix}_1_acquired")
    assert f"{prefix}_0_done" in events
    assert f"{prefix}_1_done" in events
    expected_pairs = [
        (f"{prefix}_0_registered", f"{prefix}_1_registered"),
        (acquired_0, acquired_1),
    ]
    if enforce_done_order:
        expected_pairs.append((f"{prefix}_0_done", f"{prefix}_1_done"))
    _assert_events_in_order(events, expected_pairs)


def _find_event_by_prefix(events, prefix):
    return next(event for event in events if event.startswith(prefix))


def _seed_task_state(task_id, arrival_time, *, is_priority, status):
    """Seed scheduler arrival order and registry state for FIFO checks."""
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order[task_id] = arrival_time

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry[task_id] = {
            "task_id": task_id,
            "is_priority": is_priority,
            "status": status,
        }


def _clear_task_state():
    """Clear scheduler task state after FIFO assertions."""
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order.clear()

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()


def _run_dual_accelerator_first_asr(events, lock, acquired_event, release_event):
    """Run the leading ASR task in the dual-accelerator regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                with lock:
                    events.append(f"asr1_acquired_{unit_id}")
                acquired_event.set()
                release_event.wait(timeout=10.0)
                with lock:
                    events.append("asr1_done")
    finally:
        model_manager.decrement_active_session()


def _run_dual_accelerator_detect(idx, events, lock):
    """Run one detect-language task in the dual-accelerator regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            model_manager.wait_for_priority()
            with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                with lock:
                    events.append(f"detect_{idx}_acquired_{unit_id}")
                time.sleep(0.05)
                with lock:
                    events.append(f"detect_{idx}_done")
    finally:
        model_manager.decrement_active_session()


def _run_dual_accelerator_second_asr(events, lock, acquired_event):
    """Run the trailing ASR task in the dual-accelerator regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                with lock:
                    events.append(f"asr2_acquired_{unit_id}")
                acquired_event.set()
                time.sleep(0.05)
                with lock:
                    events.append("asr2_done")
    finally:
        model_manager.decrement_active_session()


def _start_indexed_threads(worker, count, worker_args=(), delay=0.01):
    """Start indexed worker threads with a small stagger between launches."""
    threads = []
    for idx in range(count):
        thread = threading.Thread(target=worker, args=(idx, *worker_args))
        thread.start()
        threads.append(thread)
        time.sleep(delay)
    return threads


def _join_threads(threads, timeout=8.0):
    """Join a list of threads."""
    for thread in threads:
        thread.join(timeout=timeout)


def _exercise_dual_accelerator_regression():
    """Run the dual-accelerator regression and return its final state."""
    _setup_units(
        [
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ]
    )

    events = []
    lock = threading.Lock()
    asr1_acquired = threading.Event()
    asr1_release = threading.Event()
    asr2_acquired = threading.Event()

    def fake_init_unit(unit):
        """Avoid loading real engines in this concurrency regression."""
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()

    with mock.patch("modules.inference.runtime.model_manager.init_unit", side_effect=fake_init_unit):
        with _auto_confirm_priority_waits():
            t_asr1 = threading.Thread(
                target=_run_dual_accelerator_first_asr,
                args=(events, lock, asr1_acquired, asr1_release),
            )
            t_asr1.start()

            asr1_ready = asr1_acquired.wait(timeout=3.0)

            detect_threads = _start_indexed_threads(_run_dual_accelerator_detect, 3, worker_args=(events, lock))
            t_asr2 = threading.Thread(target=_run_dual_accelerator_second_asr, args=(events, lock, asr2_acquired))
            t_asr2.start()

            _join_threads(detect_threads, timeout=8.0)
            detect_threads_finished = all(not thread.is_alive() for thread in detect_threads)
            asr2_ready = asr2_acquired.wait(timeout=5.0)

            asr1_release.set()
            t_asr1.join(timeout=8.0)
            t_asr2.join(timeout=8.0)

    return {
        "asr1_ready": asr1_ready,
        "detect_threads_finished": detect_threads_finished,
        "asr2_ready": asr2_ready,
        "asr1_alive": t_asr1.is_alive(),
        "asr2_alive": t_asr2.is_alive(),
        "asr2_acquired": any(event.startswith("asr2_acquired_") for event in events),
        "asr2_done": "asr2_done" in events,
    }


def _assert_task_pair_order(events, prefix):
    """Assert two numbered tasks completed in FIFO order."""
    _assert_events_in_order(
        events,
        [
            (f"{prefix}_0_registered", f"{prefix}_1_registered"),
            (f"{prefix}_0_acquired", f"{prefix}_1_acquired"),
            (f"{prefix}_0_done", f"{prefix}_1_done"),
        ],
    )


def test_asr_tasks_processed_in_fifo_order():
    """Multiple ASR tasks must be processed in the order they arrived, not skipping."""
    _setup_units([{"id": "CPU", "type": "CPU", "name": "Host CPU"}])

    acquisition_order = []
    lock = threading.Lock()

    # Mock _try_acquire_unit_now to track acquisition order
    original_try_acquire = concurrency._try_acquire_unit_now

    def track_acquisition():
        unit = original_try_acquire()
        if unit:
            task_id = getattr(utils.THREAD_CONTEXT, "task_id", "unknown")
            with lock:
                acquisition_order.append(task_id)
        return unit

    with mock.patch("modules.inference.runtime.concurrency._try_acquire_unit_now", side_effect=track_acquisition):
        threads, proceed_events = _start_fifo_tasks(
            4,
            partial(_run_fifo_asr_task, acquisition_order=acquisition_order, lock=lock),
        )
        _release_fifo_tasks(threads, proceed_events)

    # Verify all tasks completed
    assert all(not t.is_alive() for t in threads)
    # Verify all tasks acquired units in FIFO order.
    assert [entry for entry in acquisition_order if entry.endswith("_acquired")] == [
        "asr_0_acquired",
        "asr_1_acquired",
        "asr_2_acquired",
        "asr_3_acquired",
    ]


def test_detect_language_tasks_processed_in_fifo_order():
    """Multiple detect-language (priority) tasks must be processed in arrival order."""
    _setup_units(
        [
            {"id": "CPU", "type": "CPU", "name": "Host CPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ]
    )

    acquisition_order = []
    events = []
    lock = threading.Lock()

    # Ensure no leftover FFmpeg gating from other tests can block priority admission.
    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 0
        utils.STANDARD_FFMPEG_COND.notify_all()

    with _auto_confirm_priority_waits():
        # Start 3 priority tasks in sequence
        threads, proceed_events = _start_fifo_tasks(
            3,
            partial(
                _run_fifo_priority_task,
                events=events,
                acquisition_order=acquisition_order,
                lock=lock,
                delay=0.02,
            ),
        )
        _release_fifo_tasks(threads, proceed_events)

        # Verify all tasks completed
        assert all(not t.is_alive() for t in threads)
        # Verify all tasks went through registration and acquisition
        assert len(acquisition_order) == 3
        _assert_detect_language_fifo_events(events)
