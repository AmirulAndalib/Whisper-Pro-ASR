"""Additional FIFO ordering regressions split from test_priority_fifo_ordering.py."""

import threading
import time
from unittest import mock

from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.inference.scheduler.priority.test_priority_fifo_ordering import (
    _assert_detect_language_preempts_asr_events,
    _assert_numbered_task_pair_events,
    _clear_task_state,
    _exercise_dual_accelerator_regression,
    _run_simple_asr_task,
    _run_simple_priority_task,
    _run_task_pair,
    _seed_task_state,
    _setup_units,
)


def test_detect_language_preempts_asr_but_respects_fifo():
    """Verify priority detect-language and standard ASR tasks can both acquire resources."""
    _setup_units(
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "CPU", "type": "CPU", "name": "Host CPU"},
        ]
    )

    events = []
    lock = threading.Lock()

    t_asr, t_prio = _run_task_pair(_run_simple_asr_task, _run_simple_priority_task, events, lock)

    assert not t_asr.is_alive()
    assert not t_prio.is_alive()
    _assert_detect_language_preempts_asr_events(events)


def test_earlier_asr_task_blocks_later_asr_task():
    """Later ASR task should wait if earlier ASR task is still queued."""
    _setup_units([{"id": "CPU", "type": "CPU", "name": "Host CPU"}])

    events = []
    lock = threading.Lock()

    t_asr0, t_asr1 = _run_task_pair(
        _run_simple_asr_task,
        _run_simple_asr_task,
        events,
        lock,
        first_kwargs={"task_num": 0, "hold_time": 0.15},
        second_kwargs={"task_num": 1, "hold_time": 0.05},
    )

    t_asr0.join(timeout=10.0)
    t_asr1.join(timeout=10.0)

    assert not t_asr0.is_alive()
    assert not t_asr1.is_alive()
    _assert_numbered_task_pair_events(events, "asr")


def test_priority_task_does_not_skip_earlier_priority_task():
    """Later priority task should not acquire before earlier priority task."""
    _setup_units(
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "CPU", "type": "CPU", "name": "Host CPU"},
        ]
    )

    events = []
    lock = threading.Lock()

    t_prio0, t_prio1 = _run_task_pair(
        _run_simple_priority_task,
        _run_simple_priority_task,
        events,
        lock,
        first_kwargs={"task_num": 0, "hold_time": 0.1},
        second_kwargs={"task_num": 1, "hold_time": 0.05},
    )

    assert not t_prio0.is_alive()
    assert not t_prio1.is_alive()
    _assert_numbered_task_pair_events(events, "prio", enforce_done_order=False)


def test_has_earlier_task_correctly_identifies_earlier_tasks():
    """has_earlier_task() should respect priority levels when checking FIFO ordering."""
    _setup_units([{"id": "CPU", "type": "CPU", "name": "Host CPU"}])

    task_time_1 = time.time()
    _seed_task_state("task_1", task_time_1, is_priority=False, status="initializing")

    time.sleep(0.05)
    task_time_2 = time.time()
    _seed_task_state("task_2", task_time_2, is_priority=False, status="initializing")

    time.sleep(0.05)
    task_time_3 = time.time()
    _seed_task_state("task_3", task_time_3, is_priority=True, status="queued")

    assert [
        scheduler.has_earlier_task("task_2", is_priority=False),
        scheduler.has_earlier_task("task_1", is_priority=False),
        scheduler.has_earlier_task("task_3", is_priority=True),
    ] == [True, False, False]

    with scheduler.STATE.task_registry_lock:
        del scheduler.STATE.task_registry["task_1"]

    assert [
        scheduler.has_earlier_task("task_2", is_priority=False),
        scheduler.has_earlier_task("task_3", is_priority=True),
    ] == [False, False]

    _clear_task_state()


def test_asr_after_three_detectlang_not_stuck_waiting_hardware_dual_accelerator():
    """ASR -> 3 detectlang -> ASR must not leave the second ASR stuck waiting for hardware."""
    result = _exercise_dual_accelerator_regression()
    assert result == {
        "asr1_ready": True,
        "detect_threads_finished": True,
        "asr2_ready": True,
        "asr1_alive": False,
        "asr2_alive": False,
        "asr2_acquired": True,
        "asr2_done": True,
    }


def test_asr_starts_during_ongoing_detect_language_on_other_unit():
    """Verify that ASR can acquire an idle unit while detect-language is ongoing on another unit."""
    _setup_units(
        [
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ]
    )

    events = []
    lock = threading.Lock()
    detect_acquired = threading.Event()
    detect_release = threading.Event()
    asr_acquired = threading.Event()

    def fake_init_unit(unit):
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()

    def auto_confirm_priority_waits(stop_evt):
        while not stop_evt.is_set():
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
            for u_sync in scheduler.STATE.unit_sync.values():
                if u_sync["pause_requested"].is_set() and not u_sync["pause_confirmed"].is_set():
                    u_sync["pause_confirmed"].set()
            time.sleep(0.01)

    def run_detect():
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=True):
                model_manager.wait_for_priority()
                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    with lock:
                        events.append(f"detect_acquired_{unit_id}")
                    detect_acquired.set()
                    detect_release.wait(timeout=10.0)
                    with lock:
                        events.append("detect_done")
        finally:
            model_manager.decrement_active_session()

    def run_asr():
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=False):
                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    with lock:
                        events.append(f"asr_acquired_{unit_id}")
                    asr_acquired.set()
        finally:
            model_manager.decrement_active_session()

    with mock.patch("modules.inference.runtime.model_manager.init_unit", side_effect=fake_init_unit):
        stop_auto_confirm = threading.Event()
        auto_confirm_t = threading.Thread(target=auto_confirm_priority_waits, args=(stop_auto_confirm,), daemon=True)
        auto_confirm_t.start()

        t_detect = threading.Thread(target=run_detect)
        t_detect.start()

        try:
            assert detect_acquired.wait(timeout=3.0), "Detect task failed to acquire hardware"

            t_asr = threading.Thread(target=run_asr)
            t_asr.start()

            assert asr_acquired.wait(timeout=3.0), "ASR task remained stuck waiting for hardware while detect-language was ongoing"
            t_asr.join(timeout=2.0)
        finally:
            detect_release.set()
            stop_auto_confirm.set()
            auto_confirm_t.join(timeout=2.0)
            t_detect.join(timeout=3.0)

    with lock:
        assert any(e.startswith("detect_acquired_") for e in events)
        assert any(e.startswith("asr_acquired_") for e in events)
