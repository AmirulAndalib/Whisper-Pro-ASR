"""Scenario helpers split from test_priority_concurrency.py."""

import threading
import time
from unittest import mock

from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.inference.scheduler.priority.test_priority_concurrency import (
    helper_run_priority,
    helper_run_priority_counted,
    helper_run_transcription,
)


def _exercise_zero_hardware_units_execution():
    """Run the CPU-fallback concurrency scenario and return a compact summary."""
    from modules.inference.scheduler import SchedulerState

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", []),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        events = []
        t_trans = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_p = threading.Thread(target=helper_run_priority, args=(events, "P", 0.1))

        t_trans.start()
        time.sleep(0.1)
        t_p.start()

        t_trans.join(timeout=10.0)
        t_p.join(timeout=10.0)

        return {
            "threads_done": not t_trans.is_alive() and not t_p.is_alive(),
            "priority_on_cpu": "priority_P_running_on_CPU" in events,
            "priority_done_before_transcription": events.index("priority_P_done") < events.index("transcription_1_done"),
        }


def _exercise_priority_non_preemptive():
    """Run the idle-unit priority scenario and return a compact summary."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}, {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}]
    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()
        model_manager.MODEL_POOL["GPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["GPU.0"] = mock.MagicMock()

        events = []
        t_trans = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_prio = threading.Thread(target=helper_run_priority, args=(events, "P", 0.1))

        t_trans.start()
        time.sleep(0.1)
        t_prio.start()

        t_trans.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        return {
            "threads_done": not t_trans.is_alive() and not t_prio.is_alive(),
            "priority_on_idle_unit": "priority_P_running_on_GPU.0" in events,
            "no_pause_requested": not scheduler.STATE.pause_requested.is_set()
            and not scheduler.STATE.unit_sync["NPU.0"]["pause_requested"].is_set()
            and not scheduler.STATE.unit_sync["GPU.0"]["pause_requested"].is_set(),
        }


def _exercise_single_unit_matrix(unit):
    """Run the single-unit matrix scenario and return a compact summary."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [unit]
    unit_id = unit["id"]
    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL[unit_id] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit_id] = mock.MagicMock()

        events = []
        t_std = threading.Thread(target=helper_run_transcription, args=(events, "matrix", 3, 0.2))
        t_pri = threading.Thread(target=helper_run_priority, args=(events, "matrix", 0.1))

        t_std.start()
        time.sleep(0.05)
        t_pri.start()

        t_std.join(timeout=10.0)
        t_pri.join(timeout=10.0)

        return {
            "threads_done": not t_std.is_alive() and not t_pri.is_alive(),
            "priority_running": f"priority_matrix_running_on_{unit_id}" in events,
            "priority_done": "priority_matrix_done" in events,
            "transcription_done": "transcription_matrix_done" in events,
        }


def _exercise_dual_unit_matrix(hw_list):
    """Run the dual-unit matrix scenario and return a compact summary."""
    from modules.inference.scheduler import SchedulerState

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        unit_a = hw_list[0]["id"]
        unit_b = hw_list[1]["id"]
        model_manager.MODEL_POOL[unit_a] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit_a] = mock.MagicMock()
        model_manager.MODEL_POOL[unit_b] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit_b] = mock.MagicMock()

        events = []
        t1 = threading.Thread(target=helper_run_transcription, args=(events, "a", 3, 0.2))
        t2 = threading.Thread(target=helper_run_transcription, args=(events, "b", 3, 0.2))
        tp = threading.Thread(target=helper_run_priority, args=(events, "dual", 0.1))

        t1.start()
        t2.start()
        time.sleep(0.08)
        tp.start()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)
        tp.join(timeout=10.0)

        return {
            "threads_done": not t1.is_alive() and not t2.is_alive() and not tp.is_alive(),
            "priority_running": events.count("priority_dual_running_on_" + unit_a) + events.count("priority_dual_running_on_" + unit_b) > 0,
            "priority_done": "priority_dual_done" in events,
            "transcriptions_done": events.count("transcription_a_done") + events.count("transcription_b_done") == 2,
        }


def _exercise_parallel_detectlang_two_asr_yielded():
    """Run the dual-ASR parallel detect-language scenario and return the final state."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}, {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()

        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()
        model_manager.MODEL_POOL["GPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["GPU.0"] = mock.MagicMock()

        events = []
        lock = threading.Lock()
        counters = {"running_priority": 0, "max_running_priority": 0}

        t_asr_1 = threading.Thread(target=helper_run_transcription, args=(events, "A1", 5, 0.12))
        t_asr_2 = threading.Thread(target=helper_run_transcription, args=(events, "A2", 5, 0.12))
        t_asr_1.start()
        t_asr_2.start()
        time.sleep(0.15)

        t_p1 = threading.Thread(target=helper_run_priority_counted, args=(events, "P1", counters, lock, 0.2))
        t_p2 = threading.Thread(target=helper_run_priority_counted, args=(events, "P2", counters, lock, 0.2))
        t_p3 = threading.Thread(target=helper_run_priority_counted, args=(events, "P3", counters, lock, 0.2))
        t_p4 = threading.Thread(target=helper_run_priority_counted, args=(events, "P4", counters, lock, 0.2))
        t_p1.start()
        time.sleep(0.03)
        t_p2.start()
        time.sleep(0.03)
        t_p3.start()
        time.sleep(0.03)
        t_p4.start()

        t_p1.join(timeout=12.0)
        t_p2.join(timeout=12.0)
        t_p3.join(timeout=12.0)
        t_p4.join(timeout=12.0)
        t_asr_1.join(timeout=12.0)
        t_asr_2.join(timeout=12.0)

        return {
            "prio_threads_done": not t_p1.is_alive() and not t_p2.is_alive() and not t_p3.is_alive() and not t_p4.is_alive(),
            "asr_threads_done": not t_asr_1.is_alive() and not t_asr_2.is_alive(),
            "max_running_priority": counters["max_running_priority"],
            "priority_done_count": events.count("priority_P1_done")
            + events.count("priority_P2_done")
            + events.count("priority_P3_done")
            + events.count("priority_P4_done"),
        }
