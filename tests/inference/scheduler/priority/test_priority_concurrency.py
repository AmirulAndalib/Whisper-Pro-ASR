"""
Concurrency tests for Whisper Pro ASR scheduler preemption logic.
"""

import threading
import time
from unittest import mock

from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager


def helper_run_transcription(events, name, steps=3, step_delay=0.3):
    """Simulates running a standard transcription task with preemption checks."""
    events.append(f"transcription_{name}_start")
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx() as (_, unit_id):
                events.append(f"transcription_{name}_running_on_{unit_id}")
                # Simulate segment loops with periodic preemption checks
                for i in range(steps):
                    time.sleep(step_delay)
                    events.append(f"transcription_{name}_check_preempt_{i}")
                    model_manager._check_preemption()
                events.append(f"transcription_{name}_done")
    finally:
        model_manager.decrement_active_session()


def helper_run_priority(events, name, task_delay=0.1):
    """Simulates running a high-priority task."""
    events.append(f"priority_{name}_start")
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            events.append(f"priority_{name}_waiting")
            model_manager.wait_for_priority()
            events.append(f"priority_{name}_waited")

            with model_manager.model_lock_ctx() as (_, unit_id):
                events.append(f"priority_{name}_running_on_{unit_id}")
                time.sleep(task_delay)
                events.append(f"priority_{name}_done")
    finally:
        model_manager.decrement_active_session()


def helper_run_priority_counted(events, name, counters, lock, task_delay=0.1):
    """Simulates a priority task while tracking concurrent overlap and units used."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            model_manager.wait_for_priority()
            events.append(f"priority_{name}_waited")
            with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                with lock:
                    counters["running_priority"] += 1
                    counters["max_running_priority"] = max(counters["max_running_priority"], counters["running_priority"])
                    if "units_used" in counters:
                        counters["units_used"].add(unit_id)
                events.append(f"priority_{name}_running_on_{unit_id}")
                time.sleep(task_delay)
                events.append(f"priority_{name}_done")
                with lock:
                    counters["running_priority"] -= 1
    finally:
        model_manager.decrement_active_session()


def _run_targeted_preemption_standard_task(events, unit_name):
    """Run a standard task used by the targeted-preemption regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False) as (_, actual_unit_id):
                events.append(f"std_{unit_name}_running")
                events.append(f"std_{unit_name}_running_on_{actual_unit_id}")
                model_manager._check_preemption()
                events.append(f"std_{unit_name}_checked_1")
                time.sleep(0.3)
                model_manager._check_preemption()
                events.append(f"std_{unit_name}_checked_2")
    finally:
        model_manager.decrement_active_session()


def _run_targeted_preemption_priority_task(events, result):
    """Run the targeted priority task used by the 3-unit preemption regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            model_manager.wait_for_priority()
            target = getattr(utils.THREAD_CONTEXT, "target_unit_id", "CUDA.0")
            result["targeted_unit"] = target
            events.append(f"priority_waited_for_{target}")
            with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                events.append(f"priority_running_on_{unit_id}")
                time.sleep(0.1)
                events.append(f"priority_done_on_{unit_id}")
    finally:
        model_manager.decrement_active_session()


def _join_threads(threads, timeout=10.0):
    for thread in threads:
        thread.join(timeout=timeout)


def _exercise_targeted_preemption_npu_gpu_cuda():
    """Run the 3-unit targeted-preemption regression and return the outcome summary."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
    ]

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
        model_manager.MODEL_POOL["CUDA.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CUDA.0"] = mock.MagicMock()

        events = []
        result = {"targeted_unit": ""}

        t_npu = threading.Thread(target=_run_targeted_preemption_standard_task, args=(events, "NPU.0"))
        t_gpu = threading.Thread(target=_run_targeted_preemption_standard_task, args=(events, "GPU.0"))
        t_cuda = threading.Thread(target=_run_targeted_preemption_standard_task, args=(events, "CUDA.0"))

        t_npu.start()
        t_gpu.start()
        time.sleep(0.05)
        t_cuda.start()
        time.sleep(0.1)

        t_prio = threading.Thread(target=_run_targeted_preemption_priority_task, args=(events, result))
        t_prio.start()

        t_npu.join(timeout=5.0)
        t_gpu.join(timeout=5.0)
        t_cuda.join(timeout=5.0)
        t_prio.join(timeout=5.0)

        targeted_unit = result["targeted_unit"]
        std_started = {"std_NPU.0_running", "std_GPU.0_running", "std_CUDA.0_running"}
        std_finished = {"std_NPU.0_checked_2", "std_GPU.0_checked_2", "std_CUDA.0_checked_2"}

        return {
            "threads_done": _all_threads_done(t_npu, t_gpu, t_cuda, t_prio),
            "all_standard_started": std_started.issubset(events),
            "targeted_unit": targeted_unit,
            "prio_done_before_target_second_check": _priority_done_before_target_second_check(events, targeted_unit),
            "all_units_completed_second_check": std_finished.issubset(events),
        }


def _find_target_standard_label(events, targeted_unit):
    marker = "_running_on_"
    for event in events:
        if marker not in event or not event.startswith("std_"):
            continue
        label, _, actual_unit = event.partition(marker)
        if actual_unit == targeted_unit:
            return label.removeprefix("std_")
    return None


def _priority_done_before_target_second_check(events, targeted_unit):
    target_standard_label = _find_target_standard_label(events, targeted_unit)
    if not target_standard_label:
        return False

    priority_done_event = f"priority_done_on_{targeted_unit}"
    target_standard_second_check = f"std_{target_standard_label}_checked_2"
    if priority_done_event not in events or target_standard_second_check not in events:
        return False

    return events.index(priority_done_event) < events.index(target_standard_second_check)


def _all_threads_done(*threads):
    return all(not thread.is_alive() for thread in threads)


def _exercise_priority_burst_no_livelock():
    """Run the priority-burst regression and return a compact success summary."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]
    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

        events, threads = _start_priority_burst_threads()
        _join_threads(threads, timeout=10.0)
        pair_indices = _priority_burst_pair_indices(events)
        t_p1, t_p2, t_p3, t_p4 = threads

        return {
            "threads_done": not t_p1.is_alive() and not t_p2.is_alive() and not t_p3.is_alive() and not t_p4.is_alive(),
            "all_done_events": {"priority_p1_done", "priority_p2_done", "priority_p3_done", "priority_p4_done"}.issubset(events),
            "run_order_pairs_valid": pair_indices == sorted(pair_indices),
        }


def _start_priority_burst_threads():
    events = []
    threads = [
        threading.Thread(target=helper_run_priority, args=(events, "p1", 0.08)),
        threading.Thread(target=helper_run_priority, args=(events, "p2", 0.08)),
        threading.Thread(target=helper_run_priority, args=(events, "p3", 0.08)),
        threading.Thread(target=helper_run_priority, args=(events, "p4", 0.08)),
    ]
    for thread in threads:
        thread.start()
        time.sleep(0.02)
    return events, threads


def _priority_burst_pair_indices(events):
    p1_run = _find_event_by_prefix(events, "priority_p1_running_on_")
    p2_run = _find_event_by_prefix(events, "priority_p2_running_on_")
    p3_run = _find_event_by_prefix(events, "priority_p3_running_on_")
    p4_run = _find_event_by_prefix(events, "priority_p4_running_on_")
    return [
        events.index(p1_run),
        events.index("priority_p1_done"),
        events.index(p2_run),
        events.index("priority_p2_done"),
        events.index(p3_run),
        events.index("priority_p3_done"),
        events.index(p4_run),
        events.index("priority_p4_done"),
    ]


def _find_event_by_prefix(events, prefix):
    return next(event for event in events if event.startswith(prefix))


def _setup_priority_scheduler(hw_list):
    """Reset scheduler state and populate model pools for a hardware layout."""
    from modules.inference.scheduler import SchedulerState

    scheduler.STATE = SchedulerState()
    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    for unit in hw_list:
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()


def _run_standard_gpu_preemption(events):
    """Run the non-targeted standard task in the targeted-preemption regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False):
                events.append("std_gpu_running")
                model_manager._check_preemption()
                events.append("std_gpu_checked_no_pause")
    finally:
        model_manager.decrement_active_session()


def _run_standard_npu_preemption(events):
    """Run the targeted standard task in the targeted-preemption regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False):
                events.append("std_npu_running")
                model_manager._check_preemption()
                events.append("std_npu_checked_no_pause")
                time.sleep(0.3)
                model_manager._check_preemption()
                events.append("std_npu_resumed")
    finally:
        model_manager.decrement_active_session()


def _run_targeted_priority_preemption(events):
    """Run the priority task in the targeted-preemption regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            model_manager.wait_for_priority()
            target = getattr(utils.THREAD_CONTEXT, "target_unit_id", "NPU.0")
            events.append(f"priority_waited_{target}")
            with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                events.append(f"priority_running_on_{unit_id}")
                time.sleep(0.1)
                events.append("priority_done")
    finally:
        model_manager.decrement_active_session()


def _run_resume_standard(events, barrier):
    """Run a standard task used by the multiple-priority resume regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False) as (_, acquired_unit):
                events.append(f"std_{acquired_unit}_running")
                model_manager._check_preemption()
                events.append(f"std_{acquired_unit}_paused")
                barrier.wait()
                model_manager._check_preemption()
                events.append(f"std_{acquired_unit}_resumed")
    finally:
        model_manager.decrement_active_session()


def _run_resume_priority(events, name, target, barrier):
    """Run a priority task used by the multiple-priority resume regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            barrier.wait()
            utils.THREAD_CONTEXT.target_unit_id = target
            model_manager.wait_for_priority()
            events.append(f"prio_{name}_waited")
            with model_manager.model_lock_ctx(priority=True) as (_, acquired_unit):
                events.append(f"prio_{name}_running_on_{acquired_unit}")
                time.sleep(0.1)
                events.append(f"prio_{name}_done")
    finally:
        model_manager.decrement_active_session()


def _run_barrier_standard(events, barrier):
    """Run a standard task used by the one-priority-remains regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False) as (_, acquired_unit):
                events.append(f"std_{acquired_unit}_running")
                barrier.wait()
                model_manager._check_preemption()
                events.append(f"std_{acquired_unit}_resumed")
    finally:
        model_manager.decrement_active_session()


def _run_barrier_priority(events, name, target, barrier):
    """Run a priority task used by the one-priority-remains regression."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            barrier.wait()
            utils.THREAD_CONTEXT.target_unit_id = target
            model_manager.wait_for_priority()
            with model_manager.model_lock_ctx(priority=True) as (_, acquired_unit):
                events.append(f"prio_{name}_running_on_{acquired_unit}")
                time.sleep(0.2)
                events.append(f"prio_{name}_done")
    finally:
        model_manager.decrement_active_session()


def _exercise_one_hardware_unit():
    """Run the single-NPU concurrency scenario and return a compact summary."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]
    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

        events = []
        t_trans = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_p1 = threading.Thread(target=helper_run_priority, args=(events, "P1", 0.1))
        t_p2 = threading.Thread(target=helper_run_priority, args=(events, "P2", 0.1))

        t_trans.start()
        time.sleep(0.1)
        t_p1.start()
        time.sleep(0.05)
        t_p2.start()

        t_trans.join(timeout=10.0)
        t_p1.join(timeout=10.0)
        t_p2.join(timeout=10.0)

        return {
            "threads_done": not t_trans.is_alive() and not t_p1.is_alive() and not t_p2.is_alive(),
            "p1_before_p2": events.index("priority_P1_done") < events.index("priority_P2_running_on_NPU.0"),
            "p1_running": "priority_P1_running_on_NPU.0" in events,
            "p2_running": "priority_P2_running_on_NPU.0" in events,
        }


def _exercise_two_hardware_units():
    """Run the dual-NPU/GPU concurrency scenario and return a compact summary."""
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
        t_trans1 = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans2 = threading.Thread(target=helper_run_transcription, args=(events, "2", 3, 0.3))
        t_prio = threading.Thread(target=helper_run_priority, args=(events, "P", 0.1))

        t_trans1.start()
        t_trans2.start()
        time.sleep(0.1)
        t_prio.start()

        t_trans1.join(timeout=10.0)
        t_trans2.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        return {
            "threads_done": not t_trans1.is_alive() and not t_trans2.is_alive() and not t_prio.is_alive(),
            "priority_running": "priority_P_running_on_NPU.0" in events or "priority_P_running_on_GPU.0" in events,
            "priority_done_before_one_transcription": events.index("priority_P_done") < events.index("transcription_1_done")
            or events.index("priority_P_done") < events.index("transcription_2_done"),
        }


def _exercise_three_hardware_units():
    """Run the tri-unit concurrency scenario and return a compact summary."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
    ]
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
        model_manager.MODEL_POOL["CUDA.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CUDA.0"] = mock.MagicMock()

        events = []
        t_trans1 = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans2 = threading.Thread(target=helper_run_transcription, args=(events, "2", 3, 0.3))
        t_trans3 = threading.Thread(target=helper_run_transcription, args=(events, "3", 3, 0.3))
        t_prio = threading.Thread(target=helper_run_priority, args=(events, "P", 0.1))

        t_trans1.start()
        t_trans2.start()
        t_trans3.start()
        time.sleep(0.1)
        t_prio.start()

        t_trans1.join(timeout=10.0)
        t_trans2.join(timeout=10.0)
        t_trans3.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        return {
            "threads_done": not t_trans1.is_alive() and not t_trans2.is_alive() and not t_trans3.is_alive() and not t_prio.is_alive(),
            "priority_running": events.count("priority_P_running_on_" + hw_list[0]["id"])
            + events.count("priority_P_running_on_" + hw_list[1]["id"])
            + events.count("priority_P_running_on_" + hw_list[2]["id"])
            > 0,
            "priority_done_before_one_transcription": events.index("priority_P_done")
            < min(
                events.index("transcription_1_done"),
                events.index("transcription_2_done"),
                events.index("transcription_3_done"),
            ),
        }
