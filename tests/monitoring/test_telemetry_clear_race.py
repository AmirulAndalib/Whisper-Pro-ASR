"""Tests for telemetry race conditions and locks."""

import os
import threading
import time
from unittest import mock

from modules.monitoring import telemetry_manager

MOCK_STATS = {
    "system": {
        "cpu_percent": 12.5,
        "app_cpu_percent": 5.0,
        "memory_percent": 60.0,
        "memory_used_gb": 8.0,
        "app_memory_gb": 1.2,
    },
    "telemetry": {
        "nvidia": [{"util": 15}],
        "intel_gpu_load": 0,
        "npu_load": 0,
    },
}


def test_telemetry_clear_race(tmp_path):
    """Test racing record_snapshot and clear_telemetry_history."""
    test_telemetry_file = os.path.join(tmp_path, "test_telemetry.json")

    with mock.patch("modules.monitoring.telemetry_manager.TELEMETRY_FILE", test_telemetry_file):
        with mock.patch("modules.monitoring.telemetry_manager.config.STATE_DIR", str(tmp_path)):
            os.makedirs(os.path.dirname(test_telemetry_file), exist_ok=True)

            stop_event = threading.Event()
            errors = []

            def run_writer():
                while not stop_event.is_set():
                    try:
                        telemetry_manager.record_snapshot(MOCK_STATS)
                    except Exception as exc:  # noqa: BLE001 - surface thread errors in test assertion
                        errors.append(exc)
                        return
                    time.sleep(0.001)

            def run_clearer():
                while not stop_event.is_set():
                    try:
                        telemetry_manager.clear_telemetry_history()
                    except Exception as exc:  # noqa: BLE001 - surface thread errors in test assertion
                        errors.append(exc)
                        return
                    time.sleep(0.001)

            writer_thread = threading.Thread(target=run_writer)
            clearer_thread = threading.Thread(target=run_clearer)

            writer_thread.start()
            clearer_thread.start()

            time.sleep(0.5)
            stop_event.set()

            writer_thread.join(timeout=5)
            clearer_thread.join(timeout=5)
            assert not writer_thread.is_alive(), "writer thread did not terminate (possible deadlock)"
            assert not clearer_thread.is_alive(), "clearer thread did not terminate (possible deadlock)"
            assert not errors, f"background thread raised: {errors}"

            # Clear one final time
            telemetry_manager.clear_telemetry_history()

            # Assert file is removed and history is empty
            assert not os.path.exists(test_telemetry_file) or len(telemetry_manager.get_telemetry_history()) == 0
