import json
import os
import time
from unittest import mock

import pytest

from modules.monitoring import telemetry_manager


def _telemetry_leftovers(tmp_path):
    return [path for path in tmp_path.glob("telemetry_*.json") if path.name != "telemetry_history.json"]


@pytest.fixture
def mock_telemetry_file(tmp_path):
    temp_file = tmp_path / "telemetry_history.json"
    with mock.patch("modules.monitoring.telemetry_manager.TELEMETRY_FILE", str(temp_file)):
        with mock.patch("modules.monitoring.telemetry_manager.config.STATE_DIR", str(tmp_path)):
            yield temp_file


def test_get_telemetry_history_empty(mock_telemetry_file):
    assert telemetry_manager.get_telemetry_history() == []


def test_record_snapshot(mock_telemetry_file):
    stats = {
        "system": {
            "cpu_percent": 10.0,
            "app_cpu_percent": 5.0,
            "memory_percent": 50.0,
            "memory_used_gb": 8.0,
            "app_memory_gb": 1.0,
        },
        "telemetry": {
            "nvidia": [{"util": 20}],
            "intel_gpu_load": 10,
            "npu_load": 5,
            "hardware_util": {"GPU.0": 88, "CUDA:0": 20},
        },
    }
    telemetry_manager.record_snapshot(stats)

    history = telemetry_manager.get_telemetry_history()
    _assert_record_snapshot_payload(history[0])


def _assert_record_snapshot_payload(snapshot):
    assert snapshot["cpu_sys"] == 10.0
    assert snapshot["mem_sys_gb"] == 8.0
    assert snapshot["nvidia_util"] == [20]
    assert snapshot["hardware_util"] == {"GPU.0": 88, "CUDA:0": 20}


def test_record_snapshot_pruning(mock_telemetry_file):
    # Setup history with an old snapshot
    old_time = int(time.time()) - 100000  # ~27 hours ago
    history = [{"timestamp": old_time, "cpu_sys": 1.0}]
    with open(mock_telemetry_file, "w") as f:
        json.dump(history, f)

    stats = {
        "system": {
            "cpu_percent": 10.0,
            "app_cpu_percent": 5.0,
            "memory_percent": 50.0,
            "memory_used_gb": 8.0,
            "app_memory_gb": 1.0,
        },
        "telemetry": {},
    }
    # With default 24h retention, the old snapshot should be pruned
    telemetry_manager.record_snapshot(stats)

    new_history = telemetry_manager.get_telemetry_history()
    assert len(new_history) == 1
    assert new_history[0]["cpu_sys"] == 10.0
    assert new_history[0]["mem_sys_gb"] == 8.0


def test_record_snapshot_limit(mock_telemetry_file):
    history = [{"timestamp": int(time.time()), "cpu_sys": float(i)} for i in range(2005)]
    with open(mock_telemetry_file, "w") as f:
        json.dump(history, f)

    stats = {
        "system": {
            "cpu_percent": 99.0,
            "app_cpu_percent": 5.0,
            "memory_percent": 50.0,
            "memory_used_gb": 8.0,
            "app_memory_gb": 1.0,
        },
        "telemetry": {},
    }
    telemetry_manager.record_snapshot(stats)

    new_history = telemetry_manager.get_telemetry_history()
    assert len(new_history) == 2000
    assert new_history[-1]["cpu_sys"] == 99.0


def test_update_retention():
    telemetry_manager.update_retention(telemetry_hours=48, log_days=14)
    assert os.environ["TELEMETRY_RETENTION_HOURS"] == "48"
    assert os.environ["LOG_RETENTION_DAYS"] == "14"


def test_get_telemetry_history_corrupt(mock_telemetry_file):
    with open(mock_telemetry_file, "w") as f:
        f.write("corrupt json")
    assert telemetry_manager.get_telemetry_history() == []


def test_clear_telemetry_history(mock_telemetry_file):
    stats = {
        "system": {
            "cpu_percent": 10.0,
            "app_cpu_percent": 5.0,
            "memory_percent": 50.0,
            "memory_used_gb": 8.0,
            "app_memory_gb": 1.0,
        },
        "telemetry": {},
    }
    telemetry_manager.record_snapshot(stats)
    assert os.path.exists(mock_telemetry_file)
    assert len(telemetry_manager.get_telemetry_history()) == 1

    telemetry_manager.clear_telemetry_history()
    assert not os.path.exists(mock_telemetry_file)
    assert telemetry_manager.get_telemetry_history() == []


def test_clear_telemetry_history_oserror(mock_telemetry_file):
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.remove", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError):
                telemetry_manager.clear_telemetry_history()


def test_record_snapshot_exception(mock_telemetry_file):
    """record_snapshot should raise when stats is missing required keys and leave no partial file."""
    stats = {}
    with pytest.raises((KeyError, TypeError)):
        telemetry_manager.record_snapshot(stats)
    assert not mock_telemetry_file.exists()
    assert telemetry_manager.get_telemetry_history() == []


def test_record_snapshot_atomic_write_cleanup_on_replace_error(mock_telemetry_file, tmp_path):
    """record_snapshot should remove temporary file and re-raise when atomic replace fails."""
    stats = {
        "system": {
            "cpu_percent": 10.0,
            "app_cpu_percent": 5.0,
            "memory_percent": 50.0,
            "memory_used_gb": 8.0,
            "app_memory_gb": 1.0,
        },
        "telemetry": {},
    }

    with mock.patch("modules.monitoring.telemetry_manager.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError):
            telemetry_manager.record_snapshot(stats)

    # Temporary files created for atomic writes must be cleaned on failure.
    assert _telemetry_leftovers(tmp_path) == []


@pytest.mark.parametrize("dump_error", [TypeError("bad dump"), ValueError("bad dump")])
def test_record_snapshot_atomic_write_cleanup_on_dump_error(mock_telemetry_file, tmp_path, dump_error):
    """record_snapshot should preserve existing file and clean temp file when json.dump fails."""
    original_history = [{"timestamp": int(time.time()), "cpu_sys": 7.0}]
    with open(mock_telemetry_file, "w", encoding="utf-8") as file_obj:
        json.dump(original_history, file_obj)
    original_contents = mock_telemetry_file.read_text(encoding="utf-8")

    stats = {
        "system": {
            "cpu_percent": 10.0,
            "app_cpu_percent": 5.0,
            "memory_percent": 50.0,
            "memory_used_gb": 8.0,
            "app_memory_gb": 1.0,
        },
        "telemetry": {},
    }

    with mock.patch("modules.monitoring.telemetry_manager.json.dump", side_effect=dump_error):
        with pytest.raises(type(dump_error)):
            telemetry_manager.record_snapshot(stats)

    # Existing telemetry file should remain unchanged when dump fails.
    assert mock_telemetry_file.read_text(encoding="utf-8") == original_contents
    assert telemetry_manager.get_telemetry_history() == original_history

    # Temporary files created for atomic writes must be cleaned on failure.
    assert _telemetry_leftovers(tmp_path) == []
