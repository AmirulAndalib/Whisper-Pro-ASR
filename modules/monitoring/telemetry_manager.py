"""
Persistent Telemetry History Manager
Records system resource utilization over time.
"""

import json
import logging
import os
import tempfile
import threading
import time

from modules.core import config

logger = logging.getLogger(__name__)
_FILE_LOCK = threading.Lock()
TELEMETRY_FILE = os.path.join(config.STATE_DIR, "telemetry_history.json")


def get_telemetry_history():
    """Retrieves the list of recorded telemetry snapshots."""
    if not os.path.exists(TELEMETRY_FILE):
        return []
    try:
        with open(TELEMETRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return []


def record_snapshot(stats):
    """
    Appends a new resource snapshot to the history and prunes old data.
    'stats' should contain 'system' and 'telemetry' from model_manager.get_service_stats()
    """
    # Ensure the state directory exists
    os.makedirs(config.STATE_DIR, exist_ok=True)
    with _FILE_LOCK:
        temp_path = None
        try:
            history = get_telemetry_history()

            # Build snapshot
            snapshot = {
                "timestamp": int(time.time()),
                "cpu_sys": stats["system"]["cpu_percent"],
                "cpu_app": stats["system"]["app_cpu_percent"],
                "mem_sys": stats["system"]["memory_percent"],
                "mem_sys_gb": stats["system"].get("memory_used_gb", 0.0),
                "mem_app_gb": stats["system"]["app_memory_gb"],
                "nvidia_util": [g["util"] for g in stats["telemetry"].get("nvidia", [])],
                "intel_util": stats["telemetry"].get("intel_gpu_load", 0),
                "npu_util": stats["telemetry"].get("npu_load", 0),
            }

            history.append(snapshot)

            # Pruning based on retention (default 24h)
            retention_hours = int(os.environ.get("TELEMETRY_RETENTION_HOURS", 24))
            cutoff = int(time.time()) - (retention_hours * 3600)
            history = [s for s in history if s["timestamp"] > cutoff]

            # Limit total points to prevent JSON bloat (e.g. max 2000 points = ~33 hours of 1-min snapshots)
            if len(history) > 2000:
                history = history[-2000:]

            fd, temp_path = tempfile.mkstemp(dir=config.STATE_DIR, prefix="telemetry_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(history, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(temp_path, TELEMETRY_FILE)
            temp_path = None
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to record telemetry atomically: %s", e)
            raise
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as cleanup_error:
                    logger.warning("Failed to cleanup telemetry temp file %s: %s", temp_path, cleanup_error)


def update_retention(telemetry_hours=None, log_days=None):
    """Updates retention periods in the environment."""
    if telemetry_hours is not None:
        os.environ["TELEMETRY_RETENTION_HOURS"] = str(telemetry_hours)
    if log_days is not None:
        os.environ["LOG_RETENTION_DAYS"] = str(log_days)


def clear_telemetry_history() -> None:
    """Purge all recorded telemetry snapshots."""
    with _FILE_LOCK:
        try:
            if os.path.exists(TELEMETRY_FILE):
                os.remove(TELEMETRY_FILE)
        except OSError as e:
            logger.error("Failed to clear telemetry history: %s", e)
            raise
