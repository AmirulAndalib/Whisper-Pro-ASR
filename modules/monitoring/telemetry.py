"""
Telemetry and Statistics Collection for Whisper Pro ASR
"""

import logging
import threading
import time
from typing import Any

from modules.core import config, logging_setup, utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager
from modules.monitoring import history_manager, metrics_discovery

logger = logging.getLogger(__name__)
SERVICE_START_TIME: float = time.time()
_STOP_EVENT: threading.Event = threading.Event()
TELEMETRY_HISTORY: list[dict[str, Any]] = []
_TELEMETRY_LOCK: threading.Lock = threading.Lock()

_DISPLAYABLE_STATUSES: set[str] = {
    "initializing",
    "queued",
    "active",
    "post-processing",
    "completed",
    "failed",
}


def _normalize_status_value(status: Any) -> str:
    """Return a dashboard-safe status that never uses placeholder values."""
    status_key = str(status or "").strip().lower()
    if status_key in _DISPLAYABLE_STATUSES:
        return status_key
    return "initializing"


def _is_placeholder_stage(stage_text: Any) -> bool:
    """Return True when stage text is missing or looks like placeholder content."""
    normalized = str(stage_text or "").strip().lower()
    if not normalized:
        return True
    return _is_placeholder_token(normalized)


def _is_placeholder_token(normalized: str) -> bool:
    return (
        _is_sentinel_stage(normalized)
        or _is_ratio_placeholder(normalized)
        or "placeholder" in normalized
        or normalized in {"resume", "resuming"}
    )


def _is_sentinel_stage(normalized: str) -> bool:
    return normalized in {"none", "null", "undefined", "unknown", "na", "n/a"}


def _is_ratio_placeholder(normalized: str) -> bool:
    ratio_candidate = normalized.replace("(", "").replace(")", "").replace(" ", "")
    return ratio_candidate == "0/0"


def _default_stage_for_status(status: Any) -> str:
    """Return a deterministic dashboard stage label from task status."""
    status_key = _normalize_status_value(status)
    mapping = {
        "initializing": "Initializing",
        "queued": "Queued",
        "active": "Active",
        "post-processing": "Post-Processing",
        "completed": "Completed",
        "failed": "Failed",
    }
    return mapping.get(status_key, "Initializing")


def _normalize_stage_value(stage: Any, status: Any) -> str:
    """Ensure stage is always a concrete, non-placeholder dashboard label."""
    if stage is not None:
        normalized = str(stage).strip()
        if not _is_placeholder_stage(normalized):
            return normalized
    return _default_stage_for_status(status)


def _is_whisper_active_stage(stage_text: Any) -> bool:
    """Return True when a stage indicates Whisper is still doing ASR work."""
    normalized = str(stage_text or "").lower()
    return any(token in normalized for token in ("transcrib", "inference", "translat"))


def start_telemetry_loop() -> threading.Event:
    """Spawns the background telemetry collection thread."""
    thread = threading.Thread(target=_telemetry_worker, daemon=True)
    thread.start()
    return _STOP_EVENT


def _telemetry_worker() -> None:
    """Background worker for system metrics."""
    retention_hours = int(config.TELEMETRY_RETENTION_HOURS)
    max_points = (retention_hours * 3600) // 2

    while not _STOP_EVENT.is_set():
        try:
            metrics = utils.get_system_telemetry()
            with _TELEMETRY_LOCK:
                TELEMETRY_HISTORY.append(
                    {
                        "timestamp": time.time(),
                        "system": metrics,
                        "telemetry": {
                            "nvidia": metrics_discovery.get_nvidia_metrics(),
                            "intel_gpu_load": metrics_discovery.get_intel_gpu_load(),
                            "npu_load": metrics_discovery.get_npu_load(),
                            "hardware_util": metrics_discovery.get_all_hardware_utilization(),
                        },
                    }
                )
                if len(TELEMETRY_HISTORY) > max_points:
                    TELEMETRY_HISTORY.pop(0)
        except (OSError, ValueError, AttributeError, KeyError, TypeError, RuntimeError) as e:
            logger.debug("[Telemetry] Worker cycle failed: %s", e)
        time.sleep(2)


def get_service_stats() -> dict[str, Any]:
    """Consolidates service state for the dashboard."""
    tasks = _get_dashboard_tasks_snapshot()
    tasks.sort(key=_task_sort_key)
    history_stats = history_manager.get_history_stats()
    whisper_status, uvr_status = _resolve_engine_statuses(tasks)
    telemetry_snap = _get_telemetry_snapshot()
    latest_telemetry = _get_latest_telemetry(telemetry_snap)
    telemetry_snap = _downsample_telemetry(telemetry_snap)
    actual_active, actual_queued = _count_task_statuses(tasks)
    hw_units_with_status = _build_hardware_unit_statuses(tasks)

    return {
        "version": config.VERSION,
        "uptime_sec": time.time() - SERVICE_START_TIME,
        "scheduler": {"active": actual_active, "queued": actual_queued},
        "active_sessions": actual_active,
        "queued_sessions": actual_queued,
        "tasks": tasks,
        "telemetry_history": telemetry_snap,
        "hardware_units": hw_units_with_status,
        "history": history_stats[0],
        "history_stats": history_stats[1],
        "telemetry": latest_telemetry,
        "engines": {
            "whisper": {
                "status": whisper_status,
                "model": utils.get_pretty_model_name(config.MODEL_ID),
                "device": config.DEVICE,
                "compute_type": config.COMPUTE_TYPE,
            },
            "uvr": {"status": uvr_status, "model": utils.get_pretty_model_name(config.VOCAL_SEPARATION_MODEL)},
        },
    }


def _get_dashboard_tasks_snapshot() -> list[dict[str, Any]]:
    with scheduler.STATE.task_registry_lock:
        tasks = []
        for tid, task in scheduler.STATE.task_registry.items():
            task_copy = task.copy()
            task_copy["status"] = _normalize_status_value(task_copy.get("status"))
            task_copy["stage"] = _normalize_stage_value(task_copy.get("stage"), task_copy.get("status"))
            task_copy["logs"] = logging_setup.TASK_LOGS.get(tid, [])
            tasks.append(task_copy)
        return tasks


def _task_sort_key(task: dict[str, Any]) -> tuple[int, float, str]:
    status = task.get("status", "unknown")
    start_time = float(task.get("start_time", 0.0) or 0.0)
    task_id = str(task.get("task_id", ""))
    tier = _task_status_tier(status)
    return (tier, start_time, task_id)


def _task_status_tier(status: str) -> int:
    if status == "active":
        return 0
    return 1


def _is_uvr_active_stage(stage_text: Any) -> bool:
    return any(token in str(stage_text or "").lower() for token in ["isolation", "separation", "uvr"])


def _resolve_engine_statuses(tasks: list[dict[str, Any]]) -> tuple[str, str]:
    return _resolve_whisper_status(tasks), _resolve_uvr_status(tasks)


def _resolve_whisper_status(tasks: list[dict[str, Any]]) -> str:
    if any(t.get("status") == "active" and _is_whisper_active_stage(t.get("stage")) for t in tasks):
        return "busy"
    return "loaded" if model_manager.is_engine_actually_loaded() else "ready"


def _resolve_uvr_status(tasks: list[dict[str, Any]]) -> str:
    if any(t.get("status") == "active" and _is_uvr_active_stage(t.get("stage")) for t in tasks):
        return "busy"
    return "loaded" if model_manager.is_uvr_actually_loaded() else "ready"


def _get_telemetry_snapshot() -> list[dict[str, Any]]:
    with _TELEMETRY_LOCK:
        return TELEMETRY_HISTORY[:]


def _get_latest_telemetry(telemetry_snap: list[dict[str, Any]]) -> dict[str, Any]:
    if telemetry_snap:
        return telemetry_snap[-1].get("telemetry", {})
    return {
        "nvidia": [],
        "intel_gpu_load": 0,
        "npu_load": 0,
        "hardware_util": {},
    }


def _downsample_telemetry(telemetry_snap: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(telemetry_snap) <= 300:
        return telemetry_snap
    sampled = [telemetry_snap[int(i * len(telemetry_snap) / 299.0)] for i in range(299)]
    sampled.append(telemetry_snap[-1])
    return sampled


def _count_task_statuses(tasks: list[dict[str, Any]]) -> tuple[int, int]:
    active = sum(1 for t in tasks if t.get("status") in ["active", "initializing"])
    queued = sum(1 for t in tasks if t.get("status") == "queued")
    return active, queued


def _build_hardware_unit_statuses(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units = []
    for unit in config.HARDWARE_UNITS:
        unit_copy = unit.copy()
        unit_id = unit["id"]
        unit_copy["whisper_status"] = _resolve_whisper_unit_status(tasks, unit_id)
        unit_copy["uvr_status"] = _resolve_uvr_unit_status(tasks, unit_id)
        units.append(unit_copy)
    return units


def _resolve_whisper_unit_status(tasks: list[dict[str, Any]], unit_id: Any) -> str:
    if _is_whisper_unit_active(tasks, unit_id):
        return "busy"
    if _is_whisper_model_loaded(unit_id):
        return "loaded"
    return "ready"


def _is_whisper_unit_active(tasks: list[dict[str, Any]], unit_id: Any) -> bool:
    return any(
        t.get("status") == "active" and str(t.get("unit_id")) == str(unit_id) and _is_whisper_active_stage(t.get("stage")) for t in tasks
    )


def _is_whisper_model_loaded(unit_id: Any) -> bool:
    return bool(unit_id in model_manager.MODEL_POOL)


def _resolve_uvr_unit_status(tasks: list[dict[str, Any]], unit_id: Any) -> str:
    if _is_uvr_unit_active(tasks, unit_id):
        return "busy"
    if _is_uvr_model_loaded(unit_id):
        return "loaded"
    return "ready"


def _is_uvr_unit_active(tasks: list[dict[str, Any]], unit_id: Any) -> bool:
    return any(
        t.get("status") == "active" and str(t.get("unit_id")) == str(unit_id) and _is_uvr_active_stage(t.get("stage")) for t in tasks
    )


def _is_uvr_model_loaded(unit_id: Any) -> bool:
    return bool(unit_id in model_manager.PREPROCESSOR_POOL and model_manager.PREPROCESSOR_POOL[unit_id].separator is not None)


def get_minimal_stats() -> dict[str, Any]:
    """Fast health check stats."""
    with scheduler.STATE.task_registry_lock:
        active = sum(1 for t in scheduler.STATE.task_registry.values() if t.get("status") in ["active", "initializing"])
        queued = sum(1 for t in scheduler.STATE.task_registry.values() if t.get("status") == "queued")

    return {"status": "healthy", "active": active, "queued": queued}


# Alias for backward compatibility with tests
get_summary = get_service_stats
