"""Ordering and priority backlog helpers for scheduler state."""


def has_queued_priority_tasks(state, exclude_task_id=None):
    """Return True if any detect-language task is currently queued."""
    return get_queued_priority_count(state, exclude_task_id=exclude_task_id) > 0


def get_queued_priority_count(state, exclude_task_id=None):
    """Return count of queued non-coalesced priority tasks."""
    with state.task_registry_lock:
        return sum(1 for task_key, task in state.task_registry.items() if _is_queued_priority_task(task_key, task, exclude_task_id))


def _is_queued_priority_task(task_key, task: dict, exclude_task_id=None) -> bool:
    if _is_excluded_task(task_key, task, exclude_task_id):
        return False
    if task.get("coalesced", False):
        return False
    return task.get("is_priority", False) and task.get("status") == "queued"


def _is_excluded_task(task_key, task: dict, exclude_task_id=None) -> bool:
    if not exclude_task_id:
        return False
    return task_key == exclude_task_id or task.get("task_id") == exclude_task_id


def has_earlier_task(state, current_task_id, is_priority=None):
    """Check whether an earlier same-priority task is still waiting for hardware."""
    is_priority = _resolve_task_priority(state, current_task_id, is_priority)
    if is_priority is None:
        return False
    task_snapshot, arrival_snapshot = _scheduler_snapshot(state)
    current_arrival_time = arrival_snapshot.get(current_task_id)
    if current_arrival_time is None:
        return False
    return _has_earlier_waiting_task(arrival_snapshot, task_snapshot, current_task_id, current_arrival_time, is_priority)


def _resolve_task_priority(state, current_task_id, is_priority):
    if is_priority is not None:
        return is_priority
    with state.task_registry_lock:
        task = state.task_registry.get(current_task_id)
        if task is None:
            return None
        return task.get("is_priority", False)


def _scheduler_snapshot(state) -> tuple[dict, dict]:
    with state.cond:
        task_snapshot = {
            task_id: {
                "is_priority": task.get("is_priority", False),
                "status": task.get("status"),
                "unit_id": task.get("unit_id"),
            }
            for task_id, task in state.task_registry.items()
        }
        arrival_snapshot = dict(state.task_arrival_order)
    return task_snapshot, arrival_snapshot


def _has_earlier_waiting_task(arrival_snapshot, task_snapshot, current_task_id, current_arrival_time, is_priority: bool) -> bool:
    ordered_ids = list(arrival_snapshot.keys())
    order_index = {task_id: idx for idx, task_id in enumerate(ordered_ids)}
    current_index = order_index.get(current_task_id, -1)
    for task_id, arrival_time in arrival_snapshot.items():
        if _is_earlier_waiting_matching_task(
            task_id,
            arrival_time,
            task_snapshot,
            current_context={
                "task_id": current_task_id,
                "arrival_time": current_arrival_time,
                "index": current_index,
            },
            is_priority=is_priority,
            task_index=order_index.get(task_id, -1),
        ):
            return True
    return False


def _is_earlier_waiting_matching_task(
    task_id,
    arrival_time,
    task_snapshot,
    current_context,
    is_priority: bool,
    *,
    task_index: int,
) -> bool:
    if _is_not_earlier_task(
        task_id,
        arrival_time,
        current_context["task_id"],
        current_context["arrival_time"],
        task_index=task_index,
        current_index=current_context["index"],
    ):
        return False
    task = task_snapshot.get(task_id)
    return _is_matching_priority_task(task, is_priority) and _is_waiting_for_hardware(task)


def _is_not_earlier_task(task_id, arrival_time, current_task_id, current_arrival_time, *, task_index: int, current_index: int) -> bool:
    if task_id == current_task_id:
        return True
    if arrival_time < current_arrival_time:
        return False
    if arrival_time == current_arrival_time:
        return not 0 <= task_index < current_index
    return True


def _is_matching_priority_task(task, is_priority: bool) -> bool:
    return bool(task and task.get("is_priority", False) == is_priority)


def _is_waiting_for_hardware(task: dict) -> bool:
    return task.get("status") in {"initializing", "queued"} and not task.get("unit_id")
