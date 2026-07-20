"""Helpers for validating and logging approved local media paths."""

import logging
import os

from modules.core import config, utils


def get_approved_roots() -> list[str]:
    """Return canonical roots allowed for local-path optimization."""

    roots = [
        os.path.realpath(config.TEMP_DIR),
        os.path.realpath(config.PERSISTENT_DIR),
        os.path.realpath(os.getcwd()),
    ]
    for root in config.APPROVED_ROOTS:
        roots.append(os.path.realpath(root))
    return roots


def is_path_approved(normalized_path: str, approved_roots: list[str]) -> bool:
    """Return True when path is equal to or nested under an approved root."""

    for root in approved_roots:
        if normalized_path == root or normalized_path.startswith(os.path.join(root, "")):
            return True
    return False


def log_local_path_optimization(logger: logging.Logger, normalized_path: str):
    """Emit optimization log once per request for the resolved local path."""

    already_logged = getattr(utils.THREAD_CONTEXT, "optimized_local_path_logged", None)
    if already_logged != normalized_path:
        logger.info("[System] Optimization: Using Local Path -> %s", normalized_path)
        utils.THREAD_CONTEXT.optimized_local_path_logged = normalized_path
