"""OpenVINO provider configuration and dispatch helpers."""

from __future__ import annotations

import importlib

from modules.core import config as core_config

ProviderConfig = tuple[list[str], list[dict[str, object]]]


def _normalize_openvino_device_type(device_id: str) -> str:
    """Normalize any device id to an OpenVINO-compatible target name."""
    normalized = (device_id or "").strip().lower()
    if normalized in {"", "cpu", "openvino_cpu", "openvino"}:
        return "CPU"
    for prefix, replacement in (("gpu", "GPU"), ("openvino_gpu", "GPU"), ("npu", "NPU"), ("openvino_npu", "NPU")):
        normalized_family = _normalize_openvino_family_prefix(normalized, prefix, replacement)
        if normalized_family is not None:
            return normalized_family
    return device_id.upper()


def _normalize_openvino_family_prefix(normalized: str, prefix: str, replacement: str) -> str | None:
    if not normalized.startswith(prefix):
        return None
    suffix = normalized.removeprefix(prefix)
    return f"{replacement}{suffix}" if suffix else replacement


def _normalize_openvino_provider_options(options: dict | None) -> dict:
    """Return ONNX Runtime-compatible provider options."""
    return {str(key): str(value) for key, value in dict(options or {}).items() if value is not None}


def _cpu_provider_config() -> ProviderConfig:
    return ["CPUExecutionProvider"], [{}]


def _resolve_openvino_resolver():
    return importlib.import_module("modules.inference.pipeline.openvino_resolver")


def cuda_provider_config(device_id: str) -> ProviderConfig:
    """Return CUDA provider config with CPU fallback semantics."""
    raw = str(device_id or "0")
    lowered = raw.lower()
    if lowered == "cuda":
        parsed: int | str = "0"
    else:
        idx_token = lowered.split("cuda:", 1)[1] if lowered.startswith("cuda:") else raw
        try:
            parsed = int(idx_token)
        except (TypeError, ValueError):
            parsed = "0"
    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": parsed}]


def openvino_provider_config(device_id: str) -> ProviderConfig:
    """Return OpenVINO provider config with CPU fallback semantics."""
    resolved = _normalize_openvino_device_type(device_id)
    provider_options = _normalize_openvino_provider_options(
        {
            "device_type": resolved,
            "cache_dir": str(core_config.OV_CACHE_DIR),
            "num_streams": str(max(1, int(core_config.PREPROCESS_THREADS))),
        }
    )
    return ["OpenVINOExecutionProvider", "CPUExecutionProvider"], [provider_options]


def cpu_provider_config() -> ProviderConfig:
    """Return CPU-only provider config."""
    return _cpu_provider_config()


def auto_provider_config(available):
    """Resolve AUTO mode provider config from available runtime providers."""
    providers = list(available or [])
    if "CUDAExecutionProvider" in providers:
        return cuda_or_cpu_provider_config("0", providers)
    if "OpenVINOExecutionProvider" in providers and _resolve_openvino_resolver().has_openvino_accelerator_device():
        return openvino_or_cpu_provider_config("GPU", providers)
    return cpu_provider_config()


def cuda_or_cpu_provider_config(device_id: str, available):
    """Resolve CUDA provider config with deterministic CPU fallback."""
    if "CUDAExecutionProvider" not in (available or []):
        return cpu_provider_config()
    return cuda_provider_config(device_id)


def openvino_or_cpu_provider_config(device_id: str, available):
    """Resolve OpenVINO provider config with deterministic CPU fallback."""
    resolver = _resolve_openvino_resolver()
    if "OpenVINOExecutionProvider" not in (available or []):
        return cpu_provider_config()
    if resolver.is_openvino_family_disabled(device_id):
        return cpu_provider_config()
    return openvino_provider_config(device_id)


def provider_config_dispatch(device_type: str):
    """Return provider config resolver for the requested execution class."""
    normalized = (device_type or "AUTO").upper()
    if normalized == "CUDA":
        return cuda_or_cpu_provider_config
    if normalized in {"OPENVINO", "GPU", "NPU"}:
        return openvino_or_cpu_provider_config
    if normalized == "CPU":
        return lambda _device_id, _available: cpu_provider_config()
    return lambda _device_id, available: auto_provider_config(available)


def resolve_provider_config(
    device_type: str,
    device_id: str,
    available,
):
    """Resolve provider config for requested execution class and target device id."""
    resolver = provider_config_dispatch(device_type)
    return resolver(device_id, available)
