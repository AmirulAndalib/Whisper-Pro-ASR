"""Compatibility wrappers for OpenVINO resolver split helpers."""

from __future__ import annotations

from modules.inference.pipeline import openvino_provider_dispatch as provider_dispatch
from modules.inference.pipeline import openvino_resolver as resolver

ProviderConfig = tuple[list[str], list[dict[str, object]]]


def openvino_heuristic_retry_candidates(requested: str) -> list[str]:
    """Proxy heuristic retry candidate generation to the resolver module."""
    return resolver.openvino_heuristic_retry_candidates(requested)


def alternate_openvino_candidates(retries: list[str], requested: str):
    """Proxy alternate retry candidate iterator to the resolver module."""
    return resolver.alternate_openvino_candidates(retries, requested)


def get_openvino_retry_candidates_for_devices(requested: str, available_devices: list[str]) -> list[str]:
    """Proxy runtime-device retry candidate generation to the resolver module."""
    return resolver.get_openvino_retry_candidates_for_devices(requested, available_devices)


def get_openvino_retry_candidates(requested: str, available_devices: list[str] | None = None) -> list[str]:
    """Proxy retry candidate generation to the resolver module."""
    return resolver.get_openvino_retry_candidates(requested, available_devices)


def has_openvino_accelerator_device_in(available_devices: list[str]) -> bool:
    """Proxy accelerator detection against a prefetched device list."""
    return resolver.has_openvino_accelerator_device_in(available_devices)


def has_openvino_accelerator_device(available_devices: list[str] | None = None) -> bool:
    """Proxy accelerator detection with optional runtime discovery."""
    return resolver.has_openvino_accelerator_device(available_devices)


def cuda_provider_config(device_id: str) -> ProviderConfig:
    """Proxy CUDA provider configuration generation to the resolver module."""
    return provider_dispatch.cuda_provider_config(device_id)


def openvino_provider_config(device_id: str) -> ProviderConfig:
    """Proxy OpenVINO provider configuration generation to the resolver module."""
    return provider_dispatch.openvino_provider_config(device_id)


def cpu_provider_config() -> ProviderConfig:
    """Proxy CPU-only provider configuration generation to the resolver module."""
    return provider_dispatch.cpu_provider_config()


def auto_provider_config(available: list[str]) -> ProviderConfig:
    """Proxy AUTO provider configuration generation to the resolver module."""
    return provider_dispatch.auto_provider_config(available)


def cuda_or_cpu_provider_config(device_id: str, available: list[str]) -> ProviderConfig:
    """Proxy CUDA-or-CPU provider configuration generation to the resolver module."""
    return provider_dispatch.cuda_or_cpu_provider_config(device_id, available)


def openvino_or_cpu_provider_config(device_id: str, available: list[str]) -> ProviderConfig:
    """Proxy OpenVINO-or-CPU provider configuration generation to the resolver module."""
    return provider_dispatch.openvino_or_cpu_provider_config(device_id, available)


def provider_config_dispatch(device_type: str):
    """Proxy provider resolver dispatch selection to the resolver module."""
    return provider_dispatch.provider_config_dispatch(device_type)


def resolve_provider_config(device_type: str, device_id: str, available: list[str]) -> ProviderConfig:
    """Proxy provider configuration resolution to the resolver module."""
    return provider_dispatch.resolve_provider_config(device_type, device_id, available)
