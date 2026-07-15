"""Smoke tests for the OpenVINO provider compatibility wrappers."""

import pytest

from modules.inference.pipeline import openvino_resolver_provider as provider


def test_retry_candidate_wrappers():
    """Retry-candidate wrappers should preserve resolver ordering."""
    assert provider.openvino_heuristic_retry_candidates("NPU") == ["NPU", "NPU.0", "GPU.0", "GPU"]
    assert list(provider.alternate_openvino_candidates(["NPU", "GPU.0"], "NPU")) == ["GPU.0"]
    assert provider.get_openvino_retry_candidates_for_devices("NPU.0", ["NPU", "GPU.0"]) == [
        "NPU.0",
        "NPU",
        "GPU.0",
    ]
    assert provider.get_openvino_retry_candidates("NPU", ["NPU", "GPU.0"]) == ["NPU", "GPU.0"]


def test_accelerator_detection_wrappers():
    """Accelerator detection wrappers should report the same runtime state as the resolver."""
    assert provider.has_openvino_accelerator_device_in(["CPU", "GPU.0"]) is True
    assert provider.has_openvino_accelerator_device(["CPU", "GPU.0"]) is True


def test_provider_config_wrappers():
    """Provider config wrappers should return the expected tuples."""
    cuda_providers, cuda_options = provider.cuda_provider_config("cuda:2")
    assert cuda_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert cuda_options[0]["device_id"] == 2

    openvino_providers, openvino_options = provider.openvino_provider_config("NPU")
    assert openvino_providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert openvino_options[0]["device_type"] == "NPU"


def test_provider_config_cpu_and_auto_wrappers():
    """CPU and AUTO provider wrappers should preserve their basic tuples."""
    assert provider.cpu_provider_config() == (["CPUExecutionProvider"], [{}])
    assert provider.auto_provider_config(["CPUExecutionProvider"]) == (["CPUExecutionProvider"], [{}])


def test_provider_config_fallback_wrappers():
    """Provider fallback wrappers should mirror the resolved tuples."""
    openvino_providers, openvino_options = provider.openvino_provider_config("NPU")

    assert provider.cuda_or_cpu_provider_config("0", ["CUDAExecutionProvider"]) == (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        [{"device_id": 0}],
    )
    assert provider.openvino_or_cpu_provider_config("NPU", ["OpenVINOExecutionProvider"]) == (
        openvino_providers,
        openvino_options,
    )


def test_dispatch_wrappers():
    """Dispatch wrappers should resolve provider selection and execution classes."""
    dispatch = provider.provider_config_dispatch("CPU")
    assert callable(dispatch)
    assert provider.resolve_provider_config("CPU", "0", ["CPUExecutionProvider"]) == (["CPUExecutionProvider"], [{}])


@pytest.fixture(autouse=True)
def reset_openvino_state():
    """Reset resolver circuit-breaker state before and after each wrapper smoke test."""
    provider.resolver.clear_openvino_disabled_families()
    yield
    provider.resolver.clear_openvino_disabled_families()
