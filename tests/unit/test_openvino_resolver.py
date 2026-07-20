"""Tests for OpenVINO resolution and device patching."""

from unittest import mock

from modules.inference.pipeline import openvino_resolver


def test_is_openvino_target_npu():
    """Verify OpenVINO target device detection logic for NPU."""
    assert openvino_resolver.is_openvino_target("NPU") is True
    assert openvino_resolver.is_openvino_target("NPU.1") is True


def test_is_openvino_target_gpu():
    """Verify OpenVINO target device detection logic for GPU."""
    assert openvino_resolver.is_openvino_target("GPU") is True
    assert openvino_resolver.is_openvino_target("GPU.0") is True


def test_is_openvino_target_negative():
    """Verify OpenVINO target device detection logic for negative cases."""
    assert openvino_resolver.is_openvino_target("CPU") is False
    assert openvino_resolver.is_openvino_target("") is False


def test_openvino_device_family():
    """Verify extraction of device family from device string."""
    assert openvino_resolver.openvino_device_family("NPU.0") == "NPU"
    assert openvino_resolver.openvino_device_family("GPU.1") == "GPU"
    assert openvino_resolver.openvino_device_family("CPU") is None


def test_resolve_openvino_device_type_no_available_devices():
    """Test device resolution gracefully falls back when no devices are available."""
    with mock.patch("modules.inference.pipeline.openvino_resolver.get_available_openvino_devices", return_value=[]):
        assert openvino_resolver.resolve_openvino_device_type("NPU") == "NPU"
        assert openvino_resolver.resolve_openvino_device_type("GPU") == "GPU"
        assert openvino_resolver.resolve_openvino_device_type("CPU") == "CPU"


def test_resolve_openvino_device_type_exact_match():
    """Test device resolution accurately returns exact matches without appending suffix."""
    # If OpenVINO returns "NPU", and we ask for "NPU", it should return "NPU"
    # and NOT "NPU.0". This ensures OpenVINO 2026.2 compatibility.
    with mock.patch("modules.inference.pipeline.openvino_resolver.get_available_openvino_devices", return_value=["NPU", "GPU"]):
        assert openvino_resolver.resolve_openvino_device_type("NPU") == "NPU"
        assert openvino_resolver.resolve_openvino_device_type("GPU") == "GPU"


def test_resolve_openvino_device_type_prefixed_match():
    """Test device resolution handles prefixed matching correctly."""
    with mock.patch("modules.inference.pipeline.openvino_resolver.get_available_openvino_devices", return_value=["NPU.0", "GPU.1"]):
        assert openvino_resolver.resolve_openvino_device_type("NPU") == "NPU.0"
        assert openvino_resolver.resolve_openvino_device_type("GPU") == "GPU.1"


def test_resolve_openvino_device_type_alternate_fallback():
    """Test device resolution falls back to alternate family if primary is missing."""
    with mock.patch("modules.inference.pipeline.openvino_resolver.get_available_openvino_devices", return_value=["GPU.0"]):
        # Fallback to alternate family (GPU) if NPU is requested but unavailable
        assert openvino_resolver.resolve_openvino_device_type("NPU") == "GPU.0"


def test_get_openvino_retry_candidates():
    """Test retry candidates are properly ordered based on availability and request."""
    with mock.patch("modules.inference.pipeline.openvino_resolver.get_available_openvino_devices", return_value=["GPU.1", "NPU", "CPU"]):
        candidates = openvino_resolver.get_openvino_retry_candidates("NPU")
        # Should prioritize exact match, then family matches, then alternates
        assert candidates == ["NPU", "GPU.1"]


def test_get_openvino_retry_candidates_with_prefetched_devices():
    """Retry candidate builder should accept pre-fetched devices without discovery."""
    with mock.patch("modules.inference.pipeline.openvino_resolver.get_available_openvino_devices") as mock_discovery:
        candidates = openvino_resolver.get_openvino_retry_candidates("NPU", ["NPU", "GPU.1", "CPU"])
    assert candidates == ["NPU", "GPU.1"]
    mock_discovery.assert_not_called()


def test_openvino_heuristic_retry_candidates():
    """Test fallback order when runtime enumeration is unavailable."""
    candidates = openvino_resolver.openvino_heuristic_retry_candidates("NPU")
    # Order: Requested -> Requested.0 -> Requested Family -> Alternate.0 -> Alternate Family
    assert candidates == ["NPU", "NPU.0", "GPU.0", "GPU"]


def test_has_openvino_accelerator_device_with_prefetched_devices():
    """Accelerator detector should accept pre-fetched OpenVINO devices without discovery."""
    with mock.patch("modules.inference.pipeline.openvino_resolver.get_available_openvino_devices") as mock_discovery:
        assert openvino_resolver.has_openvino_accelerator_device(["CPU"]) is False
        assert openvino_resolver.has_openvino_accelerator_device(["CPU", "GPU.0"]) is True
    mock_discovery.assert_not_called()


def test_is_openvino_session_fallback_error():
    """Test exception parsing for OpenVINO session fallback errors."""
    assert openvino_resolver.is_openvino_session_fallback_error(ValueError("Fallback to CPUExecutionProvider")) is True
    assert openvino_resolver.is_openvino_session_fallback_error(RuntimeError("Failed to create OpenVINOExecutionProvider")) is True
    assert openvino_resolver.is_openvino_session_fallback_error(ValueError("Other error")) is False


def test_mark_openvino_families():
    """Test disabling specific functionality for runtime OpenVINO families."""
    openvino_resolver.clear_openvino_disabled_families()
    assert openvino_resolver.is_openvino_family_disabled("NPU") is False

    openvino_resolver.mark_openvino_family_unavailable("NPU")
    assert openvino_resolver.is_openvino_family_disabled("NPU") is True
    assert openvino_resolver.is_openvino_family_disabled("GPU") is False


def test_disable_all_openvino_families():
    """Test blanket disabling of runtime OpenVINO families."""
    openvino_resolver.disable_all_openvino_families_for_runtime()
    assert openvino_resolver.is_openvino_family_disabled("GPU") is True

    openvino_resolver.clear_openvino_disabled_families()
    assert openvino_resolver.is_openvino_family_disabled("NPU") is False
