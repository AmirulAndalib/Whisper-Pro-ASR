"""Regression tests for Intel fallback detection branches in config loading."""

import importlib
import os
from unittest import mock

import modules.core.config as config_module


def _exists_with_intel_nodes(path: str, real_exists):
    if path in {"/dev/accel/accel0", "/dev/dri"}:
        return True
    return real_exists(path)


def test_intel_node_fallback_used_when_openvino_device_list_is_empty():
    """Empty OpenVINO enumeration should still register Intel node-fallback resources."""
    real_exists = os.path.exists

    with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO", "MAX_GPU_UNITS": "1", "MAX_NPU_UNITS": "1"}):
        with (
            mock.patch("ctranslate2.get_cuda_device_count", return_value=0),
            mock.patch("openvino.Core") as mock_core_ctor,
            mock.patch("os.path.exists", side_effect=lambda p: _exists_with_intel_nodes(p, real_exists)),
        ):
            mock_core = mock.MagicMock()
            mock_core.available_devices = []
            mock_core_ctor.return_value = mock_core
            importlib.reload(config_module)

            unit_types = {u["type"] for u in config_module.HARDWARE_UNITS}
            assert "GPU" in unit_types
            assert "NPU" in unit_types


def test_intel_node_fallback_used_when_openvino_has_only_cpu():
    """OpenVINO CPU-only enumeration should still trigger Intel node fallback."""
    real_exists = os.path.exists

    with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO", "MAX_GPU_UNITS": "1", "MAX_NPU_UNITS": "1"}):
        with (
            mock.patch("ctranslate2.get_cuda_device_count", return_value=0),
            mock.patch("openvino.Core") as mock_core_ctor,
            mock.patch("os.path.exists", side_effect=lambda p: _exists_with_intel_nodes(p, real_exists)),
        ):
            mock_core = mock.MagicMock()
            mock_core.available_devices = ["CPU"]
            mock_core_ctor.return_value = mock_core
            importlib.reload(config_module)

            unit_types = {u["type"] for u in config_module.HARDWARE_UNITS}
            assert "GPU" in unit_types
            assert "NPU" in unit_types
