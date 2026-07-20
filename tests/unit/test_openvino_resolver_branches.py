"""Branch coverage tests for OpenVINO resolver helpers."""

from unittest import mock

from modules.inference.pipeline import openvino_provider_dispatch as provider_dispatch
from modules.inference.pipeline import openvino_resolver


def test_purge_onnxruntime_modules_removes_prefixed_entries():
    """Purge helper should remove all onnxruntime-prefixed module cache entries."""
    with mock.patch.dict(
        openvino_resolver.sys.modules,
        {"onnxruntime": object(), "onnxruntime.capi": object(), "something_else": object()},
        clear=False,
    ):
        openvino_resolver.purge_onnxruntime_modules()
        assert "onnxruntime" not in openvino_resolver.sys.modules
        assert "onnxruntime.capi" not in openvino_resolver.sys.modules
        assert "something_else" in openvino_resolver.sys.modules


def test_prepend_intel_path_moves_existing_path_to_front():
    """Intel path prepend should deduplicate and place Intel runtime first."""
    prepend = getattr(openvino_resolver, "_prepend_intel_path")
    with mock.patch.object(openvino_resolver.sys, "path", ["/x", "/app/libs/intel", "/y"]):
        prepend("/app/libs/intel")
        assert openvino_resolver.sys.path[0] == "/app/libs/intel"
        assert openvino_resolver.sys.path.count("/app/libs/intel") == 1


def test_reload_openvino_capable_onnxruntime_checks_provider_presence():
    """Reload should return True only when OpenVINO provider is available."""
    reload_openvino_capable = getattr(openvino_resolver, "_reload_openvino_capable_onnxruntime")
    fake_ort = mock.MagicMock()
    fake_ort.get_available_providers.return_value = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
    with mock.patch.object(openvino_resolver.importlib, "import_module", return_value=fake_ort):
        assert reload_openvino_capable() is True

    fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    with mock.patch.object(openvino_resolver.importlib, "import_module", return_value=fake_ort):
        assert reload_openvino_capable() is False


def test_reload_onnxruntime_from_intel_path_handles_missing_path_and_import_failure():
    """Intel runtime reload should fail gracefully for missing path and import errors."""
    with mock.patch.object(openvino_resolver.os.path, "exists", return_value=False):
        assert openvino_resolver.reload_onnxruntime_from_intel_path() is False

    with (
        mock.patch.object(openvino_resolver.os.path, "exists", return_value=True),
        mock.patch.object(openvino_resolver, "_reload_openvino_capable_onnxruntime", side_effect=ImportError("missing")),
    ):
        assert openvino_resolver.reload_onnxruntime_from_intel_path() is False


def test_has_openvino_provider_handles_attribute_error():
    """Provider capability checks should tolerate broken ONNX runtime shims."""
    broken = mock.MagicMock()
    broken.get_available_providers.side_effect = AttributeError("boom")
    assert openvino_resolver.has_openvino_provider(broken) is False


def test_ensure_openvino_onnxruntime_reload_flow_and_non_target_noop():
    """OpenVINO ensure helper should skip non-targets and reload when provider missing."""
    with mock.patch.object(openvino_resolver, "reload_onnxruntime_from_intel_path") as mock_reload:
        openvino_resolver.ensure_openvino_onnxruntime("CPU")
        mock_reload.assert_not_called()

    fake_ort = mock.MagicMock()
    fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    with (
        mock.patch.object(openvino_resolver.importlib, "import_module", return_value=fake_ort),
        mock.patch.object(openvino_resolver, "reload_onnxruntime_from_intel_path") as mock_reload,
    ):
        openvino_resolver.ensure_openvino_onnxruntime("GPU")
        mock_reload.assert_called_once()


def test_fallback_helpers_and_retry_match_extensions():
    """Device fallback and retry match utilities should preserve deterministic ordering."""
    resolve_when_no_devices = getattr(openvino_resolver, "_resolve_when_no_openvino_devices")
    should_return_generic = getattr(openvino_resolver, "_should_return_generic_family")

    with mock.patch.object(openvino_resolver.logger, "warning") as mock_warn:
        assert resolve_when_no_devices("NPU", "NPU") == "NPU"
        mock_warn.assert_called_once()

    assert should_return_generic("NPU.0", "NPU", "NPU") is True
    assert should_return_generic("NPU", "NPU", "NPU") is False

    ordered = ["NPU"]
    normalized = [("NPU.0", "NPU.0"), ("GPU.0", "GPU.0"), ("NPU", "NPU")]
    openvino_resolver.extend_openvino_retry_matches(ordered, normalized, lambda upper: upper.startswith("NPU"))
    assert ordered == ["NPU", "NPU.0"]


def test_alternate_candidates_and_accelerator_detection_helpers():
    """Alternate candidate iterator and accelerator detection should behave as expected."""
    retries = ["NPU", "NPU.0", "GPU.0"]
    assert list(openvino_resolver.alternate_openvino_candidates(retries, "NPU")) == ["NPU.0", "GPU.0"]

    with mock.patch.object(openvino_resolver, "get_available_openvino_devices", return_value=["CPU"]):
        assert openvino_resolver.has_openvino_accelerator_device() is False
    with mock.patch.object(openvino_resolver, "get_available_openvino_devices", return_value=["CPU", "GPU.0"]):
        assert openvino_resolver.has_openvino_accelerator_device() is True


def test_runtime_loader_error_detection_and_family_marking_noop_for_cpu():
    """Runtime loader detector should match only intended failures and CPU should not mark families."""
    msg = "INTEL_OPENVINO_DIR is set but OpenVINO library wasn't able to be loaded"
    assert openvino_resolver.is_openvino_runtime_loader_error(RuntimeError(msg)) is True
    assert openvino_resolver.is_openvino_runtime_loader_error(RuntimeError("other")) is False

    openvino_resolver.clear_openvino_disabled_families()
    openvino_resolver.mark_openvino_family_unavailable("CPU")
    assert openvino_resolver.is_openvino_family_disabled("NPU") is False


def test_log_openvino_cpu_fallback_branches():
    """Fallback logger should no-op without context, tolerate session errors, and mark family on CPU fallback."""
    session = mock.MagicMock()

    openvino_resolver.log_openvino_cpu_fallback(session, None)
    session.get_providers.assert_not_called()

    session.reset_mock()
    session.get_providers.side_effect = RuntimeError("no session")
    openvino_resolver.log_openvino_cpu_fallback(session, {"device_type": "NPU"})

    session.reset_mock()
    session.get_providers.side_effect = None
    session.get_providers.return_value = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    openvino_resolver.clear_openvino_disabled_families()
    openvino_resolver.log_openvino_cpu_fallback(session, {"device_type": "NPU"})
    assert openvino_resolver.is_openvino_family_disabled("NPU") is False

    session.get_providers.return_value = ["CPUExecutionProvider"]
    openvino_resolver.log_openvino_cpu_fallback(session, {"device_type": "NPU"})
    assert openvino_resolver.is_openvino_family_disabled("NPU") is True


def test_force_openvino_provider_if_needed_for_cpu_fallback():
    """Provider forcing should rewrite CPU-only fallback when OpenVINO context is present."""
    assert openvino_resolver.force_openvino_provider_if_needed(["CPUExecutionProvider"], {"device_type": "GPU"}) == [
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert openvino_resolver.force_openvino_provider_if_needed(["CPUExecutionProvider"], None) == ["CPUExecutionProvider"]


def test_merge_openvino_provider_options_normalizes_option_slots():
    """Merge helper should expand provider options and inject generic GPU context options."""
    merged = openvino_resolver.merge_openvino_provider_options(
        ["OpenVINOExecutionProvider", "CPUExecutionProvider"], [None], {"device_type": "GPU"}
    )
    assert merged[0]["device_type"] == "GPU"
    assert isinstance(merged[1], dict)


def test_merge_openvino_provider_options_skips_cpu_only_lists():
    """Merge helper should leave CPU-only provider option lists unchanged."""
    assert openvino_resolver.merge_openvino_provider_options(["CPUExecutionProvider"], [{}], {"device_type": "GPU"}) == [{}]


def test_merge_openvino_provider_options_encodes_npu_slot_in_load_config():
    """Merge helper should preserve explicit NPU slots through load_config DEVICE_ID."""
    merged_dotted = openvino_resolver.merge_openvino_provider_options(
        ["OpenVINOExecutionProvider", "CPUExecutionProvider"], [{}], {"device_type": "NPU.0"}
    )
    assert merged_dotted[0]["device_type"] == "NPU"
    assert merged_dotted[0]["load_config"] == '{"NPU":{"DEVICE_ID":"0"}}'


def test_set_openvino_context_options_assigns_and_clears_thread_context():
    """Context option setter should store recognized OpenVINO device options and clear otherwise."""
    openvino_resolver.set_openvino_context_options([{"device_type": "NPU.0"}])
    assert openvino_resolver.utils.THREAD_CONTEXT.ov_options == {
        "device_type": "NPU",
        "load_config": '{"NPU":{"DEVICE_ID":"0"}}',
    }
    openvino_resolver.set_openvino_context_options([{}])
    assert openvino_resolver.utils.THREAD_CONTEXT.ov_options is None


def test_dedupe_openvino_retry_candidates_preserves_distinct_npu_slots():
    """Retry-candidate dedupe should keep distinct NPU device ids available for retry."""
    candidates = openvino_resolver.dedupe_openvino_retry_candidates(["NPU.0", "NPU.1", "NPU.0", "GPU.0"])
    assert candidates == ["NPU.0", "NPU.1", "GPU.0"]


def test_provider_config_parses_invalid_cuda_device_index_to_default_zero():
    """CUDA config helper should normalize invalid cuda device suffixes to default index 0."""
    providers, options = provider_dispatch.cuda_provider_config("cuda:not-int")
    assert providers[0] == "CUDAExecutionProvider"
    assert options[0]["device_id"] == "0"


def test_provider_config_preserves_explicit_multi_gpu_cuda_indexes():
    """CUDA config helper should preserve explicit GPU indexes on multi-GPU hosts."""
    providers, options = provider_dispatch.cuda_provider_config("cuda:2")
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert options[0]["device_id"] == 2

    digit_providers, digit_options = provider_dispatch.cuda_provider_config("3")
    assert digit_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert digit_options[0]["device_id"] == 3


def test_auto_provider_config_prefers_openvino_when_accelerator_present():
    """AUTO provider helper should choose OpenVINO when accelerator detection succeeds."""
    with mock.patch.object(openvino_resolver, "has_openvino_accelerator_device", return_value=True):
        auto_providers, _ = provider_dispatch.auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        assert auto_providers[0] == "OpenVINOExecutionProvider"


def test_openvino_or_cpu_provider_config_honors_disabled_family_circuit_breaker():
    """OpenVINO provider helper should fall back to CPU when the family is disabled."""
    openvino_resolver.clear_openvino_disabled_families()
    openvino_resolver.mark_openvino_family_unavailable("NPU")
    ov_or_cpu, _ = provider_dispatch.openvino_or_cpu_provider_config("NPU", ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert ov_or_cpu == ["CPUExecutionProvider"]


def test_resolve_provider_config_unknown_device_uses_auto_policy():
    """Unknown provider types should dispatch through AUTO policy and return CPU when that is all that is available."""
    unknown, _ = provider_dispatch.resolve_provider_config("SOMETHING", "0", ["CPUExecutionProvider"])
    assert unknown == ["CPUExecutionProvider"]
