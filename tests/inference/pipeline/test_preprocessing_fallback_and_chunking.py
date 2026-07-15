"""Additional preprocessing tests split from test_preprocessing.py."""

from unittest import mock

import pytest

from modules.inference.pipeline import preprocessing
from modules.inference.pipeline.preprocessing import PreprocessingManager


@pytest.fixture
def prep_manager():
    """Fixture to provide a clean PreprocessingManager instance."""
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    return PreprocessingManager(assigned_unit=unit)


class TestSeparateWithFallback:
    """Tests for _separate_with_fallback()."""

    def _enospc(self, path=None, **_kwargs):
        """Helper: raise ENOSPC OSError."""
        import errno as _errno

        raise OSError(_errno.ENOSPC, "No space left on device")

    def test_primary_succeeds(self):
        """Should return immediately on first try without calling sep_factory."""
        mock_sep = mock.MagicMock()
        mock_sep.separate.return_value = ["/tmp/vocal.wav"]
        factory = mock.MagicMock()

        with mock.patch("os.makedirs"):
            stems, used_sep = preprocessing._separate_with_fallback(mock_sep, factory, "audio.wav")

        assert stems == ["/tmp/vocal.wav"]
        assert used_sep is mock_sep
        factory.assert_not_called()

    def test_fallback_on_enospc(self):
        """Should retry with a new separator when ENOSPC is raised."""
        import errno as _errno

        mock_sep = mock.MagicMock()
        mock_sep.separate.side_effect = OSError(_errno.ENOSPC, "No space")

        mock_fallback_sep = mock.MagicMock()
        mock_fallback_sep.separate.return_value = ["/persistent/vocal.wav"]
        factory = mock.MagicMock(return_value=mock_fallback_sep)

        candidates = [str(preprocessing.CACHE_DIR), "/persistent/tmp"]
        with (
            mock.patch("os.makedirs"),
            mock.patch("modules.inference.pipeline.preprocessing._candidate_output_dirs", return_value=candidates),
        ):
            stems, used_sep = preprocessing._separate_with_fallback(mock_sep, factory, "audio.wav")

        assert stems == ["/persistent/vocal.wav"]
        assert used_sep is mock_fallback_sep
        factory.assert_called_once_with("/persistent/tmp")

    def test_non_enospc_propagates_immediately(self):
        """Non-ENOSPC OSError should not trigger fallback."""
        mock_sep = mock.MagicMock()
        mock_sep.separate.side_effect = OSError(5, "Input/output error")
        factory = mock.MagicMock()

        with (
            mock.patch("os.makedirs"),
            mock.patch("modules.inference.pipeline.preprocessing._candidate_output_dirs", return_value=[str(preprocessing.CACHE_DIR)]),
        ):
            with pytest.raises(OSError) as exc_info:
                preprocessing._separate_with_fallback(mock_sep, factory, "audio.wav")
        assert exc_info.value.errno == 5
        factory.assert_not_called()

    def test_disk_usage_failure_during_enospc_logging(self):
        """shutil.disk_usage failing during error logging should not crash."""
        import errno as _errno

        mock_sep = mock.MagicMock()
        mock_sep.separate.side_effect = OSError(_errno.ENOSPC, "No space")

        mock_fallback_sep = mock.MagicMock()
        mock_fallback_sep.separate.return_value = ["/ok/vocal.wav"]
        factory = mock.MagicMock(return_value=mock_fallback_sep)

        candidates = [str(preprocessing.CACHE_DIR), "/ok/dir"]
        with (
            mock.patch("os.makedirs"),
            mock.patch("modules.inference.pipeline.preprocessing._candidate_output_dirs", return_value=candidates),
            mock.patch("modules.inference.pipeline.preprocessing.shutil.disk_usage", side_effect=OSError("stat failed")),
        ):
            stems, used_sep = preprocessing._separate_with_fallback(mock_sep, factory, "audio.wav")
        assert stems == ["/ok/vocal.wav"]
        assert used_sep is mock_fallback_sep

    def test_all_candidates_exhausted_raises(self):
        """Should raise ENOSPC OSError when every directory fails."""
        import errno as _errno

        mock_sep = mock.MagicMock()
        mock_sep.separate.side_effect = OSError(_errno.ENOSPC, "No space")

        mock_fallback_sep = mock.MagicMock()
        mock_fallback_sep.separate.side_effect = OSError(_errno.ENOSPC, "No space")
        factory = mock.MagicMock(return_value=mock_fallback_sep)

        candidates = [str(preprocessing.CACHE_DIR), "/another/dir"]
        with (
            mock.patch("os.makedirs"),
            mock.patch("modules.inference.pipeline.preprocessing._candidate_output_dirs", return_value=candidates),
            mock.patch("modules.inference.pipeline.preprocessing.shutil.disk_usage", side_effect=OSError("stat failed")),
        ):
            with pytest.raises(OSError) as exc_info:
                preprocessing._separate_with_fallback(mock_sep, factory, "audio.wav")
        assert exc_info.value.args[0] == _errno.ENOSPC

    def test_preserves_chunk_duration_for_non_priority_preemption(self):
        """Non-priority separation should keep the configured chunk duration intact."""
        mock_sep = mock.MagicMock()
        mock_sep.chunk_duration = 600
        mock_sep._separate_file = mock.MagicMock(return_value=["/tmp/vocal.wav"])

        captured = {}

        def _separate(path):
            captured["chunk_duration"] = mock_sep.chunk_duration
            return ["/tmp/vocal.wav"]

        mock_sep.separate.side_effect = _separate
        factory = mock.MagicMock()

        with (
            mock.patch("os.makedirs"),
            mock.patch("modules.inference.pipeline.preprocessing.utils.get_audio_duration", return_value=1200.0),
            mock.patch("modules.inference.pipeline.preprocessing.utils.THREAD_CONTEXT") as mock_ctx,
        ):
            mock_ctx.is_priority = False
            stems, _used_sep = preprocessing._separate_with_fallback(mock_sep, factory, "audio.wav", yield_cb=lambda: None)

        assert stems == ["/tmp/vocal.wav"]
        assert captured["chunk_duration"] == 600

    def test_preserves_chunk_duration_for_priority_tasks(self):
        """Priority separation should also keep the configured chunk duration."""
        mock_sep = mock.MagicMock()
        mock_sep.chunk_duration = 600
        mock_sep._separate_file = mock.MagicMock(return_value=["/tmp/vocal.wav"])

        captured = {}

        def _separate(path):
            captured["chunk_duration"] = mock_sep.chunk_duration
            return ["/tmp/vocal.wav"]

        mock_sep.separate.side_effect = _separate
        factory = mock.MagicMock()

        with (
            mock.patch("os.makedirs"),
            mock.patch("modules.inference.pipeline.preprocessing.utils.get_audio_duration", return_value=1200.0),
            mock.patch("modules.inference.pipeline.preprocessing.utils.THREAD_CONTEXT") as mock_ctx,
        ):
            mock_ctx.is_priority = True
            stems, _used_sep = preprocessing._separate_with_fallback(mock_sep, factory, "audio.wav", yield_cb=lambda: None)

        assert stems == ["/tmp/vocal.wav"]
        assert captured["chunk_duration"] == 600


def test_apply_onnx_optimizations_no_separator():
    """Cover lines 186-187 where audio_separator import fails during optimization."""

    class MockSession:
        is_patched = False

        def __init__(self, *args, **kwargs):
            pass

    mock_ort = mock.MagicMock()
    mock_ort.InferenceSession = MockSession
    mock_ort.InferenceSession.is_patched = False

    original_import = __import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "audio_separator.separator":
            raise ImportError("Mocked ImportError")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
        with mock.patch("builtins.__import__", side_effect=failing_import):
            preprocessing.apply_onnx_optimizations()
            assert MockSession.is_patched is True


def test_init_separator_already_initialized(prep_manager):
    """Cover line 226 where separator is returned immediately if already initialized."""
    mock_sep = mock.MagicMock()
    prep_manager.separator = mock_sep
    assert prep_manager._init_separator() is mock_sep


def test_init_separator_accelerated(prep_manager):
    """Cover lines 270-271 where hardware_acceleration_enabled is forced."""
    mock_ort = mock.MagicMock()
    mock_ort.get_available_providers.return_value = ["CUDAExecutionProvider"]
    mock_ort.__version__ = "1.24.1"

    with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
        with mock.patch("modules.inference.pipeline.preprocessing._lazy_import_separator") as mock_imp:
            mock_sep_cls = mock.MagicMock()
            mock_imp.return_value = mock_sep_cls

            prep_manager._device_type = "CUDA"
            prep_manager._device_id = "CUDA"
            sep = prep_manager._init_separator()

            assert sep is not None
            assert sep.hardware_acceleration_enabled is True


def test_preprocess_audio_with_yield_cb(prep_manager):
    """Cover lines 349 and 402 where yield_cb is called."""
    with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
        mock_cfg.ENABLE_VOCAL_SEPARATION = True
        mock_sep = mock.MagicMock()
        mock_sep.separate.return_value = ["vocal.wav"]
        prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        yield_called = [0]

        def yield_cb():
            yield_called[0] += 1

        with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
            with mock.patch(
                "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path
            ) as mock_prepare:
                res = prep_manager.preprocess_audio("test.wav", yield_cb=yield_cb)
                assert "vocal.wav" in res
                assert yield_called[0] == 2
                assert mock_prepare.call_args.kwargs.get("yield_cb") is yield_cb


def test_preprocess_audio_non_cpu_make_separator(prep_manager):
    """Cover lines 370-379 (_make_separator) and 386 (non-CPU path)."""
    with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
        mock_cfg.ENABLE_VOCAL_SEPARATION = True
        mock_sep = mock.MagicMock()
        import errno

        mock_sep.separate.side_effect = OSError(errno.ENOSPC, "No space")
        prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = ["CUDAExecutionProvider"]
        prep_manager._device_type = "CUDA"

        with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
            with mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path):
                with mock.patch("modules.inference.pipeline.preprocessing._lazy_import_separator") as mock_lazy_sep:
                    mock_fallback_cls = mock.MagicMock()
                    mock_lazy_sep.return_value = mock_fallback_cls
                    mock_fallback_inst = mock.MagicMock()
                    mock_fallback_inst.separate.return_value = ["fallback_vocal.wav"]
                    mock_fallback_cls.return_value = mock_fallback_inst

                    with mock.patch("modules.inference.pipeline.preprocessing._candidate_output_dirs", return_value=["/dir1", "/dir2"]):
                        with mock.patch("os.makedirs"):
                            with mock.patch("shutil.disk_usage", return_value=mock.MagicMock(free=0)):
                                res = prep_manager.preprocess_audio("test.wav")
                                assert "fallback_vocal.wav" in res
                                mock_fallback_cls.assert_called_once()


def test_preprocess_audio_no_stems(prep_manager):
    """Cover line 422 where stems list is empty."""
    with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
        mock_cfg.ENABLE_VOCAL_SEPARATION = True
        mock_sep = mock.MagicMock()
        mock_sep.separate.return_value = []
        prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
            with mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path):
                res = prep_manager.preprocess_audio("test.wav")
                assert res == "test.wav"


def _make_chunking_separator():
    """Create a separator mock configured for chunking tests."""

    class _ChunkingSep:
        chunk_duration = 300

    mock_sep = _ChunkingSep()
    mock_sep._separate_file = mock.MagicMock(return_value=["vocals.wav"])
    mock_sep._orig_separate_file = mock_sep._separate_file

    def _separate(path):
        preprocessing.utils.THREAD_CONTEXT.uvr_in_chunk_processing = True
        try:
            return mock_sep._separate_file(path)
        finally:
            preprocessing.utils.THREAD_CONTEXT.uvr_in_chunk_processing = False

    mock_sep.separate = _separate
    return mock_sep


def test_preprocess_audio_with_chunking_progress_metadata(prep_manager):
    """Test chunking metadata and original separator invocation."""
    mock_sep = _make_chunking_separator()
    orig_sep_file = mock_sep._separate_file

    with (
        mock.patch("modules.inference.pipeline.preprocessing.utils.get_audio_duration", return_value=700.0),
        mock.patch("modules.inference.scheduler.update_task_metadata") as mock_update_meta,
        mock.patch("modules.inference.scheduler.update_task_progress") as mock_update_prog,
    ):
        stems, used_sep = preprocessing._separate_with_fallback(mock_sep, mock.MagicMock(), "test.wav")

    assert (stems, used_sep is mock_sep, mock_sep._chunk_paths_len, mock_sep._chunk_index, mock_sep._audio_dur) == (
        ["vocals.wav"],
        True,
        3,
        1,
        700.0,
    )
    assert mock_update_meta.call_args_list == [mock.call(current_position=0.0), mock.call(current_position=300.0)]
    assert mock_update_prog.call_count == 2
    start_args, _ = mock_update_prog.call_args_list[0]
    end_args, _ = mock_update_prog.call_args_list[1]
    assert (orig_sep_file.call_args, start_args[0], end_args[0], "1/3 segments" in start_args[1], "1/3 segments" in end_args[1]) == (
        mock.call("test.wav", None),
        5,
        6,
        True,
        True,
    )


def test_preprocess_audio_with_chunking_progress_scheduler_updates(prep_manager):
    """Test chunking scheduler progress updates only."""
    mock_sep = _make_chunking_separator()

    with (
        mock.patch("modules.inference.pipeline.preprocessing.utils.get_audio_duration", return_value=700.0),
        mock.patch("modules.inference.scheduler.update_task_metadata") as mock_update_meta,
        mock.patch("modules.inference.scheduler.update_task_progress") as mock_update_prog,
    ):
        preprocessing._separate_with_fallback(mock_sep, mock.MagicMock(), "test.wav")

    assert mock_update_meta.call_count == 2
    assert mock_update_prog.call_count == 2


def test_preprocess_audio_chunking_yields_at_segment_boundaries():
    """Chunked separation should invoke the preemption callback before and after each segment."""

    class _ChunkingSep:
        chunk_duration = 300

    mock_sep = _ChunkingSep()
    mock_sep._separate_file = mock.MagicMock(return_value=["vocals.wav"])
    mock_sep._orig_separate_file = mock_sep._separate_file

    def _separate(path):
        result = None
        for _ in range(3):
            preprocessing.utils.THREAD_CONTEXT.uvr_in_chunk_processing = True
            try:
                result = mock_sep._separate_file(path)
            finally:
                preprocessing.utils.THREAD_CONTEXT.uvr_in_chunk_processing = False
        return result

    mock_sep.separate = _separate

    yield_calls = []

    with (
        mock.patch("modules.inference.pipeline.preprocessing.utils.get_audio_duration", return_value=700.0),
        mock.patch("modules.inference.scheduler.update_task_metadata"),
        mock.patch("modules.inference.scheduler.update_task_progress"),
    ):
        stems, used_sep = preprocessing._separate_with_fallback(
            mock_sep, mock.MagicMock(), "test.wav", yield_cb=lambda: yield_calls.append("yield")
        )

    assert stems == ["vocals.wav"]
    assert used_sep is mock_sep
    assert mock_sep._chunk_index == 3
    assert len(yield_calls) == 6


def test_preprocessing_helpers_candidate_output_dirs_deduplicates():
    """Candidate output directories should keep order while removing duplicates."""
    with mock.patch("modules.inference.pipeline.preprocessing.helpers.config.PERSISTENT_TEMP_DIR", str(preprocessing.CACHE_DIR)):
        from modules.inference.pipeline.preprocessing import helpers as preprocessing_helpers

        dirs = preprocessing_helpers.candidate_output_dirs()
    assert isinstance(dirs, list)
    assert len(dirs) == len(set(dirs))


def test_preprocessing_helpers_separator_attr_paths():
    """Separator patch helpers should use explicit attribute presence for patch state."""

    class _Sep:
        def _separate_file(self, *_args, **_kwargs):
            return ["ok.wav"]

    sep = _Sep()

    from modules.inference.pipeline.preprocessing import helpers as preprocessing_helpers

    preprocessing_helpers._ensure_orig_separate_file_attr(sep)
    assert hasattr(sep, "_orig_separate_file")
    assert preprocessing_helpers._separator_already_patched(sep) is False

    sep._is_permanently_patched = True
    assert preprocessing_helpers._separator_already_patched(sep) is True


def test_preprocessing_helpers_delegate_outer_chunk_call_true_branch():
    """Outer chunk delegation should trigger when chunking is active and original separator is not a mock."""
    from modules.inference.pipeline.preprocessing import helpers as preprocessing_helpers

    preprocessing_helpers.utils.THREAD_CONTEXT.uvr_chunk_paths_len = 2
    preprocessing_helpers.utils.THREAD_CONTEXT.uvr_in_chunk_processing = False
    assert preprocessing_helpers._should_delegate_outer_chunk_call() is True


def test_preprocessing_helpers_run_outer_chunk_delegate_restores_flag():
    """Outer chunk delegate should set and always clear the in-chunk flag."""
    from modules.inference.pipeline.preprocessing import helpers as preprocessing_helpers

    class _Sep:
        def _orig_separate_file(self, *_args, **_kwargs):
            return ["done.wav"]

    sep = _Sep()
    preprocessing_helpers.utils.THREAD_CONTEXT.uvr_in_chunk_processing = False
    result = preprocessing_helpers._run_outer_chunk_delegate(sep, "audio.wav", None)
    assert result == ["done.wav"]
    assert preprocessing_helpers.utils.THREAD_CONTEXT.uvr_in_chunk_processing is False


def test_preprocessing_helpers_patch_audio_separator_and_provider_dict_normalization():
    """Cover audio-separator patch path and provider-options dict normalization helper."""
    from modules.inference.pipeline.preprocessing import helpers as preprocessing_helpers

    class _Separator:
        is_patched = False

    fake_module = mock.MagicMock()
    fake_module.Separator = _Separator

    with mock.patch("importlib.import_module", return_value=fake_module):
        preprocessing_helpers._patch_audio_separator_onnx_check()

    assert _Separator.is_patched is True

    provider_options = [None]
    normalized = preprocessing_helpers._ensure_provider_option_entry_dict(provider_options, 0)
    assert isinstance(normalized[0], dict)
