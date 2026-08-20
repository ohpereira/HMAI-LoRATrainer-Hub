"""Tests for handler.py — end-to-end orchestration with all deps mocked."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import TrainingJob, TrainingResult


def _run_handler(event, train_result=None, upload_result=None):
    """Run the handler with all external deps mocked. Returns (result, mock_train, mock_upload)."""
    if train_result is None:
        train_result = TrainingResult(
            ok=True,
            output_files=[{"filename": "test.safetensors", "url": "https://s3/test"}],
        )
    if upload_result is None:
        upload_result = {
            "output_files": [{"filename": "test.safetensors", "url": "https://s3/test"}],
            "presigned_urls": ["https://presigned"],
        }

    with patch("handler._create_workspace"), \
         patch("handler.download_dataset"), \
         patch("handler.extract_zip"), \
         patch("handler.validate_dataset", return_value=[]), \
         patch("handler.count_dataset_media", return_value=20), \
         patch("handler.ensure_model"), \
         patch("handler.generate_config", return_value={}), \
         patch("handler.run_training") as mock_train, \
         patch("handler.maybe_upload_outputs") as mock_upload, \
         patch("handler._cleanup_dataset"):
        from handler import handler

        mock_train.return_value = train_result
        mock_upload.return_value = upload_result

        return handler(event), mock_train, mock_upload


class TestHandler:
    def test_wan22_job(self):
        event = {
            "id": "job-123",
            "input": {
                "model_type": "wan2.2",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "testword",
                "noise_variant": "high",
            },
        }
        result, mock_train, _ = _run_handler(event)
        assert result["ok"] is True
        assert result["model_type"] == "wan2.2"
        assert result["trigger_word"] == "testword"

    def test_sdxl_job(self):
        event = {
            "id": "job-456",
            "input": {
                "model_type": "sdxl",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "person01",
            },
        }
        result, _, _ = _run_handler(event)
        assert result["ok"] is True
        assert result["model_type"] == "sdxl"

    def test_ideogram4_job(self):
        event = {
            "id": "job-ideo",
            "input": {
                "model_type": "ideogram4",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "artword",
            },
        }
        result, _, _ = _run_handler(event)
        assert result["ok"] is True
        assert result["model_type"] == "ideogram4"

    def test_flux_klein_9b_job(self):
        event = {
            "id": "job-flux",
            "input": {
                "model_type": "flux_klein_9b",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "fluxword",
            },
        }
        result, _, _ = _run_handler(event)
        assert result["ok"] is True
        assert result["model_type"] == "flux_klein_9b"

    def test_invalid_payload_unknown_model(self):
        event = {"id": "bad", "input": {"model_type": "flux"}}
        with patch("handler._create_workspace"), \
             patch("handler.download_dataset"), \
             patch("handler.extract_zip"), \
             patch("handler.validate_dataset", return_value=[]), \
             patch("handler.count_dataset_media", return_value=0), \
             patch("handler.ensure_model"), \
             patch("handler.generate_config"), \
             patch("handler.run_training"), \
             patch("handler.maybe_upload_outputs"), \
             patch("handler._cleanup_dataset"):
            from handler import handler

            result = handler(event)
        assert result["ok"] is False
        assert result["error_type"] == "VALIDATION"

    def test_invalid_payload_wan22_missing_noise_variant(self):
        """wan2.2 without noise_variant should fail validation."""
        event = {
            "id": "bad-wan",
            "input": {
                "model_type": "wan2.2",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "tw",
                # noise_variant intentionally absent
            },
        }
        with patch("handler._create_workspace"), \
             patch("handler.download_dataset"), \
             patch("handler.extract_zip"), \
             patch("handler.validate_dataset", return_value=[]), \
             patch("handler.count_dataset_media", return_value=0), \
             patch("handler.ensure_model"), \
             patch("handler.generate_config"), \
             patch("handler.run_training"), \
             patch("handler.maybe_upload_outputs"), \
             patch("handler._cleanup_dataset"):
            from handler import handler

            result = handler(event)
        assert result["ok"] is False
        assert result["error_type"] == "VALIDATION"

    def test_noise_variant_rejected_for_non_wan22(self):
        """noise_variant is only valid for wan2.2; other models should fail."""
        event = {
            "id": "bad-nv",
            "input": {
                "model_type": "sdxl",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "tw",
                "noise_variant": "high",
            },
        }
        with patch("handler._create_workspace"), \
             patch("handler.download_dataset"), \
             patch("handler.extract_zip"), \
             patch("handler.validate_dataset", return_value=[]), \
             patch("handler.count_dataset_media", return_value=0), \
             patch("handler.ensure_model"), \
             patch("handler.generate_config"), \
             patch("handler.run_training"), \
             patch("handler.maybe_upload_outputs"), \
             patch("handler._cleanup_dataset"):
            from handler import handler

            result = handler(event)
        assert result["ok"] is False
        assert result["error_type"] == "VALIDATION"

    def test_dataset_download_failure(self):
        event = {
            "id": "job-err",
            "input": {
                "model_type": "sdxl",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "tw",
            },
        }
        with patch("handler._create_workspace"), \
             patch("handler.download_dataset", side_effect=Exception("Network error")), \
             patch("handler.extract_zip"), \
             patch("handler.validate_dataset", return_value=[]), \
             patch("handler.count_dataset_media", return_value=0), \
             patch("handler.ensure_model"), \
             patch("handler.generate_config"), \
             patch("handler.run_training"), \
             patch("handler.maybe_upload_outputs"), \
             patch("handler._cleanup_dataset"):
            from handler import handler

            result = handler(event)
        assert result["ok"] is False
        assert "Network error" in result["error"]

    def test_training_failure(self):
        event = {
            "id": "job-fail",
            "input": {
                "model_type": "sdxl",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "tw",
            },
        }
        result, _, _ = _run_handler(
            event,
            train_result=TrainingResult(ok=False, error="CUDA OOM", error_type="TRAINING"),
        )
        assert result["ok"] is False
        assert result["error_type"] == "TRAINING"

    def test_response_includes_timing(self):
        event = {
            "id": "job-time",
            "input": {
                "model_type": "sdxl",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "tw",
            },
        }
        result, _, _ = _run_handler(event)
        assert "timing" in result
        assert "dataset_download_s" in result["timing"]
        assert "total_s" in result["timing"]

    def test_response_includes_trigger_word(self):
        event = {
            "id": "job-tw",
            "input": {
                "model_type": "sdxl",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "daniella01",
            },
        }
        result, _, _ = _run_handler(event)
        assert result["trigger_word"] == "daniella01"

    def test_generate_config_called_with_img_count(self):
        """generate_config must be called with (job, img_count) — not with model_spec."""
        event = {
            "id": "job-gc",
            "input": {
                "model_type": "sdxl",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "tw",
            },
        }
        with patch("handler._create_workspace"), \
             patch("handler.download_dataset"), \
             patch("handler.extract_zip"), \
             patch("handler.validate_dataset", return_value=[]), \
             patch("handler.count_dataset_media", return_value=42) as mock_count, \
             patch("handler.ensure_model"), \
             patch("handler.generate_config") as mock_gen, \
             patch("handler.run_training") as mock_train, \
             patch("handler.maybe_upload_outputs") as mock_upload, \
             patch("handler._cleanup_dataset"):
            from handler import handler

            mock_train.return_value = TrainingResult(ok=True, output_files=[])
            mock_upload.return_value = {"output_files": [], "presigned_urls": []}
            handler(event)

        mock_gen.assert_called_once()
        call_args = mock_gen.call_args
        # Second positional arg is the img_count
        assert call_args[0][1] == 42

    def test_civitai_skips_ensure_model(self):
        """When civitai_model_id is set, ensure_model should NOT be called."""
        event = {
            "id": "job-civitai",
            "input": {
                "model_type": "sdxl",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "tw",
                "civitai_model_id": "12345",
            },
        }
        with patch("handler._create_workspace"), \
             patch("handler.download_dataset"), \
             patch("handler.extract_zip"), \
             patch("handler.validate_dataset", return_value=[]), \
             patch("handler.count_dataset_media", return_value=10), \
             patch("handler.ensure_model") as mock_ensure, \
             patch("handler._maybe_download_civitai", return_value=True) as mock_civitai, \
             patch("handler.generate_config", return_value={}), \
             patch("handler.run_training") as mock_train, \
             patch("handler.maybe_upload_outputs") as mock_upload, \
             patch("handler._cleanup_dataset"):
            from handler import handler

            mock_train.return_value = TrainingResult(ok=True, output_files=[])
            mock_upload.return_value = {"output_files": [], "presigned_urls": []}

            result = handler(event)

        assert result["ok"] is True
        mock_civitai.assert_called_once()
        mock_ensure.assert_not_called()

    def test_smoke_returns_early_without_training(self):
        """A smoke=True event validates + resolves the model, returns ok/smoke,
        and skips download_dataset/ensure_model/run_training entirely."""
        event = {
            "id": "job-smoke",
            "input": {
                "smoke": True,
                "model_type": "sdxl",
                "dataset_zip_url": "x",
                "trigger_word": "hmtest",
            },
        }
        with patch("handler._create_workspace"), \
             patch("handler.download_dataset") as mock_download, \
             patch("handler.extract_zip"), \
             patch("handler.validate_dataset", return_value=[]), \
             patch("handler.count_dataset_media", return_value=0), \
             patch("handler.ensure_model") as mock_ensure, \
             patch("handler.generate_config"), \
             patch("handler.run_training") as mock_train, \
             patch("handler.maybe_upload_outputs"), \
             patch("handler._cleanup_dataset"):
            from handler import handler

            result = handler(event)

        assert result["ok"] is True
        assert result["smoke"] is True
        assert result["model_type"] == "sdxl"
        assert result["trigger_word"] == "hmtest"
        mock_download.assert_not_called()
        mock_ensure.assert_not_called()
        mock_train.assert_not_called()

    def test_contract_probe_returns_before_any_worker_action(self):
        event = {"id": "job-probe", "input": {"contract_probe": "hervora-validation-v2"}}
        with patch("handler.get_gpu_info") as mock_gpu, \
             patch("handler.validate_payload") as mock_validate, \
             patch("handler.download_dataset") as mock_download, \
             patch("handler.ensure_model") as mock_ensure, \
             patch("handler.run_training") as mock_train:
            from handler import handler

            result = handler(event)

        assert result == {
            "ok": True,
            "contract_version": "hervora-validation-v2",
            "supports_validate_only": True,
        }
        mock_gpu.assert_not_called()
        mock_validate.assert_not_called()
        mock_download.assert_not_called()
        mock_ensure.assert_not_called()
        mock_train.assert_not_called()

    def test_no_ltx_support(self):
        """LTX model_type is gone — should fail validation."""
        event = {
            "id": "job-ltx",
            "input": {
                "model_type": "ltx_2.3",
                "dataset_zip_url": "https://example.com/data.zip",
                "trigger_word": "tw",
            },
        }
        with patch("handler._create_workspace"), \
             patch("handler.download_dataset"), \
             patch("handler.extract_zip"), \
             patch("handler.validate_dataset", return_value=[]), \
             patch("handler.count_dataset_media", return_value=0), \
             patch("handler.ensure_model"), \
             patch("handler.generate_config"), \
             patch("handler.run_training"), \
             patch("handler.maybe_upload_outputs"), \
             patch("handler._cleanup_dataset"):
            from handler import handler

            result = handler(event)
        assert result["ok"] is False
        assert result["error_type"] == "VALIDATION"


class TestMaybeDownloadCivitai:
    def test_noop_when_no_civitai_id(self):
        """Returns False when civitai_model_id is not set."""
        from handler import _maybe_download_civitai

        job = TrainingJob(
            job_id="test",
            model_type="sdxl",
            dataset_zip_url="https://example.com/d.zip",
            trigger_word="tw",
        )
        assert _maybe_download_civitai(job) is False
        assert job.civitai_checkpoint_path is None

    def test_clones_repo_and_downloads(self, tmp_path):
        """Clones the downloader repo and runs the download script."""
        from handler import _maybe_download_civitai

        job = TrainingJob(
            job_id="test",
            model_type="sdxl",
            dataset_zip_url="https://example.com/d.zip",
            trigger_word="tw",
            civitai_model_id="12345",
        )

        fake_downloader = tmp_path / "downloader"

        def fake_run(cmd, **kwargs):
            """Simulate git clone + download_with_aria.py."""
            if cmd[0] == "git":
                fake_downloader.mkdir(parents=True, exist_ok=True)
            elif cmd[0] == "python":
                # Create a fake safetensors output
                output_dir = Path(cmd[-1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "model.safetensors").write_bytes(b"fake")
            return MagicMock(returncode=0)

        import config as cfg
        original_jobs = cfg.JOBS_DIR
        cfg.JOBS_DIR = tmp_path / "jobs"
        try:
            with patch("handler.CIVITAI_DOWNLOADER_DIR", fake_downloader), \
                 patch("subprocess.run", side_effect=fake_run) as mock_run:
                assert not fake_downloader.exists()
                result = _maybe_download_civitai(job)
        finally:
            cfg.JOBS_DIR = original_jobs

        assert result is True
        assert job.civitai_checkpoint_path is not None
        assert job.civitai_checkpoint_path.endswith(".safetensors")
        # which aria2c (dep check) + git clone + python download
        assert mock_run.call_count == 3

    def test_raises_on_no_safetensors(self, tmp_path):
        """Raises RuntimeError if download produces no .safetensors file."""
        from handler import _maybe_download_civitai

        job = TrainingJob(
            job_id="test",
            model_type="sdxl",
            dataset_zip_url="https://example.com/d.zip",
            trigger_word="tw",
            civitai_model_id="99999",
        )

        fake_downloader = tmp_path / "downloader"
        fake_downloader.mkdir()

        def fake_run(cmd, **kwargs):
            if cmd[0] == "python":
                output_dir = Path(cmd[-1])
                output_dir.mkdir(parents=True, exist_ok=True)
            return MagicMock(returncode=0)

        import config as cfg
        original_jobs = cfg.JOBS_DIR
        cfg.JOBS_DIR = tmp_path / "jobs"
        try:
            with patch("handler.CIVITAI_DOWNLOADER_DIR", fake_downloader), \
                 patch("subprocess.run", side_effect=fake_run):
                with pytest.raises(RuntimeError, match="no .safetensors file"):
                    _maybe_download_civitai(job)
        finally:
            cfg.JOBS_DIR = original_jobs


class TestCountDatasetMedia:
    def test_counts_image_files(self, tmp_path):
        from dataset import count_dataset_media

        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        (tmp_path / "c.txt").write_bytes(b"x")  # not media
        assert count_dataset_media(tmp_path) == 2

    def test_counts_video_files(self, tmp_path):
        from dataset import count_dataset_media

        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "b.mov").write_bytes(b"x")
        assert count_dataset_media(tmp_path) == 2

    def test_missing_dir_returns_zero(self, tmp_path):
        from dataset import count_dataset_media

        assert count_dataset_media(tmp_path / "nonexistent") == 0

    def test_empty_dir_returns_zero(self, tmp_path):
        from dataset import count_dataset_media

        assert count_dataset_media(tmp_path) == 0

    def test_mixed_extensions(self, tmp_path):
        from dataset import count_dataset_media

        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.mp4").write_bytes(b"x")
        (tmp_path / "c.webp").write_bytes(b"x")
        (tmp_path / "d.txt").write_bytes(b"x")
        (tmp_path / "e.toml").write_bytes(b"x")
        assert count_dataset_media(tmp_path) == 3
