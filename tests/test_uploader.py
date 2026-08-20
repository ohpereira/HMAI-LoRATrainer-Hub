"""Tests for uploader.py — S3 upload + presigned URL + _uploads/ sweep logic."""

import os
from pathlib import Path
from unittest.mock import MagicMock as _MagicMock, patch

import pytest

import config as cfg
from config import UPLOADS_SUBDIR, TrainingJob, validate_payload


def MagicMock(*args, **kwargs):
    """Provide successful HeadObject metadata for upload verification tests."""
    client = _MagicMock(*args, **kwargs)
    client.head_object.return_value = {
        "ContentLength": 100,
        "ETag": '"test-etag"',
    }
    return client


@pytest.fixture(autouse=True)
def _patch_config(tmp_path):
    """Patch config paths for all uploader tests."""
    original_jobs = cfg.JOBS_DIR
    yield
    cfg.JOBS_DIR = original_jobs


def _make_test_job(tmp_path, model_type="sdxl", noise_variant=None):
    cfg.JOBS_DIR = tmp_path / "jobs"

    payload = {
        "model_type": model_type,
        "dataset_zip_url": "https://example.com/data.zip",
        "trigger_word": "testword",
        "job_id": "test-job",
    }
    if noise_variant is not None:
        payload["noise_variant"] = noise_variant
    job = validate_payload(payload)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    return job


def _make_uploads_dir(job: TrainingJob) -> Path:
    """Create the _uploads staging dir sibling of output_dir."""
    uploads_dir = job.output_dir.parent / UPLOADS_SUBDIR
    uploads_dir.mkdir(parents=True, exist_ok=True)
    return uploads_dir


class TestUploadFile:
    def test_upload_called_correctly(self, mock_env_s3, tmp_path):
        from uploader import upload_file

        mock_s3 = MagicMock()
        path = tmp_path / "test.safetensors"
        path.write_bytes(b"test")

        upload_file(mock_s3, "test-bucket", path, "lora/test.safetensors")

        mock_s3.put_object.assert_called_once()
        assert mock_s3.put_object.call_args.kwargs["Bucket"] == "test-bucket"
        assert mock_s3.put_object.call_args.kwargs["Key"] == "lora/test.safetensors"


class TestPresignedUrl:
    def test_presigned_url_expiry(self, mock_env_s3):
        from uploader import generate_presigned_url

        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "https://presigned-url"

        url = generate_presigned_url(mock_s3, "test-bucket", "key.safetensors")

        mock_s3.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "key.safetensors"},
            ExpiresIn=604799,
        )
        assert url == "https://presigned-url"


class TestMaybeUploadOutputs:
    def test_scans_uploads_dir_not_output_dir(self, tmp_path, mock_env_s3):
        """Final sweep reads from _uploads/, not output_dir directly."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)

        # Put a file in _uploads/ and a decoy in output_dir — only _uploads/ file is returned
        renamed = uploads_dir / "testword_sdxl_epoch100.safetensors"
        renamed.write_bytes(b"\x00" * 100)
        decoy = job.output_dir / "aitk_lora_000000100.safetensors"
        decoy.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        filenames = [f["filename"] for f in result["output_files"]]
        assert "testword_sdxl_epoch100.safetensors" in filenames
        assert "aitk_lora_000000100.safetensors" not in filenames

    def test_finds_safetensors_in_uploads_dir(self, tmp_path, mock_env_s3):
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)
        sf = uploads_dir / "testword_sdxl_epoch100.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        assert len(result["output_files"]) == 1
        assert result["output_files"][0]["filename"] == "testword_sdxl_epoch100.safetensors"

    def test_no_adapter_filter(self, tmp_path, mock_env_s3):
        """adapter_ prefix must NOT be filtered out (spec §B7)."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)

        # If watcher placed a file whose name starts with adapter_ in uploads_dir,
        # it should still be uploaded (no filter).
        adapter_file = uploads_dir / "adapter_model.safetensors"
        adapter_file.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        filenames = [f["filename"] for f in result["output_files"]]
        assert "adapter_model.safetensors" in filenames

    def test_no_subdir_filter(self, tmp_path, mock_env_s3):
        """Files are NOT filtered by high/low parent dir — flat scan only."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path, model_type="wan2.2", noise_variant="high")
        uploads_dir = _make_uploads_dir(job)

        # Flat file with high variant in name — should be picked up
        sf = uploads_dir / "testword_wan2.2high_epoch80.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        filenames = [f["filename"] for f in result["output_files"]]
        assert "testword_wan2.2high_epoch80.safetensors" in filenames

    def test_variant_from_filename_high(self, tmp_path, mock_env_s3):
        """noise_variant detected from filename suffix — high."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path, model_type="wan2.2", noise_variant="high")
        uploads_dir = _make_uploads_dir(job)
        sf = uploads_dir / "testword_wan2.2high_epoch80.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        assert result["output_files"][0]["noise_variant"] == "high"

    def test_variant_from_filename_low(self, tmp_path, mock_env_s3):
        """noise_variant detected from filename suffix — low."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path, model_type="wan2.2", noise_variant="low")
        uploads_dir = _make_uploads_dir(job)
        sf = uploads_dir / "testword_wan2.2low_epoch80.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        assert result["output_files"][0]["noise_variant"] == "low"

    def test_variant_empty_for_non_wan(self, tmp_path, mock_env_s3):
        """noise_variant is empty string for non-wan models."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)
        sf = uploads_dir / "testword_sdxl_epoch100.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        assert result["output_files"][0]["noise_variant"] == ""

    def test_skips_non_safetensors(self, tmp_path, mock_env_s3):
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)
        (uploads_dir / "log.txt").write_text("training log")
        (uploads_dir / "testword_sdxl_epoch100.safetensors").write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        filenames = [f["filename"] for f in result["output_files"]]
        assert all(f.endswith(".safetensors") for f in filenames)

    def test_no_s3_configured(self, tmp_path):
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)
        sf = uploads_dir / "testword_sdxl_epoch100.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch.dict(os.environ, {}, clear=True):
            result = maybe_upload_outputs(job)

        assert result["storage"] == "local_only"
        assert len(result["presigned_urls"]) == 0

    def test_no_s3_configured_empty_uploads_dir(self, tmp_path):
        """No uploads dir → local_only with empty list (no crash)."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        # Do NOT create uploads_dir — should not crash

        with patch.dict(os.environ, {}, clear=True):
            result = maybe_upload_outputs(job)

        assert result["storage"] == "local_only"
        assert result["output_files"] == []

    def test_upload_failure_fallback(self, tmp_path, mock_env_s3):
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)
        sf = uploads_dir / "testword_sdxl_epoch100.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.put_object.side_effect = Exception("Network error")
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        assert any("upload_error" in f for f in result["output_files"])

    def test_idempotent_multiple_files(self, tmp_path, mock_env_s3):
        """Multiple files in uploads_dir all appear in output (idempotent re-upload ok)."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)
        for name in [
            "testword_sdxl_epoch50.safetensors",
            "testword_sdxl_epoch100.safetensors",
            "testword_sdxl_epoch100.safetensors",  # duplicate write (same name)
        ]:
            (uploads_dir / name).write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = maybe_upload_outputs(job)

        # Unique filenames (filesystem deduplicates)
        filenames = [f["filename"] for f in result["output_files"]]
        assert "testword_sdxl_epoch50.safetensors" in filenames
        assert "testword_sdxl_epoch100.safetensors" in filenames

    def test_s3_key_uses_job_id(self, tmp_path, mock_env_s3):
        """S3 key must include job.job_id."""
        from uploader import maybe_upload_outputs

        job = _make_test_job(tmp_path)
        uploads_dir = _make_uploads_dir(job)
        sf = uploads_dir / "testword_sdxl_epoch100.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            maybe_upload_outputs(job)

        call_args = mock_s3.put_object.call_args_list[0]
        s3_key = call_args.kwargs["Key"]
        assert "test-job" in s3_key


class TestRecoverOutputs:
    def test_reuploads_staged_checkpoints_from_previous_job(self, tmp_path, mock_env_s3, monkeypatch):
        from uploader import recover_outputs

        source_job_id = "11111111-1111-4111-8111-111111111111-u1"
        uploads_dir = tmp_path / "jobs" / source_job_id / UPLOADS_SUBDIR
        uploads_dir.mkdir(parents=True)
        (uploads_dir / "testword_krea2_epoch100.safetensors").write_bytes(b"\x00" * 100)
        monkeypatch.setenv("VOLUME_ROOT", str(tmp_path))

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = recover_outputs(
                source_job_id,
                "private/lora-training/user/job/outputs/",
            )

        assert result["output_files"][0]["s3_key"] == (
            "private/lora-training/user/job/outputs/testword_krea2_epoch100.safetensors"
        )
        assert mock_s3.put_object.call_count == 1


class TestUploadAndPresign:
    def test_uploads_and_returns_urls(self, tmp_path, mock_env_s3):
        from uploader import upload_and_presign

        sf = tmp_path / "model.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://presigned"
            mock_boto.return_value = mock_s3

            result = upload_and_presign(sf, "test-job")

        assert result is not None
        assert result["filename"] == "model.safetensors"
        assert "url" in result
        assert result["presigned_url"] == "https://presigned"

    def test_returns_none_without_s3(self, tmp_path):
        from uploader import upload_and_presign

        sf = tmp_path / "model.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch.dict(os.environ, {}, clear=True):
            result = upload_and_presign(sf, "test-job")

        assert result is None

    def test_returns_none_on_upload_error(self, tmp_path, mock_env_s3):
        from uploader import upload_and_presign

        sf = tmp_path / "model.safetensors"
        sf.write_bytes(b"\x00" * 100)

        with patch("uploader.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_s3.upload_file.side_effect = Exception("Network error")
            mock_boto.return_value = mock_s3

            result = upload_and_presign(sf, "test-job")

        assert result is None
