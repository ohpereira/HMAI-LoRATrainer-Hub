"""Tests for dataset.py — download, extraction, validation."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataset import download_dataset, extract_zip, find_orphan_captions, validate_dataset


class TestExtractZip:
    def test_flat_structure(self, sample_zip, tmp_path):
        dest = tmp_path / "extracted"
        extract_zip(sample_zip, dest)
        # Should have 10 files (5 images + 5 captions)
        files = list(dest.iterdir())
        assert len(files) == 10

    def test_nested_single_folder(self, sample_zip_nested, tmp_path):
        dest = tmp_path / "extracted"
        extract_zip(sample_zip_nested, dest)
        # Should unwrap the nested folder
        files = list(dest.iterdir())
        assert len(files) == 10
        # No nested subfolder should remain
        dirs = [f for f in dest.iterdir() if f.is_dir()]
        assert len(dirs) == 0

    def test_macosx_skipped(self, sample_zip_macosx, tmp_path):
        dest = tmp_path / "extracted"
        extract_zip(sample_zip_macosx, dest)
        # Should not have __MACOSX folder
        assert not (dest / "__MACOSX").exists()
        files = list(dest.iterdir())
        assert len(files) == 10

    def test_path_traversal_is_rejected(self, tmp_path):
        archive = tmp_path / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../outside.txt", "blocked")

        with pytest.raises(ValueError, match="unsafe path"):
            extract_zip(archive, tmp_path / "extracted")


class TestValidateDataset:
    def test_all_have_captions(self, sample_dataset_dir):
        unmatched = validate_dataset(sample_dataset_dir)
        assert unmatched == []

    def test_missing_captions(self, sample_dataset_dir_missing_captions):
        unmatched = validate_dataset(sample_dataset_dir_missing_captions)
        assert len(unmatched) == 2
        assert all(name.endswith(".png") for name in unmatched)

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No media files"):
            validate_dataset(empty_dir)

    def test_mixed_extensions(self, sample_dataset_dir_mixed):
        unmatched = validate_dataset(sample_dataset_dir_mixed)
        assert unmatched == []

    def test_orphan_caption_is_reported(self, tmp_path):
        (tmp_path / "image.jpg").write_bytes(b"image")
        (tmp_path / "image.txt").write_text("caption")
        (tmp_path / "orphan.txt").write_text("caption")
        assert find_orphan_captions(tmp_path) == ["orphan.txt"]


class TestDownloadDataset:
    def test_successful_download(self, tmp_path):
        dest = tmp_path / "downloaded.zip"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_bytes.return_value = [b"x" * 100]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("dataset.httpx.Client", return_value=mock_client):
            download_dataset("https://example.com/data.zip", dest)
            assert dest.exists()

    def test_http_error(self, tmp_path):
        import httpx

        dest = tmp_path / "fail.zip"
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404)
        )
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("dataset.httpx.Client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                download_dataset("https://example.com/missing.zip", dest)

    def test_corrupt_zip(self, tmp_path):
        corrupt_zip = tmp_path / "corrupt.zip"
        corrupt_zip.write_bytes(b"this is not a zip file")
        dest = tmp_path / "extracted"
        with pytest.raises(zipfile.BadZipFile):
            extract_zip(corrupt_zip, dest)
