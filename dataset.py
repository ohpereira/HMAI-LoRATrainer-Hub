"""Dataset download, extraction, and validation."""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import httpx
from loguru import logger

from config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

DOWNLOAD_TIMEOUT = 300
CHUNK_SIZE = 256 * 1024  # 256KB
MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_DATASET_ZIP_BYTES", str(300 * 1024 * 1024)))
MAX_ARCHIVE_FILES = int(os.environ.get("MAX_DATASET_ARCHIVE_FILES", "250"))
MAX_ARCHIVE_FILE_BYTES = int(os.environ.get("MAX_DATASET_FILE_BYTES", str(100 * 1024 * 1024)))
MAX_ARCHIVE_UNCOMPRESSED_BYTES = int(
    os.environ.get("MAX_DATASET_UNCOMPRESSED_BYTES", str(1024 * 1024 * 1024))
)


def download_dataset(url: str, dest_path: Path) -> None:
    """Download a file from url to dest_path with streaming."""
    logger.info("Downloading dataset from a private signed URL")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("Dataset ZIP exceeds the compressed size limit")
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Dataset ZIP exceeds the compressed size limit")
                    f.write(chunk)
                    if total > 0 and downloaded % (CHUNK_SIZE * 40) < CHUNK_SIZE:
                        pct = downloaded / total * 100
                        logger.info(f"Download progress: {pct:.0f}%")

    logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB to {dest_path}")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file, handling nested single-folder zips and skipping __MACOSX."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination_root = dest_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = []
        total_uncompressed = 0
        seen_paths = set()
        for info in zf.infolist():
            normalized_name = info.filename.replace("\\", "/")
            path = PurePosixPath(normalized_name)
            if normalized_name.startswith("__MACOSX/") or path.name.startswith("._"):
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Dataset ZIP contains an unsafe path")
            if info.flag_bits & 0x1:
                raise ValueError("Encrypted ZIP entries are not supported")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError("Dataset ZIP contains a symbolic link")
            if info.is_dir():
                continue
            if info.file_size > MAX_ARCHIVE_FILE_BYTES:
                raise ValueError("Dataset ZIP contains an oversized file")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError("Dataset ZIP exceeds the uncompressed size limit")
            canonical = normalized_name.casefold()
            if canonical in seen_paths:
                raise ValueError("Dataset ZIP contains duplicate paths")
            seen_paths.add(canonical)
            members.append((info, path))

        if len(members) > MAX_ARCHIVE_FILES:
            raise ValueError("Dataset ZIP contains too many files")

        for info, path in members:
            target = dest_dir.joinpath(*path.parts).resolve()
            if destination_root not in target.parents:
                raise ValueError("Dataset ZIP contains an unsafe path")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=CHUNK_SIZE)

    # If everything extracted into a single subfolder, unwrap it
    children = [c for c in dest_dir.iterdir() if not c.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        nested_dir = children[0]
        logger.info(f"Unwrapping nested folder: {nested_dir.name}")
        for item in nested_dir.iterdir():
            target = dest_dir / item.name
            if target.exists():
                raise ValueError("Dataset ZIP contains colliding paths")
            shutil.move(str(item), str(target))
        nested_dir.rmdir()


def validate_dataset(dataset_dir: Path) -> list[str]:
    """Validate that all media files have matching caption .txt files.

    Returns a list of media filenames that are missing captions.
    Returns empty list if all files have captions.
    Raises ValueError if no media files found.
    """
    all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    media_files = [
        f for f in dataset_dir.iterdir()
        if f.is_file() and f.suffix.lower() in all_extensions
    ]

    if not media_files:
        raise ValueError(f"No media files found in {dataset_dir}")

    unmatched = []
    for media_file in media_files:
        caption_file = media_file.with_suffix(".txt")
        if not caption_file.exists():
            unmatched.append(media_file.name)

    logger.info(
        f"Dataset: {len(media_files)} media files, "
        f"{len(media_files) - len(unmatched)} with captions"
    )
    return unmatched


def count_dataset_media(dataset_dir: Path) -> int:
    """Count the number of media files in dataset_dir.

    Returns 0 if the directory does not exist. The count drives the
    epoch→step math in the YAML generator (steps = epochs * img_count /
    batch_size * gradient_accumulation).
    """
    if not dataset_dir.exists():
        return 0
    all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return sum(
        1 for f in dataset_dir.iterdir()
        if f.is_file() and f.suffix.lower() in all_extensions
    )


def find_orphan_captions(dataset_dir: Path) -> list[str]:
    """Return caption files that do not have a matching media file."""
    all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    media_stems = {
        file.stem.casefold()
        for file in dataset_dir.iterdir()
        if file.is_file() and file.suffix.lower() in all_extensions
    }
    return [
        file.name
        for file in dataset_dir.iterdir()
        if file.is_file() and file.suffix.lower() == ".txt"
        and file.stem.casefold() not in media_stems
    ]
