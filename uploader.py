"""S3 upload with presigned URL generation and graceful fallback."""

from __future__ import annotations

import os
import re
from pathlib import Path

import boto3
from loguru import logger

from config import UPLOADS_SUBDIR, TrainingJob


def _get_s3_client():
    """Create S3 client from environment variables. Returns None if not configured.

    Supports Cloudflare R2 (R2_* vars) and AWS S3 (AWS_* vars) — R2 takes
    precedence when both are present.
    """
    r2_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_account = os.environ.get("R2_ACCOUNT_ID")
    r2_bucket = os.environ.get("R2_BUCKET")
    if all([r2_key, r2_secret, r2_account, r2_bucket]):
        client = boto3.client(
            "s3",
            aws_access_key_id=r2_key,
            aws_secret_access_key=r2_secret,
            endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
            region_name="auto",
        )
        return client, r2_bucket, "r2"

    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_bucket = os.environ.get("S3_BUCKET")
    aws_region = os.environ.get("S3_REGION", "us-east-1")
    if all([aws_key, aws_secret, aws_bucket]):
        client = boto3.client(
            "s3",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region,
        )
        return client, aws_bucket, aws_region

    return None, None, None


def upload_file(s3_client, bucket: str, local_path: Path, s3_key: str) -> str:
    """Upload a file and return a stable URL for it.

    For R2 we return the presigned URL as the canonical link (R2 objects
    aren't public by default). For AWS S3 we return the virtual-hosted URL.
    """
    s3_client.upload_file(str(local_path), bucket, s3_key)
    region = os.environ.get("S3_REGION", "us-east-1")
    if os.environ.get("R2_BUCKET"):
        url = generate_presigned_url(s3_client, bucket, s3_key)
    else:
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
    logger.info(f"Uploaded {local_path.name} -> {bucket}/{s3_key}")
    return url


def generate_presigned_url(
    s3_client,
    bucket: str,
    s3_key: str,
    expiration: int = 604799,  # 7 days minus 1 second
) -> str:
    """Generate a presigned URL for downloading from S3."""
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=expiration,
    )
    return url


def upload_and_presign(
    local_path: Path,
    job_id: str,
    output_prefix: str | None = None,
) -> dict | None:
    """Upload a single file to S3 and return metadata with presigned URL.

    Returns None if S3 is not configured.
    """
    s3_client, bucket, region = _get_s3_client()
    if s3_client is None:
        return None

    prefix = output_prefix or f"lora-outputs/{job_id}/"
    s3_key = f"{prefix}{local_path.name}"
    try:
        url = upload_file(s3_client, bucket, local_path, s3_key)
        metadata = _verified_object_metadata(s3_client, bucket, local_path, s3_key)
        presigned = generate_presigned_url(s3_client, bucket, s3_key)
        return {
            "filename": local_path.name,
            "url": url,
            "presigned_url": presigned,
            "s3_key": s3_key,
            "epoch": _extract_epoch(local_path),
            "size_bytes": metadata["size_bytes"],
            "etag": metadata["etag"],
            "noise_variant": _detect_variant(local_path),
        }
    except Exception as e:
        logger.error(f"Incremental upload failed for {local_path.name}: {e}")
        return None


def maybe_upload_outputs(job: TrainingJob) -> dict:
    """Find and upload all .safetensors files from the _uploads staging dir.

    The trainer watcher copies renamed checkpoints into
    job.output_dir.parent / UPLOADS_SUBDIR before upload, so they live outside
    AI-Toolkit's prune glob ({name}_*). We scan that dir here.

    Variant detection comes from the filename suffix (_high_noise / _low_noise)
    via job.noise_variant — no adapter_ filter, no high/low subdir filter.
    Idempotent re-upload of the same S3 key is fine.

    Returns a dict with output_files and presigned_urls.
    Falls back to local paths if S3 is not configured.
    """
    uploads_dir = job.output_dir.parent / UPLOADS_SUBDIR

    # Collect all .safetensors from the uploads staging dir (flat — watcher
    # copies files directly into uploads_dir, not into subdirs).
    if uploads_dir.exists():
        found_files = sorted(uploads_dir.glob("*.safetensors"), key=_checkpoint_sort_key)
    else:
        found_files = []

    s3_client, bucket, region = _get_s3_client()

    if s3_client is None:
        logger.warning("S3 not configured — returning local paths only")
        return {
            "storage": "local_only",
            "output_files": [
                {"filename": f.name, "local_path": str(f)}
                for f in found_files
            ],
            "presigned_urls": [],
        }

    output_files = []
    presigned_urls = []
    for f in found_files:
        prefix = job.output_prefix or f"lora-outputs/{job.job_id}/"
        s3_key = f"{prefix}{f.name}"
        try:
            url = upload_file(s3_client, bucket, f, s3_key)
            metadata = _verified_object_metadata(s3_client, bucket, f, s3_key)
            presigned = generate_presigned_url(s3_client, bucket, s3_key)
            output_files.append({
                "filename": f.name,
                "url": url,
                "s3_key": s3_key,
                "epoch": _extract_epoch(f),
                "size_bytes": metadata["size_bytes"],
                "etag": metadata["etag"],
                "noise_variant": _detect_variant(f),
            })
            presigned_urls.append(presigned)
        except Exception as e:
            logger.error(f"Failed to upload {f.name}: {e}")
            output_files.append({
                "filename": f.name,
                "local_path": str(f),
                "upload_error": str(e),
            })

    return {
        "output_files": output_files,
        "presigned_urls": presigned_urls,
    }


def _detect_variant(filepath: Path) -> str:
    """Detect noise variant from filename suffix."""
    name = filepath.name.lower()
    if "high" in name:
        return "high"
    if "low" in name:
        return "low"
    return ""


def _extract_epoch(filepath: Path) -> int | None:
    """Extract a numeric epoch when the checkpoint filename contains one."""
    match = re.search(r"epoch(\d+)", filepath.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _checkpoint_sort_key(filepath: Path) -> tuple[bool, int, str]:
    """Sort epoch checkpoints numerically while preserving legacy filenames."""
    epoch = _extract_epoch(filepath)
    return (epoch is None, epoch or 0, filepath.name)


def _verified_object_metadata(s3_client, bucket: str, local_path: Path, s3_key: str) -> dict:
    """Verify the uploaded object exists and matches the local checkpoint size."""
    expected_size = local_path.stat().st_size
    uploaded = s3_client.head_object(Bucket=bucket, Key=s3_key)
    size_bytes = int(uploaded.get("ContentLength", 0))
    etag = str(uploaded.get("ETag", "")).strip('"')
    if size_bytes != expected_size or not etag:
        raise RuntimeError("Uploaded object metadata does not match the local checkpoint")
    return {"size_bytes": size_bytes, "etag": etag}
