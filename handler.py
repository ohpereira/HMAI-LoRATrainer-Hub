"""RunPod serverless entry point — thin orchestrator."""

import re
import subprocess
import time
import traceback
from pathlib import Path

import runpod
from loguru import logger

from gpu import get_gpu_info

from config import (
    CLEANUP_DATASET_AFTER,
    PayloadError,
    TrainingResult,
    validate_payload,
)
from dataset import (
    count_dataset_media,
    download_dataset,
    extract_zip,
    find_orphan_captions,
    validate_dataset,
)
from model_downloader import ensure_aria2c, ensure_model
from yaml_generator import generate_config
from trainer import run_training
from uploader import maybe_upload_outputs, recover_outputs

CIVITAI_DOWNLOADER_DIR = Path("/app/CivitAI_Downloader")
CIVITAI_DOWNLOADER_REPO = "https://github.com/Hearmeman24/CivitAI_Downloader.git"
HERVORA_CONTRACT_VERSION = "hervora-validation-v2"
RECOVERY_SOURCE_JOB_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}-u[0-9]+$",
    re.IGNORECASE,
)


def _create_workspace(job):
    """Create the job workspace directories."""
    for d in [job.job_dir, job.dataset_dir, job.configs_dir, job.output_dir, job.logs_dir]:
        d.mkdir(parents=True, exist_ok=True)


def _cleanup_dataset(job):
    """Remove dataset files to free space after training."""
    import shutil
    try:
        zip_path = job.job_dir / "dataset.zip"
        if zip_path.exists():
            zip_path.unlink()
        if job.dataset_dir.exists():
            shutil.rmtree(job.dataset_dir)
        logger.info("Cleaned up dataset files")
    except Exception as e:
        logger.warning(f"Dataset cleanup failed: {e}")


def _ensure_civitai_deps():
    """Install aria2c and requests if not already available."""
    ensure_aria2c()

    try:
        import requests  # noqa: F401
    except ImportError:
        logger.info("Installing requests...")
        result = subprocess.run(
            ["pip", "install", "-q", "requests"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install requests: {result.stderr}")
        logger.info("requests installed")


def _maybe_download_civitai(job):
    """Download a CivitAI model if civitai_model_id is set. Returns True if downloaded."""
    if not job.civitai_model_id:
        return False

    logger.info(f"CivitAI model requested: {job.civitai_model_id}")

    _ensure_civitai_deps()

    # Clone the downloader tool if not present
    if not CIVITAI_DOWNLOADER_DIR.exists():
        logger.info("Cloning CivitAI_Downloader...")
        clone_result = subprocess.run(
            ["git", "clone", CIVITAI_DOWNLOADER_REPO, str(CIVITAI_DOWNLOADER_DIR)],
            capture_output=True,
            text=True,
        )
        if clone_result.returncode != 0:
            raise RuntimeError(
                f"Failed to clone CivitAI_Downloader: {clone_result.stderr or clone_result.stdout}"
            )

    # Download to a per-job directory (cleaned up with the job)
    download_dir = job.job_dir / "civitai_model"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading CivitAI model {job.civitai_model_id} to {download_dir}")
    result = subprocess.run(
        [
            "python", str(CIVITAI_DOWNLOADER_DIR / "download_with_aria.py"),
            "-m", job.civitai_model_id,
            "-o", str(download_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"CivitAI download failed (exit {result.returncode})")
        if result.stdout:
            logger.error(f"stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"stderr: {result.stderr}")
        raise RuntimeError(
            f"CivitAI download failed for model {job.civitai_model_id}: "
            f"{result.stderr or result.stdout or 'unknown error'}"
        )

    # Find the .safetensors file
    safetensors_files = list(download_dir.glob("**/*.safetensors"))
    if not safetensors_files:
        raise RuntimeError(
            f"CivitAI download for model {job.civitai_model_id} produced no .safetensors file"
        )

    checkpoint_path = str(safetensors_files[0])
    job.civitai_checkpoint_path = checkpoint_path
    logger.info(f"CivitAI checkpoint ready: {checkpoint_path}")
    return True


def handler(event):
    """Main RunPod handler."""
    return _handler_inner(event)


def _handler_inner(event):
    timing = {}
    t_start = time.time()
    raw_input = event.get("input", {})
    if raw_input.get("contract_probe") == HERVORA_CONTRACT_VERSION:
        logger.info(f"Hervora contract probe: {HERVORA_CONTRACT_VERSION}")
        return {
            "ok": True,
            "contract_version": HERVORA_CONTRACT_VERSION,
            "supports_validate_only": True,
        }
    raw_input["job_id"] = event.get("id", "local")
    job = None

    try:
        recovery_source_job_id = raw_input.get("recover_outputs_from_job_id")
        if recovery_source_job_id is not None:
            output_prefix = raw_input.get("output_prefix")
            if (
                not isinstance(recovery_source_job_id, str)
                or not RECOVERY_SOURCE_JOB_ID_PATTERN.fullmatch(recovery_source_job_id)
                or not isinstance(output_prefix, str)
                or not output_prefix.endswith("/")
                or output_prefix.startswith("/")
                or ".." in output_prefix.split("/")
                or "\\" in output_prefix
            ):
                raise PayloadError("Invalid output recovery request")
            logger.info(f"Recovering staged outputs for {recovery_source_job_id}")
            upload_result = recover_outputs(recovery_source_job_id, output_prefix)
            verified_files = [
                item for item in upload_result.get("output_files", [])
                if item.get("s3_key")
            ]
            if not verified_files:
                raise RuntimeError("No verified checkpoints were recovered")
            timing["total_s"] = round(time.time() - t_start, 1)
            return {
                "ok": True,
                "recovery_only": True,
                "output_files": verified_files,
                "presigned_urls": upload_result.get("presigned_urls", []),
                "timing": timing,
            }

        # Never log dataset_zip_url: presigned query parameters are credentials.
        gpu_info = get_gpu_info()
        logger.info(f"GPUs: {gpu_info.count}x {gpu_info.name} ({gpu_info.vram_gb}GB each)")

        # --- Validate ---
        job = validate_payload(raw_input)
        logger.info(f"Job {job.job_id}: model={job.model_type}, trigger={job.trigger_word}")
        if job.config_overrides:
            logger.info(f"Config overrides: {job.config_overrides}")

        # --- Smoke test: validate + resolve only, no download/train ---
        if job.smoke:
            logger.info("Smoke test: payload valid, model resolved — skipping training.")
            timing["total_s"] = round(time.time() - t_start, 1)
            return {
                "ok": True,
                "smoke": True,
                "model_type": job.model_type,
                "trigger_word": job.trigger_word,
                "model": job.model_spec.name_or_path,
                "timing": timing,
            }

        # --- Workspace ---
        _create_workspace(job)

        # --- Dataset ---
        t0 = time.time()
        zip_path = job.job_dir / "dataset.zip"
        download_dataset(job.dataset_zip_url, zip_path)
        extract_zip(zip_path, job.dataset_dir)
        unmatched = validate_dataset(job.dataset_dir)
        if unmatched:
            logger.warning(f"Unmatched files (no captions): {unmatched}")
        timing["dataset_download_s"] = round(time.time() - t0, 1)

        # Count media files for the config generator (step/save_every math).
        img_count = count_dataset_media(job.dataset_dir)
        logger.info(f"Dataset media count: {img_count}")

        # --- Dataset validation only: no model download, config or training ---
        if job.validate_only:
            orphan_captions = find_orphan_captions(job.dataset_dir)
            timing["total_s"] = round(time.time() - t_start, 1)
            result = {
                "ok": len(unmatched) == 0 and len(orphan_captions) == 0,
                "validate_only": True,
                "contract_version": HERVORA_CONTRACT_VERSION,
                "image_count": img_count,
                "unmatched_captions": len(unmatched),
                "orphan_captions": len(orphan_captions),
                "validator_version": "hervora-dataset-v1",
                "timing": timing,
            }
            _cleanup_dataset(job)
            return result

        # --- Model ---
        t0 = time.time()
        if not _maybe_download_civitai(job):
            ensure_model(job)
        timing["model_download_s"] = round(time.time() - t0, 1)

        # --- Config ---
        generate_config(job, img_count)

        # --- Train ---
        t0 = time.time()
        logger.info(f"Training started: {job.model_type}")
        result = run_training(job)
        timing["training_s"] = round(time.time() - t0, 1)

        if not result.ok:
            timing["total_s"] = round(time.time() - t_start, 1)
            logger.error(f"Training failed: {result.error}")
            return {
                "ok": False,
                "error": result.error,
                "error_type": result.error_type or "TRAINING",
                "timing": timing,
            }

        # --- Upload ---
        t0 = time.time()
        upload_result = maybe_upload_outputs(job)
        timing["upload_s"] = round(time.time() - t0, 1)
        timing["total_s"] = round(time.time() - t_start, 1)
        logger.info(
            f"Job complete: {len(upload_result.get('output_files', []))} files, "
            f"total={timing['total_s']}s"
        )

        # --- Cleanup ---
        if CLEANUP_DATASET_AFTER:
            _cleanup_dataset(job)

        return {
            "ok": True,
            "model_type": job.model_type,
            "trigger_word": job.trigger_word,
            "output_files": upload_result.get("output_files", []),
            "presigned_urls": upload_result.get("presigned_urls", []),
            "timing": timing,
        }

    except PayloadError as e:
        logger.error(f"Payload validation error: {e}")
        return {"ok": False, "error": str(e), "error_type": "VALIDATION"}
    except Exception as e:
        if job is not None and job.validate_only:
            _cleanup_dataset(job)
        logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")
        timing["total_s"] = round(time.time() - t_start, 1)
        return {
            "ok": False,
            "error": str(e),
            "error_type": "UNKNOWN",
            "timing": timing,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
