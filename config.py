"""Core configuration, dataclasses, constants, and payload validation.

AI-Toolkit backend (hard cut from diffusion-pipe). All AI-Toolkit-shaped
values (arch strings, repo ids, qtype, adapter sinks) are byte-exact from
SOURCE_FINDINGS.md §4 (cited to /tmp/aitk_src @ 7a089fd, v0.10.18).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOLUME_ROOT = Path(os.environ.get("VOLUME_ROOT", "/runpod-volume"))
JOBS_DIR = VOLUME_ROOT / "jobs"
MODELS_DIR = VOLUME_ROOT / "models"
HF_CACHE_DIR = VOLUME_ROOT / "hf_cache"

# AI-Toolkit lives here; run.py is invoked with cwd=AI_TOOLKIT_DIR because it
# does sys.path.insert(0, getcwd()) before importing toolkit (SOURCE_FINDINGS §0).
AI_TOOLKIT_DIR = Path(os.environ.get("AI_TOOLKIT_DIR", "/ai-toolkit"))
RUN_SCRIPT = str(AI_TOOLKIT_DIR / "run.py")

# AI-Toolkit config `name` → the LoRA base filename. We use a FIXED name (not the
# trigger) so the prune glob `{name}_*` and our renamed copies never collide
# (SOURCE_FINDINGS §1/§6).
AITK_JOB_NAME = "aitk_lora"

# AI-Toolkit's clean_up_saves prunes to the newest `max_step_saves_to_keep`
# files; there is NO disable sentinel — `-1` deletes all but the OLDEST
# (SOURCE_FINDINGS §6). A large positive value keeps every checkpoint.
MAX_SAVES_KEEP = 100000

# Sibling dir (outside AI-Toolkit's `{name}_*` prune glob) where the watcher
# copy-renames checkpoints before upload, so they are prune-proof.
UPLOADS_SUBDIR = "_uploads"

MAX_TRAINING_HOURS = int(os.environ.get("MAX_TRAINING_HOURS", "12"))
CLEANUP_DATASET_AFTER = os.environ.get("CLEANUP_DATASET_AFTER", "true").lower() == "true"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PayloadError(Exception):
    """Raised when input payload validation fails."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DownloadItem:
    """One ordered download for a model: an HF repo, a single URL, or an HF
    single-file. The downloader resolves these; the generator resolves
    name_or_path from where they land (DESIGN §E)."""
    kind: str                       # "repo" | "url" | "hf_file"
    repo_id: str | None = None      # for kind=="repo" / "hf_file"
    url: str | None = None          # for kind=="url"
    filename: str | None = None     # for kind=="url"/"hf_file": local name
    local_subdir: str | None = None # where it lands under MODELS_DIR


@dataclass
class ModelDefaults:
    """Per-model base knobs the YAML generator reads when an override is absent."""
    rank: int = 32
    lr: float = 1e-4
    optimizer: str = "adamw8bit"
    epochs: int = 100
    save_every_n_epochs: int = 5
    resolution: list[int] = field(default_factory=lambda: [1024])


@dataclass
class ModelSpec:
    """AI-Toolkit-shaped model spec. arch/name_or_path/qtype are byte-exact
    from SOURCE_FINDINGS.md §4."""
    model_type: str
    arch: str                                   # AI-Toolkit model.arch (exact match in get_model_class)
    downloads: list[DownloadItem] = field(default_factory=list)
    name_or_path: str | None = None             # repo id or resolved local path (config-build time)
    quantize: bool = False
    qtype: str | None = None                    # e.g. "uint4|ostris/.../...safetensors" ('|' ARA split)
    quantize_te: bool = False
    qtype_te: str | None = None                 # e.g. "qfloat8"
    low_vram: bool = False
    assistant_lora_path: str | None = None      # z_image turbo training adapter (§4)
    unconditional_lora_path: str | None = None  # ideogram4 unconditional lora (§4)
    timestep_type: str | None = None            # e.g. "weighted" | "linear"
    defaults: ModelDefaults = field(default_factory=ModelDefaults)
    dual_noise: bool = False                    # wan2.2 only — needs noise_variant


@dataclass
class TrainingJob:
    job_id: str
    model_type: str
    dataset_zip_url: str
    trigger_word: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    civitai_model_id: str | None = None
    civitai_checkpoint_path: str | None = None
    # "high" | "low" — REQUIRED iff model_type=="wan2.2", else must be absent.
    noise_variant: str | None = None
    # When True, the handler validates + resolves the model then returns ok:true
    # WITHOUT downloading the dataset/weights or training. Used for cheap health
    # checks (RunPod Hub tests, endpoint smoke tests).
    smoke: bool = False
    # Downloads and validates the dataset, then exits before model download/training.
    validate_only: bool = False
    # Optional private object prefix supplied by the trusted orchestrator.
    output_prefix: str | None = None
    # Trusted callback for sanitized progress events only.
    progress_webhook_url: str | None = None

    @property
    def is_wan22(self) -> bool:
        return self.model_type == "wan2.2"

    @property
    def job_dir(self) -> Path:
        return JOBS_DIR / self.job_id

    @property
    def dataset_dir(self) -> Path:
        return self.job_dir / "dataset"

    @property
    def configs_dir(self) -> Path:
        return self.job_dir / "configs"

    @property
    def output_dir(self) -> Path:
        return self.job_dir / "output"

    @property
    def logs_dir(self) -> Path:
        return self.job_dir / "logs"

    @property
    def model_spec(self) -> ModelSpec:
        return MODEL_REGISTRY[self.model_type]


@dataclass
class TrainingResult:
    ok: bool
    output_files: list[dict[str, str]] = field(default_factory=list)
    presigned_urls: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None


# ---------------------------------------------------------------------------
# Model Registry — 9 models (SOURCE_FINDINGS §4, byte-exact)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelSpec] = {
    # wan2.2 — arch "wan22_14b"; dual-transformer diffusers repo; ARA uint4.
    # single-flag mechanism: generator sets exactly one model_kwargs flag,
    # keeps split_multistage_loras=true → one suffixed file (SOURCE_FINDINGS §3).
    "wan2.2": ModelSpec(
        model_type="wan2.2",
        arch="wan22_14b",
        name_or_path="ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
        downloads=[DownloadItem(kind="repo", repo_id="ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
                                local_subdir="Wan2.2-T2V-A14B-Diffusers-bf16")],
        quantize=True,
        qtype="uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors",
        quantize_te=True,
        qtype_te="qfloat8",
        low_vram=True,
        dual_noise=True,
    ),
    # sdxl — no arch class; arch "sdxl" makes StableDiffusion.is_xl true.
    # name_or_path is the base single-file OR a CivitAI .safetensors (from_single_file
    # auto-dispatches on a file path — SOURCE_FINDINGS §8). Resolved at config-build.
    "sdxl": ModelSpec(
        model_type="sdxl",
        arch="sdxl",
        name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        downloads=[DownloadItem(
            kind="url",
            url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors",
            filename="sd_xl_base_1.0_0.9vae.safetensors",
            local_subdir="sdxl-base-1.0",
        )],
        defaults=ModelDefaults(rank=32, lr=1e-4, optimizer="adamw8bit",
                               epochs=100, save_every_n_epochs=5, resolution=[1024]),
    ),
    # qwen_image — diffusers repo; ARA uint3 (24GB); forced resolution [1328].
    "qwen_image": ModelSpec(
        model_type="qwen_image",
        arch="qwen_image",
        name_or_path="Qwen/Qwen-Image",
        downloads=[DownloadItem(kind="repo", repo_id="Qwen/Qwen-Image",
                                local_subdir="Qwen-Image")],
        quantize=True,
        qtype="uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors",
        quantize_te=True,
        qtype_te="qfloat8",
        low_vram=True,
        defaults=ModelDefaults(resolution=[1328]),
    ),
    # qwen_image_2512 — SAME arch "qwen_image", different repo + 2512 ARA.
    "qwen_image_2512": ModelSpec(
        model_type="qwen_image_2512",
        arch="qwen_image",
        name_or_path="Qwen/Qwen-Image-2512",
        downloads=[DownloadItem(kind="repo", repo_id="Qwen/Qwen-Image-2512",
                                local_subdir="Qwen-Image-2512")],
        quantize=True,
        qtype="uint3|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint3.safetensors",
        quantize_te=True,
        qtype_te="qfloat8",
        low_vram=True,
        defaults=ModelDefaults(resolution=[1328]),
    ),
    # z_image — arch "zimage" (NO underscore); turbo training adapter via
    # assistant_lora_path (NOT extras_name_or_path) — SOURCE_FINDINGS §4.
    "z_image": ModelSpec(
        model_type="z_image",
        arch="zimage",
        name_or_path="Tongyi-MAI/Z-Image-Turbo",
        downloads=[DownloadItem(kind="repo", repo_id="Tongyi-MAI/Z-Image-Turbo",
                                local_subdir="Z-Image-Turbo")],
        quantize=True,
        qtype="qfloat8",
        quantize_te=True,
        qtype_te="qfloat8",
        low_vram=True,
        assistant_lora_path="ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
        timestep_type="weighted",
    ),
    # ideogram4 — arch "ideogram4"; fp8 repo; unconditional lora; linear timesteps.
    "ideogram4": ModelSpec(
        model_type="ideogram4",
        arch="ideogram4",
        name_or_path="ideogram-ai/ideogram-4-fp8",
        downloads=[DownloadItem(kind="repo", repo_id="ideogram-ai/ideogram-4-fp8",
                                local_subdir="ideogram-4-fp8")],
        quantize=True,
        quantize_te=True,
        qtype_te="qfloat8",
        low_vram=True,
        unconditional_lora_path="ostris/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors",
        timestep_type="linear",
    ),
    # flux_klein_9b — arch "flux2_klein_9b" (NOT "flux2_klein"); unified diffusers
    # repo OR local component dir both resolve via name_or_path — SOURCE_FINDINGS §4.
    "flux_klein_9b": ModelSpec(
        model_type="flux_klein_9b",
        arch="flux2_klein_9b",
        name_or_path="black-forest-labs/FLUX.2-klein-base-9B",
        downloads=[DownloadItem(kind="repo", repo_id="black-forest-labs/FLUX.2-klein-base-9B",
                                local_subdir="FLUX.2-klein-base-9B")],
        quantize=True,
        qtype="qfloat8",
        quantize_te=True,
        qtype_te="qfloat8",
        low_vram=True,
        timestep_type="weighted",
    ),
    # krea2 (raw) — arch "krea2"; raw base diffusers repo; qfloat8; linear timesteps
    # (AI-Toolkit UI option 'krea2'). conv disabled. SOURCE_FINDINGS: krea2.py:135.
    "krea2": ModelSpec(
        model_type="krea2",
        arch="krea2",
        name_or_path="krea/Krea-2-Raw",
        downloads=[DownloadItem(kind="repo", repo_id="krea/Krea-2-Raw",
                                local_subdir="Krea-2-Raw")],
        quantize=True,
        qtype="qfloat8",
        quantize_te=True,
        qtype_te="qfloat8",
        low_vram=True,
        timestep_type="linear",
    ),
    # krea2_turbo — same arch "krea2"; turbo repo + turbo training adapter via
    # assistant_lora_path (same mechanism as z_image turbo). AI-Toolkit 'krea2:turbo'.
    "krea2_turbo": ModelSpec(
        model_type="krea2_turbo",
        arch="krea2",
        name_or_path="krea/Krea-2-Turbo",
        downloads=[DownloadItem(kind="repo", repo_id="krea/Krea-2-Turbo",
                                local_subdir="Krea-2-Turbo")],
        quantize=True,
        qtype="qfloat8",
        quantize_te=True,
        qtype_te="qfloat8",
        low_vram=True,
        assistant_lora_path="ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors",
        timestep_type="linear",
    ),
}

SUPPORTED_MODELS = sorted(MODEL_REGISTRY.keys())

NOISE_VARIANTS = ("high", "low")


# ---------------------------------------------------------------------------
# Payload Validation
# ---------------------------------------------------------------------------

def validate_payload(raw: dict[str, Any]) -> TrainingJob:
    """Validate incoming RunPod payload and return a TrainingJob."""
    if not isinstance(raw, dict):
        raise PayloadError("Payload must be a JSON object")

    model_type = raw.get("model_type")
    if not model_type:
        raise PayloadError("Missing required field: model_type")
    if model_type not in MODEL_REGISTRY:
        raise PayloadError(
            f"Unknown model_type '{model_type}'. Supported: {SUPPORTED_MODELS}"
        )

    dataset_zip_url = raw.get("dataset_zip_url")
    if not dataset_zip_url:
        raise PayloadError("Missing required field: dataset_zip_url")
    if not isinstance(dataset_zip_url, str):
        raise PayloadError("dataset_zip_url must be a string")

    trigger_word = raw.get("trigger_word")
    if not trigger_word:
        raise PayloadError("Missing required field: trigger_word")
    if not isinstance(trigger_word, str) or not trigger_word.strip():
        raise PayloadError("trigger_word must be a non-empty string")

    config_overrides = raw.get("config_overrides", {})
    if not isinstance(config_overrides, dict):
        raise PayloadError("config_overrides must be a JSON object")
    for key, value in config_overrides.items():
        if not isinstance(key, str):
            raise PayloadError(f"config_overrides key must be string, got {type(key).__name__}")
        if not isinstance(value, (int, float, str, bool, list)):
            raise PayloadError(
                f"config_overrides['{key}'] has invalid type {type(value).__name__}. "
                "Allowed: int, float, str, bool, list"
            )

    civitai_model_id = raw.get("civitai_model_id")
    if civitai_model_id is not None:
        if not isinstance(civitai_model_id, str) or not civitai_model_id.strip():
            raise PayloadError("civitai_model_id must be a non-empty string")
        if model_type != "sdxl":
            raise PayloadError("civitai_model_id is only supported for model_type 'sdxl'")
        civitai_model_id = civitai_model_id.strip()

    # noise_variant gating — mirrors the civitai gating above (DESIGN §B0):
    # required+enum for wan2.2, rejected for everything else.
    noise_variant = raw.get("noise_variant")
    if model_type == "wan2.2":
        if not noise_variant:
            raise PayloadError(
                "noise_variant is required for model_type 'wan2.2' (one of 'high', 'low')"
            )
        if noise_variant not in NOISE_VARIANTS:
            raise PayloadError(
                f"noise_variant must be one of {list(NOISE_VARIANTS)}, got '{noise_variant}'"
            )
    elif noise_variant is not None:
        raise PayloadError("noise_variant is only supported for model_type 'wan2.2'")

    smoke = raw.get("smoke", False)
    if not isinstance(smoke, bool):
        raise PayloadError("smoke must be a boolean")

    validate_only = raw.get("validate_only", False)
    if not isinstance(validate_only, bool):
        raise PayloadError("validate_only must be a boolean")
    if smoke and validate_only:
        raise PayloadError("smoke and validate_only cannot both be true")

    output_prefix = raw.get("output_prefix")
    if output_prefix is not None:
        if not isinstance(output_prefix, str) or not output_prefix.strip():
            raise PayloadError("output_prefix must be a non-empty string")
        output_prefix = output_prefix.strip()
        if (
            output_prefix.startswith("/")
            or not output_prefix.endswith("/")
            or ".." in output_prefix.split("/")
            or "\\" in output_prefix
        ):
            raise PayloadError("output_prefix must be a safe relative prefix ending in '/'")

    progress_webhook_url = raw.get("progress_webhook_url")
    if progress_webhook_url is not None:
        if not isinstance(progress_webhook_url, str) or not progress_webhook_url.startswith("https://"):
            raise PayloadError("progress_webhook_url must be an HTTPS URL")
        progress_webhook_url = progress_webhook_url.strip()

    job_id = raw.get("job_id", os.environ.get("RUNPOD_POD_ID", "local"))
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise PayloadError("job_id contains invalid characters")

    return TrainingJob(
        job_id=job_id,
        model_type=model_type,
        dataset_zip_url=dataset_zip_url,
        trigger_word=trigger_word.strip(),
        config_overrides=config_overrides,
        civitai_model_id=civitai_model_id,
        noise_variant=noise_variant,
        smoke=smoke,
        validate_only=validate_only,
        output_prefix=output_prefix,
        progress_webhook_url=progress_webhook_url,
    )


# ---------------------------------------------------------------------------
# Output Naming  (rename_output / sanitize_trigger_word — reused UNCHANGED)
# ---------------------------------------------------------------------------

_MODEL_BRANCH_MAP = {
    "wan2.2": "wan2.2",
    "sdxl": "sdxl",
    "qwen_image": "qwen-image",
    "qwen_image_2512": "qwen-image-2512",
    "z_image": "z-image",
    "flux_klein_9b": "flux-klein-9b",
    "ideogram4": "ideogram-4",
    "krea2": "krea2",
    "krea2_turbo": "krea2-turbo",
}


def sanitize_trigger_word(trigger_word: str) -> str:
    """Sanitize trigger word for use in filenames."""
    sanitized = trigger_word.strip()
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"[^\w\-.]", "", sanitized)
    return sanitized


def rename_output(
    trigger_word: str,
    model_type: str,
    noise_variant: str | None,
    epoch: int,
) -> str:
    """Generate standardized output filename.

    Format: <trigger_word>_<model_branch>[noise_variant]_epoch<N>.safetensors
    """
    safe_trigger = sanitize_trigger_word(trigger_word)
    branch = _MODEL_BRANCH_MAP.get(model_type, model_type)
    variant_suffix = noise_variant if noise_variant else ""
    return f"{safe_trigger}_{branch}{variant_suffix}_epoch{epoch}.safetensors"
