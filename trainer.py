"""Training subprocess launcher — single GPU, AI-Toolkit backend.

One request → one `python run.py config.yml` subprocess (cwd=AI_TOOLKIT_DIR,
cuda:0) → 1 LoRA (or one suffixed file for wan2.2). No deepspeed, no parallel
branch, no LTX/FFT. tqdm progress is on stderr (SOURCE_FINDINGS §2); stdout and
stderr are merged into one parse loop.

Checkpoint filenames (SOURCE_FINDINGS §1):
  - per-step: aitk_lora_<step:09d>[_high_noise|_low_noise].safetensors
  - bare final (best epoch, no step token): aitk_lora[_high_noise|_low_noise].safetensors
The step-based watcher (F1/F2/F3) copy-renames each into a sibling `_uploads/`
dir OUTSIDE AI-Toolkit's `aitk_lora_*` prune glob, so a renamed copy is never
pruned (SOURCE_FINDINGS §6). Success == exit code 0 AND >=1 produced safetensors.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

import yaml
from loguru import logger

from config import (
    AITK_JOB_NAME,
    AI_TOOLKIT_DIR,
    MAX_TRAINING_HOURS,
    RUN_SCRIPT,
    UPLOADS_SUBDIR,
    TrainingJob,
    TrainingResult,
    rename_output,
)
from log_parser import TrainingProgress, parse_line, report_progress, should_report
from overrides_translator import steps_per_epoch as _steps_per_epoch
from uploader import upload_and_presign
from hervora_progress import HervoraProgressReporter

# Watcher checkpoint patterns (SOURCE_FINDINGS §1). `aitk_lora` == AITK_JOB_NAME.
_STEP_RE = re.compile(
    rf"^{re.escape(AITK_JOB_NAME)}_(\d{{9}})(?:_(high|low)_noise)?\.safetensors$"
)
_BARE_RE = re.compile(
    rf"^{re.escape(AITK_JOB_NAME)}(?:_(high|low)_noise)?\.safetensors$"
)

_POLL_INTERVAL = 5  # seconds (DESIGN §B5)

# N2 write-completion gate: AI-Toolkit's save_file is non-atomic/in-place
# (BaseSDTrainProcess.py), so a poll can grab a half-written multi-GB file. A
# checkpoint is promoted ONLY when (a) its size is stable across two consecutive
# polls AND (b) its safetensors header parses. Otherwise it stays pending and is
# rechecked next poll — never dropped.


def _is_complete_safetensors(path: Path) -> bool:
    """Validate a safetensors file is fully written via its header (N2).

    safetensors layout: 8-byte little-endian header length, then that many bytes
    of JSON header, then the tensor data. A truncated file fails one of: too
    small for the length prefix; header not yet fully flushed; or the declared
    total (8 + header_len + max tensor end-offset) exceeds the on-disk size.
    No torch/safetensors import needed (works under unit mocks).
    """
    try:
        size = path.stat().st_size
        if size < 8:
            return False
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            if header_len <= 0 or 8 + header_len > size:
                return False
            header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            return False
        import json
        header = json.loads(header_bytes)
        # account for the largest tensor end-offset; data starts after the header.
        max_end = 0
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            offs = meta.get("data_offsets") if isinstance(meta, dict) else None
            if offs and len(offs) == 2:
                max_end = max(max_end, int(offs[1]))
        return 8 + header_len + max_end <= size
    except Exception:
        return False


def _build_env() -> dict[str, str]:
    """Environment for the AI-Toolkit subprocess. Single GPU, cuda:0."""
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def _epoch_for_step(step: int, steps_per_epoch: int) -> int:
    """Synthesize the cosmetic epoch from a step token (DESIGN §B3 reverse map)."""
    if steps_per_epoch <= 0:
        return 0
    return max(1, round(step / steps_per_epoch))


def _classify(name: str) -> tuple[str, int | None, str | None] | None:
    """Classify a checkpoint filename.

    Returns (kind, step, variant) where kind is "step" or "bare", step is the
    9-digit step int (None for bare), variant is "high"/"low"/None. Returns None
    if the name matches neither pattern.
    """
    m = _STEP_RE.match(name)
    if m:
        return "step", int(m.group(1)), m.group(2)
    m = _BARE_RE.match(name)
    if m:
        return "bare", None, m.group(1)
    return None


def _target_name(job: TrainingJob, epoch: int) -> str:
    """The renamed-output filename a checkpoint at `epoch` would publish to. Used
    to dedup by TARGET (a step file and the bare final at the same epoch collide)."""
    noise_variant = job.noise_variant if job.is_wan22 else None
    return rename_output(job.trigger_word, job.model_type, noise_variant, epoch)


def _publish(
    src: Path,
    epoch: int,
    variant: str | None,
    job: TrainingJob,
    progress: TrainingProgress,
    uploads_dir: Path,
) -> None:
    """Copy-rename one checkpoint into uploads_dir (prune-proof) and upload.
    noise_variant for the filename is the job's variant (wan)."""
    noise_variant = job.noise_variant if job.is_wan22 else None
    new_name = rename_output(job.trigger_word, job.model_type, noise_variant, epoch)
    renamed = uploads_dir / new_name
    uploads_dir.mkdir(parents=True, exist_ok=True)
    # copy2 (not move) so AI-Toolkit's managed originals are untouched (DESIGN §B5).
    shutil.copy2(src, renamed)

    logger.info(f"New checkpoint: {new_name}, uploading...")

    result = upload_and_presign(renamed, job.job_id, job.output_prefix)
    if result:
        progress.ready_loras.append(result)
    else:
        progress.ready_loras.append({"filename": new_name, "local_path": str(renamed)})


def _is_stable(sf: Path, sizes: dict[str, int]) -> bool:
    """N2 size-stability: True only when this poll's size matches the previous
    poll's recorded size (and a previous size existed). Updates the tracker."""
    try:
        now = sf.stat().st_size
    except OSError:
        return False
    prev = sizes.get(sf.name)
    sizes[sf.name] = now
    return prev is not None and prev == now


def _scan_once(
    save_dir: Path,
    job: TrainingJob,
    progress: TrainingProgress,
    seen: set[str],
    sizes: dict[str, int],
    steps_per_epoch: int,
    total_epochs: int,
    uploads_dir: Path,
) -> None:
    """One pass over save_dir: publish any new, fully-written step/bare checkpoints
    (F1/F2). A file is published ONLY when its size is stable across two polls AND
    its safetensors header parses (N2); otherwise it stays pending for next poll."""
    if not save_dir.exists():
        return
    for sf in sorted(save_dir.glob("*.safetensors")):
        if sf.name in seen:
            continue
        info = _classify(sf.name)
        if info is None:
            continue
        # N2: require stable size AND a complete safetensors header before promoting.
        if not _is_stable(sf, sizes):
            continue  # still being written (or first sighting) — recheck next poll
        if not _is_complete_safetensors(sf):
            continue  # header not fully flushed yet — keep pending
        kind, step, _variant = info
        # bare final → the best/last epoch (= total_epochs); step → synthesized epoch.
        epoch = total_epochs if kind == "bare" else _epoch_for_step(step, steps_per_epoch)
        # Dedup by the renamed TARGET (not just source name): the last step file and
        # the bare final can map to the SAME epoch → same target. Publishing both
        # would fire checkpoint_ready twice + overwrite the upload for one epoch.
        target = _target_name(job, epoch)
        if target in seen or (uploads_dir / target).exists():
            seen.add(sf.name)
            seen.add(target)
            continue
        seen.add(sf.name)
        seen.add(target)
        _publish(sf, epoch, _variant, job, progress, uploads_dir)


def _lora_watcher(
    save_dir: Path,
    job: TrainingJob,
    progress: TrainingProgress,
    stop_event: threading.Event,
    steps_per_epoch: int,
    total_epochs: int,
    uploads_dir: Path,
) -> None:
    """Poll save_dir for AI-Toolkit checkpoints; copy-rename + upload BEFORE
    AI-Toolkit can prune (F3). Two-pattern match (step + bare final), N2-gated."""
    seen: set[str] = set()
    sizes: dict[str, int] = {}
    while not stop_event.is_set():
        stop_event.wait(_POLL_INTERVAL)
        try:
            _scan_once(save_dir, job, progress, seen, sizes, steps_per_epoch,
                       total_epochs, uploads_dir)
        except Exception as e:  # never let the watcher die mid-run
            logger.warning(f"Lora watcher error: {e}")


def _iter_lines(stream):
    """Yield logical lines split on BOTH '\\r' and '\\n' (F5).

    tqdm renders its live bar on stderr terminated by '\\r' with no newline until
    the bar closes, so a plain `for line in stream` (which splits on '\\n' only)
    would buffer the whole bar and progress would stall at 0% all run. We read in
    chunks and split on either delimiter, emitting each carriage-return frame.
    """
    buf = ""
    while True:
        chunk = stream.read(1)
        if chunk == "":
            break
        if chunk in ("\r", "\n"):
            if buf:
                yield buf
                buf = ""
        else:
            buf += chunk
    if buf:
        yield buf


def _stream_output(proc: subprocess.Popen, progress: TrainingProgress,
                   label: str, progress_reporter: HervoraProgressReporter) -> list[str]:
    """Merge stdout+stderr into one parse loop (tqdm is on stderr — §2).

    Both streams are split on '\\r' and '\\n' (F5) so the carriage-return-only
    tqdm bar updates progress live. Returns captured tail lines for errors.
    """
    tail: list[str] = []
    started = False

    # stderr is read in a thread; stdout in the main loop. Both feed parse_line.
    def drain_stderr() -> None:
        nonlocal started
        if not proc.stderr:
            return
        for raw in _iter_lines(proc.stderr):
            line = raw.strip()
            if not line:
                continue
            tail.append(line)
            if len(tail) > 400:
                del tail[:200]
            updated = parse_line(line, progress)
            if updated and should_report(progress):
                started = True
                report_progress(progress, label)
                progress_reporter.send(progress)
                logger.info(f"[{label}] step={progress.step}/{progress.total_steps} "
                            f"epoch={progress.epoch} loss={progress.loss} "
                            f"{progress.percent:.1f}%")

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    if proc.stdout:
        for raw in _iter_lines(proc.stdout):
            line = raw.strip()
            if not line:
                continue
            logger.debug(f"[{label}] {line}")
            if not started:
                logger.info(f"[{label}] {line}")
            updated = parse_line(line, progress)
            if updated and should_report(progress):
                started = True
                report_progress(progress, label)
                progress_reporter.send(progress)

    stderr_thread.join(timeout=10)
    return tail


def _epoch_params(job: TrainingJob) -> tuple[int, int, int]:
    """Derive (steps_per_epoch, total_epochs, total_steps) from the generated
    config.yml + dataset, so step tokens map back to epoch numbers (DESIGN §B3)
    AND progress reporting uses the real job denominator (not a caching bar)."""
    config_path = job.configs_dir / "config.yml"
    try:
        proc = yaml.safe_load(open(config_path))["config"]["process"][0]
        total_steps = int(proc["train"]["steps"])
        batch = int(proc["train"].get("batch_size", 1))
        accum = int(proc["train"].get("gradient_accumulation", 1))
        num_repeats = int(proc["datasets"][0].get("num_repeats", 1))
    except Exception:
        return 1, 1, 0
    img_count = _count_media(job.dataset_dir)
    spe = _steps_per_epoch(img_count, num_repeats, max(1, batch * accum))
    total_epochs = max(1, math.ceil(total_steps / spe)) if spe else 1
    return spe, total_epochs, total_steps


def _count_media(dataset_dir: Path) -> int:
    from config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    if not dataset_dir.exists():
        return 0
    return sum(1 for f in dataset_dir.iterdir()
               if f.is_file() and f.suffix.lower() in exts)


def run_training(job: TrainingJob) -> TrainingResult:
    """Single subprocess: `python run.py <config.yml>`. Streams progress, runs the
    checkpoint watcher, collects renamed outputs from `_uploads/`.

    Success == exit code 0 AND >=1 produced .safetensors. A clean exit with zero
    safetensors is a failure (silent no-output run), not a success (DESIGN §B4).
    """
    logger.info(f"Starting AI-Toolkit training for {job.model_type}")
    config_path = job.configs_dir / "config.yml"
    # AI-Toolkit's save_root = training_folder/<name> (BaseTrainProcess.py:45), i.e.
    # job.output_dir/aitk_lora — that's where it writes + prunes (SOURCE_FINDINGS §1/§6).
    save_dir = job.output_dir / AITK_JOB_NAME
    # Renamed copies live OUTSIDE the prune glob, at the frozen cross-slice location
    # job.output_dir.parent/_uploads (DESIGN §B5) — handler/uploader read here too.
    uploads_dir = job.output_dir.parent / UPLOADS_SUBDIR
    uploads_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch, total_epochs, total_steps = _epoch_params(job)

    # Pre-set total_steps + steps_per_epoch so progress reporting uses the real
    # job denominator and synthesizes the epoch. Without this the parser locks
    # onto the first tqdm bar it sees — a setup/caching bar (e.g. .../28) — and
    # reports a bogus total_steps / percent / epoch=0.
    progress = TrainingProgress(total_steps=total_steps, total_epochs=total_epochs,
                                steps_per_epoch=steps_per_epoch, job_id=job.job_id,
                                label=job.model_type)
    progress_reporter = HervoraProgressReporter(job.progress_webhook_url)

    cmd = ["python", RUN_SCRIPT, str(config_path)]
    logger.info(f"[{job.model_type}] Launching: {' '.join(cmd)} (cwd={AI_TOOLKIT_DIR})")

    watcher_stop = threading.Event()
    watcher = threading.Thread(
        target=_lora_watcher,
        args=(save_dir, job, progress, watcher_stop, steps_per_epoch,
              total_epochs, uploads_dir),
        daemon=True,
    )
    watcher.start()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_build_env(),
        text=True,
        bufsize=1,
        cwd=str(AI_TOOLKIT_DIR),
    )

    tail: list[str] = []
    try:
        tail = _stream_output(proc, progress, job.model_type, progress_reporter)
        proc.wait(timeout=MAX_TRAINING_HOURS * 3600)
    except subprocess.TimeoutExpired:
        logger.warning(f"[{job.model_type}] Exceeded MAX_TRAINING_HOURS, killing")
        proc.kill()
        proc.wait()
        tail.append("Killed after exceeding MAX_TRAINING_HOURS")

    rc = proc.returncode

    # Stop the watcher, then run one final scan to catch anything (incl. the bare
    # final file) saved between the last poll and exit. Dedup is by the renamed
    # target already present in uploads_dir, so re-publishing is avoided.
    watcher_stop.set()
    watcher.join(timeout=30)
    _final_scan(save_dir, job, progress, steps_per_epoch, total_epochs, uploads_dir)

    output_files = _collect_outputs(uploads_dir, job)

    if rc != 0:
        tail_text = "\n".join(tail[-100:]) if tail else "No output captured"
        return TrainingResult(
            ok=False,
            error=f"Training failed (exit code {rc}): {tail_text}",
            error_type="TRAINING",
        )
    if not output_files:
        return TrainingResult(
            ok=False,
            error="Training exited cleanly but produced no .safetensors output",
            error_type="TRAINING",
        )
    return TrainingResult(ok=True, output_files=output_files)


def _final_scan(save_dir: Path, job: TrainingJob, progress: TrainingProgress,
                steps_per_epoch: int, total_epochs: int, uploads_dir: Path) -> None:
    """Final sweep over save_dir using the SAME two-pattern match, deduped by the
    renamed target already present in uploads_dir. N2: a truncated final file is
    NOT published, so D6's '>=1 safetensors == success' can't count it."""
    if not save_dir.exists():
        return
    existing = {p.name for p in uploads_dir.glob("*.safetensors")}
    for sf in sorted(save_dir.glob("*.safetensors")):
        info = _classify(sf.name)
        if info is None:
            continue
        if not _is_complete_safetensors(sf):
            logger.warning(f"final-sweep: {sf.name} has an incomplete safetensors "
                           "header; skipping (not counted as output)")
            continue
        kind, step, variant = info
        epoch = total_epochs if kind == "bare" else _epoch_for_step(step, steps_per_epoch)
        noise_variant = job.noise_variant if job.is_wan22 else None
        target = rename_output(job.trigger_word, job.model_type, noise_variant, epoch)
        if target in existing:
            continue
        _publish(sf, epoch, variant, job, progress, uploads_dir)
        existing.add(target)


def _collect_outputs(uploads_dir: Path, job: TrainingJob) -> list[dict[str, str]]:
    """Build the OUTPUT file list from the renamed copies in uploads_dir."""
    files: list[dict[str, str]] = []
    if not uploads_dir.exists():
        return files
    noise_variant = job.noise_variant if job.is_wan22 else ""
    for sf in sorted(uploads_dir.glob("*.safetensors")):
        files.append({
            "filename": sf.name,
            "local_path": str(sf),
            "noise_variant": noise_variant or "",
        })
    return files
