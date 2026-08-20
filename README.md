# HMAI LoRA Trainer

Train a LoRA on RunPod serverless. Send a zip of images + captions, get back a LoRA file.
Works for SDXL, Wan 2.2, Qwen-Image, FLUX.2 Klein, Krea 2, Z-Image, and Ideogram 4.

---

## 🤖 Easiest way: let your coding agent do it

This repo ships an agent skill ([`skills/hmai-lora-trainer/SKILL.md`](skills/hmai-lora-trainer/SKILL.md))
that knows the whole workflow — deployment, storage setup, dataset prep, job submission,
monitoring, and download. Start training with your favorite coding agent (Claude Code,
Cursor, Codex, ...) by pasting this prompt:

```
Fetch and follow this skill:
https://raw.githubusercontent.com/Hearmeman24/HMAI-LoRATrainer-Hub/main/skills/hmai-lora-trainer/SKILL.md

I want to train a LoRA. Walk me through it end-to-end: deploying the HMAI LoRA Trainer
serverless endpoint on RunPod (including S3/R2 storage so my files actually get saved),
preparing my dataset zip, picking a model, submitting the training job, monitoring it,
and downloading the finished .safetensors. Ask me for whatever you need — RunPod API key,
model choice, dataset location, storage credentials.
```

The agent reads the skill and guides you through every step below.

---

## Step 1 — Deploy the endpoint

On the RunPod Hub, click **Deploy**. Pick a GPU (48 GB or larger — A100 / H100 recommended).
That's it. No build, no setup.

If you want your trained LoRAs uploaded to your own storage, open **Environment Variables**
and fill in either your **S3** keys or your **Cloudflare R2** keys (all optional — see the
bottom of this page). Without storage, the job still runs and returns the file paths.

## Step 2 — Prepare your dataset

Make a **zip** file. Inside it, put your training images and a matching `.txt` caption next
to each image (same filename):

```
my_dataset.zip
├── 01.png
├── 01.txt
├── 02.png
├── 02.txt
└── ...
```

Upload the zip somewhere the endpoint can download it (an S3/R2 presigned URL, a public link,
etc.) and copy that **download URL**.

## Step 3 — Start a training job

Send this to your endpoint (replace the URL, the endpoint id, and your API key):

```bash
curl -X POST https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/run \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "model_type": "sdxl",
          "dataset_zip_url": "https://your-storage/my_dataset.zip",
          "trigger_word": "mystyle"
        }
      }'
```

You get back a **job id**.

## Step 4 — Check on it

```bash
curl https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/status/<JOB_ID> \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>"
```

When `status` is `COMPLETED`, the `output` contains your LoRA file name(s) and a download
URL for each.

## Step 5 — Get your LoRA

Download the `.safetensors` from the URL in the output. Done.

---

## Quick health check (no training)

Add `"smoke": true` to test that the endpoint is alive — it validates your request and
returns instantly without downloading or training:

```bash
-d '{"input": {"smoke": true, "model_type": "sdxl", "dataset_zip_url": "x", "trigger_word": "x"}}'
```

Returns `{"ok": true, "smoke": true, ...}`.

---

## The request fields

| Field | Required | What it is |
|---|---|---|
| `model_type` | yes | One of the models below. |
| `dataset_zip_url` | yes | Download URL of your dataset zip. |
| `trigger_word` | yes | The word that activates your LoRA in prompts. |
| `config_overrides` | no | Tweak training, e.g. `{"epochs": 50, "save_every_n_epochs": 5}`. |
| `noise_variant` | **wan2.2 only** | `"high"` or `"low"` — required for Wan 2.2, one per job. |
| `civitai_model_id` | sdxl only | Train on a CivitAI base checkpoint instead of stock SDXL. |
| `smoke` | no | `true` = health check only, no training. |

## Models (`model_type`)

| `model_type` | Model |
|---|---|
| `sdxl` | Stable Diffusion XL |
| `wan2.2` | Wan 2.2 T2V A14B (needs `noise_variant`) |
| `qwen_image` | Qwen-Image |
| `qwen_image_2512` | Qwen-Image 2512 |
| `z_image` | Z-Image Turbo |
| `ideogram4` | Ideogram 4 |
| `flux_klein_9b` | FLUX.2 Klein 9B |
| `krea2` | Krea 2 |
| `krea2_turbo` | Krea 2 Turbo |

Krea 2, Krea 2 Turbo, FLUX.2 Klein 9B, and Ideogram 4 need `HF_TOKEN` and an
accepted license. See [Gated models](#gated-models-accepting-the-license).

## Optional: where your LoRAs get uploaded

Set these as endpoint environment variables. Use **S3** or **R2** (R2 wins if both are set).
Leave them blank to skip upload (the job returns local file paths instead).

**S3:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_REGION`
**Cloudflare R2:** `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
**Gated models:** `HF_TOKEN` (needed for Krea 2, Krea 2 Turbo, FLUX.2 Klein 9B, and Ideogram 4)

## Gated models: accepting the license

Four models download from gated Hugging Face repos. Before your first job on one
of them:

1. Sign in to Hugging Face, open the model page below, and click
   **Agree and access repository**. These repos auto-approve, so you get access
   immediately.
2. Create a token at <https://huggingface.co/settings/tokens> with the **Read** role.
3. Set that token on your endpoint as the `HF_TOKEN` environment variable.

| `model_type` | Accept the license at |
|---|---|
| `krea2` | <https://huggingface.co/krea/Krea-2-Raw> |
| `krea2_turbo` | <https://huggingface.co/krea/Krea-2-Turbo> |
| `flux_klein_9b` | <https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B> |
| `ideogram4` | <https://huggingface.co/ideogram-ai/ideogram-4-fp8> |

Accept the license with the same account that issued the token. `krea2` and
`krea2_turbo` are separate repos, so accepting one does not cover the other.

Without this, the job fails during model download with
`GatedRepoError: 401 Client Error`.

## Deployment note

Publishing a new worker commit triggers a fresh RunPod build for the connected
`main` branch. This note does not change training behavior.
