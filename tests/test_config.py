"""Tests for config.py — payload validation, noise_variant gating, civitai gating,
and model registry shape (7 in-scope models, exact arches).
"""

import pytest

from config import (
    MODEL_REGISTRY,
    SUPPORTED_MODELS,
    PayloadError,
    TrainingJob,
    validate_payload,
)


# ---------------------------------------------------------------------------
# Basic payload validation
# ---------------------------------------------------------------------------

class TestValidatePayload:
    def test_valid_sdxl(self, valid_payload_sdxl):
        job = validate_payload(valid_payload_sdxl)
        assert isinstance(job, TrainingJob)
        assert job.model_type == "sdxl"
        assert job.trigger_word == "testword01"

    def test_valid_qwen(self, valid_payload_qwen):
        job = validate_payload(valid_payload_qwen)
        assert job.model_type == "qwen_image"

    def test_valid_qwen_2512(self, valid_payload_qwen_2512):
        job = validate_payload(valid_payload_qwen_2512)
        assert job.model_type == "qwen_image_2512"

    def test_valid_z_image(self, valid_payload_z_image):
        job = validate_payload(valid_payload_z_image)
        assert job.model_type == "z_image"

    def test_missing_model_type(self):
        with pytest.raises(PayloadError, match="model_type"):
            validate_payload({"dataset_zip_url": "http://x.com/d.zip", "trigger_word": "tw"})

    def test_missing_dataset_zip_url(self):
        with pytest.raises(PayloadError, match="dataset_zip_url"):
            validate_payload({"model_type": "sdxl", "trigger_word": "tw"})

    def test_missing_trigger_word(self):
        with pytest.raises(PayloadError, match="trigger_word"):
            validate_payload({"model_type": "sdxl", "dataset_zip_url": "http://x.com/d.zip"})

    def test_unknown_model_type(self):
        with pytest.raises(PayloadError):
            validate_payload({
                "model_type": "flux",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
            })

    def test_empty_trigger_word(self):
        with pytest.raises(PayloadError, match="non-empty"):
            validate_payload({
                "model_type": "sdxl",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "   ",
            })

    def test_config_overrides_invalid_type(self):
        with pytest.raises(PayloadError, match="invalid type"):
            validate_payload({
                "model_type": "sdxl",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "config_overrides": {"epochs": {"nested": "bad"}},
            })

    def test_config_overrides_valid_dot_notation(self):
        job = validate_payload({
            "model_type": "sdxl",
            "dataset_zip_url": "http://x.com/d.zip",
            "trigger_word": "tw",
            "config_overrides": {"adapter.rank": 64, "optimizer.lr": 1e-4},
        })
        assert job.config_overrides["adapter.rank"] == 64
        assert job.config_overrides["optimizer.lr"] == 1e-4

    def test_payload_not_dict(self):
        with pytest.raises(PayloadError, match="JSON object"):
            validate_payload("not a dict")

    def test_smoke_defaults_false(self, valid_payload_sdxl):
        job = validate_payload(valid_payload_sdxl)
        assert job.smoke is False

    def test_smoke_true_parses(self, valid_payload_sdxl):
        job = validate_payload({**valid_payload_sdxl, "smoke": True})
        assert job.smoke is True

    def test_smoke_non_bool_raises(self, valid_payload_sdxl):
        with pytest.raises(PayloadError, match="smoke"):
            validate_payload({**valid_payload_sdxl, "smoke": "yes"})

    def test_validate_only_is_boolean_and_exclusive_with_smoke(self, valid_payload_sdxl):
        job = validate_payload({**valid_payload_sdxl, "validate_only": True})
        assert job.validate_only is True
        with pytest.raises(PayloadError, match="validate_only"):
            validate_payload({**valid_payload_sdxl, "validate_only": "true"})
        with pytest.raises(PayloadError, match="cannot both be true"):
            validate_payload({**valid_payload_sdxl, "smoke": True, "validate_only": True})

    def test_output_prefix_must_be_safe(self, valid_payload_sdxl):
        job = validate_payload({
            **valid_payload_sdxl,
            "output_prefix": "private/lora-training/user/job/outputs/",
        })
        assert job.output_prefix == "private/lora-training/user/job/outputs/"
        with pytest.raises(PayloadError):
            validate_payload({**valid_payload_sdxl, "output_prefix": "../private/"})


# ---------------------------------------------------------------------------
# noise_variant gating (wan2.2 required+enum; non-wan rejected)
# ---------------------------------------------------------------------------

class TestNoiseVariantGating:
    def test_wan22_high_accepted(self):
        job = validate_payload({
            "model_type": "wan2.2",
            "dataset_zip_url": "http://x.com/d.zip",
            "trigger_word": "tw",
            "noise_variant": "high",
        })
        assert job.noise_variant == "high"

    def test_wan22_low_accepted(self):
        job = validate_payload({
            "model_type": "wan2.2",
            "dataset_zip_url": "http://x.com/d.zip",
            "trigger_word": "tw",
            "noise_variant": "low",
        })
        assert job.noise_variant == "low"

    def test_wan22_missing_noise_variant_raises(self):
        with pytest.raises(PayloadError, match="noise_variant"):
            validate_payload({
                "model_type": "wan2.2",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                # no noise_variant
            })

    def test_wan22_invalid_noise_variant_raises(self):
        with pytest.raises(PayloadError, match="noise_variant"):
            validate_payload({
                "model_type": "wan2.2",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "noise_variant": "medium",
            })

    def test_non_wan_noise_variant_rejected_sdxl(self):
        with pytest.raises(PayloadError, match="only supported for model_type 'wan2.2'"):
            validate_payload({
                "model_type": "sdxl",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "noise_variant": "high",
            })

    def test_non_wan_noise_variant_rejected_qwen(self):
        with pytest.raises(PayloadError, match="only supported for model_type 'wan2.2'"):
            validate_payload({
                "model_type": "qwen_image",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "noise_variant": "high",
            })

    def test_non_wan_noise_variant_rejected_z_image(self):
        with pytest.raises(PayloadError, match="only supported for model_type 'wan2.2'"):
            validate_payload({
                "model_type": "z_image",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "noise_variant": "low",
            })

    def test_non_wan_noise_variant_rejected_ideogram4(self):
        with pytest.raises(PayloadError, match="only supported for model_type 'wan2.2'"):
            validate_payload({
                "model_type": "ideogram4",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "noise_variant": "high",
            })

    def test_non_wan_noise_variant_rejected_flux_klein_9b(self):
        with pytest.raises(PayloadError, match="only supported for model_type 'wan2.2'"):
            validate_payload({
                "model_type": "flux_klein_9b",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "noise_variant": "low",
            })

    def test_non_wan_no_noise_variant_ok(self, valid_payload_sdxl):
        """non-wan with no noise_variant passes validation cleanly."""
        job = validate_payload(valid_payload_sdxl)
        assert job.noise_variant is None


# ---------------------------------------------------------------------------
# CivitAI gating (sdxl only)
# ---------------------------------------------------------------------------

class TestCivitaiModelId:
    def test_civitai_accepted_for_sdxl(self):
        job = validate_payload({
            "model_type": "sdxl",
            "dataset_zip_url": "http://x.com/d.zip",
            "trigger_word": "tw",
            "civitai_model_id": "12345",
        })
        assert job.civitai_model_id == "12345"

    def test_civitai_not_set_by_default(self, valid_payload_sdxl):
        job = validate_payload(valid_payload_sdxl)
        assert job.civitai_model_id is None

    def test_civitai_rejected_for_wan22(self):
        with pytest.raises(PayloadError, match="only supported for model_type 'sdxl'"):
            validate_payload({
                "model_type": "wan2.2",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "civitai_model_id": "12345",
                "noise_variant": "high",
            })

    def test_civitai_rejected_for_qwen(self):
        with pytest.raises(PayloadError, match="only supported for model_type 'sdxl'"):
            validate_payload({
                "model_type": "qwen_image",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "civitai_model_id": "12345",
            })

    def test_civitai_rejected_for_z_image(self):
        with pytest.raises(PayloadError, match="only supported for model_type 'sdxl'"):
            validate_payload({
                "model_type": "z_image",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "civitai_model_id": "12345",
            })

    def test_civitai_must_be_non_empty_string(self):
        with pytest.raises(PayloadError, match="non-empty string"):
            validate_payload({
                "model_type": "sdxl",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "civitai_model_id": "   ",
            })

    def test_civitai_must_be_string(self):
        with pytest.raises(PayloadError, match="non-empty string"):
            validate_payload({
                "model_type": "sdxl",
                "dataset_zip_url": "http://x.com/d.zip",
                "trigger_word": "tw",
                "civitai_model_id": 12345,
            })

    def test_civitai_strips_whitespace(self):
        job = validate_payload({
            "model_type": "sdxl",
            "dataset_zip_url": "http://x.com/d.zip",
            "trigger_word": "tw",
            "civitai_model_id": "  12345  ",
        })
        assert job.civitai_model_id == "12345"


# ---------------------------------------------------------------------------
# TrainingJob property smoke tests
# ---------------------------------------------------------------------------

class TestTrainingJob:
    def test_is_wan22_true(self):
        job = validate_payload({
            "model_type": "wan2.2",
            "dataset_zip_url": "http://x.com/d.zip",
            "trigger_word": "tw",
            "noise_variant": "high",
        })
        assert job.is_wan22 is True

    def test_is_wan22_false(self, valid_payload_sdxl):
        job = validate_payload(valid_payload_sdxl)
        assert job.is_wan22 is False

    def test_job_dir_path(self, valid_payload_sdxl):
        job = validate_payload(valid_payload_sdxl)
        assert "jobs" in str(job.job_dir)

    def test_dataset_dir_path(self, valid_payload_sdxl):
        job = validate_payload(valid_payload_sdxl)
        assert str(job.dataset_dir).endswith("dataset")

    def test_configs_dir_path(self, valid_payload_sdxl):
        job = validate_payload(valid_payload_sdxl)
        assert str(job.configs_dir).endswith("configs")

    def test_model_spec_from_registry(self):
        job = validate_payload({
            "model_type": "wan2.2",
            "dataset_zip_url": "http://x.com/d.zip",
            "trigger_word": "tw",
            "noise_variant": "high",
        })
        spec = job.model_spec
        assert spec.model_type == "wan2.2"
        assert spec.dual_noise is True


# ---------------------------------------------------------------------------
# Model registry — EXACTLY the 7 in-scope models; ltx_2.3 + z_image_fft absent
# ---------------------------------------------------------------------------

EXPECTED_MODELS = {
    "wan2.2",
    "sdxl",
    "qwen_image",
    "qwen_image_2512",
    "z_image",
    "ideogram4",
    "flux_klein_9b",
    "krea2",
    "krea2_turbo",
}


class TestModelRegistry:
    def test_registry_has_exactly_9_models(self):
        assert set(MODEL_REGISTRY.keys()) == EXPECTED_MODELS, (
            f"Registry has unexpected models. "
            f"Extra: {set(MODEL_REGISTRY.keys()) - EXPECTED_MODELS}. "
            f"Missing: {EXPECTED_MODELS - set(MODEL_REGISTRY.keys())}"
        )

    def test_flux_klein_9b_present(self):
        assert "flux_klein_9b" in MODEL_REGISTRY

    def test_krea2_variants(self):
        # raw + turbo share arch "krea2"; only turbo carries the training adapter.
        assert MODEL_REGISTRY["krea2"].arch == "krea2"
        assert MODEL_REGISTRY["krea2_turbo"].arch == "krea2"
        assert MODEL_REGISTRY["krea2"].assistant_lora_path is None
        assert "krea2_turbo_training_adapter" in MODEL_REGISTRY["krea2_turbo"].assistant_lora_path
        assert MODEL_REGISTRY["krea2"].name_or_path == "krea/Krea-2-Raw"
        assert MODEL_REGISTRY["krea2_turbo"].name_or_path == "krea/Krea-2-Turbo"

    def test_ltx_23_absent(self):
        assert "ltx_2.3" not in MODEL_REGISTRY

    def test_z_image_fft_absent(self):
        assert "z_image_fft" not in MODEL_REGISTRY

    def test_wan22_dual_noise(self):
        assert MODEL_REGISTRY["wan2.2"].dual_noise is True

    def test_others_not_dual_noise(self):
        for key in EXPECTED_MODELS - {"wan2.2"}:
            assert MODEL_REGISTRY[key].dual_noise is False, (
                f"{key} should not have dual_noise=True"
            )

    def test_supported_models_sorted(self):
        assert SUPPORTED_MODELS == sorted(SUPPORTED_MODELS)

    # Exact arch strings (byte-exact from SOURCE_FINDINGS §4)
    def test_wan22_arch(self):
        assert MODEL_REGISTRY["wan2.2"].arch == "wan22_14b"

    def test_sdxl_arch(self):
        assert MODEL_REGISTRY["sdxl"].arch == "sdxl"

    def test_qwen_image_arch(self):
        assert MODEL_REGISTRY["qwen_image"].arch == "qwen_image"

    def test_qwen_image_2512_arch(self):
        assert MODEL_REGISTRY["qwen_image_2512"].arch == "qwen_image"

    def test_z_image_arch(self):
        # no underscore: "zimage" (SOURCE_FINDINGS §4)
        assert MODEL_REGISTRY["z_image"].arch == "zimage"

    def test_ideogram4_arch(self):
        assert MODEL_REGISTRY["ideogram4"].arch == "ideogram4"

    def test_flux_klein_9b_arch(self):
        # NOT "flux2_klein" — must be "flux2_klein_9b" (SOURCE_FINDINGS §4)
        assert MODEL_REGISTRY["flux_klein_9b"].arch == "flux2_klein_9b"
