from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "590"
SCREEN_FOLDS = (0, 3)
SEED = 42
GRAPH_VERSION = "semantic_family_v3"
DRAFT_VERSION = "semantic_family_graph_audit_draft_v3"
ALLOWED_STAGES = frozenset({"exact_full_text", "exact_first_image", "exact_name_corroborated"})
FORBIDDEN_STAGES = frozenset(
    {
        "exact_auxiliary_images",
        "perceptual_corroborated",
        "masked_name_corroborated",
    }
)
MAX_KEY_DEGREE = 32
BLIND_AUDIT_SIZE = 300
MIN_SAME_PRODUCT_PRECISION = 0.98
MIN_KAPPA = 0.80
MIN_ELIGIBLE_PAIRS = 200
MIN_ELIGIBLE_COMPONENTS = 50
RECOMBINATION_FRACTION_OF_REPEATS = 0.25
TRAINING_VIEW_POLICY = "audited_cross_listing_first_image_25pct_repeats_v1"
INFERENCE_VIEW_POLICY = "original_first_image_448_single_pass_v1"
