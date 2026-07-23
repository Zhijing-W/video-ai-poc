"""Compatibility entry point for the schema-v3 check-in super-resolution experiment."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SUPERRES_DIR = SCRIPT_DIR.parent
EXPERIMENT_DIR = SUPERRES_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for import_root in (REPO_ROOT, EXPERIMENT_DIR, SUPERRES_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from checkin_superres.common import (  # noqa: E402,F401
    ARMS,
    CHECKIN_RE,
    IMAGE_SUFFIXES,
    SCHEMA_VERSION,
    annotation_pid_set,
    audit_prefix_coverage,
    build_image_manifest_records,
    canonical_json,
    file_sha256,
    load_checkin_front_images,
    manifest_identity,
    parse_checkin_image,
    sample_evenly_indexed,
    stable_hash,
    _relative,
    _resolve,
    _save_bgr,
)
from checkin_superres.embeddings import (  # noqa: E402,F401
    select_arm_embeddings,
    _compute_embedding_cache,
    _load_embedding_cache,
    _normalise,
    _pack_vectors,
    _unpack_vectors,
    _verify_manifest,
)
from checkin_superres.metrics import (  # noqa: E402,F401
    paired_uncertainty,
    pid_cluster_bootstrap_rate,
    summarize_scores,
    _exact_paired_p,
    _score,
    _templates,
)
from checkin_superres.matrix import (  # noqa: E402,F401
    ALL_ARMS,
    BACKEND_SPECS,
    CONTROL_SPECS,
    FROZEN_MANIFEST_ID,
    MAIN_ARMS,
    NORMALIZATION,
    build_matrix_run_spec,
    derive_backend_arms,
    embed_normalized112,
    evaluate_matrix,
    evaluate_matrix_payload,
    fiqa_normalized112,
    matrix_cache_key,
    normalize112,
    p_off_original,
)
from checkin_superres.orchestration import (  # noqa: E402,F401
    build_parser,
    evaluate,
    main,
    prepare,
)
from checkin_superres.preparation import (  # noqa: E402,F401
    _best_face_in_image,
    _freeze_query,
    _gallery_candidate,
    _model_provenance,
    _product_config,
    _provenance_compatible,
    _query_candidates,
)
from checkin_superres.visualization import (  # noqa: E402,F401
    _comparison,
    _font,
    _panel,
)

if __name__ == "__main__":
    raise SystemExit(main())
