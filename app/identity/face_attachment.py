from __future__ import annotations

import numpy as np
from PIL import Image

from .. import face as face_mod
from ..identity import embedding_gallery as gallery_mod
from ..core.config import settings


def attach_faces(
    frames: list[Frame], tracks: dict[int, dict], identities: dict[int, dict],
    session_id: str,
) -> None:
    """对每条 track 的最佳帧跑一次人脸检测，关联到该 track；提 512 维人脸指纹查/建**人脸库**。

    人脸库(face gallery)和人形/步态同套路：清晰正脸入库且高置信命中 → 直接定身份；糊脸/侧脸
    质量不过关则**不入库**(避免污染)但仍记录存在、降权(身份退人形/步态)。这就是"越清晰越能拍板"。
    """
    face_sess = f"{session_id}-face"
    gallery_mod.reset_gallery(face_sess)
    by_frame: dict[int, list[int]] = {}
    for tid, t in tracks.items():
        if identities.get(tid, {}).get("skipped"):
            continue  # 门控掉的 track（太短/太低质）不跑人脸
        by_frame.setdefault(t["best_idx"], []).append(tid)
    for fidx, tids in sorted(by_frame.items()):
        try:
            pil = Image.open(frames[fidx].local_path).convert("RGB")
            faces = face_mod.detect(pil)
        except Exception as exc:
            for tid in tids:
                identities[tid]["face"] = {
                    "quality": "unavailable",
                    "matched": False,
                    "face_subject_id": None,
                    "match_score": None,
                    "face_error": f"{type(exc).__name__}: {exc}",
                }
            continue
        person_dets = [{"box": tracks[tid]["best_box"], "track_id": tid, "label": "person"} for tid in tids]
        assoc = face_mod.associate_to_persons(faces, person_dets)
        for tid, fc in assoc.items():
            q = fc.get("quality", {}) or {}
            category = q.get("category", "poor")
            can_enroll = bool(q.get("can_enroll", category == "clear"))
            can_match = bool(q.get("can_match", True))
            rec = {
                "score": q.get("det_score"),
                "quality": category,
                "quality_score": q.get("quality"),  # 连续质量分(0~1)→ 融合软性加权用
                "fiqa_score": q.get("fiqa"),
                "defects": q.get("defects") or [],
                "can_enroll": can_enroll,
                "can_match": can_match,
                "matched": False,
                "face_subject_id": None,
                "match_score": None,
            }
            emb = fc.get("embedding")
            if emb is not None and can_match:
                try:
                    fvec = np.asarray(emb, dtype=np.float32).reshape(-1)
                    # 清晰脸才允许建档入库（auto_enroll）；糊脸只查不建（不污染人脸库）
                    fres = gallery_mod.with_gallery_locked(
                        face_sess, face_mod.FACE_DIM,
                        lambda g: g.identify_or_enroll(
                            fvec, None, auto_enroll=can_enroll,
                            hit_thresh=settings.face_hit_thresh,
                            new_thresh=settings.face_new_thresh,
                        ),
                    )
                    rec["face_subject_id"] = fres.get("subject_id")
                    rec["match_score"] = fres.get("score")
                    rec["matched"] = fres.get("decision") == "hit"
                except Exception as exc:  # 人脸库比对失败不致命
                    rec["face_error"] = str(exc)
            identities[tid]["face"] = rec

__all__ = ["attach_faces"]
