"""三路身份融合（Phase 4 · A 汇聚）—— 把人脸 / 人形 ReID / 步态合成一个统一身份。

定位：logic flow 里"A 汇聚 → 稳定 subject_id"节点的代码兑现。前面三路各自查库认人：
  - 人脸（ArcFace/AdaFace + 人脸 gallery）→ face.matched / face_subject_id / score
  - 人形（OSNet + 主体记忆库）→ subject_id / decision / score（含灰区轨迹缝合）
  - 步态（SkeletonGait++ + 步态 gallery）→ gait.score / subject_id

但它们是**各认各的**；本模块把三路**按质量加权融合**成一个统一身份置信度，并体现：
  - **质量自适应**：清晰正脸权重最高、糊脸降权退人形/步态（攻"人脸模糊"的最后一环）；
  - **多路一致加成**：人脸+人形+步态都命中 → 置信度再加成（互相印证）；
  - **互补兜底**：背身/远处看不到脸时，靠人形/步态撑住身份。

纯聚合、无副作用：只读 per-track 已算好的三路结果，产出 {confidence, sources, agreed, resolved}。
"""
from __future__ import annotations

from ..core.config import settings


def _face_cue(face: dict | None) -> tuple[float, float]:
    """人脸线索 → (强度 0~1, 有效权重)。

    权重用**软性连续加权**（文献最优，SER-FIQ/CR-FIQA 风格）：
        w = w_face × (floor + (1-floor) × 质量分)
    连续质量分平滑降权——中等/微糊脸不再被一刀切压死，poor 也保底不完全归零。
    无连续质量分时回退到旧的清晰/糊两档（向后兼容）。
    """
    if not face:
        return 0.0, 0.0
    qs = face.get("quality_score")
    if qs is not None:
        q = max(0.0, min(1.0, float(qs)))
        floor = settings.identity_face_quality_floor
        w = settings.identity_w_face * (floor + (1.0 - floor) * q)
    else:  # 回退：无连续质量分 → 清晰满权、糊脸折扣
        clear = face.get("quality") == "clear"
        w = settings.identity_w_face * (1.0 if clear else settings.identity_face_blurry_factor)
    # 强度：库命中分优先；否则用检测分粗略代理
    score = face.get("match_score")
    if score is None:
        score = face.get("score")
    s = float(score) if score is not None else 0.0
    if face.get("matched"):
        s = max(s, 0.6)  # 命中人脸库 → 至少中高强度
    return max(0.0, min(1.0, s)), w


def _body_cue(ident: dict) -> tuple[float, float]:
    """人形 ReID 线索 → (强度, 权重)。命中/缝合算有效，分数为 ReID 余弦。"""
    score = ident.get("score")
    s = float(score) if score is not None else 0.0
    decision = ident.get("decision")
    # 缝合(stitched)/命中(hit) 视为有效身份；new(开山主体) 给低强度（它定义了主体但无可比）
    if decision in {"hit", "stitched"}:
        s = max(s, 0.5)
    return max(0.0, min(1.0, s)), settings.identity_w_body


def _gait_cue(gait: dict | None) -> tuple[float, float]:
    """步态线索 → (强度, 权重)。需足够帧；分数为步态库余弦。"""
    if not gait:
        return 0.0, 0.0
    score = gait.get("score")
    s = float(score) if score is not None else 0.0
    return max(0.0, min(1.0, s)), settings.identity_w_gait


def fuse_multimodal_identity(ident: dict) -> dict:
    """对一条 track 的三路识别结果做质量加权融合，返回统一身份判定。

    Args:
        ident: per-track 身份字典，含 subject_id/decision/score(人形) + face{} + gait{}。

    Returns:
        dict：{confidence, resolved, sources:[...], agreed:bool, primary} 写回 ident['fused']。
          - confidence : 0~1 综合身份置信度（质量加权 + 一致加成）。
          - resolved   : confidence ≥ 阈值（可采信为稳定身份）。
          - sources    : 实际参与的线索（face/body/gait）及各自强度。
          - agreed     : 是否 ≥2 路都给出有效身份（互相印证）。
          - primary    : 主导线索（贡献最大的那路）。
    """
    cues = {
        "face": _face_cue(ident.get("face")),
        "body": _body_cue(ident),
        "gait": _gait_cue(ident.get("gait")),
    }
    contrib = {k: s * w for k, (s, w) in cues.items()}
    wsum = sum(w for (_, w) in cues.values() if w > 0)
    active = [k for k, (s, w) in cues.items() if w > 0 and s > 0]

    confidence = (sum(contrib.values()) / wsum) if wsum > 0 else 0.0
    # 多路一致加成：≥2 路给出有效身份 → 互相印证，提升置信
    agreed = len(active) >= 2
    if agreed:
        confidence = min(1.0, confidence + settings.identity_agree_bonus)

    primary = max(contrib, key=contrib.get) if contrib else None
    sources = [
        {"cue": k, "strength": round(cues[k][0], 3), "weight": round(cues[k][1], 3)}
        for k in ("face", "body", "gait") if cues[k][1] > 0 and cues[k][0] > 0
    ]
    fused = {
        "confidence": round(float(confidence), 4),
        "resolved": bool(confidence >= settings.identity_resolve_thresh),
        "agreed": agreed,
        "primary": primary if (primary and contrib.get(primary, 0) > 0) else None,
        "sources": sources,
    }
    ident["fused"] = fused
    return fused


__all__ = ["fuse_multimodal_identity"]
