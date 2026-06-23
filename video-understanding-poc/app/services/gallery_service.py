"""主体记忆 / ReID 服务层（Phase 3 · Step 14）：把 reid + gallery 暴露给 routers。

与 tracker_service 同理——routers 只调本层，不直接碰底层。本层负责：
解码整帧 → 按 box 裁出每个目标的 crop → 提 ReID 指纹 + 评质量 → 在该 session 的
向量库里"查库认人 / 开放集登记"，回传每个目标的归属裁决。

注意：本层**不做编排**（什么时候来认人由三时钟 Step 12 决定），只在被调用时完成"认这一批 crop"。
"""
from __future__ import annotations

from .. import gallery as _gallery
from .. import reid as _reid
from ..detector import _decode_image
from ..utils.image_utils import crop_box_region


def gallery_backend_info() -> dict:
    """当前 ReID 后端与维度（便于前端/排查显示走的是哪档指纹）。"""
    return {"backend": _reid.active_backend(), "dim": _reid.embed_dim()}


def identify_detections(
    image: str,
    detections: list[dict],
    session_id: str = "default",
    auto_enroll: bool = True,
) -> dict:
    """对一帧里的若干目标框做"查库认人 / 开放集登记"。

    Args:
        image: data URI / base64 / 字节（整帧）。
        detections: [{box:[x1,y1,x2,y2], track_id?, label?, attributes?}]。
        session_id: 记忆库会话隔离标识（换视频用不同 id）。
        auto_enroll: 未命中的新主体是否自动建档登记。

    Returns:
        dict：backend, dim, session_id, results:[每个目标的裁决], gallery_size。
    """
    img = _decode_image(image)
    dim = _reid.embed_dim()
    results: list[dict] = []

    def _run(g):
        out = []
        for det in detections:
            box = det.get("box")
            track_id = det.get("track_id")
            if not box:
                out.append({"track_id": track_id, "box": box, "decision": "invalid",
                            "reason": "missing_box"})
                continue
            crop = crop_box_region(img, box, "whole")
            if crop is None:
                out.append({"track_id": track_id, "box": box, "decision": "invalid",
                            "reason": "empty_crop"})
                continue
            vec = _reid.embed(crop)
            quality = _reid.assess_quality(crop)
            res = g.identify_or_enroll(
                vec, quality,
                label=det.get("label"),
                attributes=det.get("attributes"),
                auto_enroll=auto_enroll,
            )
            res["track_id"] = track_id
            res["box"] = box
            res["quality"] = quality
            out.append(res)
        return out

    results = _gallery.with_gallery_locked(session_id, dim, _run)
    return {
        "backend": _reid.active_backend(),
        "dim": dim,
        "session_id": session_id,
        "results": results,
        "gallery_size": _gallery.get_gallery(session_id, dim).stats()["subjects"],
    }


def gallery_stats(session_id: str = "default") -> dict:
    dim = _reid.embed_dim()
    return _gallery.get_gallery(session_id, dim).stats()


# 直接转发的运维能力（与 tracker_service 的导出风格一致）
reset_gallery = _gallery.reset_gallery
reset_all_galleries = _gallery.reset_all_galleries
active_gallery_sessions = _gallery.active_gallery_sessions

__all__ = [
    "active_gallery_sessions",
    "gallery_backend_info",
    "gallery_stats",
    "identify_detections",
    "reset_all_galleries",
    "reset_gallery",
]
