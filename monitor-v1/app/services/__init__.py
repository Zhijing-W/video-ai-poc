"""Service-layer helpers for routers."""
from .cruise_service import apply_plan
from .fusion_service import (
    active_fusion_sessions,
    add_observation,
    reset_all_fusion,
    reset_fusion,
    resolve_session,
    resolve_track,
)
from .gallery_service import (
    active_gallery_sessions,
    gallery_backend_info,
    gallery_stats,
    identify_detections,
    reset_all_galleries,
    reset_gallery,
)
from .gate_service import decide_gate, synthesize_result_from_yolo, yolo_signature
from .identity_integration import enrich_with_identity, reset_identity
from .llm_service import analyze_frame_content, compile_target_rule, summarize_session_events, summarize_video_frames
from .track_gate import decide_track_gate, gate_stats, record_llm_conclusion, record_reuse, reset_track_gate
from .tracker_service import active_sessions, reset_all_trackers, reset_tracker, track_objects
from .yolo_service import class_names, decode_image, detect_objects, enrich_detection_colors

__all__ = [
    "active_fusion_sessions",
    "active_sessions",
    "add_observation",
    "analyze_frame_content",
    "apply_plan",
    "class_names",
    "compile_target_rule",
    "decide_gate",
    "decide_track_gate",
    "decode_image",
    "detect_objects",
    "enrich_detection_colors",
    "enrich_with_identity",
    "gallery_backend_info",
    "gallery_stats",
    "gate_stats",
    "identify_detections",
    "record_llm_conclusion",
    "record_reuse",
    "reset_all_fusion",
    "reset_all_galleries",
    "reset_all_trackers",
    "reset_fusion",
    "reset_gallery",
    "reset_identity",
    "reset_track_gate",
    "reset_tracker",
    "resolve_session",
    "resolve_track",
    "summarize_session_events",
    "summarize_video_frames",
    "synthesize_result_from_yolo",
    "track_objects",
    "yolo_signature",
]
