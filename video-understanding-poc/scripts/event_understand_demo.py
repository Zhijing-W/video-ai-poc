"""Phase 4 · Step 24 竖切脚本：身份感知·多帧事件理解 端到端 Demo。

流程（一条流）：
  视频 → 选帧①(定时密采样) → 逐帧 YOLO+ByteTrack(稳定 track_id) → 语义事件标注
       → 人形 ReID + 主体记忆库(认人=身份) → 流式分窗 → 选帧②(事件驱动)
       → 身份打包 → 多模态 LLM 跨帧事件理解 → 事件时间线

用法：
    # 先不花额度，验证整条链路 + 看清"会喂给 LLM 什么"（强烈建议先跑这个）
    python scripts/event_understand_demo.py --dry-run

    # 真调 gpt-4o 做事件理解（会消耗 Azure 额度）
    python scripts/event_understand_demo.py

    # 指定视频 / 采样率 / 启用人脸分支
    python scripts/event_understand_demo.py path/to/video.mp4 --fps 3 --face
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows 控制台默认 cp1252，打印中文会报错；强制 stdout 用 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.event_pipeline import analyze_event_stream  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "data" / "samples" / "mixkit_31372.mp4"
OUT_DIR = ROOT / "out" / "event_demo"


def main() -> int:
    ap = argparse.ArgumentParser(description="身份感知·多帧事件理解 端到端 Demo")
    ap.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO), help="输入视频路径")
    ap.add_argument("--fps", type=float, default=2.0, help="选帧① 定时采样率（帧/秒），默认 2")
    ap.add_argument("--max-frames", type=int, default=300, help="抽帧硬上限，默认 300")
    ap.add_argument("--max-keyframes", type=int, default=None,
                    help="喂 LLM 的关键帧上限（默认 settings.keyframe_max=24）。低配额/撞 429 时调小，如 8")
    ap.add_argument("--max-window-seconds", type=float, default=None,
                    help="单个事件窗时长上限（秒），超过则切新窗、多调一次 LLM（默认 30）。长连续视频调小可拿到更细叙述")
    ap.add_argument("--stitch-thresh", type=float, default=None,
                    help="同视频内轨迹缝合余弦阈值（默认 0.45）。灰区孤立 track 与主体相似度≥此值即并入；设 0 关闭")
    ap.add_argument("--no-overall", action="store_true",
                    help="关闭跨窗整段事件总结（默认开；纯文本便宜调用）")
    ap.add_argument("--face", action="store_true", help="启用人脸分支（InsightFace，较慢）")
    ap.add_argument("--gait", action="store_true", help="启用步态分支（SkeletonGait++，CPU 较慢；需 OpenGait+权重）")
    ap.add_argument("--objective", default=None, help="给事件理解的关注点提示")
    ap.add_argument("--dry-run", action="store_true", help="只跑到 LLM 边界，不真调模型（不花额度）")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"[X] 找不到视频：{video}")
        print("    传一个路径：python scripts/event_understand_demo.py <video.mp4>")
        return 1

    mode = "DRY-RUN（不调 LLM）" if args.dry_run else "FULL（真调 gpt-4o，消耗额度）"
    print(f"[*] 视频：{video.name}   采样：{args.fps} fps   "
          f"人脸：{'开' if args.face else '关'}   步态：{'开' if args.gait else '关'}   模式：{mode}")
    print("[1/2] 抽帧 → 检测/跟踪 → 认人(ReID) → 分窗 → 选帧② ...")

    try:
        payload = analyze_event_stream(
            video, OUT_DIR,
            fps=args.fps, max_frames=args.max_frames,
            run_llm=not args.dry_run, with_face=args.face, with_gait=args.gait,
            objective=args.objective, max_keyframes=args.max_keyframes,
            max_window_seconds=args.max_window_seconds,
            stitch_thresh=args.stitch_thresh,
            overall_summary=(False if args.no_overall else None),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[X] 处理失败：{exc}")
        import traceback
        traceback.print_exc()
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "result.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 可读摘要 ----
    print(f"[2/2] 完成（{payload['elapsed_seconds']}s）。"
          f"共 {payload['frames_total']} 帧 / {len(payload['windows'])} 个事件窗 / "
          f"{len(payload['tracks'])} 条轨迹。ReID={payload['reid_backend']}({payload['reid_dim']}d)\n")

    if payload.get("with_gait"):
        print(f"   （步态分支已启用 SkeletonGait++）")
    elif args.gait and payload.get("gait_error"):
        print(f"   （步态未启用：{payload['gait_error']}）")

    print("== 主体记忆（认人结果）==")
    for tid, idn in payload["tracks"].items():
        g = idn.get("gait")
        gtxt = f"步态分={round(g['score'],3)}({g['frames']}帧)" if g and g.get("score") is not None else "步态无"
        print(f"  track {tid}: 主体#{idn.get('subject_id')}  裁决={idn.get('decision')}  "
              f"分={idn.get('score')}  回头客={idn.get('reused')}  脸={'有' if idn.get('face') else '无'}  {gtxt}")

    for win in payload["windows"]:
        print(f"\n========== 事件窗 #{win['window_index']}  时间 {win['time_range'][0]}~{win['time_range'][1]}"
              f"  帧数 {win['frame_count']} → 关键帧 {len(win['keyframe_indices'])} ==========")
        print(f"  语义事件: {', '.join(win['events']) or '（无）'}")
        print(f"  关键帧时间戳: {', '.join(win['keyframe_timestamps'])}")
        if win.get("identity_context"):
            print("  --- 注入 LLM 的身份上下文 ---")
            print("  " + win["identity_context"].strip().replace("\n", "\n  "))
        if "event" in win:
            ev = win["event"]
            print("  --- 事件理解（gpt-4o）---")
            print(f"  概述: {ev.get('summary')}")
            print(f"  告警: {ev.get('alert_level')}   通知: {ev.get('notification')}")
            for e in ev.get("events", []):
                flag = "⚠️" if e.get("abnormal") else "  "
                print(f"   {flag} [{e.get('time')}] {e.get('subject')}: {e.get('action')}")
        else:
            print("  （dry-run：未调用 LLM。以上身份上下文 + 关键帧即为将要喂给模型的内容。）")

    ov = payload.get("overall")
    if ov and not ov.get("error"):
        print("\n========== 整段事件总结（跨窗整合）==========")
        print(f"  概述: {ov.get('overall_summary')}")
        print(f"  整段告警: {ov.get('overall_alert_level')}   通知: {ov.get('notification')}")
        for s in ov.get("story", []):
            print(f"   [{s.get('time')}] {s.get('subject')}: {s.get('action')}")
        for s in ov.get("subjects", []):
            print(f"   · {s}")

    print(f"\n[✓] 完整结果已写入 {out_file}")
    if payload.get("dry_run"):
        print("    去掉 --dry-run 即真调 gpt-4o 生成事件叙述。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
