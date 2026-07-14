"""Phase 1 竖切脚本（Step 2）：本地端到端验证 LLM-first 是否可行。

流程：视频 → ffmpeg 抽帧 → Azure OpenAI vision → 结构化 JSON。
不依赖 Blob / FastAPI / Docker，最小代价验证"这条路通不通、效果够不够"。

用法：
    python scripts/vertical_slice.py                      # 用默认示例视频
    python scripts/vertical_slice.py path/to/video.mp4    # 指定视频
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Windows 控制台默认 cp1252，打印中文会报错；强制 stdout 用 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# 允许从项目根直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.pipeline import analyze_video  # noqa: E402

# 默认示例视频：复用 ffmpeg-learning 里已有的素材
DEFAULT_VIDEO = (
    Path(__file__).resolve().parents[2].parent
    / "ffmpeg-learning"
    / "output"
    / "short_video.mp4"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "out"


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not video.exists():
        print(f"[X] 找不到视频：{video}")
        print("    传一个路径：python scripts/vertical_slice.py <video.mp4>")
        return 1

    print(f"[1/3] 抽帧：{video.name}（每 {settings.frame_interval_seconds}s 一帧，"
          f"最多 {settings.max_frames} 帧，宽 {settings.frame_width}px）")
    print(f"[2/3] 调用 Azure OpenAI（部署：{settings.azure_openai_deployment}）...")
    t0 = time.time()
    try:
        payload = analyze_video(video, OUT_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"[X] 处理失败：{exc}")
        return 2
    dt = time.time() - t0

    for f in payload["frames_used"]:
        print(f"      - {f['frame_id']} @ {f['timestamp']}  {f['local_path']}")

    result = payload["llm_result"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "result.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[3/3] 完成（{dt:.1f}s）。结果已写入 {out_file}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
