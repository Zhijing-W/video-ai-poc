"""视频抽帧（Phase 1：规则抽帧，不做 ViT/CLIP）。

用 ffmpeg 把视频按固定间隔抽成若干 jpg，并返回每帧的元数据。
ffmpeg 优先用 PATH 里的；找不到时回退到 imageio-ffmpeg 自带的静态二进制
（这样部署到没有 ffmpeg 的环境也能跑，免 Docker）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .core.config import settings
from .utils.image_utils import seconds_to_timestamp

# 智能抽帧安全上限：再多场景突变也不会让 ffmpeg 无界产出（PoC 足够）。
_SMART_SAFETY_CAP = 300
_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


@dataclass
class Frame:
    frame_id: str
    timestamp: str
    local_path: str


def _resolve_ffmpeg() -> str:
    """返回可用的 ffmpeg 可执行文件路径。"""
    cand = settings.ffmpeg_path
    if shutil.which(cand):
        return cand
    # 回退：imageio-ffmpeg 自带静态 ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "找不到 ffmpeg。请安装 ffmpeg 并加入 PATH，"
            "或 `pip install imageio-ffmpeg`。"
        ) from exc


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    interval_seconds: int | None = None,
    max_frames: int | None = None,
    width: int | None = None,
) -> list[Frame]:
    """从视频抽帧。

    Args:
        video_path: 输入视频。
        out_dir: 帧输出目录（会被创建/清空）。
        interval_seconds: 每隔几秒抽一帧（默认取 settings）。
        max_frames: 最多抽几帧。
        width: 输出图片宽度（高度按比例，-2 保持偶数）。

    Returns:
        Frame 列表（含 frame_id / timestamp / local_path）。
    """
    interval = interval_seconds or settings.frame_interval_seconds
    cap = max_frames or settings.max_frames
    w = width or settings.frame_width

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _resolve_ffmpeg()
    pattern = str(out_dir / "frame_%03d.jpg")
    # fps=1/interval 每 interval 秒一帧；scale 缩放宽度；-frames:v 限制数量
    vf = f"fps=1/{interval},scale={w}:-2"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-frames:v", str(cap),
        "-q:v", "3",
        pattern,
    ]
    subprocess.run(cmd, check=True)

    frames: list[Frame] = []
    for i, p in enumerate(sorted(out_dir.glob("frame_*.jpg"))):
        frames.append(
            Frame(
                frame_id=p.stem,
                timestamp=seconds_to_timestamp(i * interval),
                local_path=str(p),
            )
        )
    if not frames:
        raise RuntimeError("抽帧结果为空，请检查视频是否可解码。")
    return frames


def _even_indices(n: int, k: int) -> list[int]:
    """从 n 帧里在时间轴上均匀挑 k 个下标（k>=n 时全取），避免只截开头造成偏置。"""
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [n // 2]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def extract_frames_smart(
    video_path: str | Path,
    out_dir: str | Path,
    scene_threshold: float | None = None,
    fallback_seconds: int | None = None,
    max_frames: int | None = None,
    width: int | None = None,
) -> list[Frame]:
    """智能抽帧（Phase 2 · Step 7）：场景突变 OR 定时兜底，二者 OR 触发。

    解决纯定时抽帧的两个毛病：静止画面冗余抽、长时间静止但重要的画面漏抽。
    用 ffmpeg select 过滤器组合两个触发条件（任一成立即选帧）：
      - 画面突变：scene 分数 > 阈值（镜头切换 / 人物进出）。
      - 定时兜底：距上次选帧（prev_selected_t）>= 兜底秒数；含首帧（isnan）。
    showinfo 打印 pts_time 用于精确时间戳；最终超过 max_frames 时在时间轴上均匀
    降采样（而非只截开头）。任何异常或零产出都回落到定时抽帧，保证不空手。

    Returns:
        Frame 列表（含 frame_id / timestamp / local_path），与 extract_frames 同构。
    """
    threshold = settings.scene_threshold if scene_threshold is None else scene_threshold
    fallback = settings.fallback_interval_seconds if fallback_seconds is None else fallback_seconds
    cap = max_frames or settings.max_frames
    w = width or settings.frame_width

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _resolve_ffmpeg()
    pattern = str(out_dir / "frame_%03d.jpg")
    # 单引号包裹 select 表达式，内部逗号被 ffmpeg 过滤图解析器保护，无需再转义。
    select_expr = (
        f"gt(scene,{threshold})+isnan(prev_selected_t)+gte(t-prev_selected_t,{fallback})"
    )
    vf = f"select='{select_expr}',showinfo,scale={w}:-2"

    def _run(sync_flag: str) -> subprocess.CompletedProcess:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "info",
            "-i", str(video_path),
            "-vf", vf,
            sync_flag, "vfr",                      # 仅保留被选中的帧
            "-frames:v", str(_SMART_SAFETY_CAP),
            "-q:v", "3",
            pattern,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    proc = _run("-fps_mode")                       # 新版 ffmpeg
    if proc.returncode != 0 and not list(out_dir.glob("frame_*.jpg")):
        proc = _run("-vsync")                      # 旧版 ffmpeg 回退

    files = sorted(out_dir.glob("frame_*.jpg"))
    if not files:
        # 智能抽帧未产出（极端编码/过滤异常）→ 回落定时抽帧
        return extract_frames(video_path, out_dir, max_frames=cap, width=w)

    # showinfo 的 pts_time 顺序与输出帧一致；日志可能不齐 → 容错对齐
    times = [float(x) for x in _PTS_RE.findall(proc.stderr or "")]

    keep = set(_even_indices(len(files), cap))
    frames: list[Frame] = []
    for idx, p in enumerate(files):
        if idx not in keep:
            p.unlink(missing_ok=True)
            continue
        ts = times[idx] if idx < len(times) else idx * fallback
        frames.append(Frame(frame_id=p.stem, timestamp=seconds_to_timestamp(ts), local_path=str(p)))

    if not frames:
        return extract_frames(video_path, out_dir, max_frames=cap, width=w)
    return frames


def frames_as_dicts(frames: list[Frame]) -> list[dict]:
    return [asdict(f) for f in frames]
