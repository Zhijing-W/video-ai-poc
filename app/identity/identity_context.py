"""结构化身份打包（Phase 4 · Step 22）—— 把每个 person 整理成喂 LLM 的"身份上下文"。

定位：客户对齐的关键一环。传统 CV（人脸 / 人形 ReID / 步态 + 库比对）先得出**身份**，本模块
把每个 person 的多源识别结果，整理成大模型能读的**结构化条目**，随**多帧图**一起注入 prompt。
沿用 `llm_client._format_detections` 的"检测 grounding 注入"思路，再加上**身份**。

**核心约定**（和客户一致）：身份由外部（传统 CV）给定，大模型**看图但不重新认人**——它基于
这些身份 + 多帧画面，做**跨帧事件理解**（谁、何时、做了什么）。

本模块是**纯打包叶子**：只把"已算好的 per-track 识别数据"格式化，不查库、不调模型、不碰实时流程。
真正去填这些字段（调 gallery / face / fusion）由集成步（demo / 事件理解）完成。

输入（每个 person 一条，字段都可缺省）：
    {
      "track_id": 7, "box": [x1,y1,x2,y2],
      "subject_id": 3,                 # 主体记忆库内的稳定编号
      "db_identity": "员工A123",        # 客户人员库命中的真实身份（可选）
      "decision": "hit"|"new"|"grey",  # 识别裁决
      "reused": True,                  # 是否回头客（跨 track 复用同一主体）
      "trajectory": [[x,y], ...],      # 该 track 的中心点轨迹（归一化或像素）
      "reid":  {"score": 0.86},        # 人形 ReID 命中分
      "face":  {"score": 0.82, "quality": "clear"|"blurry"|float, "matched": True},
      "gait":  {"score": 0.71},        # 步态命中分（可选）
      "color": "red", "attributes": ["male","backpack"],
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field


def confidence_band(score: float | None) -> str:
    """把 0~1 的相似度分映射成人类可读的置信度档（给 LLM 一个定性强弱）。"""
    if score is None:
        return "未知"
    if score >= 0.75:
        return "高"
    if score >= 0.55:
        return "中"
    if score > 0:
        return "低"
    return "无"


@dataclass
class PersonIdentity:
    """一个 person 的结构化身份条目（机器可读 + 可格式化成 prompt 文本）。"""

    track_id: int
    box: list[float] = field(default_factory=list)
    subject_id: int | None = None
    db_identity: str | None = None
    decision: str | None = None
    reused: bool = False
    trajectory: list[list[float]] = field(default_factory=list)
    reid: dict | None = None
    face: dict | None = None
    gait: dict | None = None
    fused: dict | None = None
    color: str | None = None
    attributes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "PersonIdentity":
        return cls(
            track_id=int(d["track_id"]),
            box=[float(v) for v in (d.get("box") or [])],
            subject_id=d.get("subject_id"),
            db_identity=d.get("db_identity"),
            decision=d.get("decision"),
            reused=bool(d.get("reused", False)),
            trajectory=[[float(x) for x in p] for p in (d.get("trajectory") or [])],
            reid=d.get("reid"),
            face=d.get("face"),
            gait=d.get("gait"),
            fused=d.get("fused"),
            color=d.get("color"),
            attributes=list(d.get("attributes") or []),
        )

    # ---- 派生展示 ----
    def label(self) -> str:
        """这个人对 LLM 的稳定称呼：优先库内真实身份，否则主体编号，再否则未识别。"""
        if self.db_identity:
            return f"{self.db_identity}（库内身份）"
        if self.subject_id is not None:
            tag = "新主体" if self.decision == "new" else ("待定" if self.decision == "grey" else "已知")
            return f"主体#{self.subject_id}（{tag}）"
        return "未识别人物"

    def _cues_text(self) -> str:
        """把人脸/人形/步态三路命中整理成一句线索说明（含质量，体现"人脸模糊就降权"）。"""
        parts = []
        if self.face:
            q = self.face.get("quality")
            qtxt = q if isinstance(q, str) else (f"q={q}" if q is not None else "")
            band = confidence_band(self.face.get("score"))
            matched = self.face.get("matched")
            parts.append(f"人脸{('命中' if matched else '存在')}·{band}{('·' + qtxt) if qtxt else ''}")
        else:
            parts.append("人脸:无/不可用")
        if self.reid:
            parts.append(f"人形ReID·{confidence_band(self.reid.get('score'))}")
        if self.gait:
            parts.append(f"步态·{confidence_band(self.gait.get('score'))}")
        return "；".join(parts)

    def _traj_text(self) -> str:
        if not self.trajectory:
            return ""
        pts = self.trajectory
        show = [pts[0], pts[len(pts) // 2], pts[-1]] if len(pts) >= 3 else pts
        seg = " → ".join(f"({round(p[0], 2)},{round(p[1], 2)})" for p in show)
        return f"  轨迹: {seg}"

    def _fused_text(self) -> str:
        """三路融合身份置信度（让 LLM 知道这个身份有多可靠、靠哪几路撑住）。"""
        if not self.fused:
            return ""
        conf = self.fused.get("confidence")
        srcs = "+".join(s.get("cue") for s in (self.fused.get("sources") or []))
        tag = "已确认" if self.fused.get("resolved") else "待定"
        multi_source = self.fused.get("multi_source", self.fused.get("agreed"))
        agree = "·多路线参与" if multi_source else ""
        band = confidence_band(conf)
        return f"融合身份·{band}({conf}){agree}" + (f"·来源:{srcs}" if srcs else "")

    def to_line(self, idx: int) -> str:
        """格式化成 prompt 里的一行（仿 _format_detections 的 grounding 风格）。"""
        box_str = ", ".join(str(round(v, 1)) for v in self.box)
        head = f"  #{idx} {self.label()}  track={self.track_id}"
        if self.reused:
            head += " ♻回头客"
        head += f"  box=[{box_str}]"
        extra = []
        fused = self._fused_text()
        if fused:
            extra.append(fused)
        cues = self._cues_text()
        if cues:
            extra.append(cues)
        if self.color:
            extra.append(f"主色:{self.color}")
        if self.attributes:
            extra.append("属性:" + "/".join(self.attributes))
        line = head + (("\n      " + " | ".join(extra)) if extra else "")
        return line + self._traj_text()

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "box": self.box,
            "subject_id": self.subject_id,
            "db_identity": self.db_identity,
            "decision": self.decision,
            "reused": self.reused,
            "label": self.label(),
            "reid": self.reid,
            "face": self.face,
            "gait": self.gait,
            "color": self.color,
            "attributes": self.attributes,
        }


def build_identity_records(people: list[dict]) -> list[PersonIdentity]:
    """把原始 per-track dict 列表规范成 PersonIdentity 列表。"""
    return [PersonIdentity.from_dict(p) for p in people if p.get("track_id") is not None]


def format_identity_grounding(
    people: list[dict] | list[PersonIdentity],
    img_w: int | None = None,
    img_h: int | None = None,
) -> str:
    """把若干 person 身份条目，格式化成喂 LLM 的"身份上下文"文本块。

    放在多帧图之前注入 prompt：告诉模型"这些身份是外部传统 CV 给定的，你不要重新认人；
    请基于这些身份 + 多帧画面，做跨帧事件理解"。无人物时返回空串。
    """
    records = build_identity_records(people) if (people and isinstance(people[0], dict)) else list(people)
    if not records:
        return ""
    dims = f"（图像尺寸约 {img_w}×{img_h} 像素，坐标原点在左上角）" if img_w and img_h else ""
    lines = [r.to_line(i) for i, r in enumerate(records, 1)]
    return (
        "\n\n【画面中的人物身份（由传统 CV：人脸 / 人形 ReID / 步态 + 库比对 得出）"
        + dims
        + "】\n"
        + "\n".join(lines)
        + "\n约定：身份为**外部给定**，请勿重新做人脸/人形识别；同一 track（或同一 主体#/库内身份）"
        "在多帧中是**同一个人**。请结合这些身份与多帧画面，理解并叙述跨帧事件（谁、何时、做了什么、"
        "是否异常）。人脸标注为模糊/低置信时，身份以人形 ReID / 步态为准。"
    )


__all__ = [
    "PersonIdentity",
    "build_identity_records",
    "confidence_band",
    "format_identity_grounding",
]
