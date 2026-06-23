"""主体记忆向量库 / ReID gallery（Phase 3 · Step 14）。

定位：Phase 3 的第二大省钱杠杆——在 MOT（Step 11，给每个目标稳定 `track_id`）之上，
让系统"**认过一次就记住**"。同一个人/物第二次出现时，提一个外观指纹向量、查这个库，
**命中即复用档案、完全不调 LLM**；跨摄像头/跨时间也能认出同一主体。

本模块只负责"**向量库**"本身（与具体 ReID 模型解耦）：接收一个已提好的归一化向量
（由 `app.reid.embed` 产出）+ 质量信息，完成检索 / 开放集登记 / 多帧投票 / 质量门控 /
负缓存。**不含编排**——"什么时候来查库"由上层三时钟（Step 12）决定，本库只被调用。

核心能力（对应设计文档 3.4）：
  - **余弦 kNN 检索**：用 FAISS `IndexIDMap2(IndexFlatIP)`，向量已 L2 归一化 → 内积即余弦。
  - **multi-shot**：每个主体存多张（不同角度/帧）shot，查询时按主体聚合（取 shot 最高分），
    比"单帧单向量"稳得多；每主体 shot 数有上限，超出按最旧淘汰。
  - **开放集登记（open-set enrollment）**：查不到（极低分）就判为新主体、自动建档登记。
  - **质量门控**：糊/太小/遮挡的 crop 不入库（避免污染向量库，越查越错）。
  - **负缓存（negative cache）**：记住"查过、确认不在库里"的查询向量，相似查询直接短路，
    避免反复白查（同样省下游 LLM）。

有状态 & 会话隔离：与 `tracker.py` 一致，按 `session_id` 隔离独立 gallery 实例 + 专属锁
（同会话串行、不同会话并行）；换视频/重新开始调用 `reset_gallery`。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .config import settings


@dataclass
class _Subject:
    """一个被记住的主体档案：多张 shot 向量 + 元数据。"""

    subject_id: int
    label: str | None = None
    row_ids: list[int] = field(default_factory=list)   # 该主体在 FAISS 中的所有 shot 行 id
    shots: int = 0
    hit_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    attributes: list[str] = field(default_factory=list)


class SessionGallery:
    """单个会话的主体记忆向量库（FAISS 余弦检索 + 开放集登记）。"""

    def __init__(self, dim: int) -> None:
        import faiss  # 懒加载：仅在真正建库时才依赖 faiss

        self.dim = int(dim)
        # IndexIDMap2 让我们能用自定义 row_id 增删（multi-shot 淘汰需要 remove_ids）。
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self._faiss = faiss
        self._subjects: dict[int, _Subject] = {}
        self._row_to_subject: dict[int, int] = {}
        self._row_vecs: dict[int, np.ndarray] = {}   # row_id -> 向量（淘汰旧 shot 时用）
        self._neg_cache: list[np.ndarray] = []       # 负缓存：确认"不在库里"的查询向量
        self._next_subject_id = 1
        self._next_row_id = 1

    # ---- 内部工具 ----
    def _prepare(self, vec: np.ndarray) -> np.ndarray:
        """转成 (1, dim) float32，并保证 L2 归一化（防御性，embed 已归一化）。"""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"向量维度不符：期望 {self.dim}，收到 {v.shape[0]}")
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v.reshape(1, self.dim)

    def _search_subjects(self, v: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """返回 [(subject_id, score)]，按 score 降序；score = 该主体所有 shot 的最高余弦。"""
        if self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(v, k)
        best: dict[int, float] = {}
        for row_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if row_id == -1:
                continue
            sid = self._row_to_subject.get(int(row_id))
            if sid is None:
                continue
            if sid not in best or score > best[sid]:
                best[sid] = float(score)
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    def _neg_hit(self, v: np.ndarray) -> bool:
        """查询向量是否落在负缓存里（与某条已确认'不在库'的向量足够像）。"""
        if not self._neg_cache:
            return False
        sims = [float(np.dot(v[0], nv)) for nv in self._neg_cache]
        return max(sims) >= settings.reid_neg_cache_thresh

    def _add_shot(self, subject: _Subject, v: np.ndarray) -> None:
        row_id = self._next_row_id
        self._next_row_id += 1
        self._index.add_with_ids(v, np.array([row_id], dtype=np.int64))
        self._row_to_subject[row_id] = subject.subject_id
        self._row_vecs[row_id] = v[0].copy()
        subject.row_ids.append(row_id)
        subject.shots += 1
        # multi-shot 上限：超了淘汰最旧 shot（FIFO），避免向量库无限膨胀。
        max_shots = settings.reid_max_shots
        while len(subject.row_ids) > max_shots:
            drop = subject.row_ids.pop(0)
            self._index.remove_ids(np.array([drop], dtype=np.int64))
            self._row_to_subject.pop(drop, None)
            self._row_vecs.pop(drop, None)

    # ---- 对外 API ----
    def identify(self, vec: np.ndarray, top_k: int = 5) -> dict:
        """只查不写：返回这条向量的归属裁决（不登记、不改库）。

        decision ∈ {hit, grey, new}：
          - hit  : 最高分 ≥ hit_thresh → 认出已知主体，可直接复用其档案、不必调 LLM。
          - new  : 最高分 < new_thresh（或库为空）→ 大概率是没见过的新主体。
          - grey : 介于两者之间 → 灰区，建议升级细粒度/多帧投票/LLM 裁决。
        """
        v = self._prepare(vec)
        ranked = self._search_subjects(v, top_k)
        best_sid, best_score = (ranked[0] if ranked else (None, 0.0))
        runner_up = ranked[1][1] if len(ranked) > 1 else None

        if best_sid is not None and best_score >= settings.reid_hit_thresh:
            decision = "hit"
        elif best_sid is None or best_score < settings.reid_new_thresh:
            decision = "new"
        else:
            decision = "grey"

        result = {
            "decision": decision,
            "subject_id": best_sid if decision == "hit" else None,
            "score": round(best_score, 4),
            "runner_up_score": round(runner_up, 4) if runner_up is not None else None,
            "gallery_size": len(self._subjects),
            "negative_cache_hit": self._neg_hit(v),
        }
        if decision == "hit":
            subj = self._subjects[best_sid]
            result["label"] = subj.label
            result["shots"] = subj.shots
            result["attributes"] = list(subj.attributes)
        return result

    def identify_or_enroll(
        self,
        vec: np.ndarray,
        quality: dict | None = None,
        *,
        label: str | None = None,
        attributes: list[str] | None = None,
        auto_enroll: bool = True,
        top_k: int = 5,
    ) -> dict:
        """查库 + 按裁决处理（开放集登记的主路径，供 /identify 用）。

        - hit  : 命中 → 复用档案；若 shot 数未满且本帧质量合格，顺手补一张 shot（多角度更稳）。
        - new  : 新主体 → 质量合格则登记建档（auto_enroll=True）；否则只判定不入库（污染防护）。
        - grey : 不登记（交给上层升级裁决），但记入负缓存倾向（不主动登记，避免污染）。
        """
        v = self._prepare(vec)
        res = self.identify(v, top_k=top_k)
        decision = res["decision"]
        accept, why = quality_ok(quality)
        res["quality_ok"] = accept
        res["quality_reason"] = why
        now = time.time()

        if decision == "hit":
            subj = self._subjects[res["subject_id"]]
            subj.hit_count += 1
            subj.last_seen = now
            if accept and subj.shots < settings.reid_max_shots:
                self._add_shot(subj, v)
            res["enrolled"] = False
            res["shots"] = subj.shots
            return res

        if decision == "new" and auto_enroll:
            if not accept:
                # 质量不过关：不建档（避免低质 crop 污染库），仅返回判定。
                res["enrolled"] = False
                return res
            subj = _Subject(subject_id=self._next_subject_id, label=label,
                            attributes=list(attributes or []), first_seen=now, last_seen=now)
            self._next_subject_id += 1
            self._subjects[subj.subject_id] = subj
            self._add_shot(subj, v)
            res["subject_id"] = subj.subject_id
            res["decision"] = "new"
            res["enrolled"] = True
            res["label"] = subj.label
            res["shots"] = subj.shots
            # 新主体登记后，从负缓存里清掉与它相近的旧"否定"记录（已不再是 negative）。
            self._neg_cache = [nv for nv in self._neg_cache
                               if float(np.dot(v[0], nv)) < settings.reid_neg_cache_thresh]
            return res

        # grey，或 new 但不自动登记：不写库。把查询向量记入负缓存（确认当前不属于任何已知主体）。
        if decision in {"new", "grey"}:
            self._push_negative(v)
        res["enrolled"] = False
        return res

    def _push_negative(self, v: np.ndarray) -> None:
        self._neg_cache.append(v[0].copy())
        cap = settings.reid_neg_cache_size
        if len(self._neg_cache) > cap:
            self._neg_cache = self._neg_cache[-cap:]

    def stats(self) -> dict:
        return {
            "dim": self.dim,
            "subjects": len(self._subjects),
            "total_shots": int(self._index.ntotal),
            "negative_cache": len(self._neg_cache),
            "subject_detail": [
                {
                    "subject_id": s.subject_id,
                    "label": s.label,
                    "shots": s.shots,
                    "hit_count": s.hit_count,
                    "attributes": s.attributes,
                }
                for s in self._subjects.values()
            ],
        }


def quality_ok(quality: dict | None) -> tuple[bool, str | None]:
    """质量门控：决定一个 crop 是否够格入库/补 shot。

    quality 由 `app.reid.assess_quality` 产出（不传则默认放行）。判据：
      - 尺寸太小（像素面积过小）→ 拒：远景小目标特征不可靠。
      - 太糊（拉普拉斯方差低）→ 拒：运动模糊/失焦特征无意义。
      - 长宽比异常 → 拒：多半是半个框/遮挡严重。
    """
    if not quality:
        return True, None
    if quality.get("area", 1e9) < settings.reid_min_area:
        return False, "too_small"
    if quality.get("blur_var", 1e9) < settings.reid_min_blur_var:
        return False, "too_blurry"
    ar = quality.get("aspect_ratio")
    if ar is not None and not (settings.reid_min_aspect <= ar <= settings.reid_max_aspect):
        return False, "bad_aspect_ratio"
    return True, None


# ---- 按 session 隔离的 gallery 注册表（与 tracker.py 一致的运维形态）----
_galleries: dict[str, dict] = {}   # session_id -> {"gallery": SessionGallery, "lock": Lock}
_registry_lock = threading.Lock()


def get_gallery(session_id: str, dim: int) -> SessionGallery:
    """懒加载：按 session 取（或按给定维度新建）一个 gallery 实例。"""
    with _registry_lock:
        entry = _galleries.get(session_id)
        if entry is None:
            entry = {"gallery": SessionGallery(dim), "lock": threading.Lock()}
            _galleries[session_id] = entry
        return entry["gallery"]


def _entry(session_id: str, dim: int) -> dict:
    with _registry_lock:
        entry = _galleries.get(session_id)
        if entry is None:
            entry = {"gallery": SessionGallery(dim), "lock": threading.Lock()}
            _galleries[session_id] = entry
        return entry


def with_gallery_locked(session_id: str, dim: int, fn):
    """在该 session 的锁内执行 fn(gallery)，保证同会话串行写库。"""
    entry = _entry(session_id, dim)
    with entry["lock"]:
        return fn(entry["gallery"])


def reset_gallery(session_id: str = "default") -> bool:
    """清空某 session 的主体记忆（换视频/重新开始时调用）。"""
    with _registry_lock:
        return _galleries.pop(session_id, None) is not None


def reset_all_galleries() -> int:
    with _registry_lock:
        count = len(_galleries)
        _galleries.clear()
        return count


def active_gallery_sessions() -> list[str]:
    with _registry_lock:
        return list(_galleries)
