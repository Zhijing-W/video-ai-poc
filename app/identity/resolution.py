from __future__ import annotations

import numpy as np

from ..core.config import settings


def person_record(tid: int, t: dict, ident: dict, win_idx: list[int], img_w: int, img_h: int) -> dict:
    """把一条 track 在某窗内的信息打包成 identity_context 能吃的 person dict。"""
    centers = [c for (i, c) in t["centers"] if i in set(win_idx)]
    box = t["boxes"].get(win_idx[-1]) or t.get("best_box") or []
    return {
        "track_id": tid,
        "source_track_ids": [tid],
        "box": box,
        "subject_id": ident.get("subject_id"),
        "decision": ident.get("decision"),
        "reused": ident.get("reused", False),
        "trajectory": [list(c) for c in centers],
        "reid": {"score": ident.get("score")} if ident.get("score") is not None else None,
        "face": ident.get("face"),
        "gait": ident.get("gait"),
        "fused": ident.get("fused"),
        "evidence": ident.get("evidence"),
        "merge_routes": ident.get("merge_routes"),
        "merge_agree": ident.get("merge_agree"),
        "cross_track_merged": ident.get("cross_track_merged", False),
        "subject_conflict_split": ident.get("subject_conflict_split", False),
    }

def group_people(
    win_tracks: list[int], tracks: dict[int, dict], identities: dict[int, dict],
    win_idx: list[int], img_w: int, img_h: int,
) -> list[dict]:
    """把窗内 track 按 subject 合并成 person 条目。

    同一 subject_id 的多条 track（ByteTrack 因遮挡/漂移把一个人断成多段）合并为**一个人**：
    轨迹跨 track 按帧序拼接、ReID 取最高分、box 取代表 track 的末位框，避免 LLM 把"一个人的
    若干轨迹"误数成多个人。subject_id 为空（没认出库内主体）的 track 各自独立成条。
    """
    win_set = set(win_idx)
    groups: dict[str, list[int]] = {}
    for tid in win_tracks:
        if tid not in tracks:
            continue
        sid = identities[tid].get("subject_id")
        key = f"subject:{sid}" if sid is not None else f"track:{tid}"
        groups.setdefault(key, []).append(tid)

    people: list[dict] = []
    for key, tids in groups.items():
        if len(tids) == 1:
            tid = tids[0]
            people.append(person_record(tid, tracks[tid], identities[tid], win_idx, img_w, img_h))
            continue
        # 多 track → 同一人：选分最高者为代表，合并轨迹/取最高 ReID 分
        rep = max(tids, key=lambda t: identities[t].get("score") or 0.0)
        merged_centers = sorted(
            [(i, c) for t in tids for (i, c) in tracks[t]["centers"] if i in win_set],
            key=lambda x: x[0],
        )
        best_score = max((identities[t].get("score") or 0.0) for t in tids)
        rep_box = tracks[rep]["boxes"].get(win_idx[-1]) or tracks[rep].get("best_box") or []
        face = next((identities[t].get("face") for t in tids if identities[t].get("face")), None)
        gait = next((identities[t].get("gait") for t in tids if identities[t].get("gait")), None)
        # 融合：取置信度最高的那条 track 的融合结果作代表
        fused = max(
            (identities[t].get("fused") for t in tids if identities[t].get("fused")),
            key=lambda fz: (fz or {}).get("confidence", 0.0), default=None,
        )
        # 跨 track 合并用到了哪几路证据（人脸/人形/步态），并到代表里
        merge_routes = sorted({r for t in tids for r in (identities[t].get("merge_routes") or [])})
        route_cn = {"face": "人脸库", "body": "人形库", "gait": "步态库"}
        local_subject = any(identities[t].get("local_subject") for t in tids)
        conflict_split = any(identities[t].get("subject_conflict_split") for t in tids)
        kind = "本视频本地subject" if local_subject else ("时间冲突拆分subject" if conflict_split else "同一人")
        attrs = [f"由{len(tids)}条轨迹合并({kind})"]
        if merge_routes:
            attrs.append("跨track印证：" + "+".join(route_cn.get(r, r) for r in merge_routes))
        people.append({
            "track_id": rep,
            "source_track_ids": sorted(tids),
            "box": rep_box,
            "subject_id": identities[rep].get("subject_id"),
            "decision": "local_stitched" if local_subject else ("conflict_split" if conflict_split else "hit"),
            "reused": False if (local_subject or conflict_split) else True,
            "trajectory": [list(c) for (_, c) in merged_centers],
            "reid": {"score": round(best_score, 4)} if best_score > 0 else None,
            "face": face,
            "gait": gait,
            "fused": fused,
            "merge_routes": merge_routes or None,
            "merge_agree": len(merge_routes) or None,
            "local_subject": local_subject or None,
            "subject_conflict_split": conflict_split or None,
            "attributes": attrs,
        })
    return people

def stitch_orphans(
    tracks: dict[int, dict],
    identities: dict[int, dict],
    track_emb: dict[int, np.ndarray],
    thresh: float,
) -> None:
    """把灰区/低质孤立 track（subject_id 为空）并成同视频内本地主体（就地改 identities）。

    做法：
      1. 若已有 gallery subject：孤立 track 与最相近主体相似度 ≥ thresh 且时间不重叠，才并入。
      2. 剩余孤立 track 彼此之间用更保守的阈值做本地聚类；同一簇内 track 也不能时间重叠。

    注意：这里不写 gallery，只给事件报告/LLM 一个稳定本地称呼。这样低质 crop 不会污染长期库，
    但也不会在报告里显示成一堆 "track 17"。
    """
    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _overlap(a: int, b: int) -> bool:
        return not (tracks[a]["last"] < tracks[b]["first"] or tracks[b]["last"] < tracks[a]["first"])

    # 各主体的成员向量（来自已分配 subject_id 的 track）
    members: dict[int, list[np.ndarray]] = {}
    member_tids: dict[int, list[int]] = {}
    for tid, idn in identities.items():
        sid = idn.get("subject_id")
        if sid is not None and tid in track_emb:
            members.setdefault(sid, []).append(track_emb[tid])
            member_tids.setdefault(sid, []).append(tid)
    reps: dict[int, np.ndarray] = {sid: _norm(np.mean(vs, axis=0)) for sid, vs in members.items()}

    # 孤立 track：按首次出现时间顺序缝合
    orphans = [tid for tid, idn in identities.items()
               if idn.get("subject_id") is None and tid in track_emb]
    orphans.sort(key=lambda t: tracks[t]["first"])

    for tid in orphans:
        v = _norm(track_emb[tid])
        best_sid, best_sim = None, -1.0
        hit_thresh = thresh if identities[tid].get("quality_ok") else max(thresh, settings.event_local_stitch_thresh)
        for sid, rep in reps.items():
            if any(_overlap(tid, mt) for mt in member_tids.get(sid, [])):
                continue
            sim = float(np.dot(v, rep))
            if sim > best_sim:
                best_sid, best_sim = sid, sim
        if best_sid is not None and best_sim >= hit_thresh:
            idn = identities[tid]
            idn["subject_id"] = best_sid
            idn["decision"] = "stitched"
            idn["reused"] = True
            idn["stitch_score"] = round(best_sim, 4)
            if idn.get("score") is None:
                idn["score"] = round(best_sim, 4)
            # 并入代表，便于后续断片接力
            members[best_sid].append(track_emb[tid])
            member_tids.setdefault(best_sid, []).append(tid)
            reps[best_sid] = _norm(np.mean(members[best_sid], axis=0))

    # 仍无 subject 的低质/灰区 track：只在本视频内保守聚类，铸本地 subject_id，不污染 gallery。
    remaining = [tid for tid in orphans if identities[tid].get("subject_id") is None]
    if not remaining:
        return

    local_thresh = max(thresh, settings.event_local_stitch_thresh)
    clusters: list[dict] = []
    for tid in remaining:
        v = _norm(track_emb[tid])
        best_cluster, best_sim = None, -1.0
        for cluster in clusters:
            if any(_overlap(tid, mt) for mt in cluster["tids"]):
                continue
            sim = float(np.dot(v, cluster["rep"]))
            if sim > best_sim:
                best_cluster, best_sim = cluster, sim
        if best_cluster is not None and best_sim >= local_thresh:
            best_cluster["tids"].append(tid)
            best_cluster["vecs"].append(track_emb[tid])
            best_cluster["scores"].append(best_sim)
            best_cluster["rep"] = _norm(np.mean(best_cluster["vecs"], axis=0))
        else:
            clusters.append({"tids": [tid], "vecs": [track_emb[tid]], "scores": [], "rep": v})

    existing = [idn.get("subject_id") for idn in identities.values() if idn.get("subject_id") is not None]
    next_sid = (max(existing) + 1) if existing else 1
    for cluster in sorted(clusters, key=lambda c: tracks[min(c["tids"])]["first"]):
        tids = cluster["tids"]
        sid = next_sid
        next_sid += 1
        best_sim = round(max(cluster["scores"]), 4) if cluster["scores"] else None
        for tid in tids:
            idn = identities[tid]
            idn["subject_id"] = sid
            idn["local_subject"] = True
            idn["reused"] = False
            idn["cross_track_merged"] = len(tids) > 1
            idn["decision"] = "local_stitched" if len(tids) > 1 else "local"
            if best_sim is not None:
                idn["stitch_score"] = best_sim
                if idn.get("score") is None:
                    idn["score"] = best_sim

def split_subject_time_conflicts(tracks: dict[int, dict], identities: dict[int, dict]) -> None:
    """拆开不可能属于同一人的 subject：同一 subject 下的 track 时间重叠则必须分成不同主体。

    ReID gallery 在远景小人/多人场景里可能把很多相似低质 crop 都 hit 到同一个 subject。
    但如果两条 track 的时间区间重叠，它们在同一时刻同时出现在画面里，就不可能是同一个人。
    这里用这个物理约束做兜底拆分，避免报告出现"主体#1 · 34条轨迹"。
    """
    def _overlap(a: int, b: int) -> bool:
        return not (tracks[a]["last"] < tracks[b]["first"] or tracks[b]["last"] < tracks[a]["first"])

    by_subject: dict[int, list[int]] = {}
    for tid, ident in identities.items():
        sid = ident.get("subject_id")
        if sid is not None and tid in tracks:
            by_subject.setdefault(int(sid), []).append(tid)

    existing = [int(sid) for sid in by_subject]
    next_sid = (max(existing) + 1) if existing else 1
    for sid, tids in list(by_subject.items()):
        if len(tids) < 2:
            continue
        tids.sort(key=lambda t: (tracks[t]["first"], tracks[t]["last"]))
        clusters: list[list[int]] = []
        for tid in tids:
            placed = False
            for cluster in clusters:
                if not any(_overlap(tid, other) for other in cluster):
                    cluster.append(tid)
                    placed = True
                    break
            if not placed:
                clusters.append([tid])
        if len(clusters) <= 1:
            continue

        for idx, cluster in enumerate(clusters):
            target_sid = sid if idx == 0 else next_sid
            if idx > 0:
                next_sid += 1
            for tid in cluster:
                ident = identities[tid]
                ident["subject_id"] = target_sid
                ident["subject_conflict_split"] = True
                ident["reused"] = False
                if ident.get("decision") == "hit":
                    ident["decision"] = "conflict_split"

def merge_tracks_cross_route(identities: dict[int, dict]) -> None:
    """跨 track 三路合并：人脸库 / 人形库 / 步态库 **任一路**认出同一人 → 并成一个 subject。

    动机：人形缝合(_stitch_orphans)只用人形 ReID 一路；但同一个人在不同 track 里，可能人形
    糊了却**人脸命中同号**、或人脸糊了却**步态命中同号**。这里用并查集，把"任意一路库编号相同"
    的 track 并成同一人——多路同时印证则置信更高（写进每条 track 的 merge_routes/merge_agree）。

    实现：三路各自把 track 按各自库编号分组，同组内两两 union；并完后每个连通分量=一个人，
    统一改写 identities[tid]['subject_id'] 为该分量的规范主体号（优先沿用分量内已有人形主体号，
    取最小；若整分量都没有人形主体号则新铸一个），下游 _group_people 即可自然按统一 subject 归并。
    """
    tids = list(identities.keys())
    if len(tids) < 2:
        return

    parent = {t: t for t in tids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def _route_id(idn: dict, route: str):
        if route == "body":
            local = (idn.get("route_subject") or {}).get("local_subject_id")
            return ("body", local) if local is not None else None
        if route == "face":
            fc = idn.get("face") or {}
            local = (fc.get("route_subject") or {}).get("local_subject_id")
            if (
                fc.get("matched")
                and fc.get("match_ready")
                and fc.get("eligibility") == "direct"
                and fc.get("quality") == "clear"
                and fc.get("track_consistency_status") in {"same_frame", "passed"}
                and local is not None
            ):
                return ("face", local)
            return None
        if route == "gait":
            gt = idn.get("gait") or {}
            local = (gt.get("route_subject") or {}).get("local_subject_id")
            if gt.get("decision") == "hit" and local is not None:
                return ("gait", local)
        return None

    # 三路分别按库编号分组 → 组内两两并；记录每条 track 触发合并用到了哪几路
    routes = ("body", "face", "gait")
    route_of_edge: dict[frozenset, set] = {}
    for route in routes:
        buckets: dict = {}
        for t in tids:
            gid = _route_id(identities[t], route)
            if gid is not None:
                buckets.setdefault(gid, []).append(t)
        for members in buckets.values():
            if len(members) < 2:
                continue
            base = members[0]
            for other in members[1:]:
                union(base, other)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    route_of_edge.setdefault(frozenset((members[i], members[j])), set()).add(route)

    # 连通分量 → 统一主体号
    comps: dict[int, list[int]] = {}
    for t in tids:
        comps.setdefault(find(t), []).append(t)

    existing_body = [idn.get("subject_id") for idn in identities.values()
                     if idn.get("subject_id") is not None]
    next_synth = (max(existing_body) + 1) if existing_body else 1

    for members in comps.values():
        if len(members) < 2:
            continue
        body_ids = [identities[t].get("subject_id") for t in members
                    if identities[t].get("subject_id") is not None]
        if body_ids:
            canonical = min(body_ids)
        else:
            canonical = next_synth
            next_synth += 1
        # 该分量里实际用到了哪几路证据（用于置信标注）
        comp_routes: set = set()
        mset = set(members)
        for edge, rs in route_of_edge.items():
            if edge <= mset:
                comp_routes |= rs
        agree = len(comp_routes)
        for t in members:
            idn = identities[t]
            prev = idn.get("subject_id")
            idn["subject_id"] = canonical
            idn["merge_routes"] = sorted(comp_routes)
            idn["merge_agree"] = agree
            if prev != canonical:
                idn["cross_track_merged"] = True
                idn["reused"] = True
                if idn.get("decision") not in ("hit", "stitched"):
                    idn["decision"] = "merged"

    for idn in identities.values():
        canonical = idn.get("subject_id")
        route_subject_ids = {}
        if canonical is not None and (idn.get("route_subject") or {}).get("local_subject_id") is not None:
            route_subject_ids["body"] = canonical
        face = idn.get("face") or {}
        if (
            canonical is not None
            and face.get("matched")
            and face.get("match_ready")
            and face.get("eligibility") == "direct"
            and face.get("quality") == "clear"
            and face.get("track_consistency_status") in {"same_frame", "passed"}
            and (face.get("route_subject") or {}).get("local_subject_id") is not None
        ):
            route_subject_ids["face"] = canonical
        gait = idn.get("gait") or {}
        if canonical is not None and (gait.get("route_subject") or {}).get("local_subject_id") is not None:
            route_subject_ids["gait"] = canonical
        idn["route_subject_ids"] = route_subject_ids

__all__ = ["person_record", "group_people", "stitch_orphans", "split_subject_time_conflicts", "merge_tracks_cross_route"]
