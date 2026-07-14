"""YOLO-Pose 人体关键点 → 部件区派生（Phase 3 · Step 13「细粒度感知 v1」）。

定位：修 Phase 2「颜色误判」。Phase 2 判人衣服颜色，是拿整个 person 粗框（或粗框里
一块**写死比例**的"torso"：高 42%~82%、宽 18%~82%）去取主色——框里混了背景、皮肤、
头发、裤子，主色被污染（文档原话：黑人肤色被当成衣服颜色、橙衣判成蓝衣）。

解法（本模块）：用 **YOLO-Pose**（17 个 COCO 关键点）找到**真实的躯干区**——
两肩到两胯围成的胸腹区，水平只取肩宽、垂直只取肩到胯——这块基本就是上衣，
背景/皮肤/头发/腿都被排除在外。取色仍复用既有 HSV 主色逻辑，只是**喂给它的区域更准**。

稳健性（关键）：监控里人常坐着/半身/遮挡，关键点未必齐全。本模块对缺失/不合理几何
逐级降级：两肩+两胯齐 → 标准躯干；只有肩 → 肩下延一段；肩都没有/几何反常 → 返回 None，
由调用方回落到 Phase 2 的写死比例 torso（即最差也不劣于原来）。

模型：ultralytics YOLO-Pose（默认 yolov8n-pose，6.5MB，CPU 可跑 ~140ms）。与检测用的
yolov8m 是两个独立模型；仅在"画面有人且开启 POSE_COLOR"时才跑，避免无谓开销。
COCO-17 顺序：0鼻 1左眼 2右眼 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘 9左腕 10右腕
            11左胯 12右胯 13左膝 14右膝 15左踝 16右踝。
"""
from __future__ import annotations

import threading

from .config import settings

_model = None
_model_lock = threading.Lock()

# COCO-17 关键点下标
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
NOSE = 0


def _load_model():
    """懒加载 + 单例：进程内只加载一次 Pose 权重（首次自动下载）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO

                _model = YOLO(settings.pose_model)
    return _model


def estimate_persons(image) -> list[dict]:
    """对一张 PIL 图跑 YOLO-Pose，返回每个人的 {box, kpts}。

    Args:
        image: PIL.Image（RGB）。

    Returns:
        list[{"box": [x1,y1,x2,y2], "kpts": [[x,y,conf]*17]}]
    """
    model = _load_model()
    results = model.predict(image, conf=settings.pose_conf, verbose=False)
    r = results[0]
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes.xyxy.cpu().numpy()
    xy = r.keypoints.xy.cpu().numpy()       # (n, 17, 2)
    conf = r.keypoints.conf
    conf = conf.cpu().numpy() if conf is not None else None  # (n, 17)

    persons: list[dict] = []
    for i in range(len(boxes)):
        kpts = []
        for j in range(xy.shape[1]):
            c = float(conf[i][j]) if conf is not None else 0.0
            kpts.append([float(xy[i][j][0]), float(xy[i][j][1]), c])
        persons.append({"box": [float(v) for v in boxes[i][:4]], "kpts": kpts})
    return persons


def torso_region(kpts: list[list[float]], person_box: list[float]) -> tuple[list[float], str] | None:
    """从关键点派生躯干区（上衣区）。返回 (torso_box, source) 或 None（无法可靠派生）。

    source: "pose_full"（肩+胯齐）/ "pose_shoulders"（仅肩，向下外推）。
    返回 None 时，调用方应回落到 Phase 2 的写死比例 torso。
    """
    kc = settings.pose_kpt_conf
    px1, py1, px2, py2 = person_box
    box_h = py2 - py1

    sho = [i for i in (L_SHO, R_SHO) if kpts[i][2] >= kc]
    hip = [i for i in (L_HIP, R_HIP) if kpts[i][2] >= kc]
    if not sho:
        return None  # 连肩都没有 → 无从派生躯干

    sho_y = sum(kpts[i][1] for i in sho) / len(sho)
    # 几何理智检查①：肩膀应在人框的上半部分。若肩点落在框下部（多为低分辨率/坐姿下的
    # 误检，如把桌沿/手当肩），判为不可靠 → 返回 None 回落写死比例 torso，避免取错区域。
    if sho_y - py1 > 0.55 * box_h:
        return None
    if len(sho) == 2:
        sho_w = abs(kpts[L_SHO][0] - kpts[R_SHO][0])
        xmin, xmax = min(kpts[L_SHO][0], kpts[R_SHO][0]), max(kpts[L_SHO][0], kpts[R_SHO][0])
    else:
        sho_w = (px2 - px1) * 0.5
        cx = kpts[sho[0]][0]
        xmin, xmax = cx - sho_w * 0.5, cx + sho_w * 0.5
    if sho_w < 4:
        return None

    pad = sho_w * 0.10  # 肩宽略外扩，纳入两侧衣料
    xmin, xmax = xmin - pad, xmax + pad

    top = sho_y + box_h * 0.02  # 略低于肩线，避开脖子/锁骨阴影
    if hip:
        hip_y = sum(kpts[i][1] for i in hip) / len(hip)
        if hip_y <= sho_y + 4:
            return None  # 胯不低于肩 → 几何反常（误检/倒置），放弃
        bottom = hip_y
        source = "pose_full"
    else:
        bottom = min(py2, sho_y + sho_w * 1.5)  # 无胯：肩下延约 1.5 个肩宽
        source = "pose_shoulders"

    # 收敛到 person 框内
    xmin, xmax = max(px1, xmin), min(px2, xmax)
    top, bottom = max(py1, top), min(py2, bottom)
    # 几何理智检查②：躯干区不能是细条（太薄多半是肩点贴边/误检），否则回落。
    if xmax - xmin < 4 or (bottom - top) < max(4.0, 0.10 * box_h):
        return None
    return [xmin, top, xmax, bottom], source
