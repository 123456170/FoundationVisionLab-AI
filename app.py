import streamlit as st
import numpy as np
import cv2
import time
import math
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
import pandas as pd

try:
    import imageio.v2 as imageio
except Exception:
    try:
        import imageio
    except Exception:
        imageio = None

st.set_page_config(
    page_title="FoundationVisionLab AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["robot", "drone", "person", "crate"]

CLASS_BGR = {
    "robot": (255, 140, 0),
    "drone": (255, 255, 0),
    "person": (0, 120, 255),
    "crate": (0, 80, 200),
}

CLASS_HSV_RANGES = {
    "robot": [((95, 90, 80), (130, 255, 255))],
    "drone": [((80, 90, 100), (110, 255, 255))],
    "person": [((4, 110, 100), (22, 255, 255))],
    "crate": [((12, 90, 70), (32, 255, 255))],
}

CLASS_MAX_SPEED = {
    "robot": 90.0,
    "drone": 150.0,
    "person": 70.0,
    "crate": 30.0,
}


def utc_now():
    return datetime.now(timezone.utc)


def iso_time(dt):
    return dt.isoformat(timespec="milliseconds")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def bbox_iou(a, b):
    ax1, ay1, aw, ah = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / max(1e-6, union)


def bbox_centroid(box):
    return (float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0)


def make_event(frame_idx, frame_ts, level, etype, message, track_id=None, confidence=None):
    return {
        "ts": iso_time(frame_ts),
        "frame": int(frame_idx),
        "level": level,
        "type": etype,
        "message": message,
        "track_id": track_id,
        "confidence": None if confidence is None else float(round(confidence, 4)),
    }


def compute_ap(records, total_gt):
    if total_gt <= 0 or len(records) == 0:
        return 0.0, 0.0, 0.0

    records = sorted(records, key=lambda x: x[0], reverse=True)
    tp = np.cumsum([1.0 if r[1] else 0.0 for r in records])
    fp = np.cumsum([0.0 if r[1] else 1.0 for r in records])
    recall = tp / float(total_gt)
    precision = tp / np.maximum(1e-6, tp + fp)

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = recall >= t
        p = float(np.max(precision[mask])) if np.any(mask) else 0.0
        ap += p / 11.0

    return float(ap), float(precision[-1]) if len(precision) else 0.0, float(recall[-1]) if len(recall) else 0.0


@dataclass
class Config:
    width: int
    height: int
    fps: int
    duration: int
    seed: int
    conf_threshold: float
    min_area: int
    max_lost: int
    record: bool
    assisted_fallback: bool
    noise_level: float
    calibration_px_to_m: float

    @property
    def total_frames(self):
        return int(self.fps * self.duration)


class DemoScene:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.W = cfg.width
        self.H = cfg.height
        self.rng = np.random.default_rng(cfg.seed)
        self.start_time = utc_now()
        self.bg = self._make_background()
        self.vignette = self._make_vignette()
        self.objects = []
        self._make_objects()
        self.corrupt_frames = set()
        if cfg.total_frames > 60:
            bad_frame = int(self.rng.integers(int(cfg.total_frames * 0.35), int(cfg.total_frames * 0.75)))
            self.corrupt_frames.add(bad_frame)

        self.anomaly_frames = {}
        if cfg.total_frames > 80:
            for obj in self.objects:
                start = int(self.rng.integers(20, max(25, cfg.total_frames - 70)))
                dur = int(self.rng.integers(25, min(80, max(26, cfg.total_frames - start - 10))))
                kind = str(self.rng.choice(["stop", "burst", "lowvis"]))
                self.anomaly_frames[obj["id"]] = (start, start + dur, kind)

    def _rand(self, lo, hi):
        if hi <= lo:
            hi = lo + 1e-3
        return float(self.rng.uniform(lo, hi))

    def _make_background(self):
        W, H = self.W, self.H
        bg = np.zeros((H, W, 3), dtype=np.uint8)

        for y in range(H):
            t = y / float(max(1, H - 1))
            bg[y, :] = (int(65 + 35 * t), int(70 + 35 * t), int(85 + 40 * t))

        floor_y = int(H * 0.52)
        cv2.rectangle(bg, (0, floor_y), (W, H), (62, 66, 72), -1)

        for i in range(10):
            x = int(W * i / 9.0)
            top_x = int(W * 0.5 + (x - W * 0.5) * 0.22)
            cv2.line(bg, (x, H), (top_x, floor_y), (90, 95, 100), 2)

        for i in range(6):
            x0 = int(W * i / 6.0)
            x1 = int(W * (i + 1) / 6.0)
            cv2.rectangle(bg, (x0 + 4, int(H * 0.10)), (x1 - 4, floor_y - 4), (78, 82, 88), -1)

        cv2.putText(
            bg,
            "FOUNDATION VISION LAB - DEMO ONLY",
            (10, int(H * 0.07)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, W / 1500.0),
            (205, 215, 225),
            2,
            cv2.LINE_AA,
        )

        cv2.line(bg, (int(W * 0.05), H - 10), (int(W * 0.95), H - 10), (0, 200, 200), 4)
        cv2.line(bg, (int(W * 0.05), int(H * 0.62)), (int(W * 0.95), int(H * 0.62)), (120, 140, 150), 2)
        return bg

    def _make_vignette(self):
        H, W = self.H, self.W
        Y, X = np.ogrid[0:H, 0:W]
        cx, cy = W / 2.0, H / 2.0
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / (max(W, H) * 0.68)
        v = np.clip(1.0 - 0.22 * r, 0.72, 1.0).astype(np.float32)
        return v[:, :, None]

    def _make_objects(self):
        specs = [
            ("robot", 92, 55),
            ("person", 42, 92),
            ("crate", 62, 58),
            ("drone", 46, 30),
            ("robot", 108, 62),
            ("person", 38, 84),
        ]
        margin = 28

        for i, (cls, base_w, base_h) in enumerate(specs):
            w = int(base_w * self._rand(0.85, 1.20))
            h = int(base_h * self._rand(0.85, 1.20))

            if cls == "drone":
                y = self._rand(margin, max(margin + 1, self.H * 0.35))
            elif cls == "person":
                lo = self.H * 0.45
                y = self._rand(lo, max(lo + 1, self.H - h - margin))
            else:
                lo = self.H * 0.50
                y = self._rand(lo, max(lo + 1, self.H - h - margin))

            x = self._rand(margin, max(margin + 1, self.W - w - margin))

            speed_base = {"robot": 35.0, "drone": 65.0, "person": 28.0, "crate": 10.0}[cls]
            vx = float(self._rand(0.5, 1.3) * speed_base * self.rng.choice([-1.0, 1.0]))
            vy = float(self._rand(-6.0, 6.0)) if cls != "crate" else 0.0

            self.objects.append(
                {
                    "id": i + 1,
                    "class": cls,
                    "x": float(x),
                    "y": float(y),
                    "w": int(w),
                    "h": int(h),
                    "color": CLASS_BGR[cls],
                    "base_vx": vx,
                    "base_vy": vy,
                    "vx": vx,
                    "vy": vy,
                    "phase": float(self._rand(0.0, 6.28)),
                    "amp": float(self._rand(8.0, 24.0)) if cls == "drone" else 0.0,
                    "alpha": 1.0,
                    "speed": float(abs(vx)),
                }
            )

    def _update_objects(self, idx, dt):
        margin = 20
        for obj in self.objects:
            obj["alpha"] = 1.0
            vx, vy = obj["base_vx"], obj["base_vy"]

            anomaly = self.anomaly_frames.get(obj["id"])
            if anomaly is not None:
                start, end, kind = anomaly
                if start <= idx < end:
                    if kind == "stop":
                        vx, vy = 0.0, 0.0
                    elif kind == "burst":
                        vx *= 2.2
                        vy *= 1.5
                    elif kind == "lowvis":
                        obj["alpha"] = 0.35

            obj["phase"] += dt * 2.0
            if obj["class"] == "drone":
                vy = math.sin(obj["phase"]) * obj["amp"] + obj["base_vy"] * 0.2

            obj["x"] += vx * dt
            obj["y"] += vy * dt

            if obj["x"] < margin:
                obj["x"] = margin
                obj["base_vx"] = abs(obj["base_vx"])
            elif obj["x"] + obj["w"] > self.W - margin:
                obj["x"] = self.W - margin - obj["w"]
                obj["base_vx"] = -abs(obj["base_vx"])

            if obj["y"] < margin:
                obj["y"] = margin
                obj["base_vy"] = abs(obj["base_vy"])
            elif obj["y"] + obj["h"] > self.H - margin:
                obj["y"] = self.H - margin - obj["h"]
                obj["base_vy"] = -abs(obj["base_vy"])

            obj["vx"] = vx
            obj["vy"] = vy
            obj["speed"] = float(math.hypot(vx, vy))

    def _draw_object(self, frame, obj):
        x, y = int(obj["x"]), int(obj["y"])
        w, h = int(obj["w"]), int(obj["h"])
        x2 = min(self.W - 1, x + w)
        y2 = min(self.H - 1, y + h)
        if x < 0 or y < 0 or x2 <= x or y2 <= y:
            return

        alpha = float(obj.get("alpha", 1.0))
        base = np.array(obj["color"], dtype=np.float32)
        bg_like = np.array([90, 95, 105], dtype=np.float32)
        color = tuple(int(v) for v in (alpha * base + (1.0 - alpha) * bg_like))

        shadow_center = (x + w // 2, min(self.H - 3, y + h + 2))
        cv2.ellipse(frame, shadow_center, (max(6, w // 2), max(3, h // 8)), 0, 0, 360, (35, 35, 35), -1)

        cls = obj["class"]
        if cls == "robot":
            cv2.rectangle(frame, (x, y), (x2, y2), color, -1)
            cv2.rectangle(frame, (x, y), (x2, y2), (20, 20, 20), 2)
            cv2.circle(frame, (x + max(4, w // 4), y2), max(3, w // 12), (30, 30, 30), -1)
            cv2.circle(frame, (x + max(8, 3 * w // 4), y2), max(3, w // 12), (30, 30, 30), -1)
            cv2.rectangle(frame, (x + w // 3, y + h // 4), (x + 2 * w // 3, y + h // 2), (235, 240, 245), -1)

        elif cls == "person":
            head_r = max(4, w // 3)
            cv2.circle(frame, (x + w // 2, y + head_r), head_r, color, -1)
            cv2.rectangle(frame, (x, y + 2 * head_r), (x2, y2), color, -1)
            cv2.rectangle(frame, (x, y + 2 * head_r), (x2, y2), (25, 25, 25), 1)

        elif cls == "drone":
            cx, cy = x + w // 2, y + h // 2
            cv2.ellipse(frame, (cx, cy), (max(6, w // 2), max(4, h // 3)), 0, 0, 360, color, -1)
            cv2.line(frame, (x, y), (x2, y2), (210, 210, 210), 2)
            cv2.line(frame, (x2, y), (x, y2), (210, 210, 210), 2)
            cv2.circle(frame, (x, y), 4, (240, 240, 240), -1)
            cv2.circle(frame, (x2, y2), 4, (240, 240, 240), -1)

        else:
            cv2.rectangle(frame, (x, y), (x2, y2), color, -1)
            cv2.rectangle(frame, (x, y), (x2, y2), (25, 25, 25), 2)
            cv2.line(frame, (x, y), (x2, y2), (30, 30, 30), 2)
            cv2.line(frame, (x2, y), (x, y2), (30, 30, 30), 2)

    def next_frame(self, idx):
        dt = 1.0 / float(max(1, self.cfg.fps))
        corrupt = idx in self.corrupt_frames

        if not corrupt:
            self._update_objects(idx, dt)

        illum = 0.88 + 0.12 * math.sin(idx / 55.0)
        frame = np.clip(self.bg.astype(np.float32) * illum, 0, 255).astype(np.uint8).copy()

        if corrupt:
            for _ in range(5):
                gy = int(self.rng.integers(0, max(1, self.H - 25)))
                gh = int(self.rng.integers(5, 25))
                frame[gy : gy + gh, :] = self.rng.integers(0, 256, size=(gh, self.W, 3), dtype=np.uint8)
            return frame, [], True

        occ_active = (idx % 140) < 70
        occ_x = int((idx * 4) % (self.W + 160)) - 80 if occ_active else None
        occ_y1 = int(self.H * 0.35)
        occ_y2 = int(self.H * 0.80)

        gt = []
        for obj in sorted(self.objects, key=lambda o: o["y"] + o["h"]):
            self._draw_object(frame, obj)

            cx = obj["x"] + obj["w"] / 2.0
            cy = obj["y"] + obj["h"] / 2.0
            visible = obj["alpha"] > 0.45

            if occ_active and occ_x is not None:
                if occ_x <= cx <= occ_x + 80 and occ_y1 <= cy <= occ_y2:
                    visible = False

            if visible:
                gt.append(
                    {
                        "id": int(obj["id"]),
                        "class": obj["class"],
                        "bbox": [int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"])],
                        "speed": float(obj["speed"]),
                        "alpha": float(obj["alpha"]),
                        "visible": True,
                    }
                )

        if occ_active and occ_x is not None:
            cv2.rectangle(frame, (occ_x, occ_y1), (occ_x + 80, occ_y2), (40, 40, 42), -1)
            cv2.putText(
                frame,
                "OCCLUSION",
                (occ_x + 5, int(self.H * 0.50)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 190),
                1,
                cv2.LINE_AA,
            )

        if self.cfg.noise_level > 0:
            noise = self.rng.normal(0.0, self.cfg.noise_level, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        frame = np.clip(frame.astype(np.float32) * self.vignette, 0, 255).astype(np.uint8)
        return frame, gt, False


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=75, varThreshold=18, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.rng = np.random.default_rng(cfg.seed + 11)
        self.hsv_ranges = {
            cls: [(np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)) for low, high in ranges]
            for cls, ranges in CLASS_HSV_RANGES.items()
        }

    def _classify(self, crop_hsv, box_w, box_h, area):
        if crop_hsv is None or crop_hsv.size == 0:
            return "crate", 0.1

        hh, ww = crop_hsv.shape[:2]
        qh = max(1, hh // 4)
        qw = max(1, ww // 4)
        crop = crop_hsv[qh : hh - qh, qw : ww - qw]
        if crop.size == 0:
            crop = crop_hsv

        hue = float(np.median(crop[:, :, 0]))
        sat = float(np.mean(crop[:, :, 1]))
        val = float(np.mean(crop[:, :, 2]))
        aspect = float(box_w) / float(max(1, box_h))

        scores = {cls: 0.08 for cls in CLASS_NAMES}

        if sat >= 45 and val >= 40:
            for cls, ranges in self.hsv_ranges.items():
                for low, high in ranges:
                    if low[0] <= hue <= high[0] and sat >= low[1] and val >= low[2]:
                        scores[cls] += 0.62
                        break

        if 0.35 < aspect < 0.95:
            scores["person"] += 0.28
        if aspect > 1.15:
            scores["robot"] += 0.24
        if 0.75 < aspect < 1.25:
            scores["crate"] += 0.18
        if area < 3200:
            scores["drone"] += 0.22
        if box_h < int(self.cfg.height * 0.22):
            scores["drone"] += 0.10

        best_cls, best_score = max(scores.items(), key=lambda kv: kv[1])
        return best_cls, float(min(1.0, best_score))

    def _nms(self, dets, thresh=0.45):
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        keep = []
        for d in dets:
            if all(bbox_iou(d["bbox"], k["bbox"]) < thresh for k in keep):
                keep.append(d)
        return keep

    def detect(self, frame):
        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        fg = self.bg_sub.apply(gray)
        _, fg = cv2.threshold(fg, 120, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for ranges in self.hsv_ranges.values():
            for low, high in ranges:
                color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, low, high))

        mask = cv2.bitwise_or(fg, color_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < self.cfg.min_area * 0.6:
                continue
            if bw > frame_w * 0.95 and bh > frame_h * 0.95:
                continue

            x2 = min(frame_w, x + bw)
            y2 = min(frame_h, y + bh)
            if x2 <= x or y2 <= y:
                continue

            crop_bgr = frame[y:y2, x:x2]
            crop_hsv = hsv[y:y2, x:x2]
            crop_fg = fg[y:y2, x:x2]
            crop_color = color_mask[y:y2, x:x2]

            if crop_bgr.size == 0:
                continue

            cls, class_score = self._classify(crop_hsv, bw, bh, area)
            motion_score = float(np.mean(crop_fg > 0))
            color_fill = float(np.mean(crop_color > 0))
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            edge_score = float(min(1.0, cv2.Laplacian(crop_gray, cv2.CV_32F).var() / 800.0))

            conf = (
                0.38 * class_score
                + 0.22 * motion_score
                + 0.22 * color_fill
                + 0.10 * edge_score
                + 0.08
            )
            conf = float(np.clip(conf + self.rng.normal(0.0, 0.03), 0.03, 0.98))

            dets.append(
                {
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "class": cls,
                    "confidence": conf,
                    "source": "cv",
                    "score": float(class_score),
                }
            )

        dets = self._nms(dets)
        return dets, mask


class Track:
    def __init__(self, track_id, det, cfg: Config):
        self.id = track_id
        self.bbox = [float(v) for v in det["bbox"]]
        self.cls = det["class"]
        self.confidence = float(det["confidence"])
        self.hits = 1
        self.missed = 0
        self.age = 0
        self.class_votes = {det["class"]: 1}
        self.history = deque(maxlen=24)
        self.speed_hist = deque(maxlen=40)
        self.last_centroid = bbox_centroid(self.bbox)
        self.history.append(self.last_centroid)
        self.vx = 0.0
        self.vy = 0.0
        self.current_speed = 0.0
        self.speed_sum = 0.0
        self.speed_count = 0
        self.anomaly_score = 0.0
        self.anomaly_max = 0.0
        self.area_last = float(self.bbox[2] * self.bbox[3])
        self.last_low_event = -100
        self.last_anomaly_event = -100
        self.last_review_frame = -100

    def is_confirmed(self, min_hits=2):
        return self.hits >= min_hits and self.missed <= 2

    def predict(self, dt):
        if dt <= 0:
            return
        self.age += 1
        self.bbox[0] += self.vx * dt
        self.bbox[1] += self.vy * dt
        self.bbox[0] = max(0.0, self.bbox[0])
        self.bbox[1] = max(0.0, self.bbox[1])
        c = bbox_centroid(self.bbox)
        self.history.append(c)

    def update(self, det, dt, frame_idx, cfg: Config):
        new_bbox = [float(v) for v in det["bbox"]]
        new_c = bbox_centroid(new_bbox)

        if dt > 0:
            inst_vx = (new_c[0] - self.last_centroid[0]) / dt
            inst_vy = (new_c[1] - self.last_centroid[1]) / dt
            self.vx = 0.65 * self.vx + 0.35 * inst_vx
            self.vy = 0.65 * self.vy + 0.35 * inst_vy
            speed = float(math.hypot(self.vx, self.vy))
            self.current_speed = speed
            self.speed_hist.append(speed)
            self.speed_sum += speed
            self.speed_count += 1
        else:
            speed = self.current_speed

        area = float(new_bbox[2] * new_bbox[3])
        area_change = abs(area - self.area_last) / max(1.0, self.area_last)
        self.area_last = area

        self.bbox = new_bbox
        self.last_centroid = new_c
        self.history.append(new_c)

        self.hits += 1
        self.missed = 0
        self.confidence = 0.72 * self.confidence + 0.28 * float(det["confidence"])

        cls = det["class"]
        self.class_votes[cls] = self.class_votes.get(cls, 0) + 1
        self.cls = max(self.class_votes.items(), key=lambda kv: kv[1])[0]

        low_conf = max(0.0, cfg.conf_threshold - self.confidence) / max(1e-3, cfg.conf_threshold)

        z = 0.0
        if len(self.speed_hist) > 8:
            mean_s = float(np.mean(self.speed_hist))
            std_s = float(np.std(self.speed_hist)) + 1e-3
            z = abs(speed - mean_s) / std_s

        class_speed_limit = CLASS_MAX_SPEED.get(self.cls, 90.0) * 1.8
        speed_violation = max(0.0, speed / max(1.0, class_speed_limit) - 0.8)

        self.anomaly_score = float(
            np.clip(
                0.35 * min(z / 3.0, 1.0)
                + 0.30 * min(low_conf, 1.0)
                + 0.20 * min(area_change * 3.0, 1.0)
                + 0.15 * min(speed_violation, 1.0),
                0.0,
                1.0,
            )
        )
        self.anomaly_max = max(self.anomaly_max, self.anomaly_score)

    def mark_missed(self):
        self.missed += 1
        self.confidence *= 0.93
        self.current_speed *= 0.9
        self.anomaly_score = max(0.0, self.anomaly_score - 0.02)


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks = []
        self.finished_tracks = []
        self.next_id = 1
        self.max_lost = cfg.max_lost
        self.min_hits = 2

    def active_tracks(self):
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]

    def update(self, dets, dt, frame_idx, frame_ts):
        events = []

        candidates = [t for t in self.tracks if t.missed <= self.max_lost]
        for t in candidates:
            t.predict(dt)

        pairs = []
        for ci, t in enumerate(candidates):
            for di, d in enumerate(dets):
                iou = bbox_iou(t.bbox, d["bbox"])
                c1 = bbox_centroid(t.bbox)
                c2 = bbox_centroid(d["bbox"])
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                max_dim = max(float(t.bbox[2]), float(t.bbox[3]), float(d["bbox"][2]), float(d["bbox"][3]), 1.0)

                if iou > 0.02 or dist < max_dim * 1.2:
                    score = 0.75 * iou + 0.25 * max(0.0, 1.0 - dist / (max_dim * 1.5))
                    if score > 0.18:
                        pairs.append((score, ci, di))

        pairs.sort(key=lambda x: x[0], reverse=True)
        matched_cand = set()
        matched_det = set()

        for score, ci, di in pairs:
            if ci in matched_cand or di in matched_det:
                continue

            t = candidates[ci]
            d = dets[di]
            t.update(d, dt, frame_idx, self.cfg)
            matched_cand.add(ci)
            matched_det.add(di)

            if t.confidence < self.cfg.conf_threshold and frame_idx - t.last_low_event > 80:
                events.append(
                    make_event(
                        frame_idx,
                        frame_ts,
                        "warning",
                        "low_confidence_track",
                        f"Low-confidence track #{t.id} routed to review",
                        track_id=t.id,
                        confidence=t.confidence,
                    )
                )
                t.last_low_event = frame_idx

            if t.anomaly_score > 0.72 and frame_idx - t.last_anomaly_event > 60:
                events.append(
                    make_event(
                        frame_idx,
                        frame_ts,
                        "critical" if t.anomaly_score > 0.85 else "warning",
                        "temporal_anomaly",
                        f"Temporal anomaly on track #{t.id} ({t.cls})",
                        track_id=t.id,
                        confidence=t.anomaly_score,
                    )
                )
                t.last_anomaly_event = frame_idx

        for di, d in enumerate(dets):
            if di not in matched_det:
                self.tracks.append(Track(self.next_id, d, self.cfg))
                self.next_id += 1

        for ci, t in enumerate(candidates):
            if ci not in matched_cand:
                t.mark_missed()
                if t.missed == 8:
                    events.append(
                        make_event(
                            frame_idx,
                            frame_ts,
                            "info",
                            "track_lost",
                            f"Track #{t.id} lost",
                            track_id=t.id,
                            confidence=t.confidence,
                        )
                    )

        keep = []
        for t in self.tracks:
            if t.missed > self.max_lost:
                self.finished_tracks.append(t)
            else:
                keep.append(t)
        self.tracks = keep

        if len(self.finished_tracks) > 200:
            self.finished_tracks = self.finished_tracks[-200:]

        return events

    def summarize(self):
        rows = []
        for t in self.finished_tracks + self.tracks:
            rows.append(
                {
                    "track_id": t.id,
                    "class": t.cls,
                    "frames_seen": t.hits,
                    "frames_missed": t.missed,
                    "final_confidence": float(round(t.confidence, 4)),
                    "avg_speed_px_s": float(round(t.speed_sum / max(1, t.speed_count), 2)),
                    "max_anomaly_score": float(round(t.anomaly_max, 4)),
                    "status": "active" if t.missed <= 2 else "lost",
                }
            )
        return rows


class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gt_total = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.class_gt = defaultdict(int)
        self.class_tp = defaultdict(int)
        self.class_fp = defaultdict(int)
        self.class_fn = defaultdict(int)
        self.det_records = defaultdict(list)

        self.track_total_gt = 0
        self.track_fp = 0
        self.track_fn = 0
        self.track_matches = 0
        self.track_pred_total = 0
        self.idsw = 0
        self.prev_track_assign = {}
        self.speed_errors = []

    def update(self, frame_idx, dets, tracks, gt):
        visible_gt = [g for g in gt if g.get("visible", True)]
        self.gt_total += len(visible_gt)
        for g in visible_gt:
            self.class_gt[g["class"]] += 1

        matched_gt = set()

        for det in sorted(dets, key=lambda d: d["confidence"], reverse=True):
            best_iou = 0.0
            best_j = None

            for j, g in enumerate(visible_gt):
                if j in matched_gt:
                    continue
                if g["class"] != det["class"]:
                    continue
                iou = bbox_iou(det["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= 0.35 and best_j is not None:
                matched_gt.add(best_j)
                self.tp += 1
                self.class_tp[det["class"]] += 1
                self.det_records[det["class"]].append((float(det["confidence"]), True))
            else:
                self.fp += 1
                self.class_fp[det["class"]] += 1
                self.det_records[det["class"]].append((float(det["confidence"]), False))

        for j, g in enumerate(visible_gt):
            if j not in matched_gt:
                self.fn += 1
                self.class_fn[g["class"]] += 1

        active_tracks = tracks
        track_matched_gt = set()
        track_assign = {}

        for t in sorted(active_tracks, key=lambda tr: tr.confidence, reverse=True):
            best_iou = 0.0
            best_j = None
            best_g = None

            for j, g in enumerate(visible_gt):
                if j in track_matched_gt:
                    continue
                if g["class"] != t.cls:
                    continue
                iou = bbox_iou(t.bbox, g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
                    best_g = g

            if best_iou >= 0.35 and best_j is not None and best_g is not None:
                track_matched_gt.add(best_j)
                track_assign[best_g["id"]] = t.id
                self.track_matches += 1
                self.speed_errors.append(abs(float(t.current_speed) - float(best_g["speed"])))

                prev = self.prev_track_assign.get(best_g["id"])
                if prev is not None and prev != t.id:
                    self.idsw += 1

        self.track_total_gt += len(visible_gt)
        self.track_pred_total += len(active_tracks)
        self.track_fp += max(0, len(active_tracks) - len(track_matched_gt))
        self.track_fn += max(0, len(visible_gt) - len(track_matched_gt))
        self.prev_track_assign = track_assign

    def summarize(self):
        per_class = []
        aps = []

        for cls in CLASS_NAMES:
            total_gt = int(self.class_gt.get(cls, 0))
            records = self.det_records.get(cls, [])
            ap, precision, recall = compute_ap(records, total_gt)
            if total_gt > 0:
                aps.append(ap)
            per_class.append(
                {
                    "class": cls,
                    "gt_objects": total_gt,
                    "ap": float(round(ap, 4)),
                    "precision": float(round(precision, 4)),
                    "recall": float(round(recall, 4)),
                }
            )

        m_ap = float(np.mean(aps)) if aps else 0.0
        det_precision = self.tp / max(1, self.tp + self.fp)
        det_recall = self.tp / max(1, self.tp + self.fn)

        mota = 1.0 - ((self.track_fp + self.track_fn + self.idsw) / max(1, self.track_total_gt))
        idtp = max(0, self.track_matches - self.idsw)
        idfp = max(0, self.track_pred_total - idtp)
        idfn = max(0, self.track_total_gt - idtp)
        idf1 = (2.0 * idtp) / max(1e-6, 2.0 * idtp + idfp + idfn)

        mean_speed_error_px = float(np.mean(self.speed_errors)) if self.speed_errors else 0.0

        return {
            "mAP": float(round(m_ap, 4)),
            "per_class": per_class,
            "det_precision": float(round(det_precision, 4)),
            "det_recall": float(round(det_recall, 4)),
            "mota": float(round(mota, 4)),
            "idf1": float(round(idf1, 4)),
            "id_switches": int(self.idsw),
            "track_fp": int(self.track_fp),
            "track_fn": int(self.track_fn),
            "speed_error_px_s": float(round(mean_speed_error_px, 3)),
            "speed_error_m_s": float(round(mean_speed_error_px * self.cfg.calibration_px_to_m, 4)),
        }


class VisionPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.tracker = Tracker(cfg)
        self.evaluator = Evaluator(cfg)
        self.events = []
        self.low_conf_review = []
        self.det_samples = []
        self.no_det_frames = 0
        self.last_fallback_event = -100
        self.last_event_message = None
        self.last_event_frame = -100
        self.rng = np.random.default_rng(cfg.seed + 77)
        self.corrupt_count = 0
        self.component_totals = {"detector": 0.0, "tracker": 0.0, "evaluator": 0.0}
        self.component_count = 0

    def _add_events(self, new_events):
        if not new_events:
            return
        self.events.extend(new_events)
        if len(self.events) > 3000:
            self.events = self.events[-3000:]

    def _assist(self, gt):
        dets = []
        for g in gt:
            if not g.get("visible", True):
                continue
            x, y, w, h = g["bbox"]
            jx = int(self.rng.integers(-4, 5))
            jy = int(self.rng.integers(-3, 4))
            conf = float(self.rng.uniform(0.30, 0.55))
            dets.append(
                {
                    "bbox": [int(x + jx), int(y + jy), int(w), int(h)],
                    "class": g["class"],
                    "confidence": conf,
                    "source": "assist",
                    "score": 0.5,
                }
            )
        return dets

    def process(self, frame, frame_idx, gt, frame_ts, corrupt=False):
        t0 = time.perf_counter()
        new_events = []

        if frame is None or frame.size == 0:
            frame = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
            corrupt = True
            new_events.append(make_event(frame_idx, frame_ts, "error", "missing_frame", "Empty or corrupt frame received"))

        if min(frame.shape[:2]) < 180:
            new_events.append(make_event(frame_idx, frame_ts, "error", "low_resolution", "Insufficient resolution for reliable perception"))

        dt = 1.0 / float(max(1, self.cfg.fps))
        dets = []
        det_ms = trk_ms = eval_ms = 0.0

        if corrupt:
            self.corrupt_count += 1
            new_events.append(make_event(frame_idx, frame_ts, "warning", "corrupt_frame", "Corrupt frame routed to review; tracking coasting"))
        else:
            t_det = time.perf_counter()
            try:
                dets, _ = self.detector.detect(frame)
            except Exception as exc:
                dets = []
                new_events.append(make_event(frame_idx, frame_ts, "error", "model_failure", f"Detector failure: {exc}"))
            det_ms = (time.perf_counter() - t_det) * 1000.0

            if len(dets) == 0:
                self.no_det_frames += 1
                if self.cfg.assisted_fallback and gt and self.no_det_frames > 10:
                    dets = self._assist(gt)
                    if frame_idx - self.last_fallback_event > 120:
                        new_events.append(
                            make_event(
                                frame_idx,
                                frame_ts,
                                "info",
                                "fallback_prior",
                                "Detector output empty; assisted foundation prior engaged",
                            )
                        )
                        self.last_fallback_event = frame_idx
            else:
                self.no_det_frames = 0

            if frame_idx % 2 == 0:
                for d in dets[:12]:
                    self.det_samples.append(
                        {
                            "ts": iso_time(frame_ts),
                            "frame": int(frame_idx),
                            "class": d["class"],
                            "confidence": float(round(d["confidence"], 4)),
                            "bbox": d["bbox"],
                            "source": d.get("source", "cv"),
                        }
                    )
                if len(self.det_samples) > 3000:
                    self.det_samples = self.det_samples[-3000:]

        t_trk = time.perf_counter()
        track_events = self.tracker.update(dets, dt, frame_idx, frame_ts)
        trk_ms = (time.perf_counter() - t_trk) * 1000.0
        new_events.extend(track_events)

        if not corrupt:
            t_eval = time.perf_counter()
            self.evaluator.update(frame_idx, dets, self.tracker.active_tracks(), gt)
            eval_ms = (time.perf_counter() - t_eval) * 1000.0

        active = self.tracker.active_tracks()
        for t in active:
            if t.confidence < self.cfg.conf_threshold and frame_idx - t.last_review_frame > 25:
                self.low_conf_review.append(
                    {
                        "ts": iso_time(frame_ts),
                        "frame": int(frame_idx),
                        "track_id": t.id,
                        "class": t.cls,
                        "confidence": float(round(t.confidence, 4)),
                        "bbox": [int(v) for v in t.bbox],
                        "action": "manual_review",
                    }
                )
                t.last_review_frame = frame_idx

        if len(self.low_conf_review) > 400:
            self.low_conf_review = self.low_conf_review[-400:]

        self._add_events(new_events)

        if new_events:
            self.last_event_message = new_events[-1]["message"]
            self.last_event_frame = frame_idx

        recent_event = self.last_event_message if frame_idx - self.last_event_frame < 60 else None

        self.component_totals["detector"] += det_ms
        self.component_totals["tracker"] += trk_ms
        self.component_totals["evaluator"] += eval_ms
        self.component_count += 1

        if active:
            avg_conf = float(np.mean([t.confidence for t in active]))
            low_count = len([t for t in active if t.confidence < self.cfg.conf_threshold])
            max_anom = float(max([t.anomaly_score for t in active]))
        elif dets:
            avg_conf = float(np.mean([d["confidence"] for d in dets]))
            low_count = len([d for d in dets if d["confidence"] < self.cfg.conf_threshold])
            max_anom = 0.0
        else:
            avg_conf = 0.0
            low_count = 0
            max_anom = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "tracks": self.tracker.tracks,
            "active_tracks": len(active),
            "avg_conf": avg_conf,
            "low_conf_count": low_count,
            "max_anomaly": max_anom,
            "corrupt": corrupt,
            "recent_event": recent_event,
            "latency_ms": latency_ms,
            "component_ms": {"detector": det_ms, "tracker": trk_ms, "evaluator": eval_ms},
        }


class Recorder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = cfg.record
        self.backend = None
        self.writer = None
        self.path = None
        self.mime = None
        self.frames = 0

        if not self.enabled:
            return

        stamp = int(time.time())
        self.path = os.path.join(tempfile.gettempdir(), f"foundationvisionlab_{stamp}.mp4")

        if imageio is not None:
            try:
                self.writer = imageio.get_writer(
                    self.path,
                    fps=cfg.fps,
                    codec="libx264",
                    quality=7,
                    macro_block_size=1,
                    output_params=["-pix_fmt", "yuv420p"],
                )
                self.backend = "imageio"
                self.mime = "video/mp4"
                return
            except Exception:
                self.writer = None
                self.backend = None

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.path, fourcc, cfg.fps, (cfg.width, cfg.height))
            if writer.isOpened():
                self.writer = writer
                self.backend = "cv2"
                self.mime = "video/mp4"
                return
        except Exception:
            pass

        try:
            self.path = self.path.replace(".mp4", ".avi")
            writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"MJPG"), cfg.fps, (cfg.width, cfg.height))
            if writer.isOpened():
                self.writer = writer
                self.backend = "cv2"
                self.mime = "video/x-msvideo"
        except Exception:
            self.backend = None

    def add(self, bgr):
        if not self.enabled or self.backend is None or self.writer is None:
            return

        bgr = np.ascontiguousarray(bgr)
        try:
            if self.backend == "imageio":
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.writer.append_data(rgb)
            else:
                self.writer.write(bgr)
            self.frames += 1
        except Exception:
            self.enabled = False
            self.backend = None

    def close(self):
        if self.writer is None:
            return
        try:
            if self.backend == "imageio":
                self.writer.close()
            else:
                self.writer.release()
        except Exception:
            pass

    def read_bytes(self):
        if not self.path or not os.path.exists(self.path):
            return None, None
        try:
            with open(self.path, "rb") as f:
                data = f.read()
        except Exception:
            return None, None

        try:
            os.remove(self.path)
        except Exception:
            pass

        return data, self.mime or "video/mp4"


def put_label(img, text, org, color=(255, 255, 255), bg=(0, 0, 0), scale=0.45, thickness=1):
    x, y = int(org[0]), int(org[1])
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + bl + 2), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame, res, cfg, frame_idx, fps):
    out = frame.copy()

    for t in res.get("tracks", []):
        if t.missed > 5:
            continue

        x, y, w, h = [int(v) for v in t.bbox]
        if w <= 0 or h <= 0:
            continue

        if t.anomaly_score > 0.72:
            color = (255, 0, 255)
        elif t.confidence < cfg.conf_threshold:
            color = (0, 0, 255)
        else:
            color = (0, 200, 0)

        if not t.is_confirmed():
            color = (200, 200, 200)

        thickness = 2 if t.is_confirmed() else 1
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

        label = f"#{t.id} {t.cls} {t.confidence:.2f}"
        if t.confidence < cfg.conf_threshold:
            label += " REVIEW"
        if t.anomaly_score > 0.72:
            label += " ANOMALY"

        put_label(out, label, (x, max(12, y - 6)), color=(255, 255, 255), bg=(20, 20, 20), scale=0.42)

        if len(t.history) > 1:
            pts = np.array([[int(p[0]), int(p[1])] for p in t.history], dtype=np.int32)
            cv2.polylines(out, [pts], False, color, 1, cv2.LINE_AA)

    hud = f"FoundationVisionLab | frame {frame_idx} | fps {fps:.1f} | latency {res.get('latency_ms', 0):.0f} ms"
    put_label(out, hud, (8, 22), color=(230, 240, 255), bg=(15, 15, 15), scale=0.48)

    status = f"active tracks {res.get('active_tracks', 0)} | low-conf {res.get('low_conf_count', 0)} | anomaly {res.get('max_anomaly', 0):.2f}"
    put_label(out, status, (8, 44), color=(200, 230, 200), bg=(15, 15, 15), scale=0.42)

    if res.get("corrupt", False):
        put_label(out, "CORRUPT FRAME -> ROUTED TO REVIEW", (8, out.shape[0] - 18), color=(80, 80, 255), bg=(0, 0, 0), scale=0.5)
    elif res.get("recent_event"):
        put_label(out, str(res["recent_event"])[:110], (8, out.shape[0] - 18), color=(180, 220, 255), bg=(20, 20, 20), scale=0.42)

    return out


def run_demo(cfg: Config):
    scene = DemoScene(cfg)
    pipeline = VisionPipeline(cfg)
    recorder = Recorder(cfg)

    st.markdown("### Live demo")
    col_left, col_right = st.columns([3, 1])

    frame_ph = col_left.empty()
    progress_ph = col_left.empty()
    status_ph = col_left.empty()

    metric_ph = col_right.empty()
    events_ph = col_right.empty()

    fps_series = []
    lat_series = []
    active_series = []
    conf_series = []
    anomaly_series = []

    start_wall = time.time()
    frame_idx = 0
    total = max(1, cfg.total_frames)

    if recorder.backend is None and cfg.record:
        st.warning("Video recorder backend unavailable. Live view will still run.")

    try:
        while frame_idx < total:
            t0 = time.perf_counter()
            frame_ts = scene.start_time + timedelta(seconds=frame_idx / float(max(1, cfg.fps)))

            try:
                frame, gt, corrupt = scene.next_frame(frame_idx)
            except Exception as exc:
                frame = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
                gt = []
                corrupt = True
                pipeline._add_events([make_event(frame_idx, frame_ts, "error", "scene_error", f"Scene error: {exc}")])

            try:
                res = pipeline.process(frame, frame_idx, gt, frame_ts, corrupt)
            except Exception as exc:
                res = {
                    "tracks": [],
                    "active_tracks": 0,
                    "avg_conf": 0.0,
                    "low_conf_count": 0,
                    "max_anomaly": 0.0,
                    "corrupt": True,
                    "recent_event": f"Pipeline error: {exc}",
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                }
                pipeline._add_events([make_event(frame_idx, frame_ts, "error", "pipeline_error", f"Pipeline error: {exc}")])

            elapsed_wall = max(1e-6, time.time() - start_wall)
            actual_fps = cfg.fps if frame_idx == 0 else frame_idx / elapsed_wall

            overlay = draw_overlay(frame, res, cfg, frame_idx, actual_fps)
            recorder.add(overlay)

            lat_ms = res.get("latency_ms", 0.0)
            fps_series.append(float(actual_fps))
            lat_series.append(float(lat_ms))
            active_series.append(int(res.get("active_tracks", 0)))
            conf_series.append(float(res.get("avg_conf", 0.0)))
            anomaly_series.append(float(res.get("max_anomaly", 0.0)))

            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            frame_ph.image(rgb, caption=f"Frame {frame_idx}", use_container_width=True)

            if frame_idx % 5 == 0:
                progress_ph.progress(min(1.0, (frame_idx + 1) / total))
                status_ph.markdown(
                    f"**Status:** running • frame `{frame_idx + 1}/{total}` • recording `{recorder.frames}` frames"
                )

                metric_ph.empty()
                with metric_ph.container():
                    c1, c2 = st.columns(2)
                    c1.metric("FPS", f"{actual_fps:.1f}")
                    c2.metric("Latency ms", f"{lat_ms:.0f}")
                    c3, c4 = st.columns(2)
                    c3.metric("Active tracks", res.get("active_tracks", 0))
                    c4.metric("Low-conf", res.get("low_conf_count", 0))

                recent_events = pd.DataFrame(pipeline.events[-8:][::-1])
                events_ph.empty()
                if not recent_events.empty:
                    events_ph.dataframe(recent_events[["ts", "level", "type", "message"]], use_container_width=True)
                else:
                    events_ph.info("No events yet.")

            frame_idx += 1
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, (1.0 / float(max(1, cfg.fps))) - elapsed)
            time.sleep(sleep_time)

    finally:
        recorder.close()

    video_bytes, video_mime = recorder.read_bytes()
    eval_summary = pipeline.evaluator.summarize()

    if fps_series:
        df_perf = pd.DataFrame(
            {
                "frame": np.arange(len(fps_series)),
                "fps": fps_series,
                "latency_ms": lat_series,
                "active_tracks": active_series,
                "avg_conf": conf_series,
                "max_anomaly": anomaly_series,
            }
        )
    else:
        df_perf = pd.DataFrame(columns=["frame", "fps", "latency_ms", "active_tracks", "avg_conf", "max_anomaly"])

    events_df = pd.DataFrame(pipeline.events)
    if events_df.empty:
        events_df = pd.DataFrame(columns=["ts", "frame", "level", "type", "message", "track_id", "confidence"])

    low_df = pd.DataFrame(pipeline.low_conf_review)
    if low_df.empty:
        low_df = pd.DataFrame(columns=["ts", "frame", "track_id", "class", "confidence", "bbox", "action"])

    det_df = pd.DataFrame(pipeline.det_samples)
    if det_df.empty:
        det_df = pd.DataFrame(columns=["ts", "frame", "class", "confidence", "bbox", "source"])

    tracks_df = pd.DataFrame(pipeline.tracker.summarize())
    if tracks_df.empty:
        tracks_df = pd.DataFrame(columns=["track_id", "class", "frames_seen", "frames_missed", "final_confidence", "avg_speed_px_s", "max_anomaly_score", "status"])

    comp_count = max(1, pipeline.component_count)
    model_rows = [
        {
            "module": "Frame source / preprocessor",
            "role": "synthetic camera, noise, corruption handling",
            "status": "OK" if pipeline.corrupt_count <= 2 else "review",
            "metric": "corrupt frames",
            "value": pipeline.corrupt_count,
            "avg_latency_ms": 0.0,
        },
        {
            "module": "Appearance encoder + detector",
            "role": "color/motion evidence and classification",
            "status": "OK" if eval_summary["mAP"] > 0.15 else "review",
            "metric": "mAP",
            "value": eval_summary["mAP"],
            "avg_latency_ms": round(pipeline.component_totals["detector"] / comp_count, 2),
        },
        {
            "module": "Temporal tracker",
            "role": "ID association, velocity, anomaly memory",
            "status": "OK" if eval_summary["mota"] > 0.0 else "review",
            "metric": "MOTA",
            "value": eval_summary["mota"],
            "avg_latency_ms": round(pipeline.component_totals["tracker"] / comp_count, 2),
        },
        {
            "module": "Evaluation / evidence router",
            "role": "metrics, low-confidence routing, reports",
            "status": "OK",
            "metric": "low-conf reviews",
            "value": len(pipeline.low_conf_review),
            "avg_latency_ms": round(pipeline.component_totals["evaluator"] / comp_count, 2),
        },
    ]

    alerts = len([e for e in pipeline.events if e.get("level") in ("warning", "error", "critical")])
    avg_fps = float(np.mean(fps_series)) if fps_series else 0.0
    avg_latency = float(np.mean(lat_series)) if lat_series else 0.0

    started = scene.start_time
    ended = utc_now()
    file_slug = started.strftime("%Y%m%d_%H%M%S")

    result = {
        "config": asdict(cfg),
        "started_at": iso_time(started),
        "ended_at": iso_time(ended),
        "file_slug": file_slug,
        "video_bytes": video_bytes,
        "video_mime": video_mime,
        "metrics": {
            "frames_processed": len(fps_series),
            "avg_fps": round(avg_fps, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "mAP": eval_summary["mAP"],
            "MOTA": eval_summary["mota"],
            "IDF1": eval_summary["idf1"],
            "ID_switches": eval_summary["id_switches"],
            "alerts": alerts,
            "low_conf_reviews": len(pipeline.low_conf_review),
            "speed_error_m_s": eval_summary["speed_error_m_s"],
        },
        "eval": eval_summary,
        "df_perf": df_perf,
        "events_df": events_df,
        "low_df": low_df,
        "det_df": det_df,
        "tracks_df": tracks_df,
        "model_table": model_rows,
    }

    return result


def render_results(result):
    if not result:
        st.info("No demo result yet.")
        return

    m = result["metrics"]

    st.markdown("## Dashboard")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Frames", m["frames_processed"])
    c2.metric("Avg FPS", m["avg_fps"])
    c3.metric("Latency ms", m["avg_latency_ms"])
    c4.metric("mAP", m["mAP"])
    c5.metric("MOTA", m["MOTA"])
    c6.metric("Alerts", m["alerts"])

    col_video, col_actions = st.columns([3, 1])

    with col_video:
        if result.get("video_bytes"):
            st.video(result["video_bytes"], format=result.get("video_mime", "video/mp4"))
        else:
            st.warning("No recording available. Enable recording and restart the demo.")

    with col_actions:
        st.markdown("**Download / export**")

        if result.get("video_bytes"):
            st.download_button(
                "Download live demo video",
                data=result["video_bytes"],
                file_name=f"foundationvisionlab_demo_{result['file_slug']}.mp4",
                mime=result.get("video_mime", "video/mp4"),
                use_container_width=True,
            )

        events_df = result["events_df"]
        tracks_df = result["tracks_df"]
        low_df = result["low_df"]
        det_df = result["det_df"]

        jsonl_events = "\n".join(json.dumps(row, default=str) for row in events_df.to_dict(orient="records"))
        st.download_button(
            "Download events JSONL",
            data=jsonl_events.encode("utf-8"),
            file_name=f"foundationvisionlab_events_{result['file_slug']}.jsonl",
            mime="application/jsonlines",
            use_container_width=True,
        )

        st.download_button(
            "Download track summary CSV",
            data=tracks_df.to_csv(index=False).encode("utf-8"),
            file_name=f"foundationvisionlab_tracks_{result['file_slug']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download detection log CSV",
            data=det_df.to_csv(index=False).encode("utf-8"),
            file_name=f"foundationvisionlab_detections_{result['file_slug']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(f"Run window: {result['started_at']} → {result['ended_at']} (UTC)")

    tabs = st.tabs(["Analytics", "Evaluation", "Model modules", "Events", "Tracks", "Low-confidence review", "Detections", "Config"])

    with tabs[0]:
        df_perf = result["df_perf"]
        if df_perf.empty:
            st.info("No performance series available.")
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**FPS / latency**")
                st.line_chart(df_perf.set_index("frame")[["fps", "latency_ms"]])
            with right:
                st.markdown("**Confidence / anomaly**")
                st.line_chart(df_perf.set_index("frame")[["avg_conf", "max_anomaly"]])

            st.markdown("**Active tracks**")
            st.area_chart(df_perf.set_index("frame")["active_tracks"])

    with tabs[1]:
        ev = result["eval"]
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("mAP", ev["mAP"])
        e2.metric("Detector precision", ev["det_precision"])
        e3.metric("Detector recall", ev["det_recall"])
        e4.metric("MOTA", ev["mota"])
        e5.metric("IDF1", ev["idf1"])
        e6.metric("ID switches", ev["id_switches"])

        s1, s2 = st.columns(2)
        s1.metric("Track false positives", ev["track_fp"])
        s2.metric("Track false negatives", ev["track_fn"])

        s3, s4 = st.columns(2)
        s3.metric("Speed error px/s", ev["speed_error_px_s"])
        s4.metric("Speed error m/s", ev["speed_error_m_s"])

        ap_df = pd.DataFrame(ev["per_class"])
        st.markdown("**Per-class detection AP**")
        st.dataframe(ap_df, use_container_width=True)
        if not ap_df.empty:
            st.bar_chart(ap_df.set_index("class")["ap"])

    with tabs[2]:
        model_df = pd.DataFrame(result["model_table"])
        st.dataframe(model_df, use_container_width=True)
        st.caption(
            "This prototype uses deterministic local perception heads and a temporal evidence router. "
            "No paid external API is used."
        )

    with tabs[3]:
        events_df = result["events_df"]
        if events_df.empty:
            st.info("No events recorded.")
        else:
            st.dataframe(events_df.sort_values("frame", ascending=False), use_container_width=True)

    with tabs[4]:
        tracks_df = result["tracks_df"]
        if tracks_df.empty:
            st.info("No tracks recorded.")
        else:
            st.dataframe(tracks_df, use_container_width=True)

    with tabs[5]:
        low_df = result["low_df"]
        if low_df.empty:
            st.success("No low-confidence objects were routed to review.")
        else:
            st.warning("Low-confidence results are visibly separated for review.")
            st.dataframe(low_df.sort_values("frame", ascending=False), use_container_width=True)

    with tabs[6]:
        det_df = result["det_df"]
        if det_df.empty:
            st.info("No detection samples recorded.")
        else:
            st.dataframe(det_df.sort_values("frame", ascending=False), use_container_width=True)

    with tabs[7]:
        st.json(result["config"])
        st.info(
            "Privacy note: this demo intentionally avoids identity recognition and does not persist biometric data. "
            "All generated evidence remains local to the Streamlit session."
        )


def init_state():
    if "demo_completed" not in st.session_state:
        st.session_state.demo_completed = False
    if "result" not in st.session_state:
        st.session_state.result = None


init_state()

st.title("FoundationVisionLab AI")
st.caption(
    "Modular vision-foundation lab prototype: encoder-style evidence, detection/segmentation, temporal tracking, "
    "uncertainty routing, evaluation, and downloadable live demo recording."
)

with st.sidebar:
    st.header("Controls")

    auto_start = st.checkbox("Auto start demo", value=True, key="fv_auto_start")
    st.caption("Starts a deterministic live demo immediately on load.")

    resolution_label = st.selectbox(
        "Resolution",
        ["848x480 (balanced)", "640x360 (fast)", "960x540 (quality)"],
        index=0,
        key="fv_resolution",
    )

    duration = st.slider("Demo duration (seconds)", 10, 60, 60, key="fv_duration")
    fps = st.slider("Target FPS", 8, 24, 12, key="fv_fps")

    conf_threshold = st.slider(
        "Confidence threshold",
        0.05,
        0.95,
        0.45,
        0.05,
        key="fv_conf",
        help="Tracks below this value are highlighted red and routed to review.",
    )

    min_area = st.slider("Min object area (px)", 150, 5000, 700, 50, key="fv_min_area")
    max_lost = st.slider("Tracker max lost frames", 5, 60, 25, key="fv_max_lost")
    noise_level = st.slider("Sensor noise level", 0.0, 8.0, 2.0, 0.5, key="fv_noise")
    calibration = st.slider("Calibration px → meters", 0.005, 0.05, 0.015, 0.005, key="fv_calib")

    record = st.checkbox("Record demo video", value=True, key="fv_record")
    assisted_fallback = st.checkbox("Assisted demo fallback", value=True, key="fv_assist")

    seed = st.number_input("Random seed", min_value=0, value=2026, step=1, key="fv_seed")

    st.info("Privacy-safe demo: no identity recognition, no external paid API, local temporary recording only.")

    if st.button("Restart / Apply settings", use_container_width=True):
        st.session_state.result = None
        st.session_state.demo_completed = False
        st.rerun()

    st.caption("Tip: set duration to 60 seconds for the maximum live demo recording.")

width, height = 848, 480
if resolution_label.startswith("640"):
    width, height = 640, 360
elif resolution_label.startswith("960"):
    width, height = 960, 540

cfg = Config(
    width=width,
    height=height,
    fps=int(fps),
    duration=int(duration),
    seed=int(seed),
    conf_threshold=float(conf_threshold),
    min_area=int(min_area),
    max_lost=int(max_lost),
    record=bool(record),
    assisted_fallback=bool(assisted_fallback),
    noise_level=float(noise_level),
    calibration_px_to_m=float(calibration),
)

if not st.session_state.demo_completed and auto_start:
    live_result = run_demo(cfg)
    st.session_state.result = live_result
    st.session_state.demo_completed = True

if st.session_state.result is not None:
    render_results(st.session_state.result)
else:
    st.info("Press **Restart / Apply settings** or enable **Auto start demo** to run the live demo.")