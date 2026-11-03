#!/usr/bin/env python3
"""
Traffic Flow Analytics — multi-class vehicle counting with speed estimation.

Counts vehicles crossing a virtual line on a road, broken down by class
(car / truck / bus / motorcycle) and by direction (eastbound / westbound),
and estimates each vehicle's speed from how fast its tracked centroid moves
across the frame — the way a traffic-monitoring camera over a highway would.

Two modes share the exact same tracking, counting, speed and overlay
pipeline:

  * Real mode       -- YOLOv8 detects vehicles in a video file or webcam
                       feed (COCO classes car, truck, bus, motorcycle).
  * Simulation mode -- a synthetic multi-lane road is rendered with
                       OpenCV/NumPy, and a colour/shape detector reads the
                       vehicles back out of the rendered pixels. The
                       detector never sees the scene's ground truth: it
                       works only on the finished frame, exactly as the
                       YOLO detector works only on real frames. Needs only
                       opencv-python + numpy, so it runs anywhere.

Speed is an estimate, not a measurement: it converts tracked pixels per
second with --px-per-metre (default 0.05 m/px, i.e. a ~100 px car is ~5 m
long). Change it to match your camera's geometry; the README explains how.

Usage (Windows CMD):
    python traffic_flow_analytics.py --simulate
    python traffic_flow_analytics.py --simulate --output demo.mp4 --duration 28
    python traffic_flow_analytics.py --source pick
    python traffic_flow_analytics.py
    python traffic_flow_analytics.py --source 0
    python traffic_flow_analytics.py --source traffic.mp4 --output counted.mp4

Running with no arguments (or --source pick) opens a file-picker dialog to
choose a video.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# COCO class ids used in real mode.
VEHICLE_CLASSES = {"car": 2, "bus": 5, "truck": 7, "motorcycle": 3}
CLASS_NAMES = {v: k for k, v in VEHICLE_CLASSES.items()}
CLASS_ORDER = ["car", "truck", "bus", "motorcycle"]

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
COL_LINE = (0, 215, 255)       # yellow counting line
COL_TEXT = (255, 255, 255)
COL_PANEL = (0, 0, 0)
CLASS_COLOURS = {
    "car": (220, 120, 30),        # blue
    "truck": (40, 140, 240),      # orange
    "bus": (60, 190, 90),         # green
    "motorcycle": (60, 60, 230),  # red
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One detected vehicle in one frame (pixel coordinates)."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str = "car"
    confidence: float = 1.0
    track_id: int | None = None  # filled in by a tracker or by YOLO/ByteTrack

    @property
    def centroid(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


# ---------------------------------------------------------------------------
# Trackers — assign persistent IDs to detections.
# ---------------------------------------------------------------------------
class CentroidTracker:
    """Centroid tracker that only matches detections of the same class.

    Class-gated matching stops a car track from jumping onto a truck that
    happens to pass nearby, which matters on a busy multi-lane road.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 60.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 1
        self.objects: dict[int, Detection] = {}
        self.disappeared: dict[int, int] = {}

    def _register(self, det: Detection) -> None:
        det.track_id = self.next_id
        self.objects[self.next_id] = det
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, track_id: int) -> None:
        del self.objects[track_id]
        del self.disappeared[track_id]

    def update(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)
            return []

        if not self.objects:
            for det in detections:
                self._register(det)
            return [(d.track_id, d) for d in detections]

        existing_ids = list(self.objects)
        existing = [self.objects[t] for t in existing_ids]

        pairs = []
        for ei, old in enumerate(existing):
            for ni, new in enumerate(detections):
                if old.label != new.label:
                    continue  # never match across classes
                dist = math.hypot(old.centroid[0] - new.centroid[0],
                                  old.centroid[1] - new.centroid[1])
                pairs.append((dist, ei, ni))
        pairs.sort(key=lambda p: p[0])

        used_existing: set[int] = set()
        used_new: set[int] = set()
        matched: list[tuple[int, Detection]] = []
        for dist, ei, ni in pairs:
            if dist > self.max_distance:
                break
            if ei in used_existing or ni in used_new:
                continue
            tid = existing_ids[ei]
            det = detections[ni]
            det.track_id = tid
            self.objects[tid] = det
            self.disappeared[tid] = 0
            used_existing.add(ei)
            used_new.add(ni)
            matched.append((tid, det))

        for ei, tid in enumerate(existing_ids):
            if ei not in used_existing:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

        for ni, det in enumerate(detections):
            if ni not in used_new:
                self._register(det)
                matched.append((det.track_id, det))

        return matched


# ---------------------------------------------------------------------------
# Line-crossing counter + speed estimator
# ---------------------------------------------------------------------------
class FlowCounter:
    """Counts direction-aware line crossings per class, and speed per track.

    A track is counted only when its centroid moves from one side of the
    line to the other, outside a small deadband — so a vehicle idling on
    the line cannot be counted repeatedly. Speed is the track's smoothed
    centroid displacement converted with the configured metres-per-pixel.
    """

    def __init__(self, line_x: int, px_per_metre: float = 0.05):
        self.line_x = line_x
        self.px_per_metre = px_per_metre
        self.margin = 6
        self.by_class: dict[str, int] = {c: 0 for c in CLASS_ORDER}
        self.by_direction: dict[str, int] = {"EAST": 0, "WEST": 0}
        self._side: dict[int, int] = {}        # track_id -> -1 west / +1 east
        self._last_seen: dict[int, int] = {}   # track_id -> frame index
        self._prev_centroid: dict[int, tuple[int, int]] = {}
        self._speed_samples: dict[int, list[float]] = {}  # track_id -> km/h
        self.speed_by_class: dict[str, list[float]] = {c: [] for c in CLASS_ORDER}
        self.peak_simultaneous = 0

    @property
    def total(self) -> int:
        return sum(self.by_class.values())

    def _speed_kmh(self, track_id: int, cx: int, cy: int, fps: float) -> float:
        prev = self._prev_centroid.get(track_id)
        self._prev_centroid[track_id] = (cx, cy)
        if prev is None or fps <= 0:
            return 0.0
        px = math.hypot(cx - prev[0], cy - prev[1])
        return px * self.px_per_metre * fps * 3.6

    def update(self, track_id: int, det: Detection, frame_index: int,
               fps: float) -> str | None:
        """Feed one tracked detection; returns 'EAST'/'WEST' if counted."""
        cx, cy = det.centroid
        kmh = self._speed_kmh(track_id, cx, cy, fps)
        if kmh > 0:
            # Keep a rolling window so one noisy frame cannot skew the mean.
            samples = self._speed_samples.setdefault(track_id, [])
            samples.append(kmh)
            if len(samples) > 30:
                samples.pop(0)

        if cx < self.line_x - self.margin:
            side = -1
        elif cx > self.line_x + self.margin:
            side = +1
        else:
            return None  # inside the deadband — no side change

        self._last_seen[track_id] = frame_index
        previous = self._side.get(track_id)
        self._side[track_id] = side
        if previous is None or previous == side:
            return None
        direction = "EAST" if (previous == -1 and side == +1) else "WEST"
        self.by_class[det.label] += 1
        self.by_direction[direction] += 1
        samples = self._speed_samples.get(track_id, [])
        if samples:
            self.speed_by_class[det.label].append(sum(samples) / len(samples))
        return direction

    def note_simultaneous(self, n: int) -> None:
        self.peak_simultaneous = max(self.peak_simultaneous, n)

    def retain_recent(self, frame_index: int, keep_frames: int = 600) -> None:
        stale = [tid for tid, seen in self._last_seen.items()
                 if frame_index - seen > keep_frames]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._side.pop(tid, None)
            self._prev_centroid.pop(tid, None)
            self._speed_samples.pop(tid, None)

    def mean_speed(self, label: str) -> float:
        samples = self.speed_by_class[label]
        return sum(samples) / len(samples) if samples else 0.0


# ---------------------------------------------------------------------------
# Overlay — shared by real and simulation modes
# ---------------------------------------------------------------------------
def _draw_dashed_line(frame: np.ndarray, x: int, y0: int, y1: int,
                      color: tuple[int, int, int]) -> None:
    dash, gap = 18, 12
    y = y0
    while y < y1:
        cv2.line(frame, (x, y), (x, min(y + dash, y1)), color, 3)
        y += dash + gap


def draw_overlay(frame: np.ndarray,
                 tracked: list[tuple[int, Detection]],
                 counter: FlowCounter,
                 fps: float,
                 last_event: tuple[str, str, float] | None = None) -> np.ndarray:
    """Draw counting line, boxes+IDs+speeds, HUD counters and FPS."""
    h, w = frame.shape[:2]
    line_x = counter.line_x

    _draw_dashed_line(frame, line_x, 0, h, COL_LINE)
    cv2.putText(frame, "COUNT LINE", (line_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_LINE, 2)
    cv2.putText(frame, "WEST <<", (line_x - 118, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)
    cv2.putText(frame, ">> EAST", (line_x + 12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)

    for tid, det in tracked:
        color = CLASS_COLOURS.get(det.label, COL_TEXT)
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        cx, cy = det.centroid
        cv2.circle(frame, (cx, cy), 4, COL_LINE, -1)
        samples = counter._speed_samples.get(tid, [])
        speed = sum(samples) / len(samples) if samples else 0.0
        label = f"ID {tid} {det.label} {speed:.0f} km/h"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(det.y1, lh + 6)
        cv2.rectangle(frame, (det.x1, ly - lh - 6), (det.x1 + lw + 6, ly),
                      color, -1)
        cv2.putText(frame, label, (det.x1 + 3, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # HUD panel (top-left).
    panel_w, panel_h = 340, 218
    panel = frame[0:panel_h, 0:panel_w].copy()
    frame[0:panel_h, 0:panel_w] = cv2.addWeighted(panel, 0.35,
                                                  np.zeros_like(panel), 0.65, 0)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), COL_LINE, 1)
    cv2.putText(frame, "TRAFFIC FLOW ANALYTICS", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL_TEXT, 2)
    y = 64
    for label in CLASS_ORDER:
        cv2.putText(frame, f"{label.upper():<11}{counter.by_class[label]}",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    CLASS_COLOURS[label], 2)
        y += 28
    cv2.putText(frame, f"EAST: {counter.by_direction['EAST']}   "
                       f"WEST: {counter.by_direction['WEST']}",
                (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_TEXT, 2)
    y += 30
    cv2.putText(frame, f"TOTAL: {counter.total}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)

    fps_text = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, fps_text, (w - fw - 14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)

    if last_event is not None:
        label, direction, age = last_event
        if age < 0.9:
            text = f"{label.upper()} {direction} +1"
            (ew, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(frame, text, ((w - ew) // 2, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        CLASS_COLOURS.get(label, COL_TEXT), 3)

    return frame


# ---------------------------------------------------------------------------
# Real-mode detector — YOLOv8 (imported lazily; only needed for --source)
# ---------------------------------------------------------------------------
class YOLOVehicleDetector:
    """Detects cars, trucks, buses and motorcycles with YOLOv8."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.4,
                 use_bytetrack: bool = True):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Real mode needs the 'ultralytics' package. Install it with "
                "'pip install ultralytics', or run simulation mode "
                "(--simulate), which only needs opencv-python + numpy."
            ) from exc
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.use_bytetrack = use_bytetrack
        self._fallback = CentroidTracker()
        self._class_ids = sorted(VEHICLE_CLASSES.values())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.use_bytetrack:
            results = self.model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                classes=self._class_ids, conf=self.confidence, verbose=False)
        else:
            results = self.model.predict(
                frame, classes=self._class_ids, conf=self.confidence,
                verbose=False)

        boxes = results[0].boxes
        detections: list[Detection] = []
        if boxes is None or len(boxes) == 0:
            if self.use_bytetrack:
                self._fallback.update([])
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = (boxes.conf.cpu().numpy() if boxes.conf is not None
                 else np.ones(len(xyxy)))
        clss = (boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None
                else np.zeros(len(xyxy), dtype=int))
        ids = None
        if self.use_bytetrack and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            det = Detection(int(x1), int(y1), int(x2), int(y2),
                            label=CLASS_NAMES.get(int(clss[i]), "car"),
                            confidence=float(confs[i]))
            if ids is not None:
                det.track_id = int(ids[i])
            detections.append(det)

        if self.use_bytetrack and ids is None:
            self._fallback.update(detections)
        return detections


# ---------------------------------------------------------------------------
# Simulation — a synthetic multi-lane road rendered with OpenCV/NumPy
# ---------------------------------------------------------------------------
@dataclass
class SimVehicle:
    """One simulated vehicle travelling along a lane."""

    x: float            # centre x of the vehicle (px)
    lane_y: int         # centre y of its lane (px)
    direction: int      # +1 eastbound (moving right) / -1 westbound (left)
    speed: float        # px per frame
    label: str          # car / truck / bus / motorcycle
    length: int         # px along the road
    width: int          # px across the road
    colour: tuple[int, int, int]
    wheel_off: float = 0.0  # animation phase for the wheels

    def box(self) -> tuple[int, int, int, int]:
        return (int(self.x - self.length / 2), int(self.lane_y - self.width / 2),
                int(self.x + self.length / 2), int(self.lane_y + self.width / 2))


# Vehicle specs: (length px, width px, body colour BGR, dark detail BGR).
_VEHICLE_SPECS = {
    "car": (98, 46, (220, 120, 30), (150, 70, 15)),
    "truck": (162, 48, (40, 140, 240), (20, 90, 170)),
    "bus": (172, 52, (60, 190, 90), (30, 130, 55)),
    "motorcycle": (56, 24, (60, 60, 230), (40, 40, 170)),
}
# HSV ranges (H 0-179, S, V) for the pixel detector below. Each class keeps
# one distinct hue family, so the detector can tell classes apart from
# pixels alone — it never reads the scene's vehicle list.
_HSV_RANGES = {
    "car": ((100, 90, 60), (115, 255, 255)),
    "truck": ((14, 90, 60), (30, 255, 255)),
    "bus": ((45, 90, 60), (75, 255, 255)),
    "motorcycle": ((0, 90, 60), (8, 255, 255)),
}


class RoadScene:
    """Renders a top-down four-lane road and its traffic.

    Lanes 0-1 run eastbound, lanes 2-3 westbound. ``step`` returns the
    rendered frame only — detection is a separate pixel-based pass
    (``detect_vehicles``), so counts can never come from ground truth.
    """

    LANE_Y = (232, 322, 412, 502)
    ROAD_TOP, ROAD_BOTTOM = 170, 560

    def __init__(self, width: int = 1280, height: int = 720, seed: int = 11):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.vehicles: list[SimVehicle] = []
        self._cooldown = [0, 0, 0, 0]  # per-lane spawn cooldown (frames)
        self._background = self._render_background()
        # Seed the road so the demo is busy from the first second.
        self._add_vehicle(lane=0, x=width * 0.18, label="car")
        self._add_vehicle(lane=1, x=width * 0.42, label="truck")
        self._add_vehicle(lane=2, x=width * 0.78, label="bus")
        self._add_vehicle(lane=3, x=width * 0.55, label="car")
        self._add_vehicle(lane=0, x=width * 0.62, label="motorcycle")
        self._add_vehicle(lane=2, x=width * 0.30, label="car")

    def _add_vehicle(self, lane: int, x: float, label: str) -> None:
        length, width, colour, _ = _VEHICLE_SPECS[label]
        direction = +1 if lane < 2 else -1
        self.vehicles.append(SimVehicle(
            x=x, lane_y=self.LANE_Y[lane], direction=direction,
            speed=self.rng.uniform(8.0, 13.0), label=label,
            length=length, width=width, colour=colour,
            wheel_off=self.rng.uniform(0, 6)))

    def _maybe_spawn(self) -> None:
        for lane in range(4):
            if self._cooldown[lane] > 0:
                self._cooldown[lane] -= 1
                continue
            direction = +1 if lane < 2 else -1
            edge = -110.0 if direction == +1 else self.width + 110.0
            # Keep a gap from the nearest vehicle already near this edge.
            for v in self.vehicles:
                if v.lane_y == self.LANE_Y[lane] and abs(v.x - edge) < 220:
                    break
            else:
                label = self.rng.choices(
                    ["car", "car", "truck", "bus", "motorcycle"],
                    weights=[5, 5, 2, 2, 1])[0]
                self._add_vehicle(lane=lane, x=edge, label=label)
                self._cooldown[lane] = self.rng.randint(45, 130)

    def _render_background(self) -> np.ndarray:
        w, h = self.width, self.height
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (96, 104, 92)  # muted verges (low saturation)
        # Asphalt.
        cv2.rectangle(img, (0, self.ROAD_TOP), (w, self.ROAD_BOTTOM),
                      (62, 62, 62), -1)
        # Lane markings: solid edges, dashed dividers.
        cv2.line(img, (0, self.ROAD_TOP + 4), (w, self.ROAD_TOP + 4),
                 (210, 210, 210), 3)
        cv2.line(img, (0, self.ROAD_BOTTOM - 4), (w, self.ROAD_BOTTOM - 4),
                 (210, 210, 210), 3)
        for y in (self.LANE_Y[0] + 45, self.LANE_Y[2] + 45):
            for x in range(0, w, 90):
                cv2.line(img, (x, y), (x + 45, y), (210, 210, 210), 3)
        # Centre line between the two directions. Painted near-white on
        # purpose: a yellow line would fall inside the truck hue range
        # and the pixel detector would read the road paint as a vehicle.
        cy = (self.LANE_Y[1] + self.LANE_Y[2]) // 2
        cv2.line(img, (0, cy - 4), (w, cy - 4), (210, 210, 210), 3)
        cv2.line(img, (0, cy + 4), (w, cy + 4), (210, 210, 210), 3)
        # (No eastbound label up here: the translucent HUD panel sits over
        # this corner and background text would ghost through it. Direction
        # arrows are drawn in the overlay instead.)
        cv2.putText(img, "<<  WESTBOUND", (w - 250, self.ROAD_BOTTOM + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        return img

    def _draw_vehicle(self, frame: np.ndarray, v: SimVehicle) -> None:
        x1, y1, x2, y2 = v.box()
        _, _, _, detail = _VEHICLE_SPECS[v.label]
        # Body.
        cv2.rectangle(frame, (x1, y1), (x2, y2), v.colour, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), 1)
        # Cab / windshield block at the front end (direction of travel).
        front = x2 if v.direction == +1 else x1
        cab_w = max(14, v.length // 5)
        cx1 = front - cab_w if v.direction == +1 else front
        cx2 = front if v.direction == +1 else front + cab_w
        cv2.rectangle(frame, (cx1, y1 + 5), (cx2, y2 - 5), detail, -1)
        # Wheels (black stubs on both long sides).
        for fx in (x1 + 12, x2 - 12):
            cv2.rectangle(frame, (fx - 5, y1 - 3), (fx + 5, y1 + 4),
                          (20, 20, 20), -1)
            cv2.rectangle(frame, (fx - 5, y2 - 4), (fx + 5, y2 + 3),
                          (20, 20, 20), -1)

    def step(self) -> np.ndarray:
        """Advance traffic one frame and return the rendered frame."""
        self._maybe_spawn()
        for v in self.vehicles:
            v.x += v.direction * v.speed
            v.wheel_off += 1
        self.vehicles = [v for v in self.vehicles
                         if -160 < v.x < self.width + 160]
        frame = self._background.copy()
        for v in self.vehicles:
            self._draw_vehicle(frame, v)
        return frame


def detect_vehicles(frame: np.ndarray) -> list[Detection]:
    """Pixel-only vehicle detector for simulation frames.

    Thresholds each class's hue family in HSV, cleans the mask with a
    closing operation (so wheels/windshields do not split a vehicle),
    and returns one box per contour. It reads nothing but the frame —
    the scene's vehicle list is never consulted.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
    detections: list[Detection] = []
    for label, (lo, hi) in _HSV_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                           np.array(hi, dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 700:  # ignore specks (e.g. the yellow centre line)
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Vehicles travel horizontally: length must exceed height, and
            # no single vehicle is longer than ~200 px — wider blobs are
            # road paint or merged vehicles, not one vehicle.
            if w < h or w < 40 or w > 260:
                continue
            detections.append(Detection(x, y, x + w, y + h, label=label))
    return detections


# ---------------------------------------------------------------------------
# Shared pipeline: track -> count/speed -> annotate
# ---------------------------------------------------------------------------
class FlowPipeline:
    """Runs tracker + counter + overlay identically for both modes."""

    def __init__(self, line_x: int, px_per_metre: float):
        self.centroid_tracker = CentroidTracker()
        self.counter = FlowCounter(line_x, px_per_metre)
        self._fps = 0.0
        self._last_event: tuple[str, str, float] | None = None

    def process(self, frame: np.ndarray, detections: list[Detection],
                frame_index: int, now: float, dt: float) -> np.ndarray:
        if dt > 0:
            instant = 1.0 / dt
            self._fps = instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant

        if detections and all(d.track_id is not None for d in detections):
            tracked = [(d.track_id, d) for d in detections]
            self.centroid_tracker.update([])  # keep its state ageing cleanly
        else:
            tracked = self.centroid_tracker.update(detections)

        self.counter.note_simultaneous(len(tracked))
        for tid, det in tracked:
            event = self.counter.update(tid, det, frame_index, self._fps)
            if event:
                self._last_event = (det.label, event, now)
        self.counter.retain_recent(frame_index)

        last = None
        if self._last_event is not None:
            label, direction, when = self._last_event
            last = (label, direction, now - when)
        return draw_overlay(frame, tracked, self.counter, self._fps, last)

    def report(self, mode: str, frames: int) -> str:
        c = self.counter
        lines = [f"[{mode}] done: {frames} frames | TOTAL={c.total} "
                 f"EAST={c.by_direction['EAST']} WEST={c.by_direction['WEST']} "
                 f"peak simultaneous={c.peak_simultaneous}"]
        for label in CLASS_ORDER:
            lines.append(f"[{mode}] {label}: {c.by_class[label]} counted, "
                         f"mean speed {c.mean_speed(label):.1f} km/h")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def make_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open output video '{path}' for writing "
            "(the mp4v codec may be unavailable in this OpenCV build).")
    return writer


def _destroy_windows() -> None:
    """Close preview windows; a no-op on headless OpenCV builds."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def show_frame(frame: np.ndarray, title: str) -> bool:
    """Show a frame; returns False when the user pressed q / Esc.

    Silently keeps going on headless OpenCV builds (no GUI support).
    """
    try:
        cv2.imshow(title, frame)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)
    except cv2.error:
        return True


def pick_video_file() -> str | None:
    """Open a file-picker dialog; returns the chosen path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("error: tkinter is not available; pass --source <video path> "
              "instead of --source pick.", file=sys.stderr)
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Choose a traffic video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                   ("All files", "*.*")])
    root.destroy()
    return path or None


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
def run_simulation(args: argparse.Namespace) -> int:
    scene = RoadScene(width=args.width, height=args.height, seed=args.seed)
    line_x = int(args.line_x * args.width)
    pipeline = FlowPipeline(line_x, args.px_per_metre)

    writer = None
    if args.output:
        writer = make_writer(args.output, args.fps, (args.width, args.height))

    duration = args.duration if args.duration is not None else 28.0
    total_frames = int(duration * args.fps)
    print(f"[simulate] rendering {args.width}x{args.height} @ {args.fps} fps, "
          f"{duration:.0f}s")
    frame_index = 0
    prev = time.perf_counter()
    try:
        while frame_index < total_frames:
            tick = time.perf_counter()
            frame = scene.step()
            detections = detect_vehicles(frame)  # pixels only, no ground truth
            now = time.perf_counter()
            annotated = pipeline.process(frame, detections, frame_index,
                                         now, now - prev)
            prev = now
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                if not show_frame(annotated, "Traffic Flow (simulation)"):
                    break
            frame_index += 1
            elapsed = time.perf_counter() - tick
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        _destroy_windows()

    print(pipeline.report("simulate", frame_index))
    if args.output:
        print(f"[simulate] video saved to {args.output}")
    return 0


def run_real(args: argparse.Namespace) -> int:
    source: int | str = 0 if str(args.source) == "0" else args.source
    if isinstance(source, str):
        import os
        if not os.path.exists(source):
            print(f"error: video file not found: {source}", file=sys.stderr)
            return 2

    try:
        detector = YOLOVehicleDetector(
            model_name=args.model, confidence=args.conf,
            use_bytetrack=(args.tracker == "bytetrack"))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"error: could not open source: {args.source}", file=sys.stderr)
        return 2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    line_x = int(args.line_x * width)
    pipeline = FlowPipeline(line_x, args.px_per_metre)

    writer = None
    if args.output:
        writer = make_writer(args.output, src_fps, (width, height))

    max_frames = (int(args.duration * src_fps) if args.duration else None)
    print(f"[real] source={args.source} {width}x{height} — press 'q' to stop")
    frame_index = 0
    prev = time.perf_counter()
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            now = time.perf_counter()
            annotated = pipeline.process(frame, detections, frame_index,
                                         now, now - prev)
            prev = now
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                if not show_frame(annotated, "Traffic Flow Analytics"):
                    break
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        _destroy_windows()

    print(pipeline.report("real", frame_index))
    if args.output:
        print(f"[real] video saved to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analytics — count vehicles by class and "
                    "direction and estimate speed, with YOLOv8 + OpenCV, "
                    "or run the built-in road simulation.")
    parser.add_argument("--source", default=None,
                        help="video file path, 0 for the webcam, or 'pick' "
                             "for a file-picker dialog (real mode)")
    parser.add_argument("--simulate", action="store_true",
                        help="run the synthetic multi-lane road simulation "
                             "(needs only opencv-python + numpy)")
    parser.add_argument("--output", default=None,
                        help="path to save the annotated output video (.mp4)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 model for real mode (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="detection confidence threshold (default: 0.4)")
    parser.add_argument("--tracker", choices=("bytetrack", "centroid"),
                        default="bytetrack",
                        help="real-mode ID source: ByteTrack via model.track "
                             "(default) or the built-in centroid tracker")
    parser.add_argument("--line-x", type=float, default=0.5,
                        help="counting-line position as a fraction of the "
                             "frame width, 0-1 (default: 0.5)")
    parser.add_argument("--duration", type=float, default=None,
                        help="seconds to process (simulation default: 28; "
                             "real mode default: the whole video)")
    parser.add_argument("--fps", type=int, default=30,
                        help="simulation frame rate (default: 30)")
    parser.add_argument("--width", type=int, default=1280,
                        help="simulation frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="simulation frame height (default: 720)")
    parser.add_argument("--seed", type=int, default=11,
                        help="random seed for the simulation (default: 11)")
    parser.add_argument("--px-per-metre", type=float, default=0.05,
                        help="metres per pixel for speed estimation "
                             "(default: 0.05; calibrate to your camera)")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a preview window (headless use)")
    args = parser.parse_args(argv)

    if not 0.0 < args.line_x < 1.0:
        parser.error("--line-x must be between 0 and 1 (exclusive)")
    if args.simulate and args.source is not None:
        parser.error("--simulate and --source cannot be used together")
    if not args.simulate and args.source is None:
        # No mode given: open the file picker, like the other delivered
        # scripts do when run with no arguments.
        picked = pick_video_file()
        if picked is None:
            parser.error("choose a mode: --simulate, or --source <video path|0>")
        args.source = picked
    if not args.simulate and args.source == "pick":
        picked = pick_video_file()
        if picked is None:
            parser.error("no video file selected")
        args.source = picked
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.simulate:
        return run_simulation(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
