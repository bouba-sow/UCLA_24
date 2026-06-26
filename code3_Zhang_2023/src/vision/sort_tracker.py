"""SORT tracker (Bewley et al.) — Zhang stage 1b(ii).

Kalman filter on bbox + Hungarian assignment (lapjv).
"""
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from zhang2023_constants import SORT_IOU_THRESHOLD, SORT_MAX_AGE, SORT_MIN_HITS


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """IoU between Nx4 and Mx4 boxes."""
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]), dtype=np.float32)
    xx1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0])
    yy1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1])
    xx2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2])
    yy2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_t = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_g = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    union = area_t[:, None] + area_g[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / np.maximum(h, 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    w = np.sqrt(np.maximum(x[2] * x[3], 1e-6))
    h = x[2] / np.maximum(w, 1e-6)
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape(1, 4)
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape(1, 5)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: list[np.ndarray] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.frames: list[int] = []
        self.boxes: list[np.ndarray] = []

    def update(self, bbox: np.ndarray, frame_idx: int) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.frames.append(frame_idx)
        self.boxes.append(bbox.astype(np.float32))

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kf.x)


class SORTTracker:
    def __init__(
        self,
        max_age: int = SORT_MAX_AGE,
        min_hits: int = SORT_MIN_HITS,
        iou_threshold: float = SORT_IOU_THRESHOLD,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []

    def reset(self) -> None:
        KalmanBoxTracker.count = 0
        self.trackers = []

    def update(self, dets: np.ndarray, frame_idx: int) -> list[KalmanBoxTracker]:
        if dets.ndim == 1 and dets.size:
            dets = dets.reshape(1, -1)
        if dets.size == 0:
            dets = np.empty((0, 5))

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self._associate(dets[:, :4], trks[:, :4])
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4], frame_idx)
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            trk.update(dets[i, :4], frame_idx)
            self.trackers.append(trk)

        active: list[KalmanBoxTracker] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or frame_idx <= self.min_hits):
                active.append(trk)
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return active

    def _associate(self, detections: np.ndarray, trackers: np.ndarray):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)
        iou_m = iou_batch(detections, trackers)
        if min(iou_m.shape) > 0:
            a = linear_sum_assignment(-iou_m)
            matched = np.array(list(zip(*a)))
        else:
            matched = np.empty((0, 2), dtype=int)
        unmatched_dets = [d for d in range(len(detections)) if d not in matched[:, 0]]
        unmatched_trks = [t for t in range(len(trackers)) if t not in matched[:, 1]]
        for m in matched:
            if iou_m[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
        matched = np.array([m for m in matched if m[0] not in unmatched_dets and m[1] not in unmatched_trks])
        if matched.size == 0:
            matched = np.empty((0, 2), dtype=int)
        return matched, np.array(unmatched_dets), np.array(unmatched_trks)
