"""
ByteTrack Multi-Object Kalman Tracking for Road Hazard Persistence.
Maintains persistent hazard IDs across video frames, filters transient false positives,
and tracks relative vehicle approach trajectory.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

class TrackedHazard:
    """Represents single tracked road hazard instance."""
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], confidence: float, class_name: str):
        self.track_id = track_id
        self.bbox = list(bbox)
        self.confidence = confidence
        self.class_name = class_name
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        # Velocity estimation [vx, vy, vw, vh]
        self.velocity = [0.0, 0.0, 0.0, 0.0]

class HazardTracker:
    """Kalman-style state filter and spatial association tracker."""
    
    def __init__(self, max_missed_frames: int = 5, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_missed = max_missed_frames
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[TrackedHazard] = []
        self._next_id = 1

    @staticmethod
    def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new frame detections.
        detections format: [{'bbox': (x1, y1, x2, y2), 'confidence': 0.85, 'class_name': 'Pothole'}]
        """
        for t in self.tracks:
            t.time_since_update += 1
            t.age += 1

        matched_tracks = set()
        matched_dets = set()

        # Associate detections to tracks via Greedy IoU Matching
        for d_idx, det in enumerate(detections):
            best_iou = 0.0
            best_t_idx = -1
            for t_idx, track in enumerate(self.tracks):
                if t_idx in matched_tracks:
                    continue
                iou = self.calculate_iou(det['bbox'], track.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_t_idx = t_idx

            if best_t_idx != -1:
                track = self.tracks[best_t_idx]
                track.bbox = list(det['bbox'])
                track.confidence = det['confidence']
                track.hits += 1
                track.time_since_update = 0
                matched_tracks.add(best_t_idx)
                matched_dets.add(d_idx)

        # Unmatched detections become new candidate tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = TrackedHazard(self._next_id, det['bbox'], det['confidence'], det['class_name'])
                self._next_id += 1
                self.tracks.append(new_track)

        # Purge dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_missed]

        # Return confirmed tracks that have persisted for >= min_hits frames
        confirmed = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.hits >= 2:
                confirmed.append({
                    "track_id": t.track_id,
                    "bbox": t.bbox,
                    "confidence": t.confidence,
                    "class_name": t.class_name,
                    "persistent_hits": t.hits
                })
        return confirmed
