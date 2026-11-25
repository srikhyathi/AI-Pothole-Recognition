"""
Stereoscopic Depth Profiling & 3D Crater Volume Integration.
Computes metric depth from dual camera disparity and estimates physical
pothole depth, surface area, and crater volume for severity grading.
"""

from typing import Dict, Any, Tuple
import numpy as np

class StereoscopicProfiler:
    """Estimates metric depth and calculates 3D crater severity metrics."""

    def __init__(self, focal_length_px: float = 850.0, baseline_meters: float = 0.20):
        self.focal_length = focal_length_px
        self.baseline = baseline_meters

    def disparity_to_depth(self, disparity_px: float) -> float:
        """Compute depth Z = (f * B) / d in meters."""
        if disparity_px <= 0.1:
            return 100.0  # Far plane clamp
        return (self.focal_length * self.baseline) / disparity_px

    def profile_crater(
        self,
        bounding_box: Tuple[int, int, int, int],
        mean_disparity: float,
        road_surface_disparity: float,
        px_to_mm_ratio: float = 0.85
    ) -> Dict[str, Any]:
        """
        Calculate depth, surface area, and estimated volume of road crater.
        bbox: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bounding_box
        width_px = max(1, x2 - x1)
        height_px = max(1, y2 - y1)
        
        # Metric depth of crater floor vs road surface
        depth_crater_m = self.disparity_to_depth(mean_disparity)
        depth_surface_m = self.disparity_to_depth(road_surface_disparity)
        
        # Pothole depth in millimeters
        crater_depth_mm = max(5.0, (depth_surface_m - depth_crater_m) * 1000.0)
        
        # Major and minor radii in mm
        radius_a_mm = (width_px * px_to_mm_ratio) / 2.0
        radius_b_mm = (height_px * px_to_mm_ratio) / 2.0
        
        # Elliptical surface area in square centimeters
        surface_area_cm2 = (np.pi * radius_a_mm * radius_b_mm) / 100.0
        
        # Ellipsoidal cap volume: V = (2/3) * pi * a * b * h in cubic centimeters
        volume_cm3 = (2.0 / 3.0) * np.pi * (radius_a_mm / 10.0) * (radius_b_mm / 10.0) * (crater_depth_mm / 10.0)
        
        # Severity classification according to IRC (Indian Roads Congress) standards
        if crater_depth_mm < 25.0:
            severity = "LOW"
            urgency = "MONITOR"
        elif crater_depth_mm < 50.0:
            severity = "MEDIUM"
            urgency = "MAINTENANCE_REQUIRED"
        else:
            severity = "CRITICAL"
            urgency = "IMMEDIATE_ACTION_AVOID"

        return {
            "depth_mm": round(float(crater_depth_mm), 2),
            "surface_area_cm2": round(float(surface_area_cm2), 2),
            "volume_cm3": round(float(volume_cm3), 2),
            "severity_level": severity,
            "urgency": urgency,
            "distance_ahead_m": round(float(depth_surface_m), 2)
        }
