"""
Comprehensive Unit Test Suite for AI-Driven Pothole Recognition Pipeline.
Tests photometric enhancement, stereoscopic depth profiling, ByteTrack tracking,
NMEA GPS parsing, CAN bus alerting, TensorRT engine config, and FastAPI gateway.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.cv.photometric_enhancer import PhotometricEnhancer
from src.depth.stereoscopic_profiler import StereoscopicProfiler
from src.tracking.bytetrack_kalman import HazardTracker
from src.gps.nmea_geospatial import NMEAGeospatialEngine
from src.alerts.can_bus_dispatcher import CANBusAlertDispatcher
from src.edge.tensorrt_quantizer import TensorRTQuantizer
from src.api.fastapi_hazard_service import api_app
from fastapi.testclient import TestClient

client = TestClient(api_app)

def test_photometric_enhancer_clahe():
    enhancer = PhotometricEnhancer(clip_limit=2.0)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 120
    enhanced = enhancer.enhance_frame(img)
    assert enhanced.shape == (100, 100, 3)

def test_stereoscopic_profiler():
    profiler = StereoscopicProfiler(focal_length_px=800.0, baseline_meters=0.20)
    depth = profiler.disparity_to_depth(16.0)
    assert depth == 10.0 # 800 * 0.2 / 16 = 10.0 meters
    
    profile = profiler.profile_crater(
        bounding_box=(100, 100, 200, 180),
        mean_disparity=15.8,
        road_surface_disparity=16.0
    )
    assert "depth_mm" in profile
    assert "surface_area_cm2" in profile
    assert profile["severity_level"] in ("LOW", "MEDIUM", "CRITICAL")

def test_bytetrack_tracking():
    tracker = HazardTracker()
    dets_f1 = [{"bbox": (50, 50, 120, 120), "confidence": 0.92, "class_name": "Pothole"}]
    t1 = tracker.update(dets_f1)
    
    dets_f2 = [{"bbox": (52, 53, 122, 122), "confidence": 0.94, "class_name": "Pothole"}]
    t2 = tracker.update(dets_f2)
    assert len(t2) == 1
    assert t2[0]["persistent_hits"] >= 2

def test_nmea_gps_parser():
    engine = NMEAGeospatialEngine()
    # Typical GPGGA sentence
    raw = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    parsed = engine.parse_nmea_sentence(raw)
    assert parsed is not None
    assert abs(parsed["latitude"] - 48.1173) < 0.01
    assert abs(parsed["longitude"] - 11.5166) < 0.01
    
    geohash = engine.encode_geohash(16.3067, 80.4365, precision=7)
    assert len(geohash) == 7

def test_can_bus_dispatcher():
    dispatcher = CANBusAlertDispatcher(vehicle_speed_kmh=72.0) # 20 m/s
    ttc = dispatcher.compute_time_to_collision(40.0)
    assert ttc == 2.0 # 40m / 20m/s = 2.0s
    
    safety = dispatcher.calculate_braking_safety_margin(distance_ahead_m=35.0)
    assert "time_to_collision_s" in safety
    assert "recommended_action" in safety
    
    frame = dispatcher.format_j1939_can_frame(pgn=0xFEF1, severity_code=3, distance_m=25.0)
    assert len(frame["payload_bytes"]) == 8

def test_tensorrt_quantizer_config():
    quant = TensorRTQuantizer(precision="FP16")
    cfg = quant.generate_engine_config()
    assert cfg["precision_mode"] == "FP16"
    assert cfg["expected_throughput_fps"] > 50

def test_fastapi_hazard_endpoints():
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "OPERATIONAL"

    r_list = client.get("/api/v1/hazards")
    assert r_list.status_code == 200
    assert r_list.json()["count"] >= 2

    r_post = client.post("/api/v1/hazards/log", json={
        "class_name": "Pothole",
        "confidence": 0.96,
        "lat": 16.3090,
        "lon": 80.4390,
        "depth_mm": 65.0,
        "severity": "CRITICAL"
    })
    assert r_post.status_code == 200
    assert r_post.json()["status"] == "LOGGED"
