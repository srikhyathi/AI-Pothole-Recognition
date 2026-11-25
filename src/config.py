# Project configuration

PROJECT_NAME = "AI-Driven Pothole Recognition and Autonomous Vehicle Alert System"
MODEL_NAME = "YOLO"
CONFIDENCE_THRESHOLD = 0.50
IMAGE_SIZE = 640
CAMERA_SOURCE = 0
ENABLE_GPS = True
ENABLE_CLOUD_ALERTS = True

# Dictionary representation for dynamic configuration loading
CONFIG = {
    "project_name": PROJECT_NAME,
    "model_name": MODEL_NAME,
    "model_path": "best.pt",
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "image_size": IMAGE_SIZE,
    "camera_source": CAMERA_SOURCE,
    "enable_gps": ENABLE_GPS,
    "enable_cloud_alerts": ENABLE_CLOUD_ALERTS,
}
