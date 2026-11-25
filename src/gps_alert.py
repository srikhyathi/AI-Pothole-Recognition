"""
GPS Location and Vehicle Alert Simulation

AI-Driven Pothole Recognition and Autonomous Vehicle Alert System

This module simulates GPS coordinates and generates a safety
alert when a pothole is detected by the YOLO model.
"""

from datetime import datetime


# Simulated GPS coordinates for demonstration.
DEFAULT_LATITUDE = 16.3067
DEFAULT_LONGITUDE = 80.4365


def get_gps_location():
    """
    Return simulated GPS coordinates.

    In a real deployment, these values can be replaced by
    coordinates received from a GPS module such as NEO-6M.
    """
    return DEFAULT_LATITUDE, DEFAULT_LONGITUDE


def generate_alert(confidence):
    """
    Generate a vehicle safety alert for a detected pothole.
    """
    latitude, longitude = get_gps_location()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 55)
    print("             ROAD SAFETY ALERT")
    print("=" * 55)
    print("WARNING    : POTHOLE DETECTED")
    print(f"Confidence : {confidence:.2f}")
    print(f"Latitude   : {latitude:.4f}")
    print(f"Longitude  : {longitude:.4f}")
    print(f"Time       : {timestamp}")
    print("-" * 55)
    print("ALERT      : Road hazard detected ahead!")
    print("ACTION     : Slow down and avoid the pothole.")
    print("=" * 55)


if __name__ == "__main__":
    generate_alert(0.85)