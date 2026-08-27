"""
AI-Driven Pothole Recognition and Autonomous Vehicle Alert System

Main detection module using YOLO and OpenCV.

Author: Sri Khyathi Ambadipudi
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detect potholes from images, videos, or a webcam."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: 0 for webcam, image path, or video path.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="models/pothole_best.pt",
        help="Path to the trained pothole YOLO model.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum confidence threshold.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/detection",
        help="Directory where detection results are saved.",
    )

    return parser.parse_args()


def convert_source(source):
    """Convert webcam source '0' into integer 0."""
    if source.isdigit():
        return int(source)
    return source


def run_detection(model, source, confidence, output_dir):
    """Run YOLO detection and display the results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source = convert_source(source)

    results = model.predict(
        source=source,
        conf=confidence,
        save=True,
        project=str(output_path),
        name="pothole_results",
        exist_ok=False,
        show=True,
    )

    print("\nDetection completed successfully.")
    print(f"Results saved inside: {output_path}")

    return results


def main():
    args = parse_arguments()

    weights_path = Path(args.weights)

    if not weights_path.exists():
        print("\nERROR: Trained pothole model was not found.")
        print(f"Expected model location: {weights_path}")
        print("\nPlease place the trained YOLO pothole model at this location")
        print("before running pothole detection.")
        return

    print("Loading pothole detection model...")
    model = YOLO(str(weights_path))

    print("Starting pothole detection...")
    print(f"Input source: {args.source}")
    print(f"Confidence threshold: {args.confidence}")

    run_detection(
        model=model,
        source=args.source,
        confidence=args.confidence,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()