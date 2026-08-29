"""
AI-Driven Pothole Recognition and Autonomous Vehicle Alert System

YOLO + OpenCV detection module for images, videos, and webcam.

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
        help="Path to the YOLO model.",
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


def run_webcam(model, confidence, output_dir):
    """Run YOLO detection on webcam and save the output video."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    output_file = output_path / "pothole_detection.avi"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video.")
        cap.release()
        return

    print("Starting webcam detection...")
    print("Press Q to stop.")

    while True:
        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        results = model.predict(
            source=frame,
            conf=confidence,
            verbose=False
        )

        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

        cv2.imshow("AI Pothole Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nDetection completed successfully.")
    print(f"Output video saved at: {output_file}")


def run_image_or_video(model, source, confidence, output_dir):
    """Run YOLO detection on an image or video."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    output_file = output_path / "pothole_detection.avi"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        fps,
        (width, height)
    )

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            source=frame,
            conf=confidence,
            verbose=False
        )

        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

        cv2.imshow("AI Pothole Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nDetection completed successfully.")
    print(f"Output video saved at: {output_file}")


def main():
    args = parse_arguments()

    weights_path = Path(args.weights)

    if not weights_path.exists():
        print("\nERROR: Model file was not found.")
        print(f"Expected location: {weights_path}")
        return

    print("Loading YOLO model...")
    model = YOLO(str(weights_path))

    source = convert_source(args.source)

    if isinstance(source, int):
        run_webcam(
            model=model,
            confidence=args.confidence,
            output_dir=args.output
        )
    else:
        run_image_or_video(
            model=model,
            source=source,
            confidence=args.confidence,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()