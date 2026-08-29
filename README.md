# AI-Driven Pothole Recognition and Autonomous Vehicle Alert System

## Project Overview

An AI-based road-safety system designed to detect potholes and other road-surface damage from images and video using computer vision and deep-learning-based object detection.

The system uses a custom-trained YOLO model to identify different types of road damage and generates a simulated GPS-based vehicle safety alert when a pothole is detected.

The project provides the computer-vision foundation for future real-world GPS, IoT, and autonomous-vehicle safety integration.

## Objectives

- Detect potholes and road-surface damage using AI.
- Process images and video footage for road analysis.
- Identify different categories of road damage.
- Provide visual detection results using bounding boxes and labels.
- Generate GPS-based road-hazard information.
- Generate vehicle safety alerts when potholes are detected.
- Support future real-time autonomous vehicle safety applications.

## Detection Classes

The trained YOLO model recognizes five classes:

1. Longitudinal Crack
2. Transverse Crack
3. Alligator Crack
4. Pothole
5. Other

## Key Features

- Custom-trained YOLO object detection
- Pothole detection from road videos
- Road-damage classification
- OpenCV image processing
- Image, video, and webcam input support
- Configurable confidence threshold
- Detection results saved automatically
- Simulated GPS location generation
- Vehicle safety alert generation
- Git LFS support for the trained model

## Technologies Used

- Python
- Ultralytics YOLO
- OpenCV
- Computer Vision
- Deep Learning
- Git
- Git LFS

## Project Structure

```text
AI-Pothole-Recognition/
|
|-- src/
|   |-- pothole_detection.py
|   |-- gps_alert.py
|
|-- best.pt
|-- demo_result.png
|-- .gitattributes
|-- .gitignore
|-- README.md
|-- requirements.txt