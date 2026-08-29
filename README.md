# AI-Driven Pothole Recognition and Autonomous Vehicle Alert System

## Project Overview

An AI-based road-safety system designed to detect potholes and other road-surface damage from images and video using computer vision and deep-learning-based object detection.

The system uses a custom-trained YOLO model to identify different types of road damage and provides visual detection results with bounding boxes and labels.

The project is designed as a foundation for future GPS-based road-hazard alerts and autonomous vehicle safety applications.

## Objectives

- Detect potholes and road-surface damage using AI.
- Process images and video footage for road analysis.
- Identify different categories of road damage.
- Provide visual detection results using bounding boxes and labels.
- Support real-time webcam-based detection.
- Provide a foundation for GPS and vehicle alert integration.

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
|
|-- best.pt
|-- .gitattributes
|-- .gitignore
|-- README.md
|-- requirements.txt