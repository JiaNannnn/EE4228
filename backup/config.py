"""
Configuration for the Real-Time Face Recognition System

This file contains all configuration parameters for the system,
including camera settings, detection parameters, preprocessing options,
and model configurations.
"""

import os
from pathlib import Path
import cv2

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Camera config
CAMERA_CONFIG = {
    "resolution": (1280, 720),  # 720p
    "fps": 30,
    "device_id": 0,
    "buffer_size": 5,  # Frame buffer size
    "exposure_auto": True,
    "white_balance_auto": True
}

# Face detection config
DETECTION_CONFIG = {
    "frontal_cascade": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    "profile_cascade": cv2.data.haarcascades + 'haarcascade_profileface.xml',
    "lbp_cascade": cv2.data.haarcascades + 'lbpcascade_frontalface.xml',
    "min_face_size": (90, 90),
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "scale_pyramid": [1.0, 0.75, 0.5],
    "confidence_threshold": 0.8
}

# Preprocessing config
PREPROCESSING_CONFIG = {
    "target_size": (90, 90),
    "clahe_grid_size": (8, 8),
    "clahe_clip_limit": 3.0,
    "eye_position_y": 0.25,  # 25% from top
    "inter_pupillary_distance": 45,  # pixels
    "histogram_range": (0.1, 0.9)  # Quantile range
}

# Model config
MODEL_CONFIG = {
    "pca_variance": 0.95,
    "lda_regularization": 1e-4,
    "confidence_threshold": 0.8,
    "embedding_cache_size": 100
}

# Performance targets
PERFORMANCE_CONFIG = {
    "max_latency_ms": 150,
    "max_memory_mb": 2048,
    "target_accuracy": 0.9
}

# Training config
TRAINING_CONFIG = {
    "train_dataset": str(ROOT_DIR / "AT&T"),
    "test_dataset": "lfw",  # Labeled Faces in the Wild
    "test_size": 0.2,
    "random_state": 42
}

# MLflow tracking
MLFLOW_CONFIG = {
    "tracking_uri": str(ROOT_DIR / "mlruns"),
    "experiment_name": "face_recognition"
}

# REST API config
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False
}

# Web demo config
DEMO_CONFIG = {
    "title": "Real-Time Face Recognition",
    "theme": "light",
    "port": 8501
} 