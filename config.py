"""
Simplified Configuration for Face Recognition System

This file contains essential configuration parameters for the system.
"""

import os
import cv2
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Camera settings
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face Detection
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MIN_FACE_SIZE = (30, 30)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5

# Face Preprocessing
FACE_SIZE = 128
# Contrast Enhancement
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
# Additional contrast enhancement options
USE_HIST_EQUALIZATION = True  # Histogram equalization
USE_CONTRAST_STRETCHING = False  # Linear contrast stretching
CONTRAST_STRETCH_PERCENTILES = (5, 95)  # Percentiles for contrast stretching (low, high)
USE_GAMMA_CORRECTION = False  # Gamma correction
GAMMA_VALUE = 1.2  # Gamma value (< 1: brighter, > 1: darker)

# Face Alignment parameters
USE_FACE_ALIGNMENT = True  # Enable/disable face alignment
# Standard eye coordinates for alignment (relative to image size)
# These define where the eyes should be positioned in the final image
LEFT_EYE_POSITION = (0.35, 0.4)  # x, y coordinates (percentage of image width/height)
RIGHT_EYE_POSITION = (0.65, 0.4)  # x, y coordinates (percentage of image width/height)
# Method to use for facial landmark detection
# Options: 'haarcascade', 'lbp'
FACE_LANDMARK_METHOD = 'haarcascade'  # Use Haar cascades as it's available without additional dependencies

# Rotation parameters
USE_ROTATION = False  # Enable/disable rotation in preprocessing
FIXED_ROTATION = False  # Use fixed angle vs random angle
ROTATION_ANGLE = 0  # Fixed rotation angle in degrees (if FIXED_ROTATION is True)
ROTATION_RANGE = (-10, 10)  # Range for random rotation (if FIXED_ROTATION is False)
# For data augmentation during training
DATA_AUGMENTATION = False  # Enable/disable data augmentation during training
AUGMENTATION_ROTATIONS = [0, -5, 5, -10, 10]  # Rotation angles for augmentation

# Face Recognition
MODEL_PATH = str(MODELS_DIR / "face_recognition_model.pkl")
PCA_VARIANCE = 0.95
CONFIDENCE_THRESHOLD = 0.5

# Training settings
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Dataset paths
ATT_DATASET_PATH = str(DATA_DIR / "att_faces")

# UI settings
WINDOW_TITLE = "Face Recognition System"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FACE_RECT_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BG_COLOR = (0, 0, 0)  # Black 