# Simplified Face Recognition System

A streamlined face recognition system that uses PCA and LDA for accurate face recognition.

## Features

- Face detection using Haar cascades
- Face preprocessing with CLAHE normalization and histogram equalization
- Face recognition using PCA-LDA model
- Real-time recognition from webcam
- Command-line interface for training and testing

## System Architecture

The system is organized into the following components:

1. **Face Detector**: Uses OpenCV's Haar cascades to detect faces in images
2. **Face Preprocessor**: Prepares faces for recognition by normalizing illumination and size
3. **Face Recognizer**: Identifies faces using PCA for dimensionality reduction and LDA for classification
4. **Training Module**: Trains the recognition model on a dataset of face images
5. **Application**: Real-time face recognition from webcam

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- joblib

## Installation

1. Clone the repository
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

1. Download the AT&T face dataset (or prepare your own dataset in a similar format)
2. Run the training script:
   ```
   python train.py --dataset /path/to/att_faces --visualize
   ```

### Real-time Recognition

1. After training, run the application:
   ```
   python app.py
   ```
2. Press 'q' to quit

## Directory Structure

```
simplified/
  ├── app.py                  # Main application
  ├── config.py               # Configuration settings
  ├── face_detector.py        # Face detection module
  ├── face_preprocessor.py    # Face preprocessing module
  ├── face_recognizer.py      # Face recognition module
  ├── train.py                # Training script
  ├── requirements.txt        # Dependencies
  ├── README.md               # This file
  ├── models/                 # Trained models
  ├── data/                   # Datasets
  └── logs/                   # Log files
```

## How It Works

1. **Data Collection**: Faces are collected from a dataset or camera
2. **Preprocessing**: 
   - Convert to grayscale
   - Apply CLAHE for illumination normalization
   - Resize to standard dimensions
   - Apply histogram equalization
3. **Training**:
   - Apply PCA to reduce dimensionality
   - Apply LDA for optimal class separation
   - Train a k-NN classifier
4. **Recognition**:
   - Detect faces in the input image
   - Preprocess detected faces
   - Project onto PCA-LDA space
   - Classify using k-NN with confidence threshold

## Performance

The system provides a good balance between accuracy and speed:
- Face detection: ~30ms per frame
- Face recognition: ~10ms per face
- Overall accuracy: >90% on AT&T dataset

## License

MIT 