# Face Recognition System

This is a real-time face recognition system that uses computer vision techniques to detect and recognize faces using a live camera feed. The system implements both PCA (Eigenfaces) and LDA for face recognition, along with the Viola-Jones algorithm for face detection.

## Features

- Real-time face detection using Viola-Jones algorithm
- Face preprocessing including:
  - Grayscale conversion
  - Illumination normalization
  - Size normalization
  - Spatial position normalization
- Face recognition using PCA and LDA
- User-friendly GUI interface
- Gallery management for training data
- Live camera feed with recognition results

## Requirements

- Python 3.7+
- Webcam
- Required packages (install using `pip install -r requirements.txt`):
  - opencv-python
  - numpy
  - scikit-learn
  - scikit-image
  - pillow
  - matplotlib
  - tkinter

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. To add a person to the recognition system:
   - Enter the person's name in the "Person Name" field
   - Click "Start Capture"
   - The system will automatically capture 10 face images
   - Repeat for each person you want to recognize

3. After adding all persons:
   - Click "Train Model" to train the recognition system
   - The system will now recognize faces in real-time

## System Components

- `face_detector.py`: Implements face detection using Viola-Jones algorithm
- `face_preprocessor.py`: Handles image preprocessing and normalization
- `face_recognizer.py`: Implements PCA and LDA for face recognition
- `main.py`: Main application with GUI interface

## Notes

- The system requires good lighting conditions for optimal performance
- Face images should be at least 90x90 pixels
- The person should face the camera directly for best results
- At least 10 face images per person are recommended for training
- The confidence threshold for recognition is set to 0.6 (can be adjusted in the code)

## Troubleshooting

1. If the camera doesn't start:
   - Check if your webcam is properly connected
   - Make sure no other application is using the camera

2. If recognition accuracy is low:
   - Ensure good lighting conditions
   - Add more training images
   - Adjust the confidence threshold in the code
   - Make sure faces are well-aligned during capture 