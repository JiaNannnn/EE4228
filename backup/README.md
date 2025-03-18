# Real-Time Face Recognition System

A comprehensive real-time face recognition system with high performance and accuracy, implemented according to the following specifications:

## Features

### 1. Live Camera Interface
- 30 FPS processing at 720p resolution
- Multi-threaded pipeline with frame queue buffering
- Automatic exposure/white balance adaptation

### 2. Detection System
- Hybrid detector: Viola-Jones (frontal) + LBP (profile)
- Minimum face size: 90×90 pixels
- Multi-scale detection (1.0x, 0.75x, 0.5x pyramid)

### 3. Advanced Preprocessing
- CLAHE normalization (8×8 grid, clipLimit=3.0)
- Geometric normalization:
  - Eye alignment to 25% vertical position
  - Inter-pupillary distance standardization (45px ±2%)
- Histogram stretching to [0.1, 0.9] quantile ranges

### 4. Recognition Engine
- PCA-LDA hybrid model:
  - 95% variance retention in PCA
  - Regularized LDA (ε=1e-4)
- Confidence-based rejection (threshold=0.8)
- Face embedding cache (LRU, 100-entry capacity)

### 5. Performance
- <150ms end-to-end latency
- <2GB memory footprint
- 90%+ accuracy on benchmark datasets

## Installation

### Prerequisites
- Python 3.8+
- OpenCV 4.5+
- dlib 19.21+
- CUDA-capable GPU (optional, for faster performance)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/realtime-face-recognition.git
cd realtime-face-recognition
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the shape predictor model (if not already present):
```bash
python download_models.py
```

## Usage

### Training the Recognition Model

1. Prepare your dataset:
   - For AT&T dataset, place the images in the `AT&T` directory
   - For custom dataset, organize in subject folders

2. Train the hybrid model:
```bash
python train_hybrid_model.py --dataset AT&T --classifier svm --pca-variance 0.95
```

Additional training options:
```
--test-size        Fraction of data to use for testing (default: 0.2)
--limit-per-subject Maximum number of images per subject (default: None)
--no-visualize     Disable eigenface and fisherface visualization
--output-dir       Directory to save the model (default: models)
--target-width     Target width for face images (default: 90)
--target-height    Target height for face images (default: 90)
--seed             Random seed for reproducibility (default: 42)
```

### Running the Recognition System

1. Run with default settings:
```bash
python realtime_face_recognition.py
```

2. Run with custom settings:
```bash
python realtime_face_recognition.py --model models/your_model.joblib --camera 0 --width 1280 --height 720 --fps 30
```

Additional runtime options:
```
--no-fps           Hide FPS counter
--no-confidence    Hide confidence scores
--no-mirror        Don't mirror camera view
--interval         Recognition interval (frames) (default: 3)
```

### Keyboard Controls

During the recognition process:
- Press `q` to quit
- Press `s` to take a screenshot
- Press `a` to add a face to the gallery

## System Architecture

```
┌─────────────────┐      ┌───────────────┐      ┌────────────────┐      ┌──────────────┐
│  Camera Thread  │──┬──>│ Frame Queue   │──┬──>│ Detection Thread│──┬──>│Detection Queue│
└─────────────────┘  │   └───────────────┘  │   └────────────────┘  │   └──────────────┘
                     │                       │                       │
                     │                       │                       │         │
                     │                       │                       │         ▼
                     │                       │                       │
                     │                       │                       │   ┌──────────────────┐
                     │                       │                       └──>│Recognition Thread │
                     │                       │                           └──────────────────┘
                     │                       │                                     │
                     │                       │                                     ▼
                     │                       │                             ┌─────────────┐
                     │                       │                             │Result Queue │
                     │                       │                             └─────────────┘
                     │                       │                                     │
                     │                       │                                     ▼
                     │                       │                           ┌──────────────────┐
                     └───────────────────────┼───────────────────────────│  Display Thread  │
                                             └───────────────────────────┤                  │
                                                                         └──────────────────┘
```

## Performance Optimization

To achieve optimal performance:
1. Use a GPU-accelerated environment
2. Adjust the recognition interval based on your hardware capabilities
3. Set an appropriate confidence threshold (0.8 recommended)
4. Use SVM classifier for better accuracy, KNN for faster performance

## Cross-Dataset Validation

The system is tested on multiple datasets:
- AT&T dataset for training
- Labeled Faces in the Wild (LFW) for testing
- Cross-validation with 5-fold stratification

## Extending the System

### Adding New Users
1. Press 'a' during recognition to add a new face
2. Enter the name in the console prompt
3. Multiple images of the same person can be added for better accuracy
4. Retrain the model with the new gallery images

### Customizing the Pipeline
- Detection parameters can be adjusted in `config.py`
- Preprocessing settings can be modified in `config.py`
- Recognition threshold can be changed in `config.py`

## Troubleshooting

### Common Issues

1. **Slow Detection**: 
   - Reduce resolution with `--width` and `--height` flags
   - Increase recognition interval with `--interval` flag

2. **False Positives**: 
   - Increase confidence threshold in `config.py`
   - Adjust `min_neighbors` parameter in detection config

3. **Missing Landmarks**: 
   - Ensure `shape_predictor_68_face_landmarks.dat` is in the correct directory
   - Check lighting conditions, improve face illumination

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AT&T Laboratories Cambridge for the AT&T Database of Faces
- The dlib developers for the face detection and landmark detection algorithms
- The OpenCV team for the computer vision toolbox 