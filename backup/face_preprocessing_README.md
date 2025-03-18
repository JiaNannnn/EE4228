# Face Preprocessing Pipeline

A comprehensive face preprocessing pipeline for PCA-based face recognition with multiple detection and enhancement methods.

## Features

- **Multiple Face Detection Methods**:
  - Haar Cascade (traditional, fast)
  - HOG+SVM (robust to pose variations)
  - DNN-based detector (high accuracy)
  - Auto mode with fallback mechanisms

- **Precise Facial Alignment**:
  - Facial landmark detection (68 points)
  - Eye-based alignment
  - Automatic rotation correction

- **Illumination Normalization Methods**:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Gamma Correction
  - Difference of Gaussians (DoG)
  - Multi-method approach combining techniques

- **Advanced Image Enhancement**:
  - Adaptive processing based on image characteristics
  - Detail enhancement for feature preservation
  - Edge enhancement for improved recognition
  - Noise reduction techniques

- **Robust Error Handling**:
  - Comprehensive logging
  - Fallback mechanisms
  - Intermediate result visualization

## Installation Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- dlib (optional, for facial landmarks)
- tqdm (for progress bars)

### Additional Model Files

For landmark detection, download the shape predictor model:
1. Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Extract and place in the `models` directory

For DNN-based detection, download the model files:
1. Download the model and config files from OpenCV's repository
2. Place in the `models` directory

## Usage

### Basic Usage

```python
from face_preprocessor import FacePreprocessor, process_gallery_folders

# Process all images in gallery folders
process_gallery_folders()
```

### Advanced Usage with Custom Parameters

```python
# Initialize with custom parameters
preprocessor = FacePreprocessor(
    target_size=(100, 100),
    detector_type='dnn',  # 'haar', 'hog', 'dnn', or 'auto'
    enable_landmarks=True,
    illumination_method='multi'  # 'clahe', 'gamma', 'dog', or 'multi'
)

# Process a single image
processed_img = preprocessor.preprocess_image(
    'path/to/image.jpg',
    'path/to/save/processed.jpg',
    enhancement_method='adaptive',  # 'adaptive', 'basic', 'detail', or 'edge'
    add_margin=True,
    margin_ratio=0.3,
    return_intermediates=False
)

# Process all gallery folders with custom settings
process_gallery_folders(
    gallery_dir="custom_gallery",
    detector_type='hog',
    illumination_method='clahe',
    enhancement_method='detail',
    use_landmarks=True,
    target_size=(100, 100)
)
```

### Getting Intermediate Results for Debugging

```python
# Get intermediate results for visualization
processed_img, intermediates = preprocessor.preprocess_image(
    'path/to/image.jpg',
    'path/to/save/processed.jpg',
    enhancement_method='adaptive',
    return_intermediates=True
)

# Visualize intermediate results
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
for i, (name, img) in enumerate(intermediates.items()):
    plt.subplot(2, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(name)
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## Configuration Options

### Face Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `haar` | Haar Cascade Classifier | Fast detection, frontal faces |
| `hog` | Histogram of Oriented Gradients | Better with pose variations |
| `dnn` | Deep Neural Network | High accuracy, various poses |
| `auto` | Try methods in sequence | General purpose use |

### Illumination Normalization Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `clahe` | Contrast Limited Adaptive Histogram Equalization | General purpose |
| `gamma` | Gamma Correction | Dark or bright images |
| `dog` | Difference of Gaussians | Preserving edges |
| `multi` | Combines multiple methods | Difficult lighting |

### Image Enhancement Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `basic` | Simple histogram equalization | Basic enhancement |
| `adaptive` | Adapts to image characteristics | General purpose |
| `detail` | Enhances facial details | Preserving features |
| `edge` | Emphasizes edges | Improving recognition |

## Pipeline Workflow

1. **Face Detection**: Detect faces using the selected method
2. **Region Extraction**: Extract the face region with optional margin
3. **Illumination Normalization**: Apply the selected normalization method
4. **Face Alignment**: Align the face using landmarks or eye positions
5. **Resize**: Standardize to the target size (default: 100x100 pixels)
6. **Enhancement**: Apply the selected enhancement method
7. **Output**: Save the processed image and return it

## Troubleshooting

### Common Issues

- **No face detected**: Try different detection methods or adjust parameters
- **Poor alignment**: Enable landmarks if available or check lighting conditions
- **Over-enhanced images**: Use 'basic' enhancement or adjust intensity
- **Missing landmarks**: Download the shape predictor model

### Logging

The pipeline uses Python's logging module with file and console handlers:
- Log file: `face_preprocessing.log`
- Log level: INFO (can be changed in code)

## License

This project is available under the MIT License.

## Acknowledgments

- OpenCV for computer vision algorithms
- dlib for facial landmark detection
- The face recognition community for research and techniques 