# Image Preprocessing Package

A focused Python package for image preprocessing, implementing core normalization and enhancement techniques.

## Core Features

This package focuses on four essential image preprocessing techniques:

1. **Grayscale Conversion**: Convert color images to grayscale for consistent processing
2. **Illumination Normalization**: Correct for lighting variations using various techniques
3. **Image Enhancement**: Improve image quality with adaptive enhancement methods
4. **Scale Normalization**: Resize images to standard dimensions with aspect ratio preservation

## Installation

### Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- Pillow

### Installation from source

```bash
git clone https://github.com/example/face_preproc.git
cd face_preproc
pip install -e .
```

## Usage

### Command Line Tool

The package provides a command-line tool for batch processing images:

```bash
process-images --input-dir=./input_images --output-dir=./processed_images
```

Available options:

```
--input-dir=<dir>          Directory containing input images
--output-dir=<dir>         Directory to save processed images
--target-size=224x224      Target size for images
--illumination=clahe       Illumination normalization method
                           (none/hist_eq/clahe/gamma/dog/tantriggs/all)
--enhancement=adaptive     Image enhancement method
                           (none/sharpen/contrast/adaptive)
--preserve-ratio           Preserve aspect ratio when resizing
--no-grayscale             Don't convert images to grayscale
--visualize                Create visualizations of preprocessing steps
```

### Basic usage with the simplified ImagePreprocessor

```python
from face_preproc.core.image_preprocessor import ImagePreprocessor

# Initialize preprocessor
preprocessor = ImagePreprocessor(
    target_size=(224, 224),
    force_grayscale=True
)

# Process an image
import cv2
image = cv2.imread('path/to/image.jpg')
processed_img = preprocessor.preprocess(
    image,
    illumination_method='clahe',
    enhancement_method='adaptive',
    preserve_aspect_ratio=True
)

# Save processed image
cv2.imwrite('processed.jpg', processed_img)
```

### Processing all images in a directory

The package includes an example script to process all images in a directory:

```bash
python -m face_preproc.examples.image_preprocessing_example input_dir output_dir
```

## Illumination Normalization Methods

The package includes multiple illumination normalization methods:

- **hist_eq**: Simple histogram equalization
- **clahe**: Contrast Limited Adaptive Histogram Equalization (better for local contrast)
- **gamma**: Gamma correction (adapts to image brightness)
- **dog**: Difference of Gaussians (enhances edges, removes illumination gradients)
- **tantriggs**: Tan-Triggs preprocessing (robust to illumination changes)
- **all**: Apply multiple methods and combine results

## Enhancement Methods

Several image enhancement methods are available:

- **none**: No enhancement applied
- **sharpen**: Apply sharpening filter to enhance edges
- **contrast**: Global contrast enhancement with histogram equalization
- **adaptive**: Intelligent enhancement based on image properties:
  - For low contrast images: CLAHE + mild sharpening
  - For dark images: Gamma correction + CLAHE
  - For bright images: Gamma correction + denoising
  - For normal images: Denoising + mild sharpening

## Scale Normalization

Images can be resized to a target size with optional aspect ratio preservation:

- With `preserve_aspect_ratio=True`: Maintains aspect ratio and adds padding
- With `preserve_aspect_ratio=False`: Simple resize to target dimensions

## Package Structure

```
face_preproc/
├── face_preproc/
│   ├── core/
│   │   ├── image_preprocessor.py  # Simplified preprocessor
│   │   ├── preprocessor.py        # Face-specific preprocessor
│   ├── utils/
│   │   ├── visualization.py       # Visualization tools
│   │   ├── logging_utils.py       # Logging utilities
│   ├── scripts/
│   │   ├── process_images.py      # Image processing script
│   ├── examples/
│   │   ├── image_preprocessing_example.py  # Simple preprocessing example
├── requirements.txt
└── README.md 