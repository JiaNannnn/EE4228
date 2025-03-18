#!/usr/bin/env python
"""
Basic Face Preprocessing Example

This script demonstrates how to use the face preprocessing package
to preprocess a single image with a face.
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(parent_dir))
    __package__ = "face_preproc.examples"

# Import from package
from face_preproc.core.preprocessor import FacePreprocessor
from face_preproc.utils.visualization import create_preprocessing_visualization

def process_single_image(image_path, output_path=None):
    """
    Process a single image with face preprocessing
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image (optional)
        
    Returns:
        Processed image and intermediate results
    """
    # Initialize preprocessor
    preprocessor = FacePreprocessor(
        target_size=(224, 224),
        force_grayscale=True
    )
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    print(f"Processing image: {image_path}")
    print(f"Original image shape: {image.shape}")
    
    # Process image
    processed_img, intermediates = preprocessor.preprocess(
        image,
        illumination_method="all",
        enhancement_method="adaptive",
        preserve_aspect_ratio=True,
        return_intermediates=True
    )
    
    if processed_img is None:
        print("Failed to detect face in the image")
        return None, None
    
    print(f"Processed image shape: {processed_img.shape}")
    
    # Save processed image if output path provided
    if output_path and processed_img is not None:
        cv2.imwrite(output_path, processed_img)
        print(f"Saved processed image to {output_path}")
    
    return processed_img, intermediates

def main():
    """Main function to demonstrate face preprocessing"""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python basic_example.py <input_image> [output_image]")
        sys.exit(1)
    
    # Get image paths
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Process image
        processed_img, intermediates = process_single_image(input_path, output_path)
        
        if processed_img is not None:
            # Show visualization
            fig = create_preprocessing_visualization("Example", intermediates)
            plt.show()
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 