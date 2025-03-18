#!/usr/bin/env python
"""
Simple Image Preprocessing Example

This script demonstrates how to use the simplified image preprocessing module
to apply grayscale conversion, illumination normalization, enhancement, and
scale normalization to images.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(parent_dir))
    __package__ = "face_preproc.examples"

# Import package modules
from face_preproc.core.image_preprocessor import ImagePreprocessor
from face_preproc.utils.logging_utils import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

def process_images(input_dir, output_dir):
    """
    Process all images in a directory using simplified image preprocessing
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
    """
    # Set up logging
    setup_logging()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        force_grayscale=True
    )
    
    # Get all image files
    input_path = Path(input_dir)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for i, img_path in enumerate(image_files):
        img_name = img_path.name
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_processed.jpg")
        
        logger.info(f"Processing image {i+1}/{len(image_files)}: {img_name}")
        
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Failed to read image: {img_path}")
                continue
            
            # Process image
            processed_img, intermediates = preprocessor.preprocess(
                img, 
                illumination_method='clahe',
                enhancement_method='adaptive',
                preserve_aspect_ratio=True,
                return_intermediates=True
            )
            
            # Save processed image
            cv2.imwrite(output_path, processed_img)
            logger.info(f"Saved processed image to {output_path}")
            
            # Create visualization for first few images
            if i < 3:  # Only visualize first 3 images
                visualize_preprocessing_steps(img_name, intermediates, os.path.join(viz_dir, f"{os.path.splitext(img_name)[0]}_steps.png"))
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")

def visualize_preprocessing_steps(image_name, intermediates, output_path=None):
    """
    Create a visualization of preprocessing steps
    
    Args:
        image_name: Name of the image (for title)
        intermediates: Dictionary of intermediate results
        output_path: Path to save visualization (optional)
    """
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Plot original image
    if 'original' in intermediates:
        img = intermediates['original']
        if len(img.shape) == 3:
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Original Image")
    
    # Plot grayscale image
    if 'grayscale' in intermediates:
        axes[1].imshow(intermediates['grayscale'], cmap='gray')
        axes[1].set_title("Grayscale Conversion")
    
    # Plot illumination normalized image
    if 'illumination_normalized' in intermediates:
        axes[2].imshow(intermediates['illumination_normalized'], cmap='gray')
        axes[2].set_title("Illumination Normalized")
    
    # Plot enhanced image
    if 'enhanced' in intermediates:
        axes[3].imshow(intermediates['enhanced'], cmap='gray')
        axes[3].set_title("Enhanced")
    
    # Plot final image
    if 'final' in intermediates:
        axes[4].imshow(intermediates['final'], cmap='gray')
        axes[4].set_title("Final (Resized)")
    
    # Remove any unused axes
    for i in range(5, len(axes)):
        fig.delaxes(axes[i])
    
    # Set title and adjust layout
    plt.suptitle(f"Image Preprocessing Steps: {image_name}", fontsize=16)
    plt.tight_layout()
    
    # Save visualization if requested
    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()

def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python image_preprocessing_example.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    process_images(input_dir, output_dir)

if __name__ == "__main__":
    main() 