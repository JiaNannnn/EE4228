#!/usr/bin/env python
"""
Image Processing Script

This script processes a directory of images using the core image preprocessing
functionality: grayscale conversion, illumination normalization, image enhancement,
and scale normalization.

Usage:
    python -m face_preproc.scripts.process_images --input-dir=<dir> --output-dir=<dir> [options]

Options:
    --input-dir=<dir>          Directory containing input images
    --output-dir=<dir>         Directory to save processed images
    --target-size=<size>       Target size for images (default: 224x224)
    --illumination=<method>    Illumination normalization method (default: clahe)
    --enhancement=<method>     Image enhancement method (default: adaptive)
    --preserve-ratio           Preserve aspect ratio when resizing
    --no-grayscale             Don't convert images to grayscale
    --visualize                Create visualizations of preprocessing steps
    --recursive                Search for images recursively in subdirectories
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(parent_dir))
    __package__ = "face_preproc.scripts"

# Import package modules
from face_preproc.core.image_preprocessor import ImagePreprocessor
from face_preproc.utils.logging_utils import setup_logging, log_process_start, log_process_complete

# Configure logger
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Process images with preprocessing techniques")
    
    parser.add_argument("--input-dir", required=True, 
                        help="Directory containing input images")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save processed images")
    parser.add_argument("--target-size", default="224x224",
                        help="Target size for images (WxH)")
    parser.add_argument("--illumination", default="clahe",
                        choices=["none", "hist_eq", "clahe", "gamma", "dog", "tantriggs", "all"],
                        help="Illumination normalization method")
    parser.add_argument("--enhancement", default="adaptive",
                        choices=["none", "sharpen", "contrast", "adaptive"],
                        help="Image enhancement method")
    parser.add_argument("--preserve-ratio", action="store_true",
                        help="Preserve aspect ratio when resizing")
    parser.add_argument("--no-grayscale", action="store_true",
                        help="Don't convert images to grayscale")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualizations of preprocessing steps")
    parser.add_argument("--recursive", action="store_true",
                        help="Search for images recursively in subdirectories")
    parser.add_argument("--log-file", default=None,
                        help="Path to log file")
    
    args = parser.parse_args()
    
    # Set log_file if not specified
    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, "image_preprocessing.log")
    
    # Parse target size
    try:
        width, height = map(int, args.target_size.split("x"))
        args.target_size = (width, height)
    except ValueError:
        parser.error("Target size must be in format WxH, e.g., 224x224")
    
    return args

def get_image_files(directory, recursive=False):
    """Get all image files from a directory"""
    input_path = Path(directory)
    image_files = []
    
    # Define image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP']
    
    # Search pattern based on recursion option
    if recursive:
        for ext in extensions:
            # Use rglob for recursive search
            image_files.extend(list(input_path.rglob(f"*{ext}")))
    else:
        for ext in extensions:
            # Use glob for non-recursive search
            image_files.extend(list(input_path.glob(f"*{ext}")))
    
    return image_files

def process_images(args):
    """Process all images in the input directory"""
    # Set up logging
    setup_logging(args.log_file)
    
    # Log configuration
    config = {
        "Input directory": args.input_dir,
        "Output directory": args.output_dir,
        "Target size": args.target_size,
        "Illumination method": args.illumination,
        "Enhancement method": args.enhancement,
        "Preserve aspect ratio": args.preserve_ratio,
        "Force grayscale": not args.no_grayscale,
        "Create visualizations": args.visualize,
        "Recursive search": args.recursive
    }
    log_process_start(logger, config)
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize image preprocessor
    preprocessor = ImagePreprocessor(
        target_size=args.target_size,
        force_grayscale=not args.no_grayscale
    )
    
    # Get all image files from input directory
    image_files = get_image_files(args.input_dir, args.recursive)
    
    logger.info(f"Found {len(image_files)} images in {args.input_dir}")
    
    if not image_files:
        logger.error(f"No image files found in {args.input_dir}")
        return
    
    # Initialize statistics
    total_images = 0
    total_processed = 0
    total_time = 0
    
    # Process each image
    for i, img_path in enumerate(image_files):
        # Get relative path from input directory to create output filename
        if args.recursive:
            try:
                rel_path = img_path.relative_to(Path(args.input_dir)) 
                output_filename = f"{rel_path.stem}_processed{rel_path.suffix}"
            except Exception:
                # Fallback if relative_to fails
                output_filename = f"{img_path.stem}_processed{img_path.suffix}"
        else:
            output_filename = f"{img_path.stem}_processed{img_path.suffix}"
        
        # Create output path
        output_path = os.path.join(args.output_dir, output_filename)
        
        logger.info(f"Processing image {i+1}/{len(image_files)}: {img_path.name}")
        
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Failed to read image: {img_path}")
                continue
            
            # Start timing
            start_time = time.time()
            
            # Process image
            processed_img, intermediates = preprocessor.preprocess(
                img, 
                illumination_method=args.illumination,
                enhancement_method=args.enhancement,
                preserve_aspect_ratio=args.preserve_ratio,
                return_intermediates=args.visualize
            )
            
            # End timing
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Save processed image
            cv2.imwrite(output_path, processed_img)
            logger.info(f"Saved processed image to {output_path}")
            total_processed += 1
            
            # Create visualization if requested
            if args.visualize:
                viz_path = os.path.join(viz_dir, f"{img_path.stem}_steps.png")
                create_visualization(img_path.name, intermediates, viz_path)
            
            logger.info(f"Processing time: {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
        
        total_images += 1
    
    # Log completion
    stats = {
        'Total images': total_images,
        'Successfully processed': total_processed,
        'Success rate': f"{(total_processed / total_images) * 100:.2f}%" if total_images > 0 else "0%",
        'Total processing time': f"{total_time:.2f} seconds",
        'Average time per image': f"{total_time / total_images:.2f} seconds" if total_images > 0 else "0 seconds"
    }
    log_process_complete(logger, stats)

def create_visualization(image_name, intermediates, output_path):
    """Create visualization of preprocessing steps"""
    try:
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        # Plot each step
        steps = ['original', 'grayscale', 'illumination_normalized', 'enhanced', 'final']
        titles = ['Original', 'Grayscale', 'Illumination Normalized', 'Enhanced', 'Final (Resized)']
        
        for i, (step, title) in enumerate(zip(steps, titles)):
            if step in intermediates:
                img = intermediates[step]
                if step == 'original' and len(img.shape) == 3:
                    # Convert BGR to RGB for display
                    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    # Grayscale display
                    axes[i].imshow(img, cmap='gray')
                axes[i].set_title(title)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
        
        # Remove any unused axes
        for i in range(5, len(axes)):
            fig.delaxes(axes[i])
        
        # Set title and adjust layout
        plt.suptitle(f"Image Preprocessing Steps: {image_name}", fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved visualization to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")

def main():
    """Main entry point"""
    args = parse_arguments()
    process_images(args)

if __name__ == "__main__":
    main() 