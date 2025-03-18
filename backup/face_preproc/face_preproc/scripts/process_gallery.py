#!/usr/bin/env python
"""
Gallery Processing Script

This script processes a directory of images containing facial photos
and applies face preprocessing techniques to normalize and enhance them.
Images are typically organized by subject (person) in subdirectories.

Usage:
    python -m face_preproc.scripts.process_gallery --gallery-dir=<dir> --output-dir=<dir> [options]

Options:
    --gallery-dir=<dir>        Directory containing gallery images (organized by subject)
    --output-dir=<dir>         Directory to save processed images
    --target-size=<size>       Target size for face images (default: 224x224)
    --illumination=<method>    Illumination normalization method (default: all)
    --enhancement=<method>     Image enhancement method (default: adaptive)
    --color-mode=<mode>        Color mode (rgb/grayscale, default: grayscale)
    --preserve-ratio           Preserve aspect ratio when resizing
    --no-grayscale             Don't convert images to grayscale
    --visualize-first          Create visualizations for first image of each subject
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
from face_preproc.core.preprocessor import FacePreprocessor
from face_preproc.utils.visualization import (
    create_preprocessing_visualization,
    create_success_rate_visualization,
    generate_preprocessing_report
)
from face_preproc.utils.logging_utils import setup_logging, log_process_start, log_process_complete

# Configure logger
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Process gallery images for face recognition")
    
    parser.add_argument("--gallery-dir", required=True, 
                        help="Directory containing gallery images (organized by subject)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save processed images")
    parser.add_argument("--target-size", default="224x224",
                        help="Target size for face images (WxH)")
    parser.add_argument("--illumination", default="all",
                        choices=["none", "hist_eq", "clahe", "gamma", "dog", "tantriggs", "all"],
                        help="Illumination normalization method")
    parser.add_argument("--enhancement", default="adaptive",
                        choices=["none", "sharpen", "contrast", "adaptive"],
                        help="Image enhancement method")
    parser.add_argument("--color-mode", default="grayscale",
                        choices=["rgb", "grayscale"],
                        help="Color mode for processing")
    parser.add_argument("--preserve-ratio", action="store_true",
                        help="Preserve aspect ratio when resizing")
    parser.add_argument("--no-grayscale", action="store_true",
                        help="Don't convert images to grayscale")
    parser.add_argument("--visualize-first", action="store_true",
                        help="Create visualizations for first image of each subject")
    parser.add_argument("--log-file", default=None,
                        help="Path to log file (default: 'gallery_preprocessing.log' in output dir)")
    
    args = parser.parse_args()
    
    # Set log_file if not specified
    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, "gallery_preprocessing.log")
    
    # Parse target size
    try:
        width, height = map(int, args.target_size.split("x"))
        args.target_size = (width, height)
    except ValueError:
        parser.error("Target size must be in format WxH, e.g., 224x224")
    
    return args

def process_gallery(args):
    """Process all images in the gallery directory"""
    # Set up logging
    setup_logging(args.log_file)
    
    # Log configuration
    config = {
        "Gallery directory": args.gallery_dir,
        "Output directory": args.output_dir,
        "Target size": args.target_size,
        "Illumination method": args.illumination,
        "Enhancement method": args.enhancement,
        "Color mode": "RGB" if args.no_grayscale else "Grayscale",
        "Preserve aspect ratio": args.preserve_ratio,
        "Visualization for first image": args.visualize_first
    }
    log_process_start(logger, config)
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize_first:
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize face preprocessor
    preproc = FacePreprocessor(
        target_size=args.target_size,
        force_grayscale=not args.no_grayscale
    )
    
    # Find all subjects (directories) in gallery
    gallery_path = Path(args.gallery_dir)
    subject_dirs = [d for d in gallery_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(subject_dirs)} subjects in gallery")
    
    if not subject_dirs:
        logger.error(f"No subject directories found in {args.gallery_dir}")
        return
    
    # Initialize statistics
    subject_stats = {}
    total_images = 0
    total_success = 0
    total_time = 0
    
    # Process each subject
    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        logger.info(f"Processing subject: {subject_name}")
        
        # Create output directory for this subject
        subject_output_dir = os.path.join(args.output_dir, subject_name)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Get all image files for this subject
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            image_files.extend(list(subject_dir.glob(f"*{ext}")))
            image_files.extend(list(subject_dir.glob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(image_files)} images for subject {subject_name}")
        
        # Initialize statistics for this subject
        subject_stats[subject_name] = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'success_rate': 0
        }
        
        # Process each image
        for i, img_path in enumerate(image_files):
            img_name = img_path.name
            output_path = os.path.join(subject_output_dir, 
                                       f"{os.path.splitext(img_name)[0]}_processed.jpg")
            
            logger.info(f"Processing image {i+1}/{len(image_files)}: {img_name}")
            
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.error(f"Failed to read image: {img_path}")
                    subject_stats[subject_name]['failed'] += 1
                    continue
                
                # Start timing
                start_time = time.time()
                
                # Process image
                processed_img, intermediates = preproc.preprocess(
                    img, 
                    illumination_method=args.illumination,
                    enhancement_method=args.enhancement,
                    preserve_aspect_ratio=args.preserve_ratio,
                    return_intermediates=True
                )
                
                # End timing
                elapsed = time.time() - start_time
                total_time += elapsed
                
                if processed_img is not None:
                    # Save processed image
                    cv2.imwrite(output_path, processed_img)
                    logger.info(f"Saved processed image to {output_path}")
                    
                    # Update statistics
                    subject_stats[subject_name]['success'] += 1
                    total_success += 1
                    
                    # Create visualization for first image if requested
                    if args.visualize_first and i == 0:
                        viz_path = os.path.join(viz_dir, f"{subject_name}_preprocessing.png")
                        create_preprocessing_visualization(
                            subject_name, intermediates, viz_path
                        )
                else:
                    logger.warning(f"Face detection failed for {img_path}")
                    subject_stats[subject_name]['failed'] += 1
                
                logger.info(f"Processing time: {elapsed:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                subject_stats[subject_name]['failed'] += 1
            
            total_images += 1
    
    # Calculate success rate for each subject
    for subject, stats in subject_stats.items():
        if stats['total'] > 0:
            stats['success_rate'] = (stats['success'] / stats['total']) * 100
    
    # Generate report
    report_path = os.path.join(args.output_dir, "preprocessing_report.txt")
    generate_preprocessing_report(
        subject_stats, total_images, total_success, total_time, report_path
    )
    
    # Create success rate visualization
    viz_path = os.path.join(args.output_dir, "success_rate_by_subject.png")
    create_success_rate_visualization(subject_stats, viz_path)
    
    # Log completion
    stats = {
        'Total images': total_images,
        'Successful detections': total_success,
        'Success rate': f"{(total_success / total_images) * 100:.2f}%" if total_images > 0 else "0%",
        'Total processing time': f"{total_time:.2f} seconds",
        'Average time per image': f"{total_time / total_images:.2f} seconds" if total_images > 0 else "0 seconds"
    }
    log_process_complete(logger, stats)

def main():
    """Main entry point"""
    args = parse_arguments()
    process_gallery(args)

if __name__ == "__main__":
    main() 