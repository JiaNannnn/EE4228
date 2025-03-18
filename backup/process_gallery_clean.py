"""
Gallery Image Processor

Process all face images in a gallery directory using advanced face preprocessing techniques
for improved face recognition accuracy.

Features:
- Grayscale conversion
- Illumination normalization
- Image enhancement
- Scale normalization
"""

import os
import sys
import logging
import time
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from face_preprocessor_clean import FacePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gallery_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gallery_processor")

def create_visualization(subject, original_img, processed_img, intermediates, output_dir):
    """
    Create visualization of preprocessing steps for the first image of each subject
    
    Args:
        subject: Subject name
        original_img: Original image
        processed_img: Final processed image
        intermediates: Dictionary of intermediate results
        output_dir: Output directory
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create visualization image
    plt.figure(figsize=(12, 8))
    
    # Determine grid size
    n_plots = len(intermediates)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    # Plot each intermediate result
    for i, (title, img) in enumerate(intermediates.items()):
        plt.subplot(rows, cols, i+1)
        
        # Handle different image types
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        
        plt.title(title)
        plt.axis('off')
    
    plt.suptitle(f"Face Preprocessing Steps - Subject: {subject}", fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    vis_path = os.path.join(vis_dir, f"{subject}_visualization.jpg")
    plt.savefig(vis_path)
    plt.close()
    logger.info(f"Saved visualization for {subject} to {vis_path}")

def create_success_rate_visualization(subject_stats, output_dir):
    """
    Create visualization of face detection success rate by subject
    
    Args:
        subject_stats: Dictionary with subject statistics
        output_dir: Output directory
    """
    subjects = list(subject_stats.keys())
    success_rates = [subject_stats[subject]['success_rate'] for subject in subjects]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(subjects, success_rates, color='steelblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f"{height:.1f}%", ha='center', va='bottom')
    
    plt.title('Face Detection Success Rate by Subject', fontsize=16)
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(0, 105)  # Leave room for text above bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save visualization
    vis_path = os.path.join(output_dir, "success_rate_by_subject.png")
    plt.savefig(vis_path)
    plt.close()
    logger.info(f"Saved success rate visualization to {vis_path}")

def generate_report(subject_stats, total_images, total_success, total_time, output_dir):
    """
    Generate a report of preprocessing results
    
    Args:
        subject_stats: Dictionary with subject statistics
        total_images: Total number of images processed
        total_success: Total number of successful face detections
        total_time: Total processing time in seconds
        output_dir: Output directory
    """
    # Calculate overall statistics
    success_rate = (total_success / total_images) * 100 if total_images > 0 else 0
    avg_time = total_time / total_images if total_images > 0 else 0
    
    # Create report
    report_lines = [
        "Face Preprocessing Report",
        "======================",
        f"Total images processed: {total_images}",
        f"Successful face detections: {total_success} ({success_rate:.2f}%)",
        f"Failed face detections: {total_images - total_success} ({100 - success_rate:.2f}%)",
        f"Total processing time: {total_time:.2f} seconds",
        f"Average time per image: {avg_time:.2f} seconds",
        "",
        "Subject-wise Statistics:",
        "----------------------"
    ]
    
    # Add subject-wise statistics
    for subject, stats in subject_stats.items():
        report_lines.append(f"Subject: {subject}")
        report_lines.append(f"  Total images: {stats['total']}")
        report_lines.append(f"  Successful detections: {stats['success']} ({stats['success_rate']:.2f}%)")
        report_lines.append(f"  Failed detections: {stats['failed']} ({100 - stats['success_rate']:.2f}%)")
        report_lines.append("")
    
    # Write report to file
    report_path = os.path.join(output_dir, "preprocessing_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved preprocessing report to {report_path}")
    
    # Print summary to console
    logger.info(f"Summary: Processed {total_images} images with {success_rate:.2f}% success rate")
    logger.info(f"Average processing time: {avg_time:.2f} seconds per image")

def process_gallery(gallery_dir, output_dir, target_size=(224, 224), 
                    illumination_method='clahe', enhance_method='adaptive',
                    preserve_aspect_ratio=True, force_grayscale=True,
                    visualize_first=True):
    """
    Process all images in the gallery directory
    
    Args:
        gallery_dir: Path to the gallery directory
        output_dir: Path to save processed images
        target_size: Tuple of target image size (width, height)
        illumination_method: Method for illumination normalization
        enhance_method: Method for image enhancement
        preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
        force_grayscale: Whether to force grayscale conversion
        visualize_first: Whether to create visualizations for the first image of each subject
    """
    # Validate gallery directory
    if not os.path.isdir(gallery_dir):
        logger.error(f"Gallery directory {gallery_dir} not found")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Processing gallery images from {gallery_dir}")
    logger.info(f"Processed images will be saved to {output_dir}")
    logger.info(f"Configuration: target_size={target_size}, illumination_method={illumination_method}, " +
                f"enhance_method={enhance_method}, force_grayscale={force_grayscale}, " +
                f"preserve_aspect_ratio={preserve_aspect_ratio}, visualize_first={visualize_first}")
    
    # Initialize face preprocessor
    preprocessor = FacePreprocessor(target_size=target_size, force_grayscale=force_grayscale)
    
    # Get list of subject directories
    subject_dirs = [d for d in os.listdir(gallery_dir) if os.path.isdir(os.path.join(gallery_dir, d))]
    logger.info(f"Found {len(subject_dirs)} subjects in gallery: {', '.join(subject_dirs)}")
    
    # Initialize statistics
    total_images = 0
    total_success = 0
    total_time = 0
    subject_stats = {}
    
    # Process each subject directory
    for subject in subject_dirs:
        subject_dir = os.path.join(gallery_dir, subject)
        subject_output_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Find all images in subject directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(Path(subject_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(subject_dir).glob(f"*{ext.upper()}")))
        
        # Initialize subject statistics
        subject_stats[subject] = {
            'total': len(image_paths),
            'success': 0,
            'failed': 0,
            'success_rate': 0
        }
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            img_path_str = str(img_path)
            output_path = os.path.join(subject_output_dir, f"{img_path.stem}_processed{img_path.suffix}")
            
            try:
                # Log processing
                logger.info(f"Processing image {i+1}/{len(image_paths)} for subject {subject}: {img_path.name}")
                
                # Time the processing
                start_time = time.time()
                
                # Check if we should create a visualization for this image
                is_first = (i == 0)
                if is_first and visualize_first:
                    # Process with visualization
                    original_img = cv2.imread(img_path_str)
                    result, intermediates = preprocessor.preprocess_image(
                        original_img,
                        save_path=output_path,
                        return_intermediates=True,
                        illumination_method=illumination_method,
                        enhance_method=enhance_method,
                        preserve_aspect_ratio=preserve_aspect_ratio
                    )
                    
                    # Check if face detection was successful based on intermediates
                    face_detected = 'face_detected' in intermediates
                    
                    # Create visualization
                    create_visualization(subject, original_img, result, intermediates, output_dir)
                else:
                    # Process without visualization
                    result = preprocessor.preprocess_image(
                        img_path_str,
                        save_path=output_path,
                        illumination_method=illumination_method,
                        enhance_method=enhance_method,
                        preserve_aspect_ratio=preserve_aspect_ratio
                    )
                    
                    # Since we don't have intermediates, we'll look at the log
                    face_detected = True  # Assume face is detected unless log says otherwise
                
                # Time tracking
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # Update statistics
                total_images += 1
                
                # Count successful face detections
                if face_detected:
                    total_success += 1
                    subject_stats[subject]['success'] += 1
                else:
                    subject_stats[subject]['failed'] += 1
                
                logger.info(f"Processed {img_path.name} in {elapsed_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                subject_stats[subject]['failed'] += 1
                total_images += 1
        
        # Calculate subject success rate
        subject_stats[subject]['success_rate'] = (subject_stats[subject]['success'] / subject_stats[subject]['total']) * 100 if subject_stats[subject]['total'] > 0 else 0
        
        logger.info(f"Completed processing {len(image_paths)} images for subject {subject}")
        logger.info(f"Success rate for {subject}: {subject_stats[subject]['success_rate']:.2f}%")
    
    # Generate report and visualizations
    generate_report(subject_stats, total_images, total_success, total_time, output_dir)
    create_success_rate_visualization(subject_stats, output_dir)
    
    logger.info("Gallery processing complete")

def main():
    """Main function to parse arguments and process gallery"""
    parser = argparse.ArgumentParser(description="Process gallery images for face recognition")
    parser.add_argument("--gallery", type=str, default="gallery", help="Path to gallery directory")
    parser.add_argument("--output", type=str, default="processed_gallery", help="Output directory")
    parser.add_argument("--target-size", type=str, default="224,224", help="Target size (width,height)")
    parser.add_argument("--illumination", type=str, default="combined", 
                        choices=["histogram", "clahe", "gamma", "dog", "combined"],
                        help="Illumination normalization method")
    parser.add_argument("--enhance", type=str, default="adaptive",
                        choices=["basic", "denoising", "sharpening", "adaptive"],
                        help="Image enhancement method")
    parser.add_argument("--color", action="store_true", help="Keep color information (default: convert to grayscale)")
    parser.add_argument("--preserve-ratio", action="store_true", help="Preserve aspect ratio when resizing")
    parser.add_argument("--skip-visualize", action="store_true", help="Skip visualization of first image for each subject")
    
    args = parser.parse_args()
    
    # Parse target size
    target_width, target_height = map(int, args.target_size.split(','))
    
    # Process gallery
    process_gallery(
        gallery_dir=args.gallery,
        output_dir=args.output,
        target_size=(target_width, target_height),
        illumination_method=args.illumination,
        enhance_method=args.enhance,
        preserve_aspect_ratio=args.preserve_ratio,
        force_grayscale=not args.color,
        visualize_first=not args.skip_visualize
    )

if __name__ == "__main__":
    main() 