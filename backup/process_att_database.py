"""
Process the AT&T Face Database using the advanced face preprocessing system
to enhance facial recognition accuracy.

This script:
1. Processes all images in the AT&T database
2. Applies advanced preprocessing techniques
3. Saves processed images in a structured way
4. Generates a report of the preprocessing results
"""

import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time

# Import the advanced face preprocessor
from face_preprocessor import FacePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("att_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("att_processor")

def read_pgm(pgmf):
    """Read PGM file format."""
    # Skip the magic number line
    pgmf.readline()
    
    # Skip potential comment lines
    line = pgmf.readline()
    while line.startswith('#'):
        line = pgmf.readline()
    
    # Read dimensions
    width, height = map(int, line.strip().split())
    
    # Skip the max gray value line
    pgmf.readline()
    
    # Read the binary data
    buffer = pgmf.read()
    
    # Convert to numpy array
    return np.frombuffer(buffer, dtype=np.uint8).reshape((height, width))

def process_att_database(att_dir, output_dir, config=None):
    """
    Process all images in the AT&T database using the advanced face preprocessor.
    
    Args:
        att_dir: Path to the AT&T database directory
        output_dir: Path to save processed images
        config: Dictionary with preprocessing configuration parameters
    
    Returns:
        processing_stats: Dictionary with processing statistics
    """
    if config is None:
        config = {
            'target_size': (112, 112),  # Preserve original size by default
            'detector_type': 'auto',
            'enable_landmarks': True,
            'use_mediapipe': True,
            'illumination_method': 'multi',
            'enable_attention': True,
            'smart_crop': True,
            'edge_aware': True,
            'enhance_method': 'adaptive',
            'enhance_intensity': 1.0
        }
    
    # Initialize the face preprocessor with the configuration
    preprocessor = FacePreprocessor(
        target_size=config['target_size'],
        detector_type=config['detector_type'],
        enable_landmarks=config['enable_landmarks'],
        use_mediapipe=config['use_mediapipe'],
        illumination_method=config['illumination_method'],
        enable_attention=config['enable_attention'],
        smart_crop=config['smart_crop'],
        edge_aware=config['edge_aware'],
        enhance_method=config['enhance_method'],
        enhance_intensity=config['enhance_intensity']
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistics
    processing_stats = {
        'total_images': 0,
        'successful_face_detections': 0,
        'failed_face_detections': 0,
        'processing_time': 0,
        'subjects': {},
        'visualization_images': []
    }
    
    # Get all subject directories
    subject_dirs = sorted(glob.glob(os.path.join(att_dir, "s*")))
    
    logger.info(f"Found {len(subject_dirs)} subjects in the AT&T database")
    
    start_time = time.time()
    
    # Process each subject
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        subject_id = os.path.basename(subject_dir)
        
        # Create subject output directory
        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Initialize subject statistics
        processing_stats['subjects'][subject_id] = {
            'total_images': 0,
            'successful_face_detections': 0,
            'failed_face_detections': 0
        }
        
        # Get all images for this subject
        image_files = sorted(glob.glob(os.path.join(subject_dir, "*.pgm")))
        
        # Process each image
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            output_path = os.path.join(subject_output_dir, f"{os.path.splitext(image_name)[0]}_processed.jpg")
            
            processing_stats['total_images'] += 1
            processing_stats['subjects'][subject_id]['total_images'] += 1
            
            try:
                # Read PGM file
                with open(image_path, 'rb') as pgmf:
                    image = read_pgm(pgmf)
                
                # Process image with visualization enabled for the first image of each subject
                if processing_stats['subjects'][subject_id]['total_images'] == 1:
                    processed_image, intermediates = preprocessor.preprocess_image(
                        image,
                        save_path=output_path,
                        return_intermediates=True
                    )
                    
                    if processed_image is not None:
                        # Save visualization for the first image of each 5th subject
                        if int(subject_id[1:]) % 5 == 0:
                            visualization_path = os.path.join(output_dir, f"{subject_id}_visualization.jpg")
                            visualize_intermediates(intermediates, visualization_path)
                            processing_stats['visualization_images'].append(visualization_path)
                else:
                    processed_image = preprocessor.preprocess_image(
                        image,
                        save_path=output_path
                    )
                
                if processed_image is not None:
                    processing_stats['successful_face_detections'] += 1
                    processing_stats['subjects'][subject_id]['successful_face_detections'] += 1
                else:
                    processing_stats['failed_face_detections'] += 1
                    processing_stats['subjects'][subject_id]['failed_face_detections'] += 1
                    logger.warning(f"Failed to process image: {image_path}")
            
            except Exception as e:
                processing_stats['failed_face_detections'] += 1
                processing_stats['subjects'][subject_id]['failed_face_detections'] += 1
                logger.error(f"Error processing {image_path}: {str(e)}")
    
    # Calculate total processing time
    processing_stats['processing_time'] = time.time() - start_time
    
    # Generate and save report
    generate_processing_report(processing_stats, output_dir)
    
    return processing_stats

def visualize_intermediates(intermediates, save_path):
    """Visualize intermediate preprocessing results and save to file."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Determine number of subplots
        n_plots = len(intermediates)
        cols = min(4, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        # Plot each intermediate result
        for i, (name, img) in enumerate(intermediates.items()):
            plt.subplot(rows, cols, i+1)
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')
            plt.title(name)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Visualization saved to {save_path}")
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")

def generate_processing_report(stats, output_dir):
    """Generate and save a report of the preprocessing results."""
    report_path = os.path.join(output_dir, "preprocessing_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("AT&T Face Database Preprocessing Report\n")
        f.write("=====================================\n\n")
        
        f.write(f"Total images processed: {stats['total_images']}\n")
        f.write(f"Successful face detections: {stats['successful_face_detections']} " +
                f"({stats['successful_face_detections']/stats['total_images']*100:.2f}%)\n")
        f.write(f"Failed face detections: {stats['failed_face_detections']} " +
                f"({stats['failed_face_detections']/stats['total_images']*100:.2f}%)\n")
        f.write(f"Total processing time: {stats['processing_time']:.2f} seconds\n")
        f.write(f"Average processing time per image: {stats['processing_time']/stats['total_images']:.4f} seconds\n\n")
        
        f.write("Subject-wise Statistics:\n")
        f.write("----------------------\n\n")
        
        for subject_id, subject_stats in sorted(stats['subjects'].items()):
            f.write(f"Subject {subject_id}:\n")
            f.write(f"  Total images: {subject_stats['total_images']}\n")
            f.write(f"  Successful detections: {subject_stats['successful_face_detections']}\n")
            f.write(f"  Failed detections: {subject_stats['failed_face_detections']}\n")
            if subject_stats['total_images'] > 0:
                success_rate = subject_stats['successful_face_detections'] / subject_stats['total_images'] * 100
                f.write(f"  Success rate: {success_rate:.2f}%\n")
            f.write("\n")
    
    logger.info(f"Preprocessing report saved to {report_path}")
    
    # Also generate a visual report if matplotlib is available
    try:
        # Calculate success rates for each subject
        subjects = list(stats['subjects'].keys())
        success_rates = [s['successful_face_detections']/s['total_images']*100 
                        for s in stats['subjects'].values()]
        
        plt.figure(figsize=(12, 6))
        plt.bar(subjects, success_rates)
        plt.xlabel('Subject ID')
        plt.ylabel('Face Detection Success Rate (%)')
        plt.title('Face Detection Success Rate by Subject')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "success_rate_by_subject.png"))
        plt.close()
        
        logger.info(f"Visual report saved to {os.path.join(output_dir, 'success_rate_by_subject.png')}")
    except Exception as e:
        logger.error(f"Error generating visual report: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process AT&T Face Database')
    parser.add_argument('--att-dir', type=str, default='AT&T', 
                      help='Path to AT&T face database directory')
    parser.add_argument('--output-dir', type=str, default='processed_att',
                      help='Path to save processed images')
    parser.add_argument('--target-size', type=str, default='112x112',
                      help='Target size for processed images (WxH)')
    parser.add_argument('--detector', type=str, default='auto',
                      choices=['auto', 'haar', 'hog', 'dnn', 'mediapipe'],
                      help='Face detector type')
    parser.add_argument('--landmarks', action='store_true', default=True,
                      help='Enable facial landmark detection')
    parser.add_argument('--mediapipe', action='store_true', default=True,
                      help='Use MediaPipe for detection and landmarks')
    parser.add_argument('--illumination', type=str, default='multi',
                      choices=['clahe', 'gamma', 'dog', 'multi'],
                      help='Illumination normalization method')
    parser.add_argument('--attention', action='store_true', default=True,
                      help='Apply attention mechanism')
    parser.add_argument('--smart-crop', action='store_true', default=True,
                      help='Use smart cropping')
    parser.add_argument('--edge-aware', action='store_true', default=True,
                      help='Enable edge-aware cropping')
    parser.add_argument('--enhance', type=str, default='adaptive',
                      choices=['adaptive', 'basic', 'detail', 'edge'],
                      help='Enhancement method')
    parser.add_argument('--intensity', type=float, default=1.0,
                      help='Enhancement intensity (0.0-2.0)')
    
    args = parser.parse_args()
    
    # Parse target size
    try:
        w, h = map(int, args.target_size.split('x'))
        target_size = (w, h)
    except:
        print(f"Invalid size format: {args.target_size}. Using default 112x112.")
        target_size = (112, 112)
    
    # Create configuration dictionary
    config = {
        'target_size': target_size,
        'detector_type': args.detector,
        'enable_landmarks': args.landmarks,
        'use_mediapipe': args.mediapipe,
        'illumination_method': args.illumination,
        'enable_attention': args.attention,
        'smart_crop': args.smart_crop,
        'edge_aware': args.edge_aware,
        'enhance_method': args.enhance,
        'enhance_intensity': args.intensity
    }
    
    # Process the AT&T database
    stats = process_att_database(args.att_dir, args.output_dir, config)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successful face detections: {stats['successful_face_detections']} " +
          f"({stats['successful_face_detections']/stats['total_images']*100:.2f}%)")
    print(f"Total processing time: {stats['processing_time']:.2f} seconds") 