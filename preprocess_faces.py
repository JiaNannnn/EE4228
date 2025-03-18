"""
Preprocess Faces

This script applies the face preprocessing pipeline to a directory of face images.
It demonstrates the sequential application of:
1. Original image loading
2. Grayscale conversion
3. Contrast enhancement (CLAHE, histogram equalization, etc.)
4. Face alignment to a standard orientation
"""

import os
import cv2
import argparse
import logging
import numpy as np
from pathlib import Path
from face_detector import FaceDetector
from face_preprocessor import FacePreprocessor
from config import (
    DATA_DIR, ATT_DATASET_PATH, FACE_SIZE,
    USE_CLAHE, USE_HIST_EQUALIZATION, USE_CONTRAST_STRETCHING, USE_GAMMA_CORRECTION,
    USE_FACE_ALIGNMENT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("preprocess_faces")

def preprocess_directory(input_dir, output_dir, save_steps=False):
    """
    Preprocess all images in a directory
    
    Parameters:
    -----------
    input_dir : str
        Input directory containing face images
    output_dir : str
        Output directory for preprocessed images
    save_steps : bool
        Whether to save intermediate preprocessing steps
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for intermediate steps if needed
    if save_steps:
        os.makedirs(os.path.join(output_dir, "grayscale"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "aligned"), exist_ok=True)
    
    # Initialize face detector and preprocessor
    detector = FaceDetector()
    preprocessor = FacePreprocessor(
        use_clahe=USE_CLAHE,
        use_hist_equalization=USE_HIST_EQUALIZATION,
        use_contrast_stretching=USE_CONTRAST_STRETCHING,
        use_gamma_correction=USE_GAMMA_CORRECTION,
        use_face_alignment=USE_FACE_ALIGNMENT
    )
    
    # Get list of image files in input directory
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.pgm"]:
        image_files.extend(list(Path(input_dir).glob(ext)))
        image_files.extend(list(Path(input_dir).glob(ext.upper())))
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for image_path in image_files:
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                continue
                
            logger.info(f"Processing image: {image_path.name}")
            
            # Detect face
            faces = detector.detect_faces(image)
            if len(faces) == 0:
                logger.warning(f"No face detected in image: {image_path.name}")
                # Use the whole image if no face is detected
                face_img = image
            else:
                # Use the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                face_img = detector.get_face_roi(image, largest_face)
            
            # Apply preprocessing steps
            if save_steps:
                # Get all preprocessing steps
                steps = preprocessor.visualize_preprocessing_steps(face_img)
                
                # Save each step
                for step_name, step_image in steps.items():
                    if step_name == "grayscale":
                        output_path = os.path.join(output_dir, "grayscale", image_path.name)
                        cv2.imwrite(output_path, step_image)
                    elif step_name in ["clahe", "contrast_stretched", "hist_equalized", "gamma_corrected"]:
                        output_path = os.path.join(output_dir, "enhanced", f"{step_name}_{image_path.name}")
                        cv2.imwrite(output_path, step_image)
                    elif step_name in ["aligned", "rotated"]:
                        output_path = os.path.join(output_dir, "aligned", f"{step_name}_{image_path.name}")
                        cv2.imwrite(output_path, step_image)
                    elif step_name == "final":
                        output_path = os.path.join(output_dir, image_path.name)
                        cv2.imwrite(output_path, step_image)
            else:
                # Just apply the full preprocessing pipeline
                preprocessed = preprocessor.preprocess(face_img)
                if preprocessed is not None:
                    output_path = os.path.join(output_dir, image_path.name)
                    cv2.imwrite(output_path, preprocessed)
                else:
                    logger.warning(f"Failed to preprocess image: {image_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path.name}: {e}")
    
    logger.info(f"Preprocessing completed. Results saved to {output_dir}")

def preprocess_att_dataset(output_dir):
    """
    Preprocess the AT&T (ORL) face dataset
    
    Parameters:
    -----------
    output_dir : str
        Output directory for preprocessed images
    """
    # Check if the AT&T dataset exists
    if not os.path.exists(ATT_DATASET_PATH):
        logger.error(f"AT&T dataset not found at {ATT_DATASET_PATH}")
        return
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Preprocessing AT&T dataset from {ATT_DATASET_PATH}")
    
    # Initialize face detector and preprocessor
    detector = FaceDetector()
    preprocessor = FacePreprocessor(
        use_clahe=USE_CLAHE,
        use_hist_equalization=USE_HIST_EQUALIZATION,
        use_contrast_stretching=USE_CONTRAST_STRETCHING,
        use_gamma_correction=USE_GAMMA_CORRECTION,
        use_face_alignment=USE_FACE_ALIGNMENT
    )
    
    # Process each subject directory
    for subject_id in range(1, 41):
        subject_dir = os.path.join(ATT_DATASET_PATH, f"s{subject_id}")
        if not os.path.exists(subject_dir):
            continue
            
        # Create output subject directory
        output_subject_dir = os.path.join(output_dir, f"s{subject_id}")
        os.makedirs(output_subject_dir, exist_ok=True)
        
        logger.info(f"Processing subject {subject_id}")
        
        # Process each image of the subject
        for image_id in range(1, 11):
            image_path = os.path.join(subject_dir, f"{image_id}.pgm")
            if not os.path.exists(image_path):
                continue
                
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not read image: {image_path}")
                    continue
                
                # Apply preprocessing steps sequentially
                # 1. Original image loaded above
                
                # 2. Convert to grayscale (preprocessor will do this)
                
                # 3. Enhance contrast and align face
                preprocessed = preprocessor.preprocess(image)
                
                if preprocessed is not None:
                    # Save preprocessed image
                    output_path = os.path.join(output_subject_dir, f"{image_id}.pgm")
                    cv2.imwrite(output_path, preprocessed)
                else:
                    logger.warning(f"Failed to preprocess image: {image_path}")
                    
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
    
    logger.info(f"Dataset preprocessing completed. Results saved to {output_dir}")

def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess face images')
    parser.add_argument('--input', type=str, default=None,
                        help='Input directory containing face images')
    parser.add_argument('--output', type=str, default=str(DATA_DIR / "preprocessed"),
                        help='Output directory for preprocessed images')
    parser.add_argument('--att', action='store_true',
                        help='Preprocess the AT&T dataset')
    parser.add_argument('--save-steps', action='store_true',
                        help='Save intermediate preprocessing steps')
    args = parser.parse_args()
    
    if args.att:
        # Preprocess the AT&T dataset
        preprocess_att_dataset(args.output)
    elif args.input:
        # Preprocess a directory of images
        preprocess_directory(args.input, args.output, args.save_steps)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 