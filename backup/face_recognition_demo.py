"""
Face Recognition Demonstration

This script allows testing the trained face recognition model on new images.
It preprocesses input images using the advanced face preprocessor and
performs recognition using the trained model.

Usage:
    python face_recognition_demo.py --image path/to/image.jpg
"""

import os
import sys
import cv2
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
try:
    from face_preprocessor import FacePreprocessor
    from face_recognition_att import FaceRecognizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure face_preprocessor.py and face_recognition_att.py are in the current directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_recognition_demo")

def preprocess_image(image_path, target_size=(112, 92)):
    """
    Preprocess an input image for face recognition
    
    Args:
        image_path: Path to the input image
        target_size: Target size for face alignment (default: AT&T database size)
        
    Returns:
        Preprocessed face image or None if no face is detected
    """
    try:
        # Initialize face preprocessor with advanced options
        preprocessor = FacePreprocessor(
            target_size=target_size,
            detector_type='dnn',
            enable_landmarks=True,
            illumination_method='multi',
            use_mediapipe=True,
            enable_attention=True,
            smart_crop=True
        )
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        
        # Detect and preprocess faces
        faces = preprocessor.detect_faces(img)
        
        if not faces:
            logger.warning(f"No faces detected in {image_path}")
            return None
        
        # Get the largest face
        face_img = preprocessor.preprocess_face(img, faces[0])
        
        if face_img is None:
            logger.warning(f"Failed to preprocess face in {image_path}")
            return None
        
        return face_img
    
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def recognize_face(face_img, model_dir="face_models"):
    """
    Recognize a face using the trained model
    
    Args:
        face_img: Preprocessed face image
        model_dir: Directory containing the trained models
        
    Returns:
        Tuple of (subject_name, confidence)
    """
    try:
        # Initialize face recognizer
        recognizer = FaceRecognizer(processed_dir="", model_dir=model_dir)
        
        # Load models
        if not recognizer.load_models():
            logger.error(f"Failed to load models from {model_dir}")
            return "Unknown", 0.0
        
        # Recognize face
        _, confidence, subject_name = recognizer.recognize(face_img)
        
        return subject_name, confidence
    
    except Exception as e:
        logger.error(f"Error during face recognition: {str(e)}")
        return "Unknown", 0.0

def display_result(img, subject_name, confidence):
    """Display the recognition result"""
    plt.figure(figsize=(8, 6))
    
    if len(img.shape) == 2:
        # Grayscale image
        plt.imshow(img, cmap='gray')
    else:
        # Color image (convert from BGR to RGB)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.title(f"Recognized as: {subject_name}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Face Recognition Demo")
    parser.add_argument("--image", type=str, required=True,
                      help="Path to the input image")
    parser.add_argument("--model-dir", type=str, default="face_models",
                      help="Directory containing trained models")
    parser.add_argument("--display", action="store_true", default=True,
                      help="Display the recognition result")
    
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.isfile(args.image):
        logger.error(f"Image not found: {args.image}")
        return
    
    # Check if model directory exists
    if not os.path.isdir(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        return
    
    # Preprocess the image
    logger.info(f"Preprocessing image: {args.image}")
    face_img = preprocess_image(args.image)
    
    if face_img is None:
        logger.error("Failed to preprocess face. Please try a different image.")
        return
    
    # Recognize the face
    logger.info("Performing face recognition")
    subject_name, confidence = recognize_face(face_img, args.model_dir)
    
    # Print result
    print(f"\nRecognition Result:")
    print(f"  Subject: {subject_name}")
    print(f"  Confidence: {confidence:.2%}")
    
    # Display result if requested
    if args.display:
        display_result(face_img, subject_name, confidence)
        
    logger.info(f"Recognition complete: {subject_name} ({confidence:.2%})")

if __name__ == "__main__":
    main() 