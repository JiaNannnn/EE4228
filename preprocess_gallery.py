"""
Preprocess Gallery Images

This script processes images from the gallery folder and registers them as users
for facial recognition. Each subdirectory in the gallery is treated as a separate user.
"""

import os
import cv2
import numpy as np
import logging
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

# Import from our facial recognition system
from face_detector import FaceDetector
from face_preprocessor import FacePreprocessor
from face_recognizer import FaceRecognizer
from config import DATA_DIR, MODELS_DIR, MODEL_PATH, FACE_SIZE

# Configure logging
logging.basicConfig(
    filename="logs/gallery_preprocessing.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gallery_preprocessing")

def preprocess_gallery(gallery_dir="gallery", min_face_size=(100, 100), train_after=True, overwrite=False):
    """
    Process all images in the gallery folder and register them as users.
    
    Parameters:
    -----------
    gallery_dir : str
        Path to the gallery directory
    min_face_size : tuple
        Minimum size (width, height) for face detection
    train_after : bool
        Whether to train the model after preprocessing
    overwrite : bool
        Whether to overwrite existing user data
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Initialize components
        detector = FaceDetector()
        preprocessor = FacePreprocessor()
        
        # Create users directory if it doesn't exist
        users_dir = Path(DATA_DIR) / "users"
        users_dir.mkdir(exist_ok=True, parents=True)
        
        # Get list of subdirectories in gallery (each is a user)
        gallery_path = Path(gallery_dir)
        user_dirs = [d for d in gallery_path.iterdir() if d.is_dir()]
        
        if not user_dirs:
            logger.warning(f"No user directories found in {gallery_dir}")
            print(f"No user directories found in {gallery_dir}")
            return False
        
        print(f"Found {len(user_dirs)} user directories in gallery")
        logger.info(f"Found {len(user_dirs)} user directories in gallery")
        
        # Counter for processed users and images
        processed_users = 0
        processed_images = 0
        failed_images = 0
        
        # Process each user directory
        for user_dir in user_dirs:
            user_id = user_dir.name
            print(f"Processing user: {user_id}")
            logger.info(f"Processing user: {user_id}")
            
            # Create user directory in the users folder
            user_output_dir = users_dir / user_id
            
            # Check if user already exists
            if user_output_dir.exists() and not overwrite:
                print(f"User {user_id} already exists. Skipping (use --overwrite to replace).")
                logger.info(f"User {user_id} already exists. Skipping.")
                continue
            
            # Create or empty the user directory
            if user_output_dir.exists() and overwrite:
                # Remove existing files
                for file in user_output_dir.glob("*"):
                    file.unlink()
            else:
                user_output_dir.mkdir(exist_ok=True)
            
            # Get all image files in user directory
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(list(user_dir.glob(ext)))
            
            if not image_files:
                logger.warning(f"No images found for user {user_id}")
                print(f"No images found for user {user_id}")
                continue
            
            # Process each image
            successful_images = 0
            for img_path in tqdm(image_files, desc=f"Processing {user_id}"):
                try:
                    # Read the image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Failed to read image: {img_path}")
                        failed_images += 1
                        continue
                    
                    # For simplicity, we'll use direct face detection and processing
                    # First, detect faces
                    faces = detector.detect_faces(img)
                    
                    # If no faces detected with default parameters, try with more lenient parameters
                    if not faces:
                        # Try with more lenient parameters
                        faces = detector.detect_faces(img, scale_factor=1.05, min_neighbors=2)
                    
                    # If still no faces, create a synthetic face ROI from the entire image
                    if not faces:
                        # Treat the whole image as a face or a portion of it
                        h, w = img.shape[:2]
                        center_x, center_y = w // 2, h // 2
                        face_size = min(w, h)
                        padding = face_size // 4
                        
                        # Create a region that covers most of the image
                        x = max(0, center_x - face_size // 2)
                        y = max(0, center_y - face_size // 2)
                        width = min(face_size, w - x)
                        height = min(face_size, h - y)
                        
                        # Create a synthetic face rectangle
                        faces = [(x, y, width, height)]
                        logger.info(f"No face detected in {img_path}. Using image center as face.")
                    
                    # Process the largest face or the only face
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    
                    # Extract the face ROI
                    face_roi = detector.get_face_roi(img, largest_face)
                    
                    # Ensure face_roi is not None
                    if face_roi is None or face_roi.size == 0:
                        logger.warning(f"Failed to extract face ROI from image: {img_path}")
                        failed_images += 1
                        continue
                    
                    # Convert to grayscale if not already
                    if len(face_roi.shape) == 3:
                        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_face = face_roi
                    
                    # Resize to standard size
                    resized_face = cv2.resize(gray_face, (FACE_SIZE, FACE_SIZE))
                    
                    # Apply basic preprocessing (equalization)
                    equalized_face = cv2.equalizeHist(resized_face)
                    
                    # Save the processed face
                    output_path = user_output_dir / f"face_{successful_images + 1}.jpg"
                    cv2.imwrite(str(output_path), equalized_face)
                    
                    successful_images += 1
                    processed_images += 1
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    failed_images += 1
            
            print(f"Processed {successful_images}/{len(image_files)} images for user {user_id}")
            logger.info(f"Processed {successful_images}/{len(image_files)} images for user {user_id}")
            
            if successful_images > 0:
                processed_users += 1
        
        # Summary
        print(f"\nPreprocessing complete: {processed_users} users, {processed_images} images processed")
        print(f"Failed: {failed_images} images")
        logger.info(f"Preprocessing complete: {processed_users} users, {processed_images} images processed")
        logger.info(f"Failed: {failed_images} images")
        
        # Train the model if requested
        if train_after and processed_users > 0:
            print("\nTraining model with preprocessed data...")
            logger.info("Training model with preprocessed data...")
            
            if train_model():
                print("Model training completed successfully")
                logger.info("Model training completed successfully")
            else:
                print("Model training failed")
                logger.warning("Model training failed")
        
        return processed_users > 0
        
    except Exception as e:
        logger.error(f"Error preprocessing gallery: {e}")
        print(f"Error preprocessing gallery: {e}")
        return False

def train_model():
    """
    Train the face recognition model with all registered users
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        from face_recognizer import FaceRecognizer
        from face_preprocessor import FacePreprocessor
        
        # Initialize components
        recognizer = FaceRecognizer()
        preprocessor = FacePreprocessor()
        
        # Create models directory if it doesn't exist
        models_dir = Path(MODELS_DIR)
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all users
        users_dir = Path(DATA_DIR) / "users"
        users = [d.name for d in users_dir.iterdir() if d.is_dir()]
        
        if not users:
            logger.warning("No users registered. Nothing to train.")
            print("No users registered. Nothing to train.")
            return False
            
        # Collect images and labels
        all_images = []
        all_labels = []
        
        # Process each user
        for i, user_id in enumerate(users):
            print(f"Processing user {user_id}...")
            logger.info(f"Processing user {user_id}...")
            
            # Get user images
            user_dir = users_dir / user_id
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_paths.extend(list(user_dir.glob(ext)))
            
            # Skip users with no images
            if not image_paths:
                logger.warning(f"No images found for user {user_id}")
                continue
                
            # Load and preprocess images - ensure all images are already properly sized
            for img_path in image_paths:
                try:
                    # Read the image
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                    if img is None:
                        logger.warning(f"Failed to load image: {img_path}")
                        continue
                        
                    # Resize to ensure consistent size
                    # Check if the image needs resizing
                    if img.shape != (FACE_SIZE, FACE_SIZE):
                        img = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
                        
                    all_images.append(img)
                    all_labels.append(users.index(user_id))
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
        
        # Check if we have enough images
        if len(all_images) < 5:
            logger.warning("Not enough images for training. At least 5 images are required.")
            print("Not enough images for training. At least 5 images are required.")
            return False
            
        print(f"Training with {len(all_images)} images from {len(users)} users")
        logger.info(f"Training with {len(all_images)} images from {len(users)} users")
        
        # Train the model directly with the processed images
        # Skip the normalize_for_training step since we've already loaded them as grayscale and resized
        accuracy = recognizer.train(all_images, all_labels, users)
        
        # Save the model
        recognizer.save(MODEL_PATH)
        
        print(f"Model trained successfully with accuracy: {accuracy:.2%}")
        logger.info(f"Model trained successfully with accuracy: {accuracy:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        print(f"Error training model: {e}")
        return False

def main():
    """
    Main function to parse command line arguments and run preprocessing.
    """
    parser = argparse.ArgumentParser(description="Preprocess gallery images for facial recognition")
    parser.add_argument("--gallery", default="gallery", help="Path to gallery directory")
    parser.add_argument("--no-train", action="store_true", help="Skip training after preprocessing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing user data")
    
    args = parser.parse_args()
    
    print("Gallery Image Preprocessing Tool")
    print("--------------------------------")
    print(f"Gallery directory: {args.gallery}")
    print(f"Train after preprocessing: {not args.no_train}")
    print(f"Overwrite existing data: {args.overwrite}")
    print()
    
    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run preprocessing
    preprocess_gallery(
        gallery_dir=args.gallery,
        train_after=not args.no_train,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main() 