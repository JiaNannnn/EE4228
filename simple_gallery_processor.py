"""
Simple Gallery Processor

A streamlined script to process gallery images without relying on complex preprocessing
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
from config import FACE_SIZE, FACE_CASCADE_PATH, MIN_FACE_SIZE, SCALE_FACTOR, MIN_NEIGHBORS

# Configure logging
logging.basicConfig(
    filename="logs/simple_gallery_processing.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_gallery_processor")

# Constants
DATA_DIR = "data"
MODELS_DIR = "models"

def preprocess_face(face):
    """
    Apply consistent preprocessing to a face image
    
    Parameters:
    -----------
    face : numpy.ndarray
        Input face image
        
    Returns:
    --------
    numpy.ndarray
        Processed face image
    """
    try:
        # Check if face is valid
        if face is None or face.size == 0:
            logger.warning("Invalid face image: empty or None")
            return None
            
        # Check dimensions
        if face.shape[0] < 10 or face.shape[1] < 10:
            logger.warning(f"Face image too small: {face.shape}")
            return None
            
        # Ensure grayscale
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray = face
            
        # Resize to standard size
        resized = cv2.resize(gray, (FACE_SIZE, FACE_SIZE))
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(resized)
        
        return equalized
    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        return None

def force_process_image(img):
    """
    Process an entire image regardless of face detection
    This is a fallback for images where face detection completely fails
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
        
    Returns:
    --------
    numpy.ndarray
        Processed image
    """
    try:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # For very large images, resize to something more manageable first
        max_dim = max(gray.shape[0], gray.shape[1])
        if max_dim > 1000:
            scale = 1000 / max_dim
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
            
        # Center crop to get a square image
        h, w = gray.shape
        size = min(h, w)
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        cropped = gray[start_y:start_y+size, start_x:start_x+size]
        
        # Resize to target size
        resized = cv2.resize(cropped, (FACE_SIZE, FACE_SIZE))
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(resized)
        
        return equalized
    except Exception as e:
        logger.error(f"Force processing failed: {e}")
        return None

def preprocess_gallery(gallery_dir="gallery", output_dir=None, overwrite=False):
    """
    Process all images in the gallery folder using consistent preprocessing
    
    Parameters:
    -----------
    gallery_dir : str
        Path to the gallery directory
    output_dir : str
        Path to the output directory (defaults to DATA_DIR/users)
    overwrite : bool
        Whether to overwrite existing user data
    
    Returns:
    --------
    int
        Number of users processed
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(DATA_DIR, "users")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Load face detector with specified parameters
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if face_cascade.empty():
            logger.error(f"Failed to load face cascade classifier from {FACE_CASCADE_PATH}")
            print(f"Error: Failed to load face cascade classifier")
            return 0
        
        # Get list of subdirectories in gallery (each is a user)
        gallery_path = Path(gallery_dir)
        user_dirs = [d for d in gallery_path.iterdir() if d.is_dir()]
        
        if not user_dirs:
            logger.warning(f"No user directories found in {gallery_dir}")
            print(f"No user directories found in {gallery_dir}")
            return 0
        
        print(f"Found {len(user_dirs)} user directories in gallery")
        
        # Counter for processed users and images
        processed_users = 0
        processed_images = 0
        failed_images = 0
        
        # Process each user directory
        for user_dir in user_dirs:
            user_id = user_dir.name
            print(f"Processing user: {user_id}")
            
            # Create user directory in the output folder
            user_output_dir = Path(output_dir) / user_id
            
            # Check if user already exists
            if user_output_dir.exists() and not overwrite:
                print(f"User {user_id} already exists. Skipping (use --overwrite to replace).")
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
                    
                    # Log image properties for debugging
                    logger.info(f"Processing image: {img_path.name}, shape: {img.shape}")
                    
                    processed_face = None
                    
                    # Try normal face detection first
                    try:
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Try multi-scale face detection with multiple parameter sets
                        faces = []
                        
                        # First attempt with default parameters
                        faces = face_cascade.detectMultiScale(
                            gray, 
                            scaleFactor=SCALE_FACTOR, 
                            minNeighbors=MIN_NEIGHBORS,
                            minSize=MIN_FACE_SIZE
                        )
                        
                        # If no faces detected, try with more lenient parameters
                        if len(faces) == 0:
                            logger.info(f"No faces found with default parameters, trying lenient parameters for {img_path.name}")
                            faces = face_cascade.detectMultiScale(
                                gray, 
                                scaleFactor=1.05, 
                                minNeighbors=3,
                                minSize=(20, 20)
                            )
                        
                        # If still no faces, try even more lenient detection
                        if len(faces) == 0:
                            logger.info(f"Still no faces found, trying very lenient parameters for {img_path.name}")
                            faces = face_cascade.detectMultiScale(
                                gray, 
                                scaleFactor=1.03, 
                                minNeighbors=2,
                                minSize=(10, 10)
                            )
                        
                        # If still no faces, try to detect faces in different scales
                        if len(faces) == 0:
                            logger.info(f"Trying multi-scale detection for {img_path.name}")
                            # Resize image to different scales and try detection
                            for scale in [0.5, 0.75, 1.5, 2.0]:
                                resized = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
                                temp_faces = face_cascade.detectMultiScale(
                                    resized,
                                    scaleFactor=1.1,
                                    minNeighbors=3,
                                    minSize=(30, 30)
                                )
                                
                                # If faces found, adjust coordinates back to original scale
                                if len(temp_faces) > 0:
                                    for (x, y, w, h) in temp_faces:
                                        x = int(x / scale)
                                        y = int(y / scale)
                                        w = int(w / scale)
                                        h = int(h / scale)
                                        faces = np.array([[x, y, w, h]])
                                    break
                        
                        # If still no faces, use the center of the image
                        if len(faces) == 0:
                            logger.warning(f"No face detected in {img_path.name}. Using image center as face.")
                            h, w = gray.shape
                            center_x, center_y = w // 2, h // 2
                            size = min(w, h) // 2
                            faces = np.array([[center_x - size, center_y - size, size * 2, size * 2]])
                            print(f"No face detected in {img_path.name}. Using image center as face.")
                        
                        # Use the largest face
                        if len(faces) > 0:
                            if len(faces) > 1:
                                logger.info(f"Multiple faces ({len(faces)}) detected in {img_path.name}, using largest")
                            
                            # Find the largest face by area
                            largest_face_idx = np.argmax(faces[:, 2] * faces[:, 3])
                            x, y, w, h = faces[largest_face_idx]
                            
                            # Add a margin (20% of face size)
                            margin_x = int(w * 0.2)
                            margin_y = int(h * 0.2)
                            
                            # Calculate new coordinates with margins, ensuring they stay within image bounds
                            x_with_margin = max(0, x - margin_x)
                            y_with_margin = max(0, y - margin_y)
                            w_with_margin = min(gray.shape[1] - x_with_margin, w + 2 * margin_x)
                            h_with_margin = min(gray.shape[0] - y_with_margin, h + 2 * margin_y)
                            
                            # Log face detection details
                            logger.info(f"Face detected at ({x},{y},{w},{h}) with margin: ({x_with_margin},{y_with_margin},{w_with_margin},{h_with_margin})")
                            
                            # Extract the face with margins
                            face = gray[y_with_margin:y_with_margin+h_with_margin, 
                                        x_with_margin:x_with_margin+w_with_margin]
                                    
                            # Apply consistent preprocessing
                            processed_face = preprocess_face(face)
                    except Exception as e:
                        logger.error(f"Error during face detection for {img_path.name}: {e}")
                        processed_face = None
                    
                    # If all face detection methods failed, force process the entire image
                    if processed_face is None:
                        logger.warning(f"All face detection attempts failed for {img_path.name}. Force processing entire image.")
                        print(f"Force processing {img_path.name}")
                        processed_face = force_process_image(img)
                    
                    # If we have a processed face (by any method), save it
                    if processed_face is not None:
                        # Save the processed face
                        output_path = user_output_dir / f"face_{successful_images + 1}.jpg"
                        success = cv2.imwrite(str(output_path), processed_face)
                        
                        if success:
                            successful_images += 1
                            processed_images += 1
                            logger.info(f"Successfully saved processed face to {output_path}")
                        else:
                            logger.error(f"Failed to write image to {output_path}")
                            failed_images += 1
                    else:
                        logger.error(f"Could not process image {img_path.name} by any method")
                        failed_images += 1
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    print(f"Error processing {img_path.name}: {str(e)}")
                    failed_images += 1
            
            print(f"Processed {successful_images}/{len(image_files)} images for user {user_id}")
            if successful_images > 0:
                processed_users += 1
        
        # Summary
        print(f"\nPreprocessing complete: {processed_users} users, {processed_images} images processed")
        print(f"Failed: {failed_images} images")
        
        return processed_users
        
    except Exception as e:
        logger.error(f"Error preprocessing gallery: {e}")
        print(f"Error preprocessing gallery: {e}")
        return 0

def main():
    """
    Main function to parse command line arguments and run preprocessing.
    """
    parser = argparse.ArgumentParser(description="Process gallery images for facial recognition")
    parser.add_argument("--gallery", default="gallery", help="Path to gallery directory")
    parser.add_argument("--output", default=None, help="Path to output directory (default: data/users)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing user data")
    parser.add_argument("--save-debug", action="store_true", help="Save debug images showing detected faces")
    parser.add_argument("--force-all", action="store_true", help="Force process all images without face detection")
    
    args = parser.parse_args()
    
    print("Simple Gallery Image Processing Tool")
    print("----------------------------------")
    print(f"Gallery directory: {args.gallery}")
    print(f"Output directory: {args.output or os.path.join(DATA_DIR, 'users')}")
    print(f"Overwrite existing data: {args.overwrite}")
    print(f"Face image size: {FACE_SIZE}x{FACE_SIZE}")
    print()
    
    # Run preprocessing
    num_users = preprocess_gallery(
        gallery_dir=args.gallery,
        output_dir=args.output,
        overwrite=args.overwrite
    )
    
    if num_users > 0:
        print(f"\nSuccessfully processed {num_users} users.")
        print("You can now train the model using the Streamlit application.")
    else:
        print("\nNo users were processed. Please check the logs for details.")

if __name__ == "__main__":
    main() 