"""
Train Face Recognition Model

Script to train the face recognition model on a dataset of face images.
"""

import os
import numpy as np
import cv2
import argparse
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from face_detector import FaceDetector
from face_preprocessor import FacePreprocessor
from face_recognizer import FaceRecognizer
from config import (
    ATT_DATASET_PATH, MODEL_PATH, TRAIN_TEST_SPLIT, RANDOM_STATE, 
    DATA_AUGMENTATION, AUGMENTATION_ROTATIONS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train")

def load_att_dataset(dataset_path):
    """
    Load the AT&T face dataset
    
    Parameters:
    -----------
    dataset_path : str
        Path to the AT&T dataset
        
    Returns:
    --------
    tuple
        (faces, labels, subject_names)
    """
    logger.info(f"Loading AT&T dataset from {dataset_path}")
    
    faces = []
    labels = []
    subject_names = []
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    
    # Load dataset
    for subject_id in range(1, 41):  # AT&T has 40 subjects
        subject_dir = os.path.join(dataset_path, f"s{subject_id}")
        
        # Skip if subject directory doesn't exist
        if not os.path.exists(subject_dir):
            logger.warning(f"Subject directory {subject_dir} does not exist")
            continue
            
        # Add subject name
        subject_names.append(f"Subject_{subject_id}")
        
        # Process each image for this subject
        for image_id in range(1, 11):  # 10 images per subject
            image_path = os.path.join(subject_dir, f"{image_id}.pgm")
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                logger.warning(f"Image {image_path} does not exist")
                continue
                
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Skip if image couldn't be read
            if image is None:
                logger.warning(f"Could not read image {image_path}")
                continue
                
            # Add image and label
            faces.append(image)
            labels.append(subject_id - 1)  # Use 0-based labels
            
    logger.info(f"Loaded {len(faces)} images of {len(subject_names)} subjects")
    
    return faces, np.array(labels), subject_names

def visualize_dataset_samples(faces, labels, n_subjects=5, n_samples=3):
    """
    Visualize random samples from the dataset
    
    Parameters:
    -----------
    faces : list
        List of face images
    labels : numpy.ndarray
        Array of labels
    n_subjects : int
        Number of subjects to visualize
    n_samples : int
        Number of samples per subject
    """
    unique_labels = np.unique(labels)
    n_subjects = min(n_subjects, len(unique_labels))
    
    # Select random subjects
    subject_indices = np.random.choice(unique_labels, n_subjects, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(n_subjects, n_samples, figsize=(n_samples * 2, n_subjects * 2))
    
    # Plot samples
    for i, subject_idx in enumerate(subject_indices):
        # Get indices for this subject
        indices = np.where(labels == subject_idx)[0]
        
        # Select random samples
        sample_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
        
        # Plot each sample
        for j, sample_idx in enumerate(sample_indices):
            if n_subjects == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
                
            ax.imshow(faces[sample_idx], cmap='gray')
            ax.set_title(f"Subject {subject_idx}")
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("dataset_samples.png")
    logger.info("Dataset samples visualization saved to dataset_samples.png")

def train_model(faces, labels, subject_names, test_size=TRAIN_TEST_SPLIT, augment=DATA_AUGMENTATION):
    """
    Train the face recognition model
    
    Parameters:
    -----------
    faces : list
        List of face images
    labels : list
        List of labels
    subject_names : list
        List of subject names
    test_size : float
        Fraction of data to use for testing
    augment : bool
        Whether to use data augmentation
        
    Returns:
    --------
    tuple
        (recognizer, accuracy)
    """
    logger.info("Training face recognition model...")
    
    # Initialize preprocessor and recognizer
    preprocessor = FacePreprocessor()
    recognizer = FaceRecognizer()
    
    # Preprocess faces
    start_time = time.time()
    
    if augment:
        logger.info(f"Using data augmentation with rotation angles: {AUGMENTATION_ROTATIONS}")
        # With augmentation, normalized_faces is X and augmented_labels is y
        normalized_faces, augmented_labels = preprocessor.normalize_for_training(faces, augment=True)
        
        if normalized_faces is None:
            logger.error("Failed to normalize faces for training")
            return None, 0.0
            
        # Convert augmented_labels to actual subject labels
        augmented_subject_labels = [labels[idx] for idx in augmented_labels]
        
        logger.info(f"Dataset size after augmentation: {len(normalized_faces)} samples")
        
        # Train with augmented data
        accuracy = recognizer.train(normalized_faces, augmented_subject_labels, 
                                   subject_names, test_size=test_size)
    else:
        # Without augmentation, normalized_faces is X
        normalized_faces = preprocessor.normalize_for_training(faces, augment=False)
        
        if normalized_faces is None:
            logger.error("Failed to normalize faces for training")
            return None, 0.0
            
        # Train without augmentation
        accuracy = recognizer.train(normalized_faces, labels, subject_names, test_size=test_size)
    
    processing_time = time.time() - start_time
    logger.info(f"Preprocessing and training completed in {processing_time:.2f} seconds")
    
    # Save the model
    if recognizer.save(MODEL_PATH):
        logger.info(f"Model saved to {MODEL_PATH}")
    else:
        logger.error(f"Failed to save model to {MODEL_PATH}")
        
    return recognizer, accuracy

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--dataset", type=str, default=ATT_DATASET_PATH,
                        help="Path to the AT&T dataset")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to save the trained model")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize dataset samples")
    parser.add_argument("--test-size", type=float, default=TRAIN_TEST_SPLIT,
                        help="Fraction of data to use for testing")
    parser.add_argument("--augment", action="store_true", default=DATA_AUGMENTATION,
                        help="Use data augmentation with rotation")
    args = parser.parse_args()
    
    # Load the dataset
    faces, labels, subject_names = load_att_dataset(args.dataset)
    
    # Visualize dataset samples if requested
    if args.visualize:
        visualize_dataset_samples(faces, labels)
    
    # Train the model
    recognizer, accuracy = train_model(faces, labels, subject_names, 
                                       test_size=args.test_size,
                                       augment=args.augment)
    
    logger.info(f"Model training complete with accuracy: {accuracy:.2%}")
    
    return 0

if __name__ == "__main__":
    main() 