"""
Process AT&T Database and Train Face Recognition Model

This script processes the AT&T database using advanced face preprocessing
techniques and then trains a face recognition model using the processed images.

Usage:
    python process_and_train.py

The script performs the following steps:
1. Process all images in the AT&T database using advanced preprocessing
2. Train a face recognition model using PCA (eigenfaces) and optionally LDA
3. Evaluate the model performance using cross-validation
4. Generate visualizations of eigenfaces and performance metrics
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Import our modules
try:
    from process_att_database import process_att_database, read_pgm
    from face_recognition_att import FaceRecognizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure process_att_database.py and face_recognition_att.py are in the current directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("att_face_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("process_and_train")

def ensure_directories_exist(base_dir="processed_att", model_dir="face_models"):
    """Ensure necessary directories exist"""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Created directories: {base_dir}, {model_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process AT&T Database and Train Face Recognition Model")
    parser.add_argument("--att-dir", type=str, default=".",
                      help="Base directory containing AT&T database (with s1, s2, ... subdirectories)")
    parser.add_argument("--processed-dir", type=str, default="processed_att",
                      help="Directory to store processed images")
    parser.add_argument("--model-dir", type=str, default="face_models",
                      help="Directory to store trained models")
    parser.add_argument("--use-lda", action="store_true", default=True,
                      help="Use LDA after PCA for feature extraction")
    parser.add_argument("--components", type=int, default=None,
                      help="Number of PCA components (if None, 95% variance preserved)")
    parser.add_argument("--k-folds", type=int, default=5,
                      help="Number of folds for cross-validation")
    parser.add_argument("--classifier", type=str, default="svm", choices=["svm", "knn"],
                      help="Classifier type (SVM or KNN)")
    parser.add_argument("--skip-processing", action="store_true",
                      help="Skip database processing if already processed")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories_exist(args.processed_dir, args.model_dir)
    
    # Check if AT&T database exists
    subject_dirs = [d for d in os.listdir(args.att_dir) if d.startswith('s') and os.path.isdir(os.path.join(args.att_dir, d))]
    if not subject_dirs:
        logger.error(f"AT&T database not found in {args.att_dir}. Expecting directories: s1, s2, etc.")
        return
    
    logger.info(f"Found {len(subject_dirs)} subjects in AT&T database")
    
    # Step 1: Process the AT&T database
    if not args.skip_processing:
        logger.info("Processing AT&T database...")
        start_time = time.time()
        
        # Process the database
        result = process_att_database(
            att_dir=args.att_dir,
            output_dir=args.processed_dir,
            config={
                'target_size': (112, 92),  # Standard size for AT&T database
                'detector_type': 'dnn',
                'enable_landmarks': True,
                'illumination_method': 'multi',
                'use_mediapipe': True,
                'enable_attention': True,
                'smart_crop': True,
                'edge_aware': True,
                'visualize': True
            }
        )
        
        if not result:
            logger.error("Database processing failed. Check the logs.")
            return
        
        processing_time = time.time() - start_time
        logger.info(f"Database processing completed in {processing_time:.2f} seconds")
    else:
        logger.info("Skipping database processing (--skip-processing flag used)")
        
        # Check if processed directory contains data
        processed_subject_dirs = [d for d in os.listdir(args.processed_dir) 
                                if d.startswith('s') and os.path.isdir(os.path.join(args.processed_dir, d))]
        
        if not processed_subject_dirs:
            logger.error(f"No processed data found in {args.processed_dir}. Cannot skip processing.")
            return
    
    # Step 2: Train face recognition model
    logger.info("Training face recognition model...")
    start_time = time.time()
    
    # Initialize face recognizer
    recognizer = FaceRecognizer(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        use_lda=args.use_lda,
        n_components=args.components
    )
    
    # Load data
    num_images = recognizer.load_data()
    
    if num_images == 0:
        logger.error("No images loaded. Please check the processed directory.")
        return
    
    logger.info(f"Loaded {num_images} images for training")
    
    # Train model
    recognizer.train(classifier_type=args.classifier)
    
    # Step 3: Evaluate model performance
    logger.info(f"Evaluating model performance using {args.k_folds}-fold cross-validation...")
    results = recognizer.evaluate(k_folds=args.k_folds)
    
    if results is None:
        logger.error("Evaluation failed. Check the logs.")
        return
    
    # Print results
    print("\nModel Evaluation Results:")
    if args.k_folds > 1:
        print(f"Cross-validation accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    else:
        print(f"Test accuracy: {results['mean_accuracy']:.4f}")
    
    print("\nClassification Report:")
    print(results['report'])
    
    # Step 4: Generate visualizations
    logger.info("Generating visualizations...")
    recognizer.visualize_eigenfaces()
    recognizer.visualize_results(results)
    
    training_time = time.time() - start_time
    logger.info(f"Model training and evaluation completed in {training_time:.2f} seconds")
    
    # Print final message
    print("\nProcess completed successfully!")
    print(f"- Processed images saved to: {args.processed_dir}")
    print(f"- Trained models saved to: {args.model_dir}")
    print(f"- Visualizations saved to: {args.model_dir}")
    print("\nTo test the model on a new image, run:")
    print(f"  python face_recognition_demo.py --image path/to/image.jpg --model-dir {args.model_dir}")

if __name__ == "__main__":
    main() 