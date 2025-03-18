"""
Train Hybrid PCA-LDA Model

This script trains the hybrid PCA-LDA face recognition model on the AT&T dataset
and cross-validates with the LFW dataset. It performs:
1. Loading and preprocessing images from AT&T dataset
2. Training the hybrid model with 95% variance retention in PCA
3. Evaluating the model performance
4. Saving the trained model
"""

import cv2
import numpy as np
import os
import argparse
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow
from datetime import datetime
import traceback
import glob
from tqdm import tqdm
import random

# Import custom modules
from hybrid_recognition_model import HybridRecognitionModel
from advanced_preprocessor import AdvancedPreprocessor
from config import TRAINING_CONFIG, MODEL_CONFIG, PREPROCESSING_CONFIG, MLFLOW_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_hybrid_model")

def load_att_dataset(dataset_path, preprocessor=None, limit_per_subject=None, target_size=None):
    """
    Load AT&T dataset and preprocess images
    
    Parameters:
    -----------
    dataset_path : str
        Path to AT&T dataset
    preprocessor : AdvancedPreprocessor, optional
        Preprocessor to use, if None a default one will be created
    limit_per_subject : int, optional
        Max number of images to load per subject
    target_size : tuple, optional
        Target size for face images
        
    Returns:
    --------
    tuple
        (faces, labels, subject_names)
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return [], [], []
        
    if preprocessor is None:
        if target_size:
            preprocessor = AdvancedPreprocessor(target_size=target_size)
        else:
            preprocessor = AdvancedPreprocessor()
    
    logger.info(f"Loading AT&T dataset from {dataset_path}")
    
    faces = []
    labels = []
    subject_names = []
    
    # Find all subject directories (s1, s2, ...)
    subject_dirs = sorted(glob.glob(os.path.join(dataset_path, "s*")))
    
    if not subject_dirs:
        logger.error(f"No subject directories found in {dataset_path}")
        return [], [], []
        
    logger.info(f"Found {len(subject_dirs)} subjects")
    
    # Process each subject
    for subject_idx, subject_dir in enumerate(tqdm(subject_dirs, desc="Loading subjects")):
        subject_name = os.path.basename(subject_dir)
        subject_names.append(subject_name)
        
        # Find all PGM images for this subject
        image_files = sorted(glob.glob(os.path.join(subject_dir, "*.pgm")))
        
        if not image_files:
            logger.warning(f"No images found for subject {subject_name}")
            continue
            
        # Limit number of images per subject if requested
        if limit_per_subject and len(image_files) > limit_per_subject:
            image_files = image_files[:limit_per_subject]
            
        # Process each image
        for image_file in tqdm(image_files, desc=f"Processing {subject_name}", leave=False):
            try:
                # Load image
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    logger.warning(f"Failed to load image: {image_file}")
                    continue
                    
                # Preprocess image
                processed_face, _ = preprocessor.preprocess(image, cache_key=image_file)
                
                if processed_face is None:
                    logger.warning(f"Failed to preprocess image: {image_file}")
                    continue
                    
                # Add to dataset
                faces.append(processed_face)
                labels.append(subject_idx)
                
            except Exception as e:
                logger.error(f"Error processing image {image_file}: {e}")
                continue
    
    logger.info(f"Loaded {len(faces)} images for {len(subject_names)} subjects")
    return np.array(faces), np.array(labels), subject_names

def visualize_dataset_samples(faces, labels, subject_names, num_subjects=5, samples_per_subject=3):
    """
    Visualize random samples from the dataset
    
    Parameters:
    -----------
    faces : list
        List of face images
    labels : list
        List of labels
    subject_names : list
        List of subject names
    num_subjects : int
        Number of subjects to visualize
    samples_per_subject : int
        Number of samples per subject
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with visualized samples
    """
    if not faces or not labels or not subject_names:
        logger.error("Cannot visualize empty dataset")
        return None
        
    # Convert labels to numeric indices if they are strings
    if isinstance(labels[0], str):
        label_to_idx = {label: i for i, label in enumerate(set(labels))}
        numeric_labels = [label_to_idx[label] for label in labels]
    else:
        numeric_labels = labels
    
    # Get unique labels
    unique_labels = sorted(set(numeric_labels))
    
    # Select random subjects
    selected_subjects = random.sample(unique_labels, min(num_subjects, len(unique_labels)))
    
    # Create figure
    fig, axes = plt.subplots(num_subjects, samples_per_subject, figsize=(12, 10))
    
    # Flatten axes if necessary
    if num_subjects == 1:
        axes = axes.reshape(1, -1)
    
    # Plot samples
    for i, subject_idx in enumerate(selected_subjects):
        # Get samples for this subject
        subject_samples = [faces[j] for j in range(len(faces)) if numeric_labels[j] == subject_idx]
        
        # Select random samples
        selected_samples = random.sample(subject_samples, min(samples_per_subject, len(subject_samples)))
        
        # Display samples
        for j, sample in enumerate(selected_samples):
            axes[i, j].imshow(sample, cmap='gray')
            axes[i, j].set_title(f"Subject {subject_names[subject_idx]}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    return fig

def train_and_evaluate(faces, labels, classifier_type="svm", test_size=0.2, random_state=42,
                      pca_variance=0.95, model_dir="models", visualize=True):
    """
    Train and evaluate the hybrid PCA-LDA model
    
    Parameters:
    -----------
    faces : numpy.ndarray
        Array of face images
    labels : numpy.ndarray
        Array of labels
    classifier_type : str
        Type of classifier to use ("svm" or "knn")
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed for reproducibility
    pca_variance : float
        Fraction of variance to retain in PCA
    model_dir : str
        Directory to save the model
    visualize : bool
        Whether to visualize eigenfaces and fisherfaces
    
    Returns:
    --------
    tuple
        (model, metrics, figures)
    """
    if not faces.any() or not labels.any():
        logger.error("Cannot train with empty dataset")
        return None, {}, {}
        
    logger.info(f"Training hybrid PCA-LDA model with {len(faces)} images")
    
    # Create model
    model = HybridRecognitionModel(pca_variance=pca_variance)
    
    # Start MLflow run
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
    
    with mlflow.start_run(run_name=f"hybrid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Train model
        start_time = time.time()
        metrics = model.train(faces, labels, test_size=test_size, classifier_type=classifier_type, 
                            random_state=random_state)
        training_time = time.time() - start_time
        
        if not metrics["success"]:
            logger.error(f"Training failed: {metrics.get('error', 'Unknown error')}")
            return model, metrics, {}
            
        logger.info(f"Model trained in {training_time:.2f}s with accuracy: {metrics['accuracy']:.4f}")
        
        # Update metrics
        metrics["training_time"] = training_time
        
        # Log parameters
        mlflow.log_param("pca_variance", pca_variance)
        mlflow.log_param("classifier_type", classifier_type)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_samples", metrics["n_samples"])
        mlflow.log_param("n_classes", metrics["n_classes"])
        mlflow.log_param("n_components_pca", metrics["n_components_pca"])
        mlflow.log_param("n_components_lda", metrics["n_components_lda"])
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("cv_accuracy_mean", metrics["cv_accuracy_mean"])
        mlflow.log_metric("cv_accuracy_std", metrics["cv_accuracy_std"])
        
        figures = {}
        
        # Visualize eigenfaces and fisherfaces
        if visualize:
            try:
                # Eigenfaces
                eigenface_fig = model.plot_eigenfaces(n_components=16)
                if eigenface_fig:
                    eigenface_path = os.path.join(model_dir, "eigenfaces.png")
                    eigenface_fig.savefig(eigenface_path)
                    mlflow.log_artifact(eigenface_path)
                    figures["eigenfaces"] = eigenface_fig
                
                # Fisherfaces
                fisherface_fig = model.plot_fisherfaces(n_components=16)
                if fisherface_fig:
                    fisherface_path = os.path.join(model_dir, "fisherfaces.png")
                    fisherface_fig.savefig(fisherface_path)
                    mlflow.log_artifact(fisherface_path)
                    figures["fisherfaces"] = fisherface_fig
            except Exception as e:
                logger.error(f"Error visualizing faces: {e}")
        
        # Save model
        model_path = os.path.join(model_dir, f"hybrid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        os.makedirs(model_dir, exist_ok=True)
        
        success = model.save(model_path)
        if success:
            logger.info(f"Model saved to {model_path}")
            mlflow.log_artifact(model_path)
        else:
            logger.error("Failed to save model")
    
    return model, metrics, figures

def main():
    """Main function for training the hybrid PCA-LDA model"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Hybrid PCA-LDA Model")
    parser.add_argument("--dataset", type=str, default=TRAINING_CONFIG["train_dataset"],
                        help="Path to AT&T dataset")
    parser.add_argument("--classifier", type=str, default="svm", choices=["svm", "knn"],
                        help="Classifier type (svm or knn)")
    parser.add_argument("--test-size", type=float, default=TRAINING_CONFIG["test_size"],
                        help="Fraction of data to use for testing")
    parser.add_argument("--pca-variance", type=float, default=MODEL_CONFIG["pca_variance"],
                        help="Fraction of variance to retain in PCA")
    parser.add_argument("--limit-per-subject", type=int, default=None,
                        help="Maximum number of images to use per subject")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Disable eigenface and fisherface visualization")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save the model")
    parser.add_argument("--target-width", type=int, default=PREPROCESSING_CONFIG["target_size"][0],
                        help="Target width for face images")
    parser.add_argument("--target-height", type=int, default=PREPROCESSING_CONFIG["target_size"][1],
                        help="Target height for face images")
    parser.add_argument("--seed", type=int, default=TRAINING_CONFIG["random_state"],
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create preprocessor with target size
    target_size = (args.target_width, args.target_height)
    preprocessor = AdvancedPreprocessor(target_size=target_size)
    
    try:
        # Load dataset
        faces, labels, subject_names = load_att_dataset(
            args.dataset, 
            preprocessor=preprocessor,
            limit_per_subject=args.limit_per_subject,
            target_size=target_size
        )
        
        if len(faces) == 0:
            logger.error("No faces loaded from dataset")
            return
            
        # Visualize dataset samples
        if args.visualize:
            sample_fig = visualize_dataset_samples(faces, labels, subject_names)
            if sample_fig:
                os.makedirs(args.output_dir, exist_ok=True)
                sample_path = os.path.join(args.output_dir, "dataset_samples.png")
                sample_fig.savefig(sample_path)
                logger.info(f"Dataset samples saved to {sample_path}")
        
        # Train and evaluate model
        model, metrics, figures = train_and_evaluate(
            faces, 
            labels,
            classifier_type=args.classifier,
            test_size=args.test_size,
            random_state=args.seed,
            pca_variance=args.pca_variance,
            model_dir=args.output_dir,
            visualize=args.visualize
        )
        
        # Print metrics
        if metrics and metrics.get("success", False):
            print("\nTraining Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Cross-Validation Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
            print(f"PCA Components: {metrics['n_components_pca']}")
            print(f"LDA Components: {metrics['n_components_lda']}")
            print(f"Training Time: {metrics['training_time']:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 