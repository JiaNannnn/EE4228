"""
Face Recognition System using the processed AT&T Database

This script:
1. Loads processed AT&T face images
2. Extracts features using PCA (eigenfaces) and optionally LDA
3. Trains a classifier for face recognition
4. Evaluates performance using cross-validation
5. Provides visualization of eigenfaces and recognition results
"""

import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_recognition")

class FaceRecognizer:
    """
    Face Recognition System using the processed AT&T Database
    """
    
    def __init__(self, processed_dir, model_dir='models', use_lda=True, n_components=None):
        """
        Initialize the face recognition system
        
        Args:
            processed_dir: Directory containing processed face images
            model_dir: Directory to save/load trained models
            use_lda: Whether to use LDA for feature extraction after PCA
            n_components: Number of components for PCA (if None, 0.95 variance retained)
        """
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        self.use_lda = use_lda
        self.n_components = n_components
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.pca = None
        self.lda = None
        self.classifier = None
        
        # Initialize data containers
        self.images = []
        self.labels = []
        self.subject_names = {}
        
        logger.info(f"Initializing Face Recognizer with {'PCA+LDA' if use_lda else 'PCA'}")
    
    def load_data(self):
        """
        Load processed images from the AT&T database
        
        Returns:
            Number of images loaded
        """
        # Get all subject directories
        subject_dirs = sorted(glob.glob(os.path.join(self.processed_dir, "s*")))
        
        if not subject_dirs:
            logger.error(f"No subject directories found in {self.processed_dir}")
            return 0
        
        logger.info(f"Found {len(subject_dirs)} subjects in {self.processed_dir}")
        
        # Clear existing data
        self.images = []
        self.labels = []
        self.subject_names = {}
        
        # Process each subject
        for i, subject_dir in enumerate(tqdm(subject_dirs, desc="Loading subjects")):
            subject_id = os.path.basename(subject_dir)
            
            # Map subject ID to numeric label
            label = i
            self.subject_names[label] = subject_id
            
            # Get all processed images for this subject
            image_files = sorted(glob.glob(os.path.join(subject_dir, "*_processed.jpg")))
            
            if not image_files:
                logger.warning(f"No processed images found for subject {subject_id}")
                continue
            
            # Load each image
            for image_path in image_files:
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        logger.warning(f"Failed to load image: {image_path}")
                        continue
                    
                    # Flatten image to 1D array
                    img_flat = img.flatten()
                    
                    # Add to data
                    self.images.append(img_flat)
                    self.labels.append(label)
                    
                except Exception as e:
                    logger.error(f"Error loading {image_path}: {str(e)}")
        
        # Convert to numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        logger.info(f"Loaded {len(self.images)} images from {len(self.subject_names)} subjects")
        
        return len(self.images)
    
    def train(self, classifier_type='svm'):
        """
        Train the face recognition model
        
        Args:
            classifier_type: Type of classifier to use ('svm' or 'knn')
            
        Returns:
            Trained classifier
        """
        if len(self.images) == 0:
            logger.error("No images loaded. Call load_data() first.")
            return None
        
        # Determine number of components
        n_components = self.n_components
        if n_components is None:
            # Use enough components to retain 95% of variance
            n_components = min(len(self.images) - 1, self.images.shape[1])
        
        # Initialize and fit PCA
        logger.info(f"Training PCA with {n_components} components")
        self.pca = PCA(n_components=n_components, whiten=True)
        pca_features = self.pca.fit_transform(self.images)
        
        # If using LDA, apply it after PCA
        if self.use_lda:
            # Number of components for LDA is at most (num_classes - 1)
            n_components_lda = min(len(np.unique(self.labels)) - 1, pca_features.shape[1])
            logger.info(f"Training LDA with {n_components_lda} components")
            
            self.lda = LDA(n_components=n_components_lda)
            features = self.lda.fit_transform(pca_features, self.labels)
        else:
            features = pca_features
        
        # Train classifier
        if classifier_type == 'svm':
            logger.info("Training SVM classifier")
            self.classifier = SVC(kernel='rbf', gamma='scale', probability=True)
        else:  # knn
            logger.info("Training KNN classifier")
            self.classifier = KNeighborsClassifier(n_neighbors=5)
        
        self.classifier.fit(features, self.labels)
        
        # Save models
        self.save_models()
        
        return self.classifier
    
    def save_models(self):
        """Save trained models to disk"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        pca_path = os.path.join(self.model_dir, "pca_model.pkl")
        joblib.dump(self.pca, pca_path)
        logger.info(f"PCA model saved to {pca_path}")
        
        if self.use_lda and self.lda is not None:
            lda_path = os.path.join(self.model_dir, "lda_model.pkl")
            joblib.dump(self.lda, lda_path)
            logger.info(f"LDA model saved to {lda_path}")
        
        classifier_path = os.path.join(self.model_dir, "classifier_model.pkl")
        joblib.dump(self.classifier, classifier_path)
        logger.info(f"Classifier model saved to {classifier_path}")
        
        # Save subject mapping
        np.save(os.path.join(self.model_dir, "subject_names.npy"), self.subject_names)
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            pca_path = os.path.join(self.model_dir, "pca_model.pkl")
            self.pca = joblib.load(pca_path)
            logger.info(f"PCA model loaded from {pca_path}")
            
            if self.use_lda:
                lda_path = os.path.join(self.model_dir, "lda_model.pkl")
                if os.path.exists(lda_path):
                    self.lda = joblib.load(lda_path)
                    logger.info(f"LDA model loaded from {lda_path}")
            
            classifier_path = os.path.join(self.model_dir, "classifier_model.pkl")
            self.classifier = joblib.load(classifier_path)
            logger.info(f"Classifier model loaded from {classifier_path}")
            
            # Load subject mapping
            subject_path = os.path.join(self.model_dir, "subject_names.npy")
            if os.path.exists(subject_path):
                self.subject_names = np.load(subject_path, allow_pickle=True).item()
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def recognize(self, image):
        """
        Recognize a face in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (predicted label, confidence, subject name)
        """
        if self.pca is None or self.classifier is None:
            logger.error("Models not trained. Call train() or load_models() first.")
            return None, 0, "Unknown"
        
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to match training data if needed
        expected_shape = int(np.sqrt(self.pca.components_[0].shape[0]))
        if gray.shape[0] != expected_shape or gray.shape[1] != expected_shape:
            gray = cv2.resize(gray, (expected_shape, expected_shape))
        
        # Flatten image
        img_flat = gray.flatten().reshape(1, -1)
        
        # Extract features
        pca_features = self.pca.transform(img_flat)
        
        if self.use_lda and self.lda is not None:
            features = self.lda.transform(pca_features)
        else:
            features = pca_features
        
        # Predict with probabilities
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(features)[0]
            predicted_label = np.argmax(proba)
            confidence = proba[predicted_label]
        else:
            predicted_label = self.classifier.predict(features)[0]
            # For KNN, calculate confidence as inverse of mean distance
            if hasattr(self.classifier, "kneighbors"):
                distances, _ = self.classifier.kneighbors(features)
                confidence = 1.0 / (1.0 + np.mean(distances))
            else:
                confidence = 0.0
        
        # Get subject name
        subject_name = self.subject_names.get(predicted_label, "Unknown")
        
        return predicted_label, confidence, subject_name
    
    def evaluate(self, k_folds=5, test_size=0.2, random_state=42):
        """
        Evaluate model performance using k-fold cross-validation or train/test split
        
        Args:
            k_folds: Number of folds for cross-validation (if 0, uses train/test split)
            test_size: Fraction of data to use for testing (for train/test split)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(self.images) == 0:
            logger.error("No images loaded. Call load_data() first.")
            return None
        
        results = {
            'accuracy': [],
            'confusion_matrix': None,
            'report': None
        }
        
        if k_folds > 1:
            # Perform k-fold cross-validation
            logger.info(f"Performing {k_folds}-fold cross-validation")
            
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
            
            fold_accuracies = []
            all_y_true = []
            all_y_pred = []
            
            for fold, (train_index, test_index) in enumerate(kf.split(self.images)):
                logger.info(f"Processing fold {fold+1}/{k_folds}")
                
                # Split data
                X_train, X_test = self.images[train_index], self.images[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                
                # Train PCA
                n_components = self.n_components
                if n_components is None:
                    n_components = min(len(X_train) - 1, X_train.shape[1])
                
                pca = PCA(n_components=n_components, whiten=True)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)
                
                # Train LDA if required
                if self.use_lda:
                    n_components_lda = min(len(np.unique(y_train)) - 1, X_train_pca.shape[1])
                    lda = LDA(n_components=n_components_lda)
                    X_train_features = lda.fit_transform(X_train_pca, y_train)
                    X_test_features = lda.transform(X_test_pca)
                else:
                    X_train_features = X_train_pca
                    X_test_features = X_test_pca
                
                # Train classifier
                clf = SVC(kernel='rbf', gamma='scale')
                clf.fit(X_train_features, y_train)
                
                # Predict
                y_pred = clf.predict(X_test_features)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                
                # Store predictions for overall confusion matrix
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                
                logger.info(f"Fold {fold+1} accuracy: {accuracy:.4f}")
            
            # Calculate overall metrics
            results['accuracy'] = fold_accuracies
            results['mean_accuracy'] = np.mean(fold_accuracies)
            results['std_accuracy'] = np.std(fold_accuracies)
            results['confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred)
            results['report'] = classification_report(all_y_true, all_y_pred)
            
            logger.info(f"Cross-validation accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            
        else:
            # Perform single train/test split
            logger.info(f"Performing {int(test_size*100)}% test split")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.images, self.labels, test_size=test_size, random_state=random_state,
                stratify=self.labels
            )
            
            # Train PCA
            n_components = self.n_components
            if n_components is None:
                n_components = min(len(X_train) - 1, X_train.shape[1])
            
            pca = PCA(n_components=n_components, whiten=True)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            # Train LDA if required
            if self.use_lda:
                n_components_lda = min(len(np.unique(y_train)) - 1, X_train_pca.shape[1])
                lda = LDA(n_components=n_components_lda)
                X_train_features = lda.fit_transform(X_train_pca, y_train)
                X_test_features = lda.transform(X_test_pca)
            else:
                X_train_features = X_train_pca
                X_test_features = X_test_pca
            
            # Train classifier
            clf = SVC(kernel='rbf', gamma='scale')
            clf.fit(X_train_features, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_features)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            results['accuracy'] = [accuracy]
            results['mean_accuracy'] = accuracy
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            results['report'] = classification_report(y_test, y_pred)
            
            logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return results
    
    def visualize_eigenfaces(self, n_eigenfaces=5):
        """Visualize top eigenfaces"""
        if self.pca is None:
            logger.error("PCA model not trained. Call train() first.")
            return
        
        # Get dimensions of input images
        n_samples, n_features = self.images.shape
        image_shape = int(np.sqrt(n_features))
        
        n_eigenfaces = min(n_eigenfaces, len(self.pca.components_))
        
        # Display mean face
        mean_face = self.pca.mean_.reshape(image_shape, image_shape)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, n_eigenfaces + 1, 1)
        plt.imshow(mean_face, cmap='gray')
        plt.title('Mean Face')
        plt.axis('off')
        
        # Display top eigenfaces
        for i in range(n_eigenfaces):
            plt.subplot(1, n_eigenfaces + 1, i + 2)
            eigenface = self.pca.components_[i].reshape(image_shape, image_shape)
            plt.imshow(eigenface, cmap='gray')
            plt.title(f'Eigenface {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "eigenfaces.png"))
        plt.close()
        
        logger.info(f"Eigenfaces visualization saved to {os.path.join(self.model_dir, 'eigenfaces.png')}")
    
    def visualize_results(self, results):
        """Visualize evaluation results"""
        if results is None:
            logger.error("No results to visualize.")
            return
        
        try:
            # Plot confusion matrix
            if results['confusion_matrix'] is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(results['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                
                classes = sorted(self.subject_names.keys())
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=90)
                plt.yticks(tick_marks, classes)
                
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.savefig(os.path.join(self.model_dir, "confusion_matrix.png"))
                plt.close()
                
                logger.info(f"Confusion matrix saved to {os.path.join(self.model_dir, 'confusion_matrix.png')}")
            
            # Plot accuracy distribution for k-fold CV
            if len(results['accuracy']) > 1:
                plt.figure(figsize=(8, 6))
                plt.bar(range(1, len(results['accuracy']) + 1), results['accuracy'])
                plt.axhline(y=results['mean_accuracy'], color='r', linestyle='-', label=f"Mean Accuracy: {results['mean_accuracy']:.4f}")
                plt.xlabel('Fold')
                plt.ylabel('Accuracy')
                plt.title('Cross-Validation Accuracy')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.model_dir, "cv_accuracy.png"))
                plt.close()
                
                logger.info(f"Cross-validation accuracy plot saved to {os.path.join(self.model_dir, 'cv_accuracy.png')}")
            
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")

def main():
    """Main function to demonstrate the face recognition system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Recognition using AT&T Database')
    parser.add_argument('--processed-dir', type=str, default='processed_att',
                      help='Directory containing processed AT&T images')
    parser.add_argument('--model-dir', type=str, default='face_models',
                      help='Directory to save models')
    parser.add_argument('--use-lda', action='store_true', default=True,
                      help='Use LDA after PCA for feature extraction')
    parser.add_argument('--components', type=int, default=None,
                      help='Number of PCA components (if None, 95% variance preserved)')
    parser.add_argument('--k-folds', type=int, default=5,
                      help='Number of folds for cross-validation')
    parser.add_argument('--classifier', type=str, default='svm',
                      choices=['svm', 'knn'], help='Classifier type')
    parser.add_argument('--visualize', action='store_true', default=True,
                      help='Visualize results')
    
    args = parser.parse_args()
    
    # Create face recognizer
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
    
    # Train models
    recognizer.train(classifier_type=args.classifier)
    
    # Evaluate performance
    results = recognizer.evaluate(k_folds=args.k_folds)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    if args.k_folds > 1:
        print(f"Cross-validation accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    else:
        print(f"Test accuracy: {results['mean_accuracy']:.4f}")
    
    print("\nClassification Report:")
    print(results['report'])
    
    # Visualize results
    if args.visualize:
        recognizer.visualize_eigenfaces()
        recognizer.visualize_results(results)
    
    print(f"\nModels saved to {args.model_dir}")
    print(f"Run face_recognition_demo.py to test the trained models on new images")

if __name__ == "__main__":
    main() 