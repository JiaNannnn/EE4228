"""
Hybrid PCA-LDA Face Recognition Model

Implements a hybrid face recognition approach using:
1. PCA for dimensionality reduction (95% variance retention)
2. Regularized LDA for optimal class separation (ε=1e-4)
3. Confidence-based rejection for unknown faces (threshold=0.8)
4. Face embedding cache for improved performance (LRU, 100-entry)
"""

import numpy as np
import cv2
import os
import joblib
import time
import logging
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
from config import MODEL_CONFIG

logger = logging.getLogger("hybrid_recognition_model")

class HybridRecognitionModel:
    """
    Hybrid PCA-LDA model for face recognition with confidence-based rejection
    """
    
    def __init__(self,
                pca_variance=MODEL_CONFIG["pca_variance"],
                lda_regularization=MODEL_CONFIG["lda_regularization"],
                confidence_threshold=MODEL_CONFIG["confidence_threshold"],
                cache_size=MODEL_CONFIG["embedding_cache_size"]):
        """
        Initialize the hybrid PCA-LDA model
        
        Parameters:
        -----------
        pca_variance : float
            Fraction of variance to retain in PCA (0.0 to 1.0)
        lda_regularization : float
            Regularization parameter for LDA
        confidence_threshold : float
            Confidence threshold for face recognition (0.0 to 1.0)
        cache_size : int
            Size of the face embedding cache
        """
        self.pca_variance = pca_variance
        self.lda_regularization = lda_regularization
        self.confidence_threshold = confidence_threshold
        
        # Initialize PCA
        self.pca = PCA(n_components=pca_variance, whiten=True, random_state=42)
        
        # Initialize LDA with shrinkage regularization
        self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=lda_regularization)
        
        # Initialize standard scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Initialize classifier (default to SVM, can be changed during training)
        self.classifier = SVC(C=10.0, kernel="rbf", gamma="scale", probability=True)
        
        # Cache for face embeddings to avoid redundant computation
        self.embedding_cache = OrderedDict()
        self.cache_size = cache_size
        
        # Model metadata
        self.trained = False
        self.n_components_pca = None
        self.n_components_lda = None
        self.class_names = None
        self.training_time = None
        self.mean_face = None
        
    def _extract_features(self, faces):
        """
        Extract features from face images using PCA+LDA
        
        Parameters:
        -----------
        faces : list or numpy.ndarray
            List of face images or array of flattened face images
            
        Returns:
        --------
        numpy.ndarray
            Extracted features
        """
        if len(faces) == 0:
            logger.error("No faces provided for feature extraction")
            return None
            
        # Check if we need to flatten the images
        if isinstance(faces[0], np.ndarray) and len(faces[0].shape) > 1:
            # Flatten face images
            flat_faces = [face.flatten() for face in faces]
            X = np.array(flat_faces)
        else:
            # Already flattened
            X = np.array(faces)
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA
        X_pca = self.pca.transform(X_scaled)
        
        # Apply LDA if model is trained and we have more than one class
        if self.trained and hasattr(self.lda, "transform"):
            X_lda = self.lda.transform(X_pca)
            return X_lda
        else:
            return X_pca
    
    def train(self, faces, labels, test_size=0.2, classifier_type="svm", random_state=42):
        """
        Train the hybrid PCA-LDA model
        
        Parameters:
        -----------
        faces : list
            List of face images
        labels : list
            List of labels for the face images
        test_size : float
            Fraction of data to use for testing (0.0 to 1.0)
        classifier_type : str
            Type of classifier to use ("svm" or "knn")
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary with training metrics
        """
        if len(faces) == 0 or len(labels) == 0:
            logger.error("No data provided for training")
            return {"success": False, "error": "No data provided for training"}
            
        if len(faces) != len(labels):
            logger.error(f"Number of faces ({len(faces)}) does not match number of labels ({len(labels)})")
            return {"success": False, "error": "Data size mismatch"}
            
        # Check number of classes
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        if n_classes < 2:
            logger.error("Need at least 2 classes for training")
            return {"success": False, "error": "Insufficient classes for training"}
            
        start_time = time.time()
        
        # Flatten face images
        flat_faces = [face.flatten() for face in faces]
        X = np.array(flat_faces)
        y = np.array(labels)
        
        # Calculate mean face for visualization
        self.mean_face = np.mean(X, axis=0)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Fit scaler on training data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit PCA on training data
        self.pca.fit(X_train_scaled)
        self.n_components_pca = self.pca.n_components_
        
        # Transform data with PCA
        X_train_pca = self.pca.transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Fit LDA on PCA-transformed training data
        self.lda.fit(X_train_pca, y_train)
        self.n_components_lda = self.lda.n_components_
        
        # Transform data with LDA
        X_train_lda = self.lda.transform(X_train_pca)
        X_test_lda = self.lda.transform(X_test_pca)
        
        # Setup classifier
        if classifier_type.lower() == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=min(5, n_classes))
        else:
            self.classifier = SVC(C=10.0, kernel="rbf", gamma="scale", probability=True)
            
        # Train classifier
        self.classifier.fit(X_train_lda, y_train)
        
        # Make predictions on test set
        y_pred = self.classifier.predict(X_test_lda)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )
        
        # Calculate training time
        self.training_time = time.time() - start_time
        
        # Store class names
        self.class_names = list(unique_labels)
        self.trained = True
        
        # Cross-validation for robust evaluation
        cv_scores = self._cross_validation(X, y, n_splits=5, random_state=random_state)
        
        # Return metrics
        metrics = {
            "success": True,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_samples": len(X),
            "n_classes": n_classes,
            "n_components_pca": self.n_components_pca,
            "n_components_lda": self.n_components_lda,
            "training_time": self.training_time,
            "cv_accuracy_mean": np.mean(cv_scores),
            "cv_accuracy_std": np.std(cv_scores)
        }
        
        logger.info(f"Model trained with {metrics['accuracy']:.4f} accuracy on test set")
        
        return metrics
    
    def _cross_validation(self, X, y, n_splits=5, random_state=42):
        """
        Perform cross-validation for robust evaluation
        
        Returns:
        --------
        list
            List of accuracy scores for each fold
        """
        # Setup stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []
        
        for train_idx, test_idx in skf.split(X, y):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit PCA
            pca_cv = PCA(n_components=self.pca_variance, whiten=True)
            pca_cv.fit(X_train_scaled)
            
            # Transform with PCA
            X_train_pca = pca_cv.transform(X_train_scaled)
            X_test_pca = pca_cv.transform(X_test_scaled)
            
            # Fit LDA
            lda_cv = LinearDiscriminantAnalysis(
                solver="eigen", shrinkage=self.lda_regularization
            )
            lda_cv.fit(X_train_pca, y_train)
            
            # Transform with LDA
            X_train_lda = lda_cv.transform(X_train_pca)
            X_test_lda = lda_cv.transform(X_test_pca)
            
            # Train classifier
            if isinstance(self.classifier, KNeighborsClassifier):
                clf = KNeighborsClassifier(
                    n_neighbors=self.classifier.n_neighbors
                )
            else:
                clf = SVC(C=10.0, kernel="rbf", gamma="scale", probability=True)
                
            clf.fit(X_train_lda, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test_lda)
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)
            
        return scores
    
    def predict(self, face, cache_key=None):
        """
        Predict identity of a face image with confidence score
        
        Parameters:
        -----------
        face : numpy.ndarray
            Face image to recognize
        cache_key : any hashable, optional
            Key for caching the result
            
        Returns:
        --------
        str or None
            Predicted label or None if confidence below threshold
        float
            Confidence score (0.0 to 1.0)
        dict
            Additional prediction metadata
        """
        if not self.trained:
            logger.error("Model not trained, cannot make predictions")
            return None, 0.0, {"error": "Model not trained"}
            
        # Check cache first if cache_key provided
        if cache_key is not None and cache_key in self.embedding_cache:
            logger.debug(f"Using cached embedding for {cache_key}")
            embedding = self.embedding_cache[cache_key]
        else:
            # Extract features
            face_flat = face.flatten().reshape(1, -1)
            face_scaled = self.scaler.transform(face_flat)
            face_pca = self.pca.transform(face_scaled)
            embedding = self.lda.transform(face_pca)
            
            # Cache embedding if cache_key provided
            if cache_key is not None:
                # Apply LRU eviction if cache is full
                if len(self.embedding_cache) >= self.cache_size:
                    self.embedding_cache.popitem(last=False)
                    
                self.embedding_cache[cache_key] = embedding
        
        # Get prediction probabilities
        probs = self.classifier.predict_proba(embedding)[0]
        
        # Get predicted class and confidence
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # Apply confidence threshold
        if confidence >= self.confidence_threshold:
            predicted_label = self.class_names[pred_idx]
        else:
            predicted_label = None
            
        # Calculate reconstruction error using PCA
        face_flat = face.flatten().reshape(1, -1)
        face_scaled = self.scaler.transform(face_flat)
        face_pca = self.pca.transform(face_scaled)
        face_reconstructed = self.pca.inverse_transform(face_pca)
        reconstruction_error = np.mean(np.abs(face_scaled - face_reconstructed))
        
        # Return prediction results
        metadata = {
            "all_probabilities": dict(zip(self.class_names, probs)),
            "reconstruction_error": reconstruction_error,
            "threshold_applied": self.confidence_threshold
        }
        
        return predicted_label, confidence, metadata
    
    def get_eigenfaces(self, n_components=16):
        """
        Get the top eigenfaces for visualization
        
        Returns:
        --------
        list
            List of eigenface images
        """
        if not self.trained:
            logger.error("Model not trained, eigenfaces not available")
            return []
            
        # Get components from PCA
        components = self.pca.components_
        
        # Limit number of components
        n = min(n_components, len(components))
        
        # Reshape components to face images
        eigenfaces = []
        for i in range(n):
            # Normalize component
            component = components[i]
            component = (component - component.min()) / (component.max() - component.min())
            
            # Reshape to image dimensions
            height = int(np.sqrt(len(component)))
            width = len(component) // height
            eigenface = component[:height*width].reshape(height, width)
            
            # Convert to uint8
            eigenface = (eigenface * 255).astype(np.uint8)
            eigenfaces.append(eigenface)
            
        return eigenfaces
    
    def plot_eigenfaces(self, n_components=16, figsize=(12, 8)):
        """
        Plot the top eigenfaces
        """
        eigenfaces = self.get_eigenfaces(n_components)
        
        if not eigenfaces:
            return None
            
        # Setup plot
        n = len(eigenfaces)
        rows = int(np.ceil(n / 4))
        fig, axes = plt.subplots(rows, 4, figsize=figsize)
        axes = axes.flatten()
        
        # Plot eigenfaces
        for i in range(n):
            axes[i].imshow(eigenfaces[i], cmap="gray")
            axes[i].set_title(f"Eigenface {i+1}")
            axes[i].axis("off")
            
        # Hide empty subplots
        for i in range(n, len(axes)):
            axes[i].axis("off")
            
        plt.tight_layout()
        return fig
    
    def plot_fisherfaces(self, n_components=16, figsize=(12, 8)):
        """
        Plot the fisherfaces (LDA components)
        """
        if not self.trained or not hasattr(self.lda, "scalings_"):
            logger.error("Model not trained or LDA scalings not available")
            return None
            
        # Get scalings from LDA
        scalings = self.lda.scalings_
        
        # Limit number of components
        n = min(n_components, scalings.shape[1])
        
        # Setup plot
        rows = int(np.ceil(n / 4))
        fig, axes = plt.subplots(rows, 4, figsize=figsize)
        axes = axes.flatten()
        
        # Plot fisherfaces
        for i in range(n):
            # Get scaling vector
            scaling = scalings[:, i]
            
            # Project back to original space via PCA
            if hasattr(self.pca, "components_"):
                face_space_vector = self.pca.components_.T.dot(scaling)
                
                # Normalize
                face_space_vector = (face_space_vector - face_space_vector.min()) / (
                    face_space_vector.max() - face_space_vector.min()
                )
                
                # Reshape to image dimensions
                height = int(np.sqrt(len(face_space_vector)))
                width = len(face_space_vector) // height
                fisherface = face_space_vector[:height*width].reshape(height, width)
                
                # Convert to uint8
                fisherface = (fisherface * 255).astype(np.uint8)
                
                # Plot
                axes[i].imshow(fisherface, cmap="gray")
                axes[i].set_title(f"Fisherface {i+1}")
                axes[i].axis("off")
            
        # Hide empty subplots
        for i in range(n, len(axes)):
            axes[i].axis("off")
            
        plt.tight_layout()
        return fig
    
    def save(self, model_path):
        """
        Save the model to disk
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        """
        if not self.trained:
            logger.error("Cannot save untrained model")
            return False
            
        # Create directory if it doesn't exist
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
        
        # Create model data dictionary
        model_data = {
            "pca": self.pca,
            "lda": self.lda,
            "scaler": self.scaler,
            "classifier": self.classifier,
            "class_names": self.class_names,
            "n_components_pca": self.n_components_pca,
            "n_components_lda": self.n_components_lda,
            "pca_variance": self.pca_variance,
            "lda_regularization": self.lda_regularization,
            "confidence_threshold": self.confidence_threshold,
            "trained": self.trained,
            "mean_face": self.mean_face,
            "training_time": self.training_time,
            "model_version": "1.0",
            "timestamp": time.time()
        }
        
        # Save model
        try:
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, model_path):
        """
        Load the model from disk
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
            
        Returns:
        --------
        bool
            True if model loaded successfully, False otherwise
        """
        try:
            # Load model data
            model_data = joblib.load(model_path)
            
            # Set model components
            self.pca = model_data["pca"]
            self.lda = model_data["lda"]
            self.scaler = model_data["scaler"]
            self.classifier = model_data["classifier"]
            self.class_names = model_data["class_names"]
            self.n_components_pca = model_data["n_components_pca"]
            self.n_components_lda = model_data["n_components_lda"]
            self.pca_variance = model_data["pca_variance"]
            self.lda_regularization = model_data["lda_regularization"]
            self.confidence_threshold = model_data["confidence_threshold"]
            self.trained = model_data["trained"]
            self.mean_face = model_data["mean_face"]
            self.training_time = model_data["training_time"]
            
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False 