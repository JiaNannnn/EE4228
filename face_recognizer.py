"""
Face Recognizer

Implements a face recognition model using PCA and LDA:
1. PCA for dimensionality reduction
2. LDA for optimal class separation
3. Nearest neighbor classification for recognition
"""

import numpy as np
import cv2
import os
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from config import PCA_VARIANCE, CONFIDENCE_THRESHOLD, RANDOM_STATE

# Setup logging
logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Face recognizer using PCA and LDA
    """
    
    def __init__(self, pca_variance=PCA_VARIANCE, confidence_threshold=CONFIDENCE_THRESHOLD):
        """
        Initialize the face recognizer
        
        Parameters:
        -----------
        pca_variance : float
            Fraction of variance to retain in PCA (0.0 to 1.0)
        confidence_threshold : float
            Confidence threshold for recognition (0.0 to 1.0)
        """
        self.pca_variance = pca_variance
        self.confidence_threshold = confidence_threshold
        
        # Initialize models
        self.pca = PCA(n_components=pca_variance, whiten=True, random_state=RANDOM_STATE)
        self.lda = LinearDiscriminantAnalysis()
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.scaler = StandardScaler()
        
        # Model state
        self.is_trained = False
        self.classes = None
        self.n_components_pca = None
        self.mean_face = None
        
        logger.info(f"Face recognizer initialized with PCA variance: {pca_variance}")
        
    def train(self, faces, labels, test_size=0.2):
        """
        Train the face recognition model
        
        Parameters:
        -----------
        faces : numpy.ndarray
            Array of face images (flattened)
        labels : numpy.ndarray
            Array of labels
        test_size : float
            Fraction of data to use for testing (0.0 to 1.0)
            
        Returns:
        --------
        dict
            Training results
        """
        if faces.shape[0] != labels.shape[0]:
            error_msg = f"Number of faces ({faces.shape[0]}) does not match number of labels ({labels.shape[0]})"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            faces, labels, test_size=test_size, random_state=RANDOM_STATE, stratify=labels
        )
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Save mean face and number of components
        self.mean_face = self.pca.mean_
        self.n_components_pca = self.pca.n_components_
        
        # Apply LDA if more than one class
        n_classes = len(np.unique(y_train))
        if n_classes > 1:
            X_train_lda = self.lda.fit_transform(X_train_pca, y_train)
            X_test_lda = self.lda.transform(X_test_pca)
        else:
            # Skip LDA if only one class
            X_train_lda = X_train_pca
            X_test_lda = X_test_pca
            
        # Train KNN classifier
        self.knn.fit(X_train_lda, y_train)
        
        # Calculate accuracy
        y_pred = self.knn.predict(X_test_lda)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save classes
        self.classes = np.unique(labels)
        
        # Mark as trained
        self.is_trained = True
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Log training results
        logger.info(f"Model trained with {faces.shape[0]} faces, {n_classes} classes")
        logger.info(f"PCA components: {self.n_components_pca}")
        logger.info(f"Training accuracy: {accuracy:.4f}")
        
        # Return training results
        return {
            "accuracy": accuracy,
            "n_components_pca": self.n_components_pca,
            "n_classes": n_classes,
            "confusion_matrix": cm
        }
        
    def predict(self, face):
        """
        Predict the identity of a face
        
        Parameters:
        -----------
        face : numpy.ndarray
            Face image (flattened)
            
        Returns:
        --------
        tuple
            (predicted_label, confidence)
        """
        if not self.is_trained:
            logger.error("Model is not trained")
            return None, 0.0
            
        # Ensure face is in correct shape
        if face.ndim > 1:
            face = face.flatten()
            
        # Reshape to 2D array (1 sample, n features)
        face = face.reshape(1, -1)
        
        # Preprocess the face
        face_scaled = self.scaler.transform(face)
        face_pca = self.pca.transform(face_scaled)
        
        # Apply LDA if applicable
        if hasattr(self.lda, 'scalings_'):
            face_lda = self.lda.transform(face_pca)
        else:
            face_lda = face_pca
            
        # Get nearest neighbors
        distances, indices = self.knn.kneighbors(face_lda)
        
        # Get predicted label (numeric)
        numeric_label = self.knn.predict(face_lda)[0]
        
        # Calculate confidence based on distance
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        confidence = 1.0 - min_distance / (max_distance + 1e-6)
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            logger.info(f"Confidence {confidence:.2f} below threshold {self.confidence_threshold:.2f}")
            return "Unknown", confidence
        
        # Convert numeric label to named label if available
        if hasattr(self, 'reverse_map') and numeric_label in self.reverse_map:
            predicted_label = self.reverse_map[numeric_label]
        else:
            predicted_label = str(numeric_label)
            
        logger.debug(f"Predicted {predicted_label} with confidence {confidence:.2f}")
        return predicted_label, confidence
        
    def save(self, model_file):
        """
        Save the model to a file
        
        Parameters:
        -----------
        model_file : str
            Path to the model file
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if not self.is_trained:
                logger.warning("Cannot save an untrained model")
                return False
                
            # Create directory if it doesn't exist
            model_dir = os.path.dirname(model_file)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Prepare the model dictionary
            model_dict = {
                'pca': self.pca,
                'lda': self.lda,
                'knn': self.knn,
                'scaler': self.scaler,
                'classes': self.classes,
                'pca_variance': self.pca_variance,
                'confidence_threshold': self.confidence_threshold,
                'is_trained': self.is_trained,
                'mean_face': self.mean_face if hasattr(self, 'mean_face') else None,
                'n_components_pca': self.n_components_pca if hasattr(self, 'n_components_pca') else None,
                'label_map': self.label_map if hasattr(self, 'label_map') else None,
                'reverse_map': self.reverse_map if hasattr(self, 'reverse_map') else None
            }
            
            # Save the model
            joblib.dump(model_dict, model_file)
            
            logger.info(f"Model saved to {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
            
    def load(self, model_file):
        """
        Load the model from a file
        
        Parameters:
        -----------
        model_file : str
            Path to the model file
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_file):
                logger.warning(f"Model file {model_file} does not exist")
                return False
                
            # Load the model
            model_dict = joblib.load(model_file)
            
            # Restore the model components
            self.pca = model_dict['pca']
            self.lda = model_dict['lda']
            self.knn = model_dict['knn']
            self.scaler = model_dict['scaler']
            self.classes = model_dict['classes']
            self.pca_variance = model_dict['pca_variance']
            self.confidence_threshold = model_dict['confidence_threshold']
            self.is_trained = model_dict['is_trained']
            
            # Restore optional components
            if 'mean_face' in model_dict and model_dict['mean_face'] is not None:
                self.mean_face = model_dict['mean_face']
            
            if 'n_components_pca' in model_dict and model_dict['n_components_pca'] is not None:
                self.n_components_pca = model_dict['n_components_pca']
                
            # Restore label mapping if available
            if 'label_map' in model_dict and model_dict['label_map'] is not None:
                self.label_map = model_dict['label_map']
                
            if 'reverse_map' in model_dict and model_dict['reverse_map'] is not None:
                self.reverse_map = model_dict['reverse_map']
            
            logger.info(f"Model loaded from {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def get_eigenfaces(self, n_components=10):
        """
        Get the top N eigenfaces
        
        Parameters:
        -----------
        n_components : int
            Number of eigenfaces to return
            
        Returns:
        --------
        list
            List of eigenfaces
        """
        if not self.is_trained:
            logger.error("Model is not trained")
            return None
            
        # Get the eigenfaces from PCA components
        eigenfaces = []
        for i in range(min(n_components, self.n_components_pca)):
            eigenface = self.pca.components_[i].reshape(int(np.sqrt(self.pca.components_[i].shape[0])), -1)
            eigenfaces.append(eigenface)
            
        return eigenfaces 