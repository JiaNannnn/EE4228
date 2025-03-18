import numpy as np
import cv2
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class PCAFaceDetector:
    """
    PCA-based face detector that uses eigenfaces to distinguish between
    face and non-face images.
    """
    
    def __init__(self, n_components=0.95, target_size=(100, 100)):
        """
        Initialize the PCA face detector
        
        Parameters:
        -----------
        n_components : float or int
            Number of components for PCA. If float between 0 and 1, represents the
            variance to be explained. If int, represents the number of components.
        target_size : tuple
            Size to resize input images to (width, height)
        """
        self.n_components = n_components
        self.target_size = target_size
        self.pca = None
        self.scaler = None
        self.classifier = None
        self.is_trained = False
        self.threshold = None
        
    def preprocess_image(self, image):
        """
        Preprocess an image for the PCA model
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image, grayscale or color
            
        Returns:
        --------
        processed_image : numpy.ndarray
            Preprocessed grayscale image of target size
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to target size
        if gray.shape[:2] != self.target_size[::-1]:  # Note: target_size is (width, height)
            resized = cv2.resize(gray, self.target_size)
        else:
            resized = gray
            
        # Normalize the image
        normalized = cv2.equalizeHist(resized)
        
        return normalized
    
    def train(self, face_images, non_face_images):
        """
        Train the PCA face detector on face and non-face images
        
        Parameters:
        -----------
        face_images : numpy.ndarray
            Array of face images, shape (n_samples, height, width)
        non_face_images : numpy.ndarray
            Array of non-face images, shape (n_samples, height, width)
            
        Returns:
        --------
        self : PCAFaceDetector
            Returns self for method chaining
        """
        print(f"Training PCA face detector with {len(face_images)} face images and {len(non_face_images)} non-face images")
        
        # Ensure all images are processed
        processed_faces = np.array([self.preprocess_image(img) for img in face_images])
        processed_non_faces = np.array([self.preprocess_image(img) for img in non_face_images])
        
        # Flatten images
        n_face_samples = len(processed_faces)
        n_non_face_samples = len(processed_non_faces)
        h, w = processed_faces[0].shape
        
        X_faces = processed_faces.reshape(n_face_samples, h * w)
        X_non_faces = processed_non_faces.reshape(n_non_face_samples, h * w)
        
        # Combine data and create labels
        X = np.vstack([X_faces, X_non_faces])
        y = np.hstack([np.ones(n_face_samples), np.zeros(n_non_face_samples)])
        
        # Standardize the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train PCA
        self.pca = PCA(n_components=self.n_components, whiten=True)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate reconstruction error for faces
        X_faces_scaled = self.scaler.transform(X_faces)
        X_faces_pca = self.pca.transform(X_faces_scaled)
        X_faces_rec = self.pca.inverse_transform(X_faces_pca)
        reconstruction_errors = np.mean(np.square(X_faces_scaled - X_faces_rec), axis=1)
        
        # Set threshold for face detection based on reconstruction error
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        self.threshold = mean_error + 2 * std_error
        
        print(f"PCA face detector trained with {self.pca.n_components_} components")
        print(f"Reconstruction error threshold: {self.threshold:.6f}")
        
        # Train SVM classifier for additional accuracy
        self.classifier = SVC(kernel='rbf', probability=True)
        self.classifier.fit(X_pca, y)
        
        self.is_trained = True
        return self
    
    def detect(self, image):
        """
        Detect if an image contains a face
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
            
        Returns:
        --------
        is_face : bool
            True if the image contains a face, False otherwise
        confidence : float
            Confidence score (higher means more confident it's a face)
        """
        if not self.is_trained:
            raise ValueError("PCA face detector is not trained yet")
        
        # Preprocess image
        processed = self.preprocess_image(image)
        h, w = processed.shape
        X = processed.reshape(1, h * w)
        
        # Apply PCA transformation
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Compute reconstruction error
        X_rec = self.pca.inverse_transform(X_pca)
        error = np.mean(np.square(X_scaled - X_rec))
        
        # Get classifier probability
        prob = self.classifier.predict_proba(X_pca)[0][1]
        
        # Face is detected if error is below threshold and classifier agrees
        is_face = error < self.threshold and prob > 0.5
        
        # Confidence score (inverse of error, normalized)
        confidence = 1.0 - (error / (self.threshold * 2))
        confidence = max(0.0, min(1.0, confidence))
        
        # Combine with classifier probability
        confidence = 0.7 * confidence + 0.3 * prob
        
        return is_face, confidence
    
    def save_model(self, file_path):
        """
        Save the trained model to a file
        
        Parameters:
        -----------
        file_path : str
            Path to save the model to
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'threshold': self.threshold,
            'n_components': self.n_components,
            'target_size': self.target_size,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        print(f"PCA face detector model saved to {file_path}")
        
    def load_model(self, file_path):
        """
        Load a trained model from a file
        
        Parameters:
        -----------
        file_path : str
            Path to load the model from
            
        Returns:
        --------
        self : PCAFaceDetector
            Returns self for method chaining
        """
        model_data = joblib.load(file_path)
        
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.threshold = model_data['threshold']
        self.n_components = model_data['n_components']
        self.target_size = model_data['target_size']
        self.is_trained = model_data['is_trained']
        
        print(f"PCA face detector model loaded from {file_path}")
        return self 