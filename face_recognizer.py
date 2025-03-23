import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from facenet_pytorch import InceptionResnetV1
import torch

class FaceRecognizer:
    """
    Face recognition module using FaceNet embeddings and SVM/KNN classification.
    This implements the recognition approach described in the system architecture.
    """
    
    def __init__(self, model_path=None, method='svm', threshold=0.8):
        """
        Initialize face recognizer with FaceNet model.
        
        Args:
            model_path: Path to pre-trained FaceNet model (optional)
            method: Classification method ('svm' or 'knn')
            threshold: Confidence threshold for recognition
        """
        self.threshold = threshold
        self.method = method
        self.model_trained = False
        self.label_encoder = LabelEncoder()
        self.single_person_mode = False
        self.single_person_name = None
        self.reference_embeddings = None
        
        # Initialize FaceNet model
        try:
            # Try using PyTorch implementation
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
            self.backend = 'pytorch'
        except Exception as e:
            print(f"Error loading FaceNet from PyTorch: {e}")
            self.backend = 'tensorflow'
            # Fallback to loading from TensorFlow model
            try:
                if model_path and os.path.exists(model_path):
                    self.facenet = tf.saved_model.load(model_path)
                else:
                    # Use default path
                    default_paths = [
                        './facenet_keras.h5',
                        './models/facenet_keras.h5'
                    ]
                    for path in default_paths:
                        if os.path.exists(path):
                            self.facenet = tf.keras.models.load_model(path)
                            break
            except Exception as e:
                print(f"Error loading FaceNet from TensorFlow: {e}")
                # Fallback to OpenCV DNN
                self.backend = 'opencv'
                
        # Initialize classifier based on method
        if method == 'svm':
            self.classifier = SVC(kernel='linear', probability=True)
            # For single-person case
            self.one_class_classifier = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        else:  # KNN
            self.classifier = KNeighborsClassifier(n_neighbors=5)
    
    def _get_embedding(self, face_img):
        """
        Generate a 128-dimensional embedding vector for a face.
        
        Args:
            face_img: Preprocessed face image (already aligned, normalized)
            
        Returns:
            128-dimensional embedding vector
        """
        try:
            # Resize image to the expected input size
            if self.backend == 'pytorch':
                # Convert to correct format for PyTorch
                if len(face_img.shape) == 2:  # Grayscale
                    face_img = np.stack([face_img] * 3, axis=-1)  # Convert to 3 channel
                
                # Ensure right shape for PyTorch (B x C x H x W)
                if face_img.shape[2] == 3:  # If channels are last
                    face_img = np.transpose(face_img, (2, 0, 1))
                
                # Convert to PyTorch tensor
                face_tensor = torch.from_numpy(face_img).unsqueeze(0).float()
                
                # Get embedding
                with torch.no_grad():
                    embedding = self.facenet(face_tensor).numpy().flatten()
                
            elif self.backend == 'tensorflow':
                # Convert grayscale to RGB if needed
                if len(face_img.shape) == 2:
                    face_img = np.stack([face_img] * 3, axis=-1)
                
                # Ensure image is in range [0, 1]
                if face_img.max() > 1.0:
                    face_img = face_img / 255.0
                
                # Resize to FaceNet input size (160x160)
                face_img = cv2.resize(face_img, (160, 160))
                
                # Add batch dimension
                face_batch = np.expand_dims(face_img, axis=0)
                
                # Get embedding
                embedding = self.facenet.predict(face_batch)[0]
                
            else:  # OpenCV fallback - use LBPH features
                if len(face_img.shape) == 3:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                # Extract LBPH features (simple fallback)
                lbph = cv2.face.LBPHFaceRecognizer_create()
                hist = np.zeros(256)
                for i in range(face_img.shape[0]):
                    for j in range(face_img.shape[1]):
                        hist[face_img[i, j]] += 1
                
                # Normalize histogram
                hist = hist / hist.sum()
                
                # Pad to 128 dimensions for compatibility
                embedding = np.zeros(128)
                embedding[:min(256, 128)] = hist[:min(256, 128)]
            
            # L2 normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error generating face embedding: {e}")
            # Return a zero vector on failure
            return np.zeros(128)
    
    def train(self, face_images, face_labels):
        """
        Train the face recognition model using face images and their labels.
        
        Args:
            face_images: List of preprocessed face images
            face_labels: List of corresponding labels
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            if len(face_images) == 0 or len(face_labels) == 0:
                return False
                
            print(f"Training face recognizer with {len(face_images)} images")
            
            # Generate embeddings for all faces
            embeddings = []
            for face in face_images:
                embedding = self._get_embedding(face)
                embeddings.append(embedding)
                
            embeddings = np.array(embeddings)
            
            # Get unique labels
            unique_labels = np.unique(face_labels)
            
            # Check if we have only one person
            if len(unique_labels) == 1:
                print("Single person mode: Using distance-based recognition")
                self.single_person_mode = True
                self.single_person_name = unique_labels[0]
                self.reference_embeddings = embeddings
                
                # Calculate mean embedding and distances for threshold
                self.mean_embedding = np.mean(embeddings, axis=0)
                
                # Train one-class SVM just for outlier detection
                self.one_class_classifier.fit(embeddings)
                
                self.model_trained = True
                return True
            else:
                # Multi-person mode: Use standard classifier
                self.single_person_mode = False
                self.single_person_name = None
                self.reference_embeddings = None
                
                # Convert string labels to numeric using LabelEncoder
                numeric_labels = self.label_encoder.fit_transform(face_labels)
                
                # Train classifier
                self.classifier.fit(embeddings, numeric_labels)
                self.model_trained = True
                return True
            
        except Exception as e:
            print(f"Error training face recognizer: {e}")
            return False
    
    def predict(self, face_img):
        """
        Predict the identity of a face.
        
        Args:
            face_img: Preprocessed face image
            
        Returns:
            Tuple of (predicted_name, confidence_score)
        """
        try:
            if not self.model_trained:
                return "Unknown", 0.0
                
            # Get embedding for face
            embedding = self._get_embedding(face_img)
            
            # Reshape for single prediction
            embedding = embedding.reshape(1, -1)
            
            # Handle single person mode differently
            if self.single_person_mode:
                try:
                    # Method 1: Use One-Class SVM for outlier detection
                    # Negative score = outlier, positive = inlier
                    svm_score = self.one_class_classifier.score_samples(embedding)[0]
                    
                    # Convert to probability-like score (higher = more likely)
                    # Map from typically -1 to 1 range to 0 to 1
                    svm_confidence = float(min(max((svm_score + 1) / 2, 0), 1))
                    
                    # Method 2: Use cosine similarity with mean embedding
                    cosine_sim = float(np.dot(embedding.flatten(), self.mean_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(self.mean_embedding)
                    ))
                    
                    # Method 3: Use minimum distance to reference embeddings
                    distances = []
                    for ref_emb in self.reference_embeddings:
                        dist = np.linalg.norm(embedding - ref_emb.reshape(1, -1))
                        distances.append(float(dist))
                    min_distance = min(distances)
                    distance_score = float(1.0 / (1.0 + min_distance))  # Convert to similarity
                    
                    # Combine scores (weighted average)
                    confidence = float(0.4 * svm_confidence + 0.3 * cosine_sim + 0.3 * distance_score)
                    
                    if confidence >= self.threshold:
                        return self.single_person_name, confidence
                    else:
                        return "Unknown", confidence
                except Exception as e:
                    print(f"Error in single person prediction: {e}")
                    return "Unknown", 0.0
            else:
                # Multi-person mode: Use standard classifier
                if self.method == 'svm':
                    # SVM with probabilities
                    prediction = int(self.classifier.predict(embedding)[0])
                    proba = self.classifier.predict_proba(embedding)[0]
                    confidence = float(proba[prediction])
                else:
                    # KNN with distances
                    neighbors = self.classifier.kneighbors(embedding, return_distance=True)
                    distances = neighbors[0][0]
                    indices = neighbors[1][0]
                    
                    # Convert distance to confidence (closer = higher confidence)
                    avg_distance = float(np.mean(distances))
                    confidence = float(1.0 / (1.0 + avg_distance))
                    
                    # Get prediction from most common neighbor class
                    prediction = int(self.classifier.predict(embedding)[0])
                
                # Convert numeric prediction back to label
                predicted_name = self.label_encoder.inverse_transform([prediction])[0]
                
                # Apply threshold
                if confidence < self.threshold:
                    return "Unknown", confidence
                    
                return predicted_name, confidence
            
        except Exception as e:
            print(f"Error predicting face: {e}")
            return "Unknown", 0.0
            
    def save_model(self, model_path):
        """Save the trained model to disk"""
        try:
            if not self.model_trained:
                return False
                
            import joblib
            
            # Create model directory if it doesn't exist
            model_dir = os.path.dirname(model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            
            # Save the model components
            model_data = {
                'classifier': self.classifier if not self.single_person_mode else self.one_class_classifier,
                'label_encoder': self.label_encoder,
                'method': self.method,
                'threshold': self.threshold,
                'single_person_mode': self.single_person_mode,
                'single_person_name': self.single_person_name,
                'reference_embeddings': self.reference_embeddings,
                'mean_embedding': getattr(self, 'mean_embedding', None)
            }
            
            joblib.dump(model_data, model_path)
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        try:
            import joblib
            
            if not os.path.exists(model_path):
                return False
                
            # Load the model components
            model_data = joblib.load(model_path)
            
            self.single_person_mode = model_data.get('single_person_mode', False)
            
            if self.single_person_mode:
                self.one_class_classifier = model_data['classifier']
                self.single_person_name = model_data['single_person_name']
                self.reference_embeddings = model_data['reference_embeddings']
                self.mean_embedding = model_data.get('mean_embedding')
            else:
                self.classifier = model_data['classifier']
                
            self.label_encoder = model_data['label_encoder']
            self.method = model_data['method']
            self.threshold = model_data['threshold']
            
            self.model_trained = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False