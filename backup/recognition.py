import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import logging

from user_manager import UserManager
from face_utils import FaceDetector, FacePreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_recognition')

class FaceRecognizer:
    """Class for face recognition using PCA embeddings"""
    
    def __init__(self, pca_model_path='models/pca_face_model.joblib', 
                 confidence_threshold=0.6, log_dir='logs'):
        """
        Initialize face recognizer
        
        Parameters:
        -----------
        pca_model_path : str
            Path to PCA model file
        confidence_threshold : float
            Confidence threshold for recognition
        log_dir : str
            Directory to save recognition logs
        """
        self.pca_model_path = pca_model_path
        self.confidence_threshold = confidence_threshold
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Load PCA model
        self.pca_model = self._load_pca_model()
        
        # Initialize user manager
        self.user_manager = UserManager()
        
        # Set up face detector and preprocessor
        self.detector = FaceDetector()
        self.preprocessor = FacePreprocessor(target_size=(100, 100))
        
        # Initialize recognition log
        self.log_file = os.path.join(log_dir, f"recognition_log_{datetime.now().strftime('%Y%m%d')}.csv")
        self._init_log_file()
    
    def _load_pca_model(self):
        """
        Load PCA model from file
        
        Returns:
        --------
        pca_model : dict or None
            Dictionary containing PCA model components, or None if loading fails
        """
        if os.path.exists(self.pca_model_path):
            try:
                return joblib.load(self.pca_model_path)
            except Exception as e:
                logger.error(f"Error loading PCA model: {str(e)}")
                return None
        else:
            logger.warning(f"PCA model file not found: {self.pca_model_path}")
            return None
    
    def _init_log_file(self):
        """Initialize log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            headers = ['timestamp', 'face_id', 'predicted_user', 'confidence', 'threshold_passed', 'actual_user']
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.log_file, index=False)
            logger.info(f"Created recognition log file: {self.log_file}")
    
    def is_model_trained(self):
        """
        Check if PCA model is trained
        
        Returns:
        --------
        is_trained : bool
            Whether the PCA model is trained
        """
        return self.pca_model is not None
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image
        
        Parameters:
        -----------
        face_image : numpy.ndarray
            Input face image
            
        Returns:
        --------
        processed_face : numpy.ndarray
            Preprocessed face image
        """
        return self.preprocessor.preprocess(face_image)
    
    def get_face_embedding(self, face_image):
        """
        Get face embedding using PCA
        
        Parameters:
        -----------
        face_image : numpy.ndarray
            Preprocessed face image
            
        Returns:
        --------
        embedding : numpy.ndarray or None
            Face embedding vector, or None if PCA model is not trained
        """
        if not self.is_model_trained():
            logger.warning("Cannot get face embedding: PCA model not trained")
            return None
        
        try:
            # Flatten image
            face_flat = face_image.flatten().reshape(1, -1)
            
            # Standardize face
            face_std = self.pca_model['scaler'].transform(face_flat)
            
            # Project into PCA space
            embedding = self.pca_model['pca'].transform(face_std)[0]
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting face embedding: {str(e)}")
            return None
    
    def recognize_face(self, face_image, actual_user=None, log_result=True):
        """
        Recognize face
        
        Parameters:
        -----------
        face_image : numpy.ndarray
            Face image to recognize
        actual_user : str, optional
            Actual user (for logging purposes)
        log_result : bool
            Whether to log the recognition result
            
        Returns:
        --------
        result : tuple
            Tuple containing (predicted_user, confidence, passed_threshold)
        """
        # Preprocess face
        processed_face = self.preprocess_face(face_image)
        
        # Get face embedding
        embedding = self.get_face_embedding(processed_face)
        
        if embedding is None:
            logger.warning("Cannot recognize face: Failed to get embedding")
            return ("Unknown", 0.0, False)
        
        # Get user embeddings
        user_embeddings = self.user_manager.get_all_embeddings()
        
        if not user_embeddings:
            logger.warning("Cannot recognize face: No user embeddings available")
            return ("Unknown", 0.0, False)
        
        # Find closest match
        best_match = None
        best_score = -1.0
        
        # Print debug information about available embeddings
        logger.info(f"Debug - Recognition attempt with {len(user_embeddings)} user embeddings")
        logger.info(f"Debug - Current confidence threshold: {self.confidence_threshold}")
        
        all_scores = {}
        for username, user_embedding in user_embeddings.items():
            # Calculate similarity score (cosine similarity)
            similarity = cosine_similarity([embedding], [user_embedding])[0][0]
            all_scores[username] = similarity
            
            if similarity > best_score:
                best_score = similarity
                best_match = username
        
        # Log all scores for debugging
        logger.info("Debug - All confidence scores:")
        for username, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  User: {username}, Score: {score:.4f}")
        
        # Check if score passes threshold
        passed_threshold = best_score >= self.confidence_threshold
        
        # Predicted user is best match if it passes threshold, otherwise "Unknown"
        predicted_user = best_match if passed_threshold else "Unknown"
        
        # Log recognition result
        if log_result:
            self.log_recognition(predicted_user, best_score, passed_threshold, actual_user)
        
        return (predicted_user, best_score, passed_threshold)
    
    def log_recognition(self, predicted_user, confidence, threshold_passed, actual_user=None):
        """
        Log recognition result
        
        Parameters:
        -----------
        predicted_user : str
            Predicted user
        confidence : float
            Confidence score
        threshold_passed : bool
            Whether confidence passes threshold
        actual_user : str, optional
            Actual user (for evaluation purposes)
        """
        try:
            # Create log entry
            timestamp = datetime.now().isoformat()
            face_id = f"face_{timestamp.replace(':', '').replace('-', '').replace('.', '')}"
            
            log_entry = {
                'timestamp': timestamp,
                'face_id': face_id,
                'predicted_user': predicted_user,
                'confidence': confidence,
                'threshold_passed': threshold_passed,
                'actual_user': actual_user or "unknown"
            }
            
            # Append to log file
            df = pd.DataFrame([log_entry])
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            
            # Log message
            log_message = f"Recognition: {predicted_user} (conf: {confidence:.2f}, passed: {threshold_passed})"
            if actual_user:
                correct = predicted_user == actual_user
                log_message += f", Actual: {actual_user}, Correct: {correct}"
            
            if threshold_passed:
                logger.info(log_message)
            else:
                logger.warning(log_message)
        
        except Exception as e:
            logger.error(f"Error logging recognition result: {str(e)}")
    
    def recognize_faces_in_image(self, image, log_results=True):
        """
        Recognize all faces in an image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image containing faces
        log_results : bool
            Whether to log recognition results
            
        Returns:
        --------
        results : list
            List of tuples containing (face_rect, predicted_user, confidence, passed_threshold)
        """
        # Detect faces
        face_rects = self.detector.detect_faces(image)
        
        if len(face_rects) == 0:
            return []
        
        results = []
        
        for face_rect in face_rects:
            # Extract face ROI
            face_roi = self.detector.get_face_roi(image, face_rect, margin=10)
            
            # Recognize face
            predicted_user, confidence, passed_threshold = self.recognize_face(
                face_roi, log_result=log_results)
            
            results.append((face_rect, predicted_user, confidence, passed_threshold))
        
        return results
    
    def analyze_recognition_logs(self, days=7):
        """
        Analyze recognition logs
        
        Parameters:
        -----------
        days : int
            Number of days to analyze
            
        Returns:
        --------
        stats : dict
            Dictionary containing recognition statistics
        """
        try:
            all_logs = []
            
            # Get log files
            log_files = []
            for i in range(days):
                date = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y%m%d')
                log_file = os.path.join(self.log_dir, f"recognition_log_{date}.csv")
                if os.path.exists(log_file):
                    log_files.append(log_file)
            
            # Read log files
            for log_file in log_files:
                try:
                    df = pd.read_csv(log_file)
                    all_logs.append(df)
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {str(e)}")
            
            if not all_logs:
                return {"error": "No log files found"}
            
            # Combine all logs
            logs_df = pd.concat(all_logs)
            
            # Calculate statistics
            total_recognitions = len(logs_df)
            unknown_recognitions = len(logs_df[logs_df['predicted_user'] == 'Unknown'])
            known_recognitions = total_recognitions - unknown_recognitions
            
            # User recognition counts
            user_counts = logs_df[logs_df['predicted_user'] != 'Unknown']['predicted_user'].value_counts().to_dict()
            
            # Recognition accuracy (when actual_user is known)
            accuracy_df = logs_df[logs_df['actual_user'] != 'unknown']
            if len(accuracy_df) > 0:
                correct_recognitions = accuracy_df[accuracy_df['predicted_user'] == accuracy_df['actual_user']]
                accuracy = len(correct_recognitions) / len(accuracy_df)
            else:
                accuracy = None
            
            # Average confidence per user
            user_confidence = logs_df[logs_df['predicted_user'] != 'Unknown'].groupby('predicted_user')['confidence'].mean().to_dict()
            
            # Recognition trend over time (by day)
            logs_df['date'] = pd.to_datetime(logs_df['timestamp']).dt.date
            trend = logs_df.groupby('date').size().to_dict()
            
            # Compile statistics
            stats = {
                'total_recognitions': total_recognitions,
                'known_recognitions': known_recognitions,
                'unknown_recognitions': unknown_recognitions,
                'recognition_rate': known_recognitions / total_recognitions if total_recognitions > 0 else 0,
                'user_counts': user_counts,
                'accuracy': accuracy,
                'user_confidence': user_confidence,
                'trend': trend
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error analyzing recognition logs: {str(e)}")
            return {"error": str(e)}

def draw_recognition_results(image, results):
    """
    Draw recognition results on image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    results : list
        List of tuples containing (face_rect, predicted_user, confidence, passed_threshold)
        
    Returns:
    --------
    annotated_image : numpy.ndarray
        Image with recognition results drawn
    """
    # Make a copy to avoid modifying the original
    annotated_image = image.copy()
    
    for face_rect, predicted_user, confidence, passed_threshold in results:
        x, y, w, h = face_rect
        
        # Set color based on threshold
        if passed_threshold:
            color = (0, 255, 0)  # Green for known faces
        else:
            color = (0, 165, 255)  # Orange for unknown faces
        
        # Draw rectangle
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
        
        # Create label
        if predicted_user != "Unknown":
            label = f"{predicted_user} ({confidence:.2f})"
        else:
            label = "Unknown"
        
        # Draw label background
        cv2.rectangle(annotated_image, (x, y-30), (x+w, y), color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label, (x+5, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated_image

if __name__ == "__main__":
    print("This module provides face recognition with thresholds and logging.")
    print("Import this module and use its classes and functions in your application.") 