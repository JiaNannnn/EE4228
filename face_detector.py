"""
Face Detector

Simple face detection module using OpenCV's Haar cascades.
"""

import cv2
import numpy as np
import logging
from config import FACE_CASCADE_PATH, MIN_FACE_SIZE, SCALE_FACTOR, MIN_NEIGHBORS, FACE_LANDMARK_METHOD

# Setup logging
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detector using OpenCV's Haar cascades
    """
    
    def __init__(self, cascade_path=FACE_CASCADE_PATH, 
                 min_face_size=MIN_FACE_SIZE,
                 scale_factor=SCALE_FACTOR, 
                 min_neighbors=MIN_NEIGHBORS,
                 landmark_method=FACE_LANDMARK_METHOD):
        """
        Initialize the face detector
        
        Parameters:
        -----------
        cascade_path : str
            Path to the Haar cascade XML file
        min_face_size : tuple
            Minimum face size to detect (width, height)
        scale_factor : float
            Scale factor for the detection algorithm
        min_neighbors : int
            Minimum number of neighbors for the detection algorithm
        landmark_method : str
            Method to use for facial landmark detection ('haarcascade' or 'lbp')
        """
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.landmark_method = landmark_method
        
        # Load the face cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            error_msg = f"Error: Could not load face cascade from {cascade_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Load the eye cascades for face alignment
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        if self.eye_cascade.empty():
            logger.warning("Could not load eye cascade for face alignment")
        
        # Additional left and right eye cascades as fallback
        self.left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
        self.right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        
        logger.info(f"Face detector initialized with cascade: {cascade_path}")
        
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
            
        Returns:
        --------
        list
            List of face rectangle coordinates (x, y, width, height)
        """
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image for face detection")
            return []
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        logger.debug(f"Detected {len(faces)} faces")
        
        return list(faces)
    
    def detect_eyes(self, face_img):
        """
        Detect eyes in a face image
        
        Parameters:
        -----------
        face_img : numpy.ndarray
            Face image (grayscale)
            
        Returns:
        --------
        tuple
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or (None, None) if detection fails
        """
        if face_img is None or face_img.size == 0:
            logger.error("Invalid face image for eye detection")
            return None, None
        
        # Ensure image is grayscale
        if len(face_img.shape) == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img.copy()
        
        # Try to detect both eyes using the eye cascade
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If we found exactly two eyes, determine which is left and which is right
        if len(eyes) == 2:
            # Sort eyes by x-coordinate (left-to-right)
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            right_eye = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
            return left_eye, right_eye
        
        # If we didn't find exactly two eyes, try alternative approaches
        
        # Try the specific left and right eye cascades
        left_eye = self.left_eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        right_eye = self.right_eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Check if we found one of each
        left_center = None
        right_center = None
        
        if len(left_eye) > 0:
            # Use the largest left eye detection
            left_eye = sorted(left_eye, key=lambda e: e[2] * e[3], reverse=True)[0]
            left_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
        
        if len(right_eye) > 0:
            # Use the largest right eye detection
            right_eye = sorted(right_eye, key=lambda e: e[2] * e[3], reverse=True)[0]
            right_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
        
        # If we found both eyes, return them
        if left_center is not None and right_center is not None:
            # Ensure left_center is actually to the left of right_center
            if left_center[0] < right_center[0]:
                return left_center, right_center
            else:
                return right_center, left_center
        
        # If we found only one eye with the specific detectors, try to estimate the other
        # based on face geometry (this is approximate)
        if left_center is not None and right_center is None:
            # Estimate right eye position
            face_width = face_gray.shape[1]
            eye_distance = face_width * 0.3  # Approximate distance between eyes
            right_center = (left_center[0] + int(eye_distance), left_center[1])
            return left_center, right_center
            
        if right_center is not None and left_center is None:
            # Estimate left eye position
            face_width = face_gray.shape[1]
            eye_distance = face_width * 0.3  # Approximate distance between eyes
            left_center = (right_center[0] - int(eye_distance), right_center[1])
            return left_center, right_center
        
        # If all methods fail, return None
        logger.warning("Could not detect eyes in face image")
        return None, None
        
    def get_face_roi(self, image, face_rect, add_margin=True):
        """
        Get the region of interest (ROI) for a face
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        face_rect : tuple
            Face rectangle coordinates (x, y, width, height)
        add_margin : bool
            Whether to add a margin around the face
            
        Returns:
        --------
        numpy.ndarray
            Face image
        """
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image for getting face ROI")
            return None
            
        # Get face coordinates
        x, y, w, h = face_rect
        
        # Add margin if requested
        if add_margin:
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            # Calculate new coordinates with margin
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(image.shape[1] - x, w + 2 * margin_x)
            h = min(image.shape[0] - y, h + 2 * margin_y)
            
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        
        return face_roi
        
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw face rectangles on an image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        faces : list
            List of face rectangle coordinates (x, y, width, height)
        color : tuple
            Color for the face rectangles (B, G, R)
        thickness : int
            Thickness of the rectangles
            
        Returns:
        --------
        numpy.ndarray
            Image with face rectangles
        """
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image for drawing faces")
            return image
            
        # Make a copy of the image
        result = image.copy()
        
        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
        return result 