"""
Advanced Face Preprocessor

Implements a comprehensive face preprocessing pipeline for face recognition:
1. CLAHE normalization (8×8 grid, clipLimit=3.0)
2. Geometric normalization:
   - Eye alignment to 25% vertical position
   - Inter-pupillary distance standardization (45px ±2%)
3. Histogram stretching to [0.1, 0.9] quantile ranges
"""

import cv2
import numpy as np
import logging
from collections import OrderedDict
import time
from pathlib import Path
from config import PREPROCESSING_CONFIG

logger = logging.getLogger("advanced_preprocessor")

# Try to import dlib, but provide fallback if not available
DLIB_AVAILABLE = False
try:
    import dlib
    DLIB_AVAILABLE = True
    logger.info("dlib module is available and loaded successfully")
except ImportError:
    logger.warning("dlib module is not available. Using OpenCV fallback for face alignment.")

class AdvancedPreprocessor:
    """
    Advanced face preprocessor for face recognition with
    illumination normalization and geometric alignment.
    """
    
    def __init__(self,
                target_size=PREPROCESSING_CONFIG["target_size"],
                clahe_grid_size=PREPROCESSING_CONFIG["clahe_grid_size"],
                clahe_clip_limit=PREPROCESSING_CONFIG["clahe_clip_limit"],
                eye_position_y=PREPROCESSING_CONFIG["eye_position_y"],
                inter_pupillary_distance=PREPROCESSING_CONFIG["inter_pupillary_distance"],
                histogram_range=PREPROCESSING_CONFIG["histogram_range"]):
        """
        Initialize the advanced preprocessor
        
        Parameters:
        -----------
        target_size : tuple
            Target size for face images (width, height)
        clahe_grid_size : tuple
            Grid size for CLAHE normalization
        clahe_clip_limit : float
            Clip limit for CLAHE normalization
        eye_position_y : float
            Target vertical position of eyes (as fraction of image height)
        inter_pupillary_distance : int
            Target distance between pupils in pixels
        histogram_range : tuple
            Target quantile range for histogram stretching (min, max)
        """
        self.target_size = target_size
        self.clahe_grid_size = clahe_grid_size
        self.clahe_clip_limit = clahe_clip_limit
        self.eye_position_y = eye_position_y
        self.inter_pupillary_distance = inter_pupillary_distance
        self.histogram_range = histogram_range
        
        # Initialize facial landmark predictor if dlib is available
        self.landmark_predictor = None
        self.face_detector = None
        
        if DLIB_AVAILABLE:
            try:
                model_path = str(Path(__file__).parent / "shape_predictor_68_face_landmarks.dat")
                self.landmark_predictor = dlib.shape_predictor(model_path)
                self.face_detector = dlib.get_frontal_face_detector()
                logger.info("Using dlib for face landmark detection")
            except Exception as e:
                logger.error(f"Error loading facial landmark predictor: {e}")
                DLIB_AVAILABLE = False
        
        # Initialize OpenCV-based eye detector as fallback
        if not DLIB_AVAILABLE:
            logger.info("Initializing OpenCV-based face detection as fallback")
            # Initialize OpenCV's Haar cascade for eye detection
            try:
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.eye_cascade.empty() or self.face_cascade.empty():
                    logger.error("Failed to load OpenCV Haar cascades")
                else:
                    logger.info("OpenCV Haar cascades loaded successfully as fallback")
            except Exception as e:
                logger.error(f"Error loading OpenCV cascades: {e}")
                
        # Initialize CLAHE for illumination normalization
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        
        # Cache for processed faces to avoid redundant processing
        self.cache = OrderedDict()
        self.cache_size = 100
        
        # Facial landmark indices for eye coordinates
        # From the 68-point model: left eye corners (36, 39), right eye corners (42, 45)
        self.left_eye_indices = list(range(36, 42))
        self.right_eye_indices = list(range(42, 48))
        
    def convert_to_grayscale(self, image):
        """
        Convert image to grayscale if it's not already
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
        
    def apply_clahe(self, image):
        """
        Apply CLAHE normalization to enhance local contrast
        """
        gray = self.convert_to_grayscale(image)
        return self.clahe.apply(gray)
        
    def detect_landmarks(self, image):
        """
        Detect facial landmarks using dlib or OpenCV fallback
        
        Returns:
        --------
        numpy.ndarray
            Array of (x, y) coordinates for facial landmarks
            For dlib: 68 facial landmarks
            For OpenCV fallback: 4 landmarks (left eye, right eye, center points)
        """
        # If dlib is available, use it
        if DLIB_AVAILABLE and self.landmark_predictor is not None:
            # Ensure image is grayscale
            gray = self.convert_to_grayscale(image)
            
            # Convert to dlib format
            dlib_rect = None
            
            # Try to detect face with dlib
            dlib_faces = self.face_detector(gray, 1)
            if len(dlib_faces) > 0:
                dlib_rect = dlib_faces[0]
            else:
                # Fallback to creating a rect from the full image
                height, width = gray.shape
                dlib_rect = dlib.rectangle(0, 0, width, height)
                
            # Detect landmarks
            shape = self.landmark_predictor(gray, dlib_rect)
            
            # Convert to numpy array
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            return landmarks
        
        # OpenCV fallback for landmark detection
        else:
            gray = self.convert_to_grayscale(image)
            
            # Try to detect face with OpenCV
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # If no face found, use the entire image
            if len(faces) == 0:
                roi_gray = gray
            else:
                # Use the first face
                x, y, w, h = faces[0]
                roi_gray = gray[y:y+h, x:x+w]
                
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # Need at least two eyes
            if len(eyes) < 2:
                logger.warning("Failed to detect eyes with OpenCV")
                return None
            
            # Sort eyes by x-coordinate to identify left and right
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Create simulated landmarks (only 4 points for the eyes)
            landmarks = np.zeros((4, 2), dtype=np.int32)
            
            # Adjust coordinates if using a face ROI
            x_offset = 0
            y_offset = 0
            if len(faces) > 0:
                x_offset = faces[0][0]
                y_offset = faces[0][1]
            
            # Left eye
            left_eye_x = x_offset + eyes[0][0] + eyes[0][2] // 2
            left_eye_y = y_offset + eyes[0][1] + eyes[0][3] // 2
            landmarks[0] = [left_eye_x, left_eye_y]
            
            # Right eye
            right_eye_x = x_offset + eyes[1][0] + eyes[1][2] // 2
            right_eye_y = y_offset + eyes[1][1] + eyes[1][3] // 2
            landmarks[1] = [right_eye_x, right_eye_y]
            
            # Add two more points for compatibility
            landmarks[2] = [left_eye_x - eyes[0][2]//4, left_eye_y]
            landmarks[3] = [right_eye_x + eyes[1][2]//4, right_eye_y]
            
            return landmarks
            
    def get_eye_centers(self, landmarks):
        """
        Calculate eye centers from facial landmarks
        
        Returns:
        --------
        tuple
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y))
        """
        if landmarks is None:
            logger.warning("Landmarks not available for eye center calculation")
            return None
        
        # For dlib landmarks
        if DLIB_AVAILABLE and len(landmarks) >= 68:
            # Calculate eye centers
            left_eye = landmarks[self.left_eye_indices].mean(axis=0).astype("int")
            right_eye = landmarks[self.right_eye_indices].mean(axis=0).astype("int")
            
            return (tuple(left_eye), tuple(right_eye))
        
        # For OpenCV fallback landmarks
        elif len(landmarks) >= 2:
            # We already have eye centers in the first two landmarks
            left_eye = tuple(landmarks[0])
            right_eye = tuple(landmarks[1])
            
            return (left_eye, right_eye)
        
        logger.warning("Not enough landmarks for eye center calculation")
        return None
        
    def align_face(self, image, landmarks=None):
        """
        Align face based on eye centers
        """
        # If landmarks not provided, detect them
        if landmarks is None:
            landmarks = self.detect_landmarks(image)
            
        if landmarks is None:
            logger.warning("Failed to detect landmarks for face alignment")
            return cv2.resize(image, self.target_size)
            
        # Get eye centers
        eye_centers = self.get_eye_centers(landmarks)
        if eye_centers is None:
            return cv2.resize(image, self.target_size)
            
        left_eye_center, right_eye_center = eye_centers
        
        # Calculate angle between eyes
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Compute scale factor to standardize inter-pupillary distance
        current_eye_distance = np.sqrt((dx ** 2) + (dy ** 2))
        scale = self.inter_pupillary_distance / current_eye_distance
        
        # Calculate target eye positions
        target_width, target_height = self.target_size
        
        target_left_eye_x = (1.0 - self.eye_position_y) * target_width // 2
        target_right_eye_x = target_width - target_left_eye_x
        target_eye_y = int(self.eye_position_y * target_height)
        
        # Get the rotation matrix
        center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Update translation component of the matrix
        rotation_matrix[0, 2] += (target_width / 2) - center[0]
        rotation_matrix[1, 2] += target_eye_y - center[1]
        
        # Apply the affine transformation
        aligned_face = cv2.warpAffine(
            image, 
            rotation_matrix, 
            self.target_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return aligned_face
        
    def stretch_histogram(self, image, min_quantile=None, max_quantile=None):
        """
        Stretch histogram to specified quantile range
        """
        if min_quantile is None:
            min_quantile = self.histogram_range[0]
        if max_quantile is None:
            max_quantile = self.histogram_range[1]
            
        gray = self.convert_to_grayscale(image)
        
        # Calculate quantiles
        min_val = np.quantile(gray, min_quantile)
        max_val = np.quantile(gray, max_quantile)
        
        # Avoid division by zero
        if max_val == min_val:
            logger.warning("Cannot stretch histogram: min_val == max_val")
            return gray
            
        # Stretch histogram
        stretched = np.clip((gray - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
        
        return stretched
    
    def preprocess(self, image, cache_key=None):
        """
        Apply the full preprocessing pipeline to a face image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input face image
        cache_key : any hashable, optional
            Key for caching the result (e.g., image path or ID)
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed face image ready for recognition
        dict
            Preprocessing metadata
        """
        # Check cache first if cache_key provided
        if cache_key is not None and cache_key in self.cache:
            logger.debug(f"Using cached preprocessing result for {cache_key}")
            return self.cache[cache_key]["image"], self.cache[cache_key]["metadata"]
            
        if image is None:
            logger.error("Cannot preprocess None image")
            return None, {}
            
        start_time = time.time()
        metadata = {}
        
        # Step 1: Convert to grayscale
        gray = self.convert_to_grayscale(image)
        metadata["gray_shape"] = gray.shape
        
        # Step 2: Apply CLAHE normalization
        clahe_image = self.apply_clahe(gray)
        metadata["clahe_time"] = time.time() - start_time
        
        # Step 3: Detect landmarks for alignment
        landmarks = self.detect_landmarks(clahe_image)
        metadata["landmarks_found"] = landmarks is not None
        metadata["landmarks_time"] = time.time() - start_time - metadata["clahe_time"]
        
        # Step 4: Align face based on eye positions
        if landmarks is not None:
            aligned_face = self.align_face(clahe_image, landmarks)
            metadata["alignment_success"] = True
        else:
            # Fallback to simple resize if alignment fails
            aligned_face = cv2.resize(clahe_image, self.target_size)
            metadata["alignment_success"] = False
        
        metadata["alignment_time"] = time.time() - start_time - metadata["clahe_time"] - metadata["landmarks_time"]
        
        # Step 5: Stretch histogram to [0.1, 0.9] quantile range
        stretched_face = self.stretch_histogram(aligned_face)
        metadata["stretching_time"] = time.time() - start_time - metadata["clahe_time"] - metadata["landmarks_time"] - metadata["alignment_time"]
        
        # Step 6: Final resize to ensure exact target size
        final_face = cv2.resize(stretched_face, self.target_size)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        metadata["total_time"] = total_time
        
        # Cache the result if cache_key provided
        if cache_key is not None:
            # Maintain cache size limit with LRU eviction
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # Remove oldest item
                
            self.cache[cache_key] = {
                "image": final_face,
                "metadata": metadata
            }
            
        # Log processing time if it's slow
        if total_time > 0.1:  # 100ms threshold
            logger.debug(f"Face preprocessing took {total_time*1000:.1f}ms")
            
        return final_face, metadata 