"""
Face Preprocessor

Implements essential face preprocessing operations:
1. Grayscale conversion
2. CLAHE normalization for illumination correction
3. Contrast enhancement (multiple methods available)
4. Face alignment to standard orientation
5. Resizing to standard dimensions
6. Rotation (fixed or random)
7. Histogram equalization
"""

import cv2
import numpy as np
import logging
import random
import math
from config import (
    FACE_SIZE, USE_CLAHE, CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE,
    USE_ROTATION, FIXED_ROTATION, ROTATION_ANGLE, ROTATION_RANGE,
    DATA_AUGMENTATION, AUGMENTATION_ROTATIONS,
    USE_HIST_EQUALIZATION, USE_CONTRAST_STRETCHING, 
    CONTRAST_STRETCH_PERCENTILES, USE_GAMMA_CORRECTION, GAMMA_VALUE,
    USE_FACE_ALIGNMENT, LEFT_EYE_POSITION, RIGHT_EYE_POSITION
)
from face_detector import FaceDetector

# Setup logging
logger = logging.getLogger(__name__)

class FacePreprocessor:
    """
    Face preprocessor with essential operations for face recognition
    """
    
    def __init__(self, target_size=FACE_SIZE, 
                 use_clahe=USE_CLAHE, 
                 clahe_clip_limit=CLAHE_CLIP_LIMIT,
                 clahe_grid_size=CLAHE_GRID_SIZE,
                 use_rotation=USE_ROTATION,
                 fixed_rotation=FIXED_ROTATION,
                 rotation_angle=ROTATION_ANGLE,
                 rotation_range=ROTATION_RANGE,
                 use_hist_equalization=USE_HIST_EQUALIZATION,
                 use_contrast_stretching=USE_CONTRAST_STRETCHING,
                 contrast_stretch_percentiles=CONTRAST_STRETCH_PERCENTILES,
                 use_gamma_correction=USE_GAMMA_CORRECTION,
                 gamma_value=GAMMA_VALUE,
                 use_face_alignment=USE_FACE_ALIGNMENT,
                 left_eye_position=LEFT_EYE_POSITION,
                 right_eye_position=RIGHT_EYE_POSITION):
        """
        Initialize the face preprocessor
        
        Parameters:
        -----------
        target_size : tuple
            Target size for output face images (width, height)
        use_clahe : bool
            Whether to use CLAHE normalization
        clahe_clip_limit : float
            Clip limit for CLAHE normalization
        clahe_grid_size : tuple
            Grid size for CLAHE normalization
        use_rotation : bool
            Whether to apply rotation to the face images
        fixed_rotation : bool
            Whether to use a fixed rotation angle (True) or random rotation (False)
        rotation_angle : float
            Fixed rotation angle in degrees (only used if fixed_rotation is True)
        rotation_range : tuple
            Range for random rotation angle (min_angle, max_angle) in degrees
            (only used if fixed_rotation is False)
        use_hist_equalization : bool
            Whether to use histogram equalization
        use_contrast_stretching : bool
            Whether to use contrast stretching
        contrast_stretch_percentiles : tuple
            Percentiles (low, high) for contrast stretching
        use_gamma_correction : bool
            Whether to use gamma correction
        gamma_value : float
            Gamma value for gamma correction (< 1: brighter, > 1: darker)
        use_face_alignment : bool
            Whether to align faces based on eye positions
        left_eye_position : tuple
            Desired position for the left eye (x, y as percentage of image dimensions)
        right_eye_position : tuple
            Desired position for the right eye (x, y as percentage of image dimensions)
        """
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.use_rotation = use_rotation
        self.fixed_rotation = fixed_rotation
        self.rotation_angle = rotation_angle
        self.rotation_range = rotation_range
        self.use_hist_equalization = use_hist_equalization
        self.use_contrast_stretching = use_contrast_stretching
        self.contrast_stretch_percentiles = contrast_stretch_percentiles
        self.use_gamma_correction = use_gamma_correction
        self.gamma_value = gamma_value
        self.use_face_alignment = use_face_alignment
        self.left_eye_position = left_eye_position
        self.right_eye_position = right_eye_position
        
        # Initialize CLAHE if needed
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_grid_size
            )
        else:
            self.clahe = None
            
        # Initialize face detector for eye detection (used in alignment)
        if self.use_face_alignment:
            self.face_detector = FaceDetector()
            
        logger.info(f"Face preprocessor initialized with target size: {target_size}")
        if self.use_rotation:
            if self.fixed_rotation:
                logger.info(f"Fixed rotation enabled with angle: {self.rotation_angle} degrees")
            else:
                logger.info(f"Random rotation enabled with range: {self.rotation_range} degrees")
        
        # Log contrast enhancement settings
        enhancements = []
        if self.use_clahe:
            enhancements.append("CLAHE")
        if self.use_hist_equalization:
            enhancements.append("Histogram Equalization")
        if self.use_contrast_stretching:
            enhancements.append(f"Contrast Stretching ({self.contrast_stretch_percentiles})")
        if self.use_gamma_correction:
            enhancements.append(f"Gamma Correction (γ={self.gamma_value})")
        
        if enhancements:
            logger.info(f"Contrast enhancements: {', '.join(enhancements)}")
            
        if self.use_face_alignment:
            logger.info(f"Face alignment enabled with eye positions: L={self.left_eye_position}, R={self.right_eye_position}")
    
    def _rotate_image(self, image, angle=None):
        """
        Rotate an image by the given angle
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        angle : float, optional
            Rotation angle in degrees. If None, a random angle will be used
            based on the rotation_range.
            
        Returns:
        --------
        numpy.ndarray
            Rotated image
        """
        if angle is None:
            if self.fixed_rotation:
                angle = self.rotation_angle
            else:
                angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        
        # Calculate the center of the image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate the new size to avoid cropping after rotation
        # This handles cases of rotation causing the corners to be outside the image bounds
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust the rotation matrix to account for the new size
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply the rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=0)
        
        return rotated
    
    def _align_face(self, face_img):
        """
        Align a face image based on eye positions
        
        Parameters:
        -----------
        face_img : numpy.ndarray
            Input face image
            
        Returns:
        --------
        numpy.ndarray
            Aligned face image
        """
        if face_img is None or face_img.size == 0:
            logger.error("Invalid face image for alignment")
            return None
        
        # Get desired eye positions based on the target image size
        desired_left_eye_x = int(self.left_eye_position[0] * self.target_size[0])
        desired_left_eye_y = int(self.left_eye_position[1] * self.target_size[1])
        desired_right_eye_x = int(self.right_eye_position[0] * self.target_size[0])
        desired_right_eye_y = int(self.right_eye_position[1] * self.target_size[1])
        
        # Detect eyes in the face image
        left_eye, right_eye = self.face_detector.detect_eyes(face_img)
        
        # If we couldn't detect eyes, return the original image
        if left_eye is None or right_eye is None:
            logger.warning("Could not detect eyes for face alignment")
            return face_img
        
        # Calculate the angle between the eyes
        if right_eye[0] - left_eye[0] == 0:  # Avoid division by zero
            angle = 0
        else:
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate the scale factor
        # Distance between eyes in the original image
        eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        # Distance between eyes in the desired image
        desired_eye_distance = np.sqrt((desired_right_eye_x - desired_left_eye_x)**2 + 
                                        (desired_right_eye_y - desired_left_eye_y)**2)
        scale = desired_eye_distance / eye_distance
        
        # Get the rotation matrix
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Update the translation component of the matrix to place the eyes
        # at the desired positions after scaling and rotation
        tx = (desired_left_eye_x + desired_right_eye_x) / 2 - center[0]
        ty = (desired_left_eye_y + desired_right_eye_y) / 2 - center[1]
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty
        
        # Apply the affine transformation
        aligned_face = cv2.warpAffine(face_img, rotation_matrix, self.target_size, 
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
        
        return aligned_face
        
    def _apply_contrast_stretching(self, image):
        """
        Apply contrast stretching to an image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
            
        Returns:
        --------
        numpy.ndarray
            Image with enhanced contrast
        """
        # Get percentiles
        low_percentile, high_percentile = self.contrast_stretch_percentiles
        
        # Calculate percentile values from the image
        low_val = np.percentile(image, low_percentile)
        high_val = np.percentile(image, high_percentile)
        
        # Apply contrast stretching
        if high_val > low_val:
            # Clip values to avoid division by zero
            image = np.clip(image, low_val, high_val)
            # Normalize to [0, 255]
            image = 255 * (image - low_val) / (high_val - low_val)
        
        return image.astype(np.uint8)
        
    def _apply_gamma_correction(self, image):
        """
        Apply gamma correction to an image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
            
        Returns:
        --------
        numpy.ndarray
            Image with gamma correction
        """
        # Normalize to [0, 1]
        normalized = image / 255.0
        
        # Apply gamma correction
        gamma_corrected = np.power(normalized, 1.0 / self.gamma_value)
        
        # Scale back to [0, 255]
        return (gamma_corrected * 255).astype(np.uint8)
    
    def preprocess(self, image, angle=None):
        """
        Preprocess a face image for recognition
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input face image
        angle : float, optional
            Optional specific rotation angle in degrees
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed face image
        """
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image for preprocessing")
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply CLAHE if enabled (early contrast enhancement)
        if self.use_clahe and self.clahe is not None:
            gray = self.clahe.apply(gray)
            
        # Apply face alignment if enabled
        if self.use_face_alignment:
            aligned = self._align_face(gray)
            if aligned is not None:
                gray = aligned
            else:
                # If alignment fails, resize the image
                gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_CUBIC)
        else:
            # Apply rotation if enabled (when not using alignment)
            if self.use_rotation or angle is not None:
                gray = self._rotate_image(gray, angle)
                
            # Resize to target size
            gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        # Apply contrast enhancement methods
        enhanced = gray.copy()
        
        # Apply contrast stretching if enabled (before histogram equalization)
        if self.use_contrast_stretching:
            enhanced = self._apply_contrast_stretching(enhanced)
            
        # Apply histogram equalization if enabled
        if self.use_hist_equalization:
            enhanced = cv2.equalizeHist(enhanced)
            
        # Apply gamma correction if enabled (after other enhancements)
        if self.use_gamma_correction:
            enhanced = self._apply_gamma_correction(enhanced)
        
        return enhanced
        
    def preprocess_batch(self, images, angles=None):
        """
        Preprocess a batch of face images
        
        Parameters:
        -----------
        images : list
            List of input face images
        angles : list, optional
            List of rotation angles in degrees, one per image
            
        Returns:
        --------
        list
            List of preprocessed face images
        """
        if angles is not None and len(images) != len(angles):
            logger.warning("Number of images and angles don't match. Ignoring angles.")
            angles = None
            
        if angles is None:
            return [self.preprocess(img) for img in images if img is not None]
        else:
            return [self.preprocess(img, angle) for img, angle in zip(images, angles) if img is not None]
    
    def normalize_for_training(self, images, augment=DATA_AUGMENTATION):
        """
        Normalize a batch of face images for training
        
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of face images or array of face images
        augment : bool, optional
            Whether to apply data augmentation (generating additional rotated versions)
            
        Returns:
        --------
        numpy.ndarray
            Array of flattened, normalized face images
        list
            List of labels (same length as the normalized array, accounting for augmentation)
        """
        # Preprocess images
        preprocessed = []
        augmented_labels = []
        
        if not augment:
            preprocessed = self.preprocess_batch(images)
        else:
            # Apply data augmentation with multiple rotation angles
            for i, img in enumerate(images):
                if img is None:
                    continue
                
                # Add original and rotated versions
                for angle in AUGMENTATION_ROTATIONS:
                    proc_img = self.preprocess(img, angle=angle)
                    if proc_img is not None:
                        preprocessed.append(proc_img)
                        augmented_labels.append(i)  # Keep track of which original image this came from
        
        # Convert to numpy array and ensure correct shape
        if len(preprocessed) == 0:
            logger.error("No valid images for normalization")
            return None, None
            
        # Flatten the images
        height, width = self.target_size[1], self.target_size[0]
        flattened = [img.reshape(height * width) for img in preprocessed]
        
        # Convert to float and normalize to [0, 1]
        normalized = np.array(flattened, dtype=np.float32) / 255.0
        
        if augment:
            return normalized, augmented_labels
        else:
            return normalized
            
    def visualize_preprocessing_steps(self, image):
        """
        Visualize the preprocessing steps on a single image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
            
        Returns:
        --------
        dict
            Dictionary of images at different preprocessing steps
        """
        result = {}
        
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image for preprocessing")
            return result
            
        # Original image
        result['original'] = image.copy()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        result['grayscale'] = gray.copy()
        
        # Apply CLAHE if enabled
        if self.use_clahe and self.clahe is not None:
            gray = self.clahe.apply(gray)
            result['clahe'] = gray.copy()
        
        # Apply face alignment if enabled
        if self.use_face_alignment:
            aligned = self._align_face(gray)
            if aligned is not None:
                result['aligned'] = aligned.copy()
                gray = aligned
            else:
                # If alignment fails, resize the image
                gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_CUBIC)
                result['resized'] = gray.copy()
        else:
            # Apply rotation if enabled
            if self.use_rotation:
                gray = self._rotate_image(gray)
                result['rotated'] = gray.copy()
                
            # Resize to target size
            resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_CUBIC)
            result['resized'] = resized.copy()
            gray = resized
        
        # Apply contrast enhancement methods
        enhanced = gray.copy()
        
        # Apply contrast stretching if enabled
        if self.use_contrast_stretching:
            enhanced = self._apply_contrast_stretching(enhanced)
            result['contrast_stretched'] = enhanced.copy()
            
        # Apply histogram equalization if enabled
        if self.use_hist_equalization:
            enhanced = cv2.equalizeHist(enhanced)
            result['hist_equalized'] = enhanced.copy()
            
        # Apply gamma correction if enabled
        if self.use_gamma_correction:
            enhanced = self._apply_gamma_correction(enhanced)
            result['gamma_corrected'] = enhanced.copy()
        
        result['final'] = enhanced
        
        return result 