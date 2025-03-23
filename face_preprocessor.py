import cv2
import numpy as np
from face_detector import get_facial_landmarks

class FacePreprocessor:
    """
    Preprocesses face images for recognition:
    1. Resizes to target dimensions
    2. Aligns face based on facial landmarks
    3. Normalizes pixel values
    4. Applies additional preprocessing as needed
    """
    
    def __init__(self, target_size=(160, 160), normalize=True, align_face=True):
        """
        Initialize the face preprocessor.
        
        Args:
            target_size: Tuple (width, height) for output size
            normalize: Whether to normalize pixel values
            align_face: Whether to align faces using landmarks
        """
        self.target_size = target_size
        self.normalize = normalize
        self.align_face = align_face
    
    def preprocess(self, face_img):
        """
        Preprocess a face image.
        
        Args:
            face_img: Input face image
            
        Returns:
            Preprocessed face image
        """
        if face_img is None or face_img.size == 0:
            return None
            
        try:
            # Make a copy to avoid modifying the original
            img = face_img.copy()
            
            # Align face if enabled and landmarks available
            if self.align_face:
                img = self._align_face(img)
            
            # Resize image to target size
            img = cv2.resize(img, self.target_size)
            
            # Convert to grayscale if color
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            # Normalize pixel values if enabled
            if self.normalize:
                img = img.astype(np.float32) / 255.0
                
            return img
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None
    
    def _align_face(self, face_img):
        """
        Align face based on eye positions from facial landmarks.
        
        Args:
            face_img: Input face image
            
        Returns:
            Aligned face image or original if alignment fails
        """
        try:
            # Get facial landmarks
            landmarks = get_facial_landmarks(face_img)
            
            # If no landmarks found, return original image
            if not landmarks:
                return face_img
                
            # Get coordinates for eyes
            left_eye = landmarks[0]['left_eye']
            right_eye = landmarks[0]['right_eye']
            
            # Calculate angle for alignment
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate center point between eyes
            eye_center = ((left_eye[0] + right_eye[0]) // 2,
                         (left_eye[1] + right_eye[1]) // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
            
            # Determine output size
            h, w = face_img.shape[:2]
            
            # Apply affine transformation
            aligned_face = cv2.warpAffine(face_img, M, (w, h), 
                                         flags=cv2.INTER_CUBIC)
            
            return aligned_face
            
        except Exception as e:
            print(f"Error in face alignment: {str(e)}")
            return face_img

    def enhance_image(self, image):
        """Enhance image quality"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return enhanced

    def normalize_illumination(self, image):
        """Normalize illumination using local normalization"""
        # Convert to float32
        float_img = image.astype(np.float32)
        
        # Calculate local mean and standard deviation
        local_mean = cv2.GaussianBlur(float_img, (15, 15), 0)
        local_std = np.sqrt(cv2.GaussianBlur(float_img**2, (15, 15), 0) - local_mean**2)
        
        # Normalize locally
        normalized = (float_img - local_mean) / (local_std + 1e-6)
        
        # Scale to [0, 1] range
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        return normalized 