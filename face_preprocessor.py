import cv2
import numpy as np

class FacePreprocessor:
    def __init__(self, target_size=(100, 100)):
        self.target_size = target_size
        # Load facial landmark detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def align_face(self, image):
        """Align face based on eye positions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate to get left and right eye
            eyes = sorted(eyes, key=lambda x: x[0])
            left_eye, right_eye = eyes[:2]
            
            # Calculate eye centers
            left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
            right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
            
            # Calculate angle to align eyes horizontally
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotate image
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            return aligned_image
        
        return image

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

    def preprocess(self, image):
        """
        Preprocess face image with enhanced normalization steps
        """
        try:
            # Input validation
            if image is None or image.size == 0:
                raise ValueError("Empty or invalid image input")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Align face using eye detection
            aligned = self.align_face(gray)
            
            # Resize to target size
            resized = cv2.resize(aligned, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Enhance image quality
            enhanced = self.enhance_image(resized)
            
            # Normalize illumination
            normalized = self.normalize_illumination(enhanced)
            
            # Ensure output is properly shaped
            if normalized.shape != self.target_size[::-1]:
                raise ValueError(f"Output shape {normalized.shape} does not match target size {self.target_size[::-1]}")

            return normalized

        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}") 