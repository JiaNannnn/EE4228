import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Load the pre-trained Viola-Jones face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Default parameters - can be adjusted via methods
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_face_size = (30, 30)
        self.max_face_size = None  # No maximum by default

    def set_parameters(self, scale_factor=None, min_neighbors=None, min_face_size=None, max_face_size=None):
        """Set face detection parameters"""
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if min_neighbors is not None:
            self.min_neighbors = min_neighbors
        if min_face_size is not None:
            self.min_face_size = min_face_size
        if max_face_size is not None:
            self.max_face_size = max_face_size
        
        return self

    def detect_faces(self, image):
        """
        Detect faces in the input image using optimized parameters
        Returns: List of (x, y, w, h) face rectangles
        """
        if image is None or image.size == 0:
            print("Warning: Empty image in face detection")
            return []
            
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Normalize image brightness and contrast for better detection
        equalized = cv2.equalizeHist(gray)
        
        # Try initially with standard parameters
        faces = self.face_cascade.detectMultiScale(
            equalized,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
            maxSize=self.max_face_size
        )
        
        # If no faces detected, try with more lenient parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                equalized,
                scaleFactor=self.scale_factor + 0.05,  # Make slightly higher
                minNeighbors=max(2, self.min_neighbors - 2),  # Make slightly lower
                minSize=tuple(int(dim * 0.8) for dim in self.min_face_size),  # Smaller min size
                maxSize=self.max_face_size
            )
            
        # If still no faces, try with even more lenient parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,  # Use original grayscale image
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        # Apply face validation to remove false positives
        valid_faces = []
        for (x, y, w, h) in faces:
            # Check face aspect ratio - faces should be roughly square
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.4:  # Reasonable face aspect ratios
                valid_faces.append((x, y, w, h))
        
        # Return the original detection if validation filtered all faces
        return valid_faces if valid_faces else faces

    def get_face_roi(self, image, face_rect):
        """
        Extract face ROI from image using detected rectangle with better cropping
        """
        x, y, w, h = face_rect
        
        # Apply a dynamic margin based on face size
        margin_x = int(w * 0.15)  # 15% horizontal margin
        margin_y = int(h * 0.15)  # 15% vertical margin
        
        # Get image dimensions
        img_h, img_w = image.shape[:2]
        
        # Calculate new coordinates with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)
        
        # Extract ROI with margin
        face_roi = image[y1:y2, x1:x2]
        
        # If the ROI is empty, return the original cropping
        if face_roi.size == 0:
            face_roi = image[y:y+h, x:x+w]
        
        return face_roi
        
    def detect_and_process_face(self, image):
        """
        Detect face and process it in one step, returning the best face ROI
        """
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return None
            
        # Find the largest face (most likely to be the main face)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        
        # Extract face ROI
        face_roi = self.get_face_roi(image, largest_face)
        
        return face_roi 