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

    def detect_faces(self, image):
        """
        Detect faces in the input image
        Returns: List of (x, y, w, h) face rectangles
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces

    def get_face_roi(self, image, face_rect):
        """
        Extract face ROI from image using detected rectangle
        """
        x, y, w, h = face_rect
        return image[y:y+h, x:x+w] 