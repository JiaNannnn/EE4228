"""
Real-time Face Recognition App

Application for real-time face recognition using a webcam.
"""

import cv2
import numpy as np
import argparse
import time
import logging
import os
from face_detector import FaceDetector
from face_preprocessor import FacePreprocessor
from face_recognizer import FaceRecognizer
from config import (
    CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    MODEL_PATH, FACE_RECT_COLOR, FONT, FONT_SCALE, 
    FONT_THICKNESS, TEXT_COLOR, TEXT_BG_COLOR,
    WINDOW_TITLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app")

class FaceRecognitionApp:
    """
    Application for real-time face recognition
    """
    
    def __init__(self, camera_id=CAMERA_ID, model_path=MODEL_PATH,
                 camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT,
                 camera_fps=CAMERA_FPS):
        """
        Initialize the app
        
        Parameters:
        -----------
        camera_id : int
            Camera device ID
        model_path : str
            Path to the trained model
        camera_width : int
            Camera width
        camera_height : int
            Camera height
        camera_fps : int
            Camera FPS
        """
        self.camera_id = camera_id
        self.model_path = model_path
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        
        # Initialize camera
        self.cap = None
        
        # Initialize components
        self.detector = FaceDetector()
        self.preprocessor = FacePreprocessor()
        self.recognizer = FaceRecognizer()
        
        # Load model if exists
        if os.path.exists(model_path):
            if self.recognizer.load(model_path):
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.error(f"Failed to load model from {model_path}")
        else:
            logger.warning(f"Model file {model_path} does not exist")
            
        # Performance metrics
        self.fps = 0
        self.processing_time = 0
        
        logger.info("Face Recognition App initialized")
        
    def start_camera(self):
        """
        Start the camera
        
        Returns:
        --------
        bool
            True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            logger.info(f"Camera {self.camera_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
            
    def stop_camera(self):
        """
        Stop the camera
        """
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera stopped")
            
    def process_frame(self, frame):
        """
        Process a frame for face recognition
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame
            
        Returns:
        --------
        numpy.ndarray
            Processed frame with recognition results
        list
            List of recognition results (label, confidence)
        """
        # Start timer
        start_time = time.time()
        
        # Make a copy of the frame
        result = frame.copy()
        
        # Detect faces
        faces = self.detector.detect_faces(frame)
        
        # Process each face
        recognition_results = []
        
        for face_rect in faces:
            # Extract face ROI
            face_roi = self.detector.get_face_roi(frame, face_rect)
            
            # Preprocess face
            preprocessed_face = self.preprocessor.preprocess(face_roi)
            
            # Skip if preprocessing failed
            if preprocessed_face is None:
                continue
                
            # Recognize face
            if self.recognizer.is_trained:
                label, confidence = self.recognizer.predict(preprocessed_face)
            else:
                label, confidence = None, 0.0
                
            # Add to results
            recognition_results.append((face_rect, label, confidence))
            
            # Draw face rectangle
            x, y, w, h = face_rect
            cv2.rectangle(result, (x, y), (x + w, y + h), FACE_RECT_COLOR, 2)
            
            # Draw label
            if label is not None:
                label_text = f"{label} ({confidence:.2f})"
            else:
                label_text = "Unknown"
                
            # Calculate text size
            text_size = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            
            # Draw text background
            cv2.rectangle(
                result,
                (x, y - text_size[1] - 10),
                (x + text_size[0] + 10, y),
                TEXT_BG_COLOR,
                -1
            )
            
            # Draw text
            cv2.putText(
                result,
                label_text,
                (x + 5, y - 5),
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS
            )
            
        # Calculate processing time
        self.processing_time = time.time() - start_time
        
        # Calculate FPS
        if self.processing_time > 0:
            self.fps = 1.0 / self.processing_time
            
        # Draw FPS
        cv2.putText(
            result,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            FONT,
            FONT_SCALE,
            (0, 255, 0),
            FONT_THICKNESS
        )
        
        return result, recognition_results
        
    def run(self):
        """
        Run the app
        """
        if not self.start_camera():
            logger.error("Failed to start camera")
            return
            
        # Create window
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                    
                # Process frame
                processed_frame, _ = self.process_frame(frame)
                
                # Show frame
                cv2.imshow(WINDOW_TITLE, processed_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                # Press 'q' to quit
                if key == ord('q'):
                    break
                    
        finally:
            # Clean up
            self.stop_camera()
            cv2.destroyAllWindows()
            logger.info("Application stopped")
            
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time Face Recognition App")
    parser.add_argument("--camera", type=int, default=CAMERA_ID,
                        help="Camera device ID")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to the trained model")
    args = parser.parse_args()
    
    # Create and run the app
    app = FaceRecognitionApp(camera_id=args.camera, model_path=args.model)
    app.run()
    
if __name__ == "__main__":
    main() 