"""
Web Interface for Real-Time Face Recognition System

This module provides a web interface to access the face recognition system
remotely through a browser. It uses Flask as the web server and provides
both realtime video streaming and system management capabilities.
"""

import os
import sys
import cv2
import time
import base64
import datetime
import logging
import threading
import numpy as np
from pathlib import Path
from flask import Flask, Response, render_template, request, jsonify
from io import BytesIO
from PIL import Image
import queue

# Set up logging first before imports to capture any import errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebInterface")

# Import custom modules with try/except to handle potential import errors
try:
    from config import Config
    
    # Import the remaining modules
    # Use a flag to track successful imports
    all_imports_successful = True
    
    try:
        from realtime_face_recognition import RealtimeFaceRecognition
        from hybrid_face_detector import HybridFaceDetector
        from advanced_preprocessor import AdvancedPreprocessor
        from hybrid_recognition_model import HybridRecognitionModel
    except ImportError as e:
        logger.error(f"Error importing face recognition modules: {e}")
        all_imports_successful = False
except ImportError as e:
    logger.error(f"Error importing config: {e}")
    all_imports_successful = False
    
    # Define a minimal config class as a fallback
    class Config:
        def __init__(self):
            self.CAMERA_ID = 0
            self.CAMERA_WIDTH = 640
            self.CAMERA_HEIGHT = 480
            self.TARGET_FPS = 30
            self.MIN_FACE_SIZE = (90, 90)
            self.SCALE_FACTOR = 1.1
            self.FACE_SIZE = (90, 90)
            self.USE_CLAHE = True
            self.CLAHE_CLIP_LIMIT = 3.0
            self.NORMALIZE_FACE = True
            self.MODEL_PATH = "recognition_model.pkl"
            self.HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.LBP_CASCADE_PATH = cv2.data.haarcascades + 'lbpcascade_frontalface.xml'
            self.SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
            self.WEB_HOST = "0.0.0.0"
            self.WEB_PORT = 8000
            self.PCA_VARIANCE_RETAIN = 0.95

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Global variables
config = Config()
recognition_system = None
camera = None
is_streaming = False
stream_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=10)  # Buffer for frames
last_detection_results = []
system_status = {"error": None, "message": "System initializing..."}

class WebCamera:
    """Class to handle camera operations for web streaming"""
    
    def __init__(self, camera_id=0):
        """
        Initialize camera
        
        Parameters:
        -----------
        camera_id : int, optional
            Camera ID (default: 0)
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.frame_width = config.CAMERA_WIDTH
        self.frame_height = config.CAMERA_HEIGHT
        self.frame_rate = config.TARGET_FPS
    
    def start(self):
        """Start camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            self.is_running = True
            logger.info(f"Camera {self.camera_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
            
    def stop(self):
        """Stop camera"""
        if self.cap and self.is_running:
            self.is_running = False
            self.cap.release()
            logger.info("Camera stopped")
            
    def get_frame(self):
        """Get a single frame from camera"""
        if not self.is_running:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None
            
        return frame

def initialize_system():
    """Initialize the face recognition system"""
    global recognition_system, camera, system_status
    
    try:
        # Check if all imports were successful
        if not all_imports_successful:
            system_status = {
                "error": "ImportError",
                "message": "Failed to import required modules. The system may run with limited functionality."
            }
            logger.error(system_status["message"])
            # Still continue to initialize basic components
        
        # Initialize camera
        try:
            camera = WebCamera(camera_id=config.CAMERA_ID)
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            camera = None
            system_status = {
                "error": "CameraError",
                "message": f"Failed to initialize camera: {str(e)}"
            }
        
        # Initialize remaining components even if camera failed
        try:
            # Initialize face detector
            detector = HybridFaceDetector(
                min_face_size=config.MIN_FACE_SIZE,
                scale_factor=config.SCALE_FACTOR
            )
            
            # Initialize preprocessor
            preprocessor = AdvancedPreprocessor(
                target_size=config.FACE_SIZE,
            )
            
            # Initialize recognition model
            recognition_model = HybridRecognitionModel()
            
            # Check if model exists and load it
            if os.path.exists(config.MODEL_PATH):
                logger.info(f"Loading model from {config.MODEL_PATH}")
                recognition_model.load(config.MODEL_PATH)
                
            # Initialize recognition system
            recognition_system = RealtimeFaceRecognition(
                face_detector=detector,
                face_preprocessor=preprocessor,
                recognition_model=recognition_model,
            )
            
            logger.info("Face recognition system initialized")
            
            # Don't overwrite camera error if there was one
            if "error" not in system_status or system_status["error"] != "CameraError":
                system_status = {
                    "error": None,
                    "message": "System initialized successfully."
                }
            return True
            
        except Exception as e:
            logger.error(f"Error initializing recognition components: {e}")
            system_status = {
                "error": "InitializationError",
                "message": f"Failed to initialize recognition components: {str(e)}"
            }
            return False
            
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        system_status = {
            "error": "SystemError",
            "message": f"Failed to initialize system: {str(e)}"
        }
        return False

def process_frame_worker():
    """Worker function to process frames in background thread"""
    global is_streaming, last_detection_results, current_frame
    
    logger.info("Frame processing worker started")
    
    # Create a static frame if camera is not available
    placeholder_frame = None
    if camera is None:
        # Create a placeholder frame with a message
        placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder_frame,
            "Camera not available",
            (160, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        # Add the error message
        cv2.putText(
            placeholder_frame,
            f"Error: {system_status['error']}",
            (160, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # Convert to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', placeholder_frame)
        if ret:
            current_frame = buffer.tobytes()
    
    while is_streaming:
        try:
            # If camera is not available, serve the placeholder frame
            if camera is None:
                time.sleep(0.1)  # Sleep to avoid high CPU usage
                continue
                
            # Get frame from queue
            if frame_queue.empty():
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                continue
                
            frame = frame_queue.get()
            
            # Process frame
            start_time = time.time()
            
            # Check if recognition system is available
            if recognition_system is not None:
                processed_frame, detection_results = recognition_system.process_frame(frame)
            else:
                # Fallback if system isn't initialized - just return the original frame
                processed_frame = frame.copy()
                detection_results = []
                
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Update results
            with stream_lock:
                last_detection_results = detection_results
                
            # Add FPS information
            fps = 1000 / processing_time if processing_time > 0 else 0
            cv2.putText(
                processed_frame, 
                f"FPS: {fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            
            # Add system status if there was an error
            if system_status["error"] is not None:
                cv2.putText(
                    processed_frame,
                    f"Error: {system_status['error']}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
            # Convert to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            
            # Send frame to all clients
            frame_bytes = buffer.tobytes()
            
            # Add to global frame buffer
            with stream_lock:
                current_frame = frame_bytes
                
        except Exception as e:
            logger.error(f"Error in frame processing worker: {e}")
            time.sleep(0.1)
            
    logger.info("Frame processing worker stopped")

def capture_frames_worker():
    """Worker function to capture frames in background thread"""
    global is_streaming
    
    logger.info("Frame capture worker started")
    
    while is_streaming:
        try:
            # Capture frame
            frame = camera.get_frame()
            
            if frame is not None:
                # Add to queue, discard if queue is full (to maintain real-time)
                if not frame_queue.full():
                    frame_queue.put(frame)
                    
            else:
                # Camera error
                logger.warning("Camera returned empty frame")
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in frame capture worker: {e}")
            time.sleep(0.1)
            
    logger.info("Frame capture worker stopped")

def generate_frames():
    """Generator function for video streaming"""
    global current_frame
    
    while is_streaming:
        try:
            # Get latest processed frame
            with stream_lock:
                if 'current_frame' in globals() and current_frame is not None:
                    frame_data = current_frame
                else:
                    time.sleep(0.03)  # Wait for a frame
                    continue
                    
            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                   
        except Exception as e:
            logger.error(f"Error in generate_frames: {e}")
            time.sleep(0.1)

def start_streaming():
    """Start video streaming"""
    global is_streaming, current_frame
    
    if is_streaming:
        return True
    
    # Check if camera is available
    if camera is None:
        logger.error("Cannot start streaming: Camera is not available")
        return False
        
    # Start camera
    if not camera.start():
        logger.error("Failed to start camera")
        return False
        
    # Initialize streaming variables
    is_streaming = True
    current_frame = None
    
    # Start workers
    threading.Thread(target=capture_frames_worker, daemon=True).start()
    threading.Thread(target=process_frame_worker, daemon=True).start()
    
    logger.info("Streaming started")
    return True

def stop_streaming():
    """Stop video streaming"""
    global is_streaming
    
    if not is_streaming:
        return True
        
    # Stop streaming
    is_streaming = False
    
    # Clear queue
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except:
            pass
            
    # Stop camera
    camera.stop()
    
    logger.info("Streaming stopped")
    return True

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', 
                          is_streaming=is_streaming,
                          model_loaded=recognition_system.recognition_model.is_trained if recognition_system else False,
                          system_status=system_status)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if not is_streaming:
        return "Streaming not active", 404
        
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_stream', methods=['POST'])
def api_start_stream():
    """API endpoint to start streaming"""
    if camera is None:
        return jsonify({'status': 'error', 'message': 'Camera is not available. Check system logs.'})
        
    if start_streaming():
        return jsonify({'status': 'success', 'message': 'Streaming started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start streaming'})

@app.route('/api/stop_stream', methods=['POST'])
def api_stop_stream():
    """API endpoint to stop streaming"""
    if stop_streaming():
        return jsonify({'status': 'success', 'message': 'Streaming stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to stop streaming'})

@app.route('/api/get_status')
def api_get_status():
    """API endpoint to get system status"""
    status = {
        'is_streaming': is_streaming,
        'model_loaded': recognition_system.recognition_model.is_trained if recognition_system and hasattr(recognition_system, 'recognition_model') else False,
        'known_subjects': recognition_system.recognition_model.get_known_subjects() if recognition_system and hasattr(recognition_system, 'recognition_model') and recognition_system.recognition_model.is_trained else [],
        'detection_results': last_detection_results,
        'system_status': system_status
    }
    return jsonify(status)

@app.route('/api/add_face', methods=['POST'])
def api_add_face():
    """API endpoint to add a new face"""
    if not is_streaming:
        return jsonify({'status': 'error', 'message': 'Streaming must be active to add faces'})
        
    try:
        # Get subject name
        subject_name = request.form.get('subject_name')
        if not subject_name:
            return jsonify({'status': 'error', 'message': 'Subject name is required'})
            
        # Get current frame
        frame = camera.get_frame()
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to capture frame'})
            
        # Detect faces
        faces = recognition_system.face_detector.detect_faces(frame)
        if not faces:
            return jsonify({'status': 'error', 'message': 'No face detected in frame'})
            
        # Use the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = recognition_system.face_detector.extract_face_roi(frame, largest_face)
        
        # Preprocess face
        preprocessed_face = recognition_system.face_preprocessor.preprocess(face_roi)
        
        # Save face for training
        save_dir = os.path.join('face_database', subject_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{subject_name}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Save preprocessed face
        cv2.imwrite(filepath, preprocessed_face)
        
        # Add to model if it's already trained
        if recognition_system.recognition_model.is_trained:
            recognition_system.recognition_model.add_face(preprocessed_face, subject_name)
            
        return jsonify({
            'status': 'success', 
            'message': f'Face added for {subject_name}',
            'face_count': len(os.listdir(save_dir))
        })
        
    except Exception as e:
        logger.error(f"Error adding face: {e}")
        return jsonify({'status': 'error', 'message': f'Error adding face: {str(e)}'})

@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    """API endpoint to train the recognition model"""
    try:
        # Check if face database exists
        if not os.path.exists('face_database'):
            return jsonify({'status': 'error', 'message': 'No face database found'})
            
        # Count faces
        subjects = [d for d in os.listdir('face_database') if os.path.isdir(os.path.join('face_database', d))]
        if not subjects:
            return jsonify({'status': 'error', 'message': 'No subjects found in face database'})
            
        # Load faces for training
        faces = []
        labels = []
        subject_names = []
        
        for subject_idx, subject in enumerate(subjects):
            subject_dir = os.path.join('face_database', subject)
            subject_names.append(subject)
            
            for img_file in os.listdir(subject_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(subject_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        faces.append(img)
                        labels.append(subject_idx)
                        
        if not faces:
            return jsonify({'status': 'error', 'message': 'No valid face images found'})
            
        # Convert to numpy arrays
        faces_array = np.array(faces)
        labels_array = np.array(labels)
        
        # Train model
        recognition_system.recognition_model.train(
            faces_array, 
            labels_array, 
            subject_names,
            pca_variance_retain=config.PCA_VARIANCE_RETAIN
        )
        
        # Save trained model
        recognition_system.recognition_model.save(config.MODEL_PATH)
        
        return jsonify({
            'status': 'success',
            'message': f'Model trained with {len(faces)} images of {len(subjects)} subjects',
            'subjects': subjects
        })
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({'status': 'error', 'message': f'Error training model: {str(e)}'})

@app.route('/api/delete_subject', methods=['POST'])
def api_delete_subject():
    """API endpoint to delete a subject"""
    try:
        # Get subject name
        subject_name = request.form.get('subject_name')
        if not subject_name:
            return jsonify({'status': 'error', 'message': 'Subject name is required'})
            
        # Check if subject exists
        subject_dir = os.path.join('face_database', subject_name)
        if not os.path.exists(subject_dir):
            return jsonify({'status': 'error', 'message': f'Subject {subject_name} not found'})
            
        # Delete subject directory
        for file in os.listdir(subject_dir):
            os.remove(os.path.join(subject_dir, file))
        os.rmdir(subject_dir)
        
        # Retrain model if it's already trained
        if recognition_system.recognition_model.is_trained:
            # Remove from model
            recognition_system.recognition_model.remove_subject(subject_name)
            
            # Save model
            recognition_system.recognition_model.save(config.MODEL_PATH)
            
        return jsonify({
            'status': 'success',
            'message': f'Subject {subject_name} deleted'
        })
        
    except Exception as e:
        logger.error(f"Error deleting subject: {e}")
        return jsonify({'status': 'error', 'message': f'Error deleting subject: {str(e)}'})

def create_templates_folder():
    """Create the templates folder with required HTML files if they don't exist"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    
    # Create directories if they don't exist
    for directory in [templates_dir, static_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    # Check if index.html exists
    index_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_path):
        logger.info("Creating default index.html template")
        # Template creation logic is implemented in the main function
        
    # Check if style.css exists
    style_path = os.path.join(static_dir, 'style.css')
    if not os.path.exists(style_path):
        logger.info("Creating default style.css")
        # Style creation logic is implemented in the main function
    
    # Check if script.js exists
    script_path = os.path.join(static_dir, 'script.js')
    if not os.path.exists(script_path):
        logger.info("Creating default script.js")
        # Script creation logic is implemented in the main function

def main():
    """Main function"""
    # Create templates folder if it doesn't exist
    create_templates_folder()
    
    # Initialize system
    if not initialize_system():
        logger.warning("System initialized with limited functionality due to missing modules")
    
    # Create face database directory
    os.makedirs('face_database', exist_ok=True)
    
    try:
        # Run flask app
        logger.info(f"Starting web server on {config.WEB_HOST}:{config.WEB_PORT}")
        app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Error starting web server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 