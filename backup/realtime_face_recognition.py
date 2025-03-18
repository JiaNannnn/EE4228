"""
Real-Time Face Recognition System

A multi-threaded face recognition system with these specifications:
1. Live Camera Interface
   - 30 FPS processing at 720p resolution
   - Multi-threaded pipeline with frame queue buffering
   - Automatic exposure/white balance adaptation

2. Detection + Recognition Pipeline
   - Hybrid face detector (Viola-Jones + LBP)
   - Advanced preprocessing with CLAHE and geometric normalization
   - PCA-LDA hybrid model with confidence-based rejection
"""

import cv2
import numpy as np
import time
import argparse
import threading
import queue
from pathlib import Path
import os
import logging
from collections import OrderedDict, deque
import uuid
import json
import platform
import psutil
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Import custom modules
from config import CAMERA_CONFIG, DETECTION_CONFIG, PREPROCESSING_CONFIG, MODEL_CONFIG
from hybrid_face_detector import HybridFaceDetector
from advanced_preprocessor import AdvancedPreprocessor
from hybrid_recognition_model import HybridRecognitionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("realtime_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("realtime_recognition")


class RealtimeFaceRecognition:
    """
    Real-time face recognition system with multi-threaded pipeline
    """
    
    def __init__(self, model_path=None, camera_id=None, resolution=None, fps=None,
                 buffer_size=None, show_fps=True, show_confidence=True, 
                 mirror=True, recognition_interval=3):
        """
        Initialize the real-time face recognition system
        
        Parameters:
        -----------
        model_path : str, optional
            Path to pre-trained recognition model
        camera_id : int, optional
            Camera device ID
        resolution : tuple, optional
            Camera resolution (width, height)
        fps : int, optional
            Camera frames per second
        buffer_size : int, optional
            Frame buffer size
        show_fps : bool
            Whether to show FPS counter
        show_confidence : bool
            Whether to show confidence scores
        mirror : bool
            Whether to mirror the camera view
        recognition_interval : int
            Number of frames between recognition attempts (higher = better performance)
        """
        # Set parameters from config or arguments
        self.camera_id = camera_id if camera_id is not None else CAMERA_CONFIG["device_id"]
        self.resolution = resolution if resolution is not None else CAMERA_CONFIG["resolution"]
        self.fps = fps if fps is not None else CAMERA_CONFIG["fps"]
        self.buffer_size = buffer_size if buffer_size is not None else CAMERA_CONFIG["buffer_size"]
        self.show_fps = show_fps
        self.show_confidence = show_confidence
        self.mirror = mirror
        self.recognition_interval = recognition_interval
        
        # Working directory
        self.work_dir = Path(__file__).parent
        
        # Frame buffer and threads
        self.frame_queue = queue.Queue(maxsize=self.buffer_size)
        self.detection_queue = queue.Queue(maxsize=self.buffer_size)
        self.result_queue = queue.Queue(maxsize=self.buffer_size)
        self.frame_lock = threading.Lock()
        self.running = False
        self.threads = []
        
        # Initialize components
        self.camera = None
        self.detector = HybridFaceDetector()
        self.preprocessor = AdvancedPreprocessor()
        self.recognizer = HybridRecognitionModel()
        
        # Load recognition model if provided
        if model_path:
            model_path = str(self.work_dir / model_path)
            self.load_model(model_path)
            
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.preprocessing_times = deque(maxlen=30)
        self.recognition_times = deque(maxlen=30)
        
        # Recognition cache for smooth results
        # Format: {face_id: (label, confidence, timestamp, face_location)}
        self.recognition_cache = OrderedDict()
        self.cache_timeout = 1.0  # seconds
        
        # Frame counter for skipping recognition on some frames
        self.frame_count = 0
        
        # User gallery
        self.gallery = {}  # {label: [face_embeddings]}
        self.gallery_metadata = {}  # {label: {name, count, last_updated}}
        
        # System info
        self.system_info = self._get_system_info()
        logger.info(f"System info: {self.system_info}")
        
    def _get_system_info(self):
        """Get system information"""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "memory": f"{psutil.virtual_memory().total / (1024 ** 3):.1f} GB",
            "python": platform.python_version(),
            "opencv": cv2.__version__
        }
    
    def load_model(self, model_path):
        """
        Load a pre-trained recognition model
        
        Parameters:
        -----------
        model_path : str
            Path to the model file
            
        Returns:
        --------
        bool
            True if model loaded successfully, False otherwise
        """
        try:
            success = self.recognizer.load(model_path)
            if success:
                logger.info(f"Loaded recognition model from {model_path}")
                logger.info(f"Model has {len(self.recognizer.class_names)} classes")
                return True
            else:
                logger.error(f"Failed to load model from {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _camera_thread(self):
        """
        Camera thread function for capturing frames
        """
        # Open camera
        self.camera = cv2.VideoCapture(self.camera_id)
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Enable auto exposure and white balance if supported
        if CAMERA_CONFIG["exposure_auto"]:
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = auto mode
            
        if CAMERA_CONFIG["white_balance_auto"]:
            self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)  # 1 = auto mode
            
        # Check if camera opened successfully
        if not self.camera.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            self.running = False
            return
            
        logger.info(f"Started camera thread with resolution {self.resolution} @ {self.fps} FPS")
        
        # Main capture loop
        prev_frame_time = time.time()
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                    
                # Mirror frame if requested
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                    
                # Calculate actual FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_frame_time)
                prev_frame_time = current_time
                self.fps_counter.append(fps)
                
                # Add frame to queue
                if not self.frame_queue.full():
                    frame_data = {
                        "frame": frame.copy(),
                        "timestamp": current_time,
                        "frame_id": str(uuid.uuid4()),
                        "fps": fps
                    }
                    self.frame_queue.put(frame_data, block=False)
                else:
                    logger.debug("Frame queue full, dropping frame")
            
            except Exception as e:
                logger.error(f"Error in camera thread: {e}")
                
        # Release camera
        if self.camera:
            self.camera.release()
            logger.info("Camera released")
    
    def _detection_thread(self):
        """
        Detection thread function for detecting faces in frames
        """
        logger.info("Started detection thread")
        
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=1.0)
                frame = frame_data["frame"]
                
                # Detect faces
                start_time = time.time()
                faces = self.detector.detect_faces(frame)
                detection_time = time.time() - start_time
                
                # Update timing stats
                self.detection_times.append(detection_time)
                
                # Add detection results to frame data
                frame_data["faces"] = faces
                frame_data["detection_time"] = detection_time
                
                # Put in detection queue
                if not self.detection_queue.full():
                    self.detection_queue.put(frame_data, block=False)
                    
                # Mark as done in frame queue
                self.frame_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in detection thread: {e}")
                
        logger.info("Detection thread stopped")
    
    def _recognition_thread(self):
        """
        Recognition thread function for processing detected faces
        """
        logger.info("Started recognition thread")
        
        while self.running:
            try:
                # Get detection result from queue
                frame_data = self.detection_queue.get(timeout=1.0)
                frame = frame_data["frame"]
                faces = frame_data["faces"]
                
                # Skip recognition on some frames for performance
                self.frame_count += 1
                do_recognition = self.frame_count % self.recognition_interval == 0
                
                # Process each face
                recognition_results = []
                recognition_time = 0
                
                for face_idx, face_data in enumerate(faces):
                    x, y, w, h, conf = face_data
                    
                    # Extract face ROI
                    face_roi = self.detector.get_face_roi(frame, face_data)
                    
                    # Skip if face ROI is None
                    if face_roi is None:
                        continue
                        
                    # Generate a unique identifier for this face based on position
                    face_id = f"{x}_{y}_{w}_{h}"
                    
                    # Check if we have recent recognition for this face_id
                    if face_id in self.recognition_cache:
                        cached_result = self.recognition_cache[face_id]
                        label, confidence, timestamp, _ = cached_result
                        
                        # Use cached result if it's recent
                        if time.time() - timestamp < self.cache_timeout:
                            recognition_results.append({
                                "face_id": face_id,
                                "face_rect": (x, y, w, h),
                                "label": label,
                                "confidence": confidence,
                                "cached": True
                            })
                            continue
                    
                    # Preprocess face
                    start_time = time.time()
                    processed_face, _ = self.preprocessor.preprocess(face_roi, cache_key=face_id)
                    preprocessing_time = time.time() - start_time
                    self.preprocessing_times.append(preprocessing_time)
                    
                    # Skip recognition on some frames or if not trained
                    if not do_recognition or not self.recognizer.trained:
                        # Add placeholder result
                        recognition_results.append({
                            "face_id": face_id,
                            "face_rect": (x, y, w, h),
                            "label": None,
                            "confidence": 0.0,
                            "cached": False
                        })
                        continue
                        
                    # Recognize face
                    start_time = time.time()
                    label, confidence, _ = self.recognizer.predict(processed_face, cache_key=face_id)
                    recog_time = time.time() - start_time
                    recognition_time += recog_time
                    
                    # Update recognition cache
                    self.recognition_cache[face_id] = (label, confidence, time.time(), (x, y, w, h))
                    
                    # Limit cache size with LRU eviction
                    if len(self.recognition_cache) > 100:
                        self.recognition_cache.popitem(last=False)
                        
                    # Add recognition result
                    recognition_results.append({
                        "face_id": face_id,
                        "face_rect": (x, y, w, h),
                        "label": label,
                        "confidence": confidence,
                        "cached": False
                    })
                
                # Update timing stats if recognition performed
                if do_recognition and recognition_time > 0:
                    self.recognition_times.append(recognition_time)
                
                # Add recognition results to frame data
                frame_data["recognition_results"] = recognition_results
                
                # Put in result queue
                if not self.result_queue.full():
                    self.result_queue.put(frame_data, block=False)
                    
                # Mark as done in detection queue
                self.detection_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in recognition thread: {e}")
                traceback.print_exc()
                
        logger.info("Recognition thread stopped")
    
    def _display_thread(self):
        """
        Display thread function for showing recognition results
        """
        logger.info("Started display thread")
        
        # Create display window
        window_name = "Real-Time Face Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while self.running:
            try:
                # Get recognition result from queue
                frame_data = self.result_queue.get(timeout=1.0)
                frame = frame_data["frame"].copy()  # Copy to avoid modifying original
                recognition_results = frame_data["recognition_results"]
                timestamp = frame_data["timestamp"]
                fps = frame_data["fps"]
                
                # Draw each face with recognition result
                for result in recognition_results:
                    x, y, w, h = result["face_rect"]
                    label = result["label"]
                    confidence = result["confidence"]
                    
                    # Determine color based on confidence and label
                    if label is None:
                        color = (128, 128, 128)  # Gray for unknown
                    else:
                        # Gradient from red to green based on confidence
                        green = int(255 * min(confidence / 0.8, 1.0))
                        red = int(255 * (1.0 - min(confidence / 0.8, 1.0)))
                        color = (0, green, red)
                        
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Prepare label text
                    if label:
                        label_text = f"{label}"
                        if self.show_confidence:
                            label_text += f" ({confidence:.0%})"
                    else:
                        label_text = "Unknown"
                        
                    # Draw label text
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y), color, -1)
                    cv2.putText(frame, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw FPS
                if self.show_fps:
                    avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
                    fps_text = f"FPS: {avg_fps:.1f}"
                    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw performance metrics
                    if self.detection_times:
                        det_time = np.mean(self.detection_times) * 1000
                        det_text = f"Detection: {det_time:.1f}ms"
                        cv2.putText(frame, det_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    if self.preprocessing_times:
                        pre_time = np.mean(self.preprocessing_times) * 1000
                        pre_text = f"Preprocess: {pre_time:.1f}ms"
                        cv2.putText(frame, pre_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    if self.recognition_times:
                        rec_time = np.mean(self.recognition_times) * 1000
                        rec_text = f"Recognition: {rec_time:.1f}ms"
                        cv2.putText(frame, rec_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                # Show model info
                if self.recognizer.trained:
                    model_text = f"Model: {len(self.recognizer.class_names)} subjects"
                    cv2.putText(frame, model_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                # Press 'q' to quit
                if key == ord('q'):
                    self.running = False
                    break
                    
                # Press 's' to take a screenshot
                elif key == ord('s'):
                    screenshot_dir = self.work_dir / "screenshots"
                    screenshot_dir.mkdir(exist_ok=True)
                    
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = str(screenshot_dir / f"screenshot_{timestamp_str}.jpg")
                    
                    cv2.imwrite(screenshot_path, frame)
                    logger.info(f"Screenshot saved to {screenshot_path}")
                    
                # Press 'a' to add a face to the gallery
                elif key == ord('a'):
                    # Check if there's a face to add
                    if recognition_results:
                        self._add_face_to_gallery(frame, recognition_results[0])
                
                # Mark as done in result queue
                self.result_queue.task_done()
                
            except queue.Empty:
                # Show blank frame if queue is empty
                blank_frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Waiting for camera...", (50, self.resolution[1] // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, blank_frame)
                cv2.waitKey(100)
                
            except Exception as e:
                logger.error(f"Error in display thread: {e}")
                traceback.print_exc()
                
        # Cleanup
        cv2.destroyAllWindows()
        logger.info("Display thread stopped")
    
    def _add_face_to_gallery(self, frame, face_result):
        """
        Add a face to the gallery interactively
        """
        x, y, w, h = face_result["face_rect"]
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess face
        processed_face, _ = self.preprocessor.preprocess(face_roi)
        
        # Get name from user
        name = None
        
        # Create input dialog window
        input_window = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(input_window, "Enter name in console", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Add Face to Gallery", input_window)
        cv2.waitKey(1)
        
        # Get input from console
        try:
            name = input("Enter name for this face (or press Enter to cancel): ")
        except:
            logger.error("Failed to get input from console")
            
        cv2.destroyWindow("Add Face to Gallery")
        
        if not name:
            logger.info("Face addition cancelled")
            return
            
        # Save face to gallery
        gallery_dir = self.work_dir / "gallery"
        gallery_dir.mkdir(exist_ok=True)
        
        # Create subject directory
        subject_dir = gallery_dir / name
        subject_dir.mkdir(exist_ok=True)
        
        # Save face image
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_path = str(subject_dir / f"{timestamp_str}.jpg")
        
        cv2.imwrite(face_path, processed_face)
        logger.info(f"Face added to gallery: {face_path}")
        
        # Update gallery metadata
        if name in self.gallery_metadata:
            self.gallery_metadata[name]["count"] += 1
            self.gallery_metadata[name]["last_updated"] = timestamp_str
        else:
            self.gallery_metadata[name] = {
                "name": name,
                "count": 1,
                "last_updated": timestamp_str
            }
            
        # Save gallery metadata
        metadata_path = str(gallery_dir / "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.gallery_metadata, f, indent=2)
            
        # Prompt to retrain model
        try:
            retrain = input("Retrain recognition model with new face? (y/n): ")
            if retrain.lower() == 'y':
                logger.info("Initiating model retraining...")
                # TODO: Implement incremental model retraining
                logger.info("Model retraining not implemented yet")
        except:
            logger.error("Failed to get input from console")
    
    def start(self):
        """
        Start the real-time face recognition system
        """
        # Check if already running
        if self.running:
            logger.warning("System is already running")
            return
            
        # Set running flag
        self.running = True
        
        # Start threads
        self.threads = []
        
        # Camera thread
        camera_thread = threading.Thread(target=self._camera_thread)
        camera_thread.daemon = True
        camera_thread.start()
        self.threads.append(camera_thread)
        
        # Detection thread
        detection_thread = threading.Thread(target=self._detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
        self.threads.append(detection_thread)
        
        # Recognition thread
        recognition_thread = threading.Thread(target=self._recognition_thread)
        recognition_thread.daemon = True
        recognition_thread.start()
        self.threads.append(recognition_thread)
        
        # Display thread (runs in main thread)
        self._display_thread()
        
        # Stop system when display thread exits
        self.stop()
    
    def stop(self):
        """
        Stop the real-time face recognition system
        """
        # Set running flag
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
            
        # Release camera
        if self.camera:
            self.camera.release()
            
        # Clear queues
        self._clear_queue(self.frame_queue)
        self._clear_queue(self.detection_queue)
        self._clear_queue(self.result_queue)
        
        logger.info("System stopped")
    
    def _clear_queue(self, q):
        """Safely clear a queue"""
        try:
            while not q.empty():
                q.get_nowait()
                q.task_done()
        except:
            pass


def main():
    """Main function for running the real-time face recognition system"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Real-Time Face Recognition System")
    parser.add_argument("--model", type=str, default="models/recognition_model.joblib",
                        help="Path to pre-trained recognition model")
    parser.add_argument("--camera", type=int, default=CAMERA_CONFIG["device_id"],
                        help="Camera device ID")
    parser.add_argument("--width", type=int, default=CAMERA_CONFIG["resolution"][0],
                        help="Camera frame width")
    parser.add_argument("--height", type=int, default=CAMERA_CONFIG["resolution"][1],
                        help="Camera frame height")
    parser.add_argument("--fps", type=int, default=CAMERA_CONFIG["fps"],
                        help="Camera frames per second")
    parser.add_argument("--no-fps", action="store_false", dest="show_fps",
                        help="Hide FPS counter")
    parser.add_argument("--no-confidence", action="store_false", dest="show_confidence",
                        help="Hide confidence scores")
    parser.add_argument("--no-mirror", action="store_false", dest="mirror",
                        help="Don't mirror camera view")
    parser.add_argument("--interval", type=int, default=3,
                        help="Recognition interval (frames)")
    
    args = parser.parse_args()
    
    # Create and start system
    resolution = (args.width, args.height)
    system = RealtimeFaceRecognition(
        model_path=args.model,
        camera_id=args.camera,
        resolution=resolution,
        fps=args.fps,
        show_fps=args.show_fps,
        show_confidence=args.show_confidence,
        mirror=args.mirror,
        recognition_interval=args.interval
    )
    
    try:
        system.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        system.stop()


if __name__ == "__main__":
    main() 