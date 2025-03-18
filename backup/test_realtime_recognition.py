"""
Test and Benchmark Real-Time Face Recognition System

This script runs a comprehensive test of the real-time face recognition system,
including performance benchmarking and validation with different test cases.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue
import pandas as pd
import psutil
from collections import defaultdict

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime_face_recognition import RealtimeFaceRecognition
from hybrid_face_detector import HybridFaceDetector
from advanced_preprocessor import AdvancedPreprocessor
from hybrid_recognition_model import HybridRecognitionModel
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestRecognition")

class SystemTester:
    """Class to test and benchmark the real-time face recognition system"""
    
    def __init__(self, config_file=None, model_path=None, test_data_path=None):
        """
        Initialize the tester
        
        Parameters:
        -----------
        config_file : str, optional
            Path to configuration file
        model_path : str, optional
            Path to trained recognition model
        test_data_path : str, optional
            Path to test data directory
        """
        self.config = Config(config_file) if config_file else Config()
        self.model_path = model_path or "trained_hybrid_model.pkl"
        self.test_data_path = test_data_path
        
        # Performance metrics
        self.metrics = {
            'detection_times': [],
            'preprocessing_times': [],
            'recognition_times': [],
            'total_times': [],
            'memory_usage': [],
            'accuracy': [],
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Initialize system components
        self._init_components()
        
    def _init_components(self):
        """Initialize system components"""
        try:
            # Initialize face detector
            self.detector = HybridFaceDetector(
                haarcascade_path=self.config.HAARCASCADE_PATH,
                lbp_cascade_path=self.config.LBP_CASCADE_PATH,
                shape_predictor_path=self.config.SHAPE_PREDICTOR_PATH,
                min_face_size=self.config.MIN_FACE_SIZE,
                scale_factor=self.config.SCALE_FACTOR
            )
            
            # Initialize preprocessor
            self.preprocessor = AdvancedPreprocessor(
                target_size=self.config.FACE_SIZE,
                use_clahe=self.config.USE_CLAHE,
                clip_limit=self.config.CLAHE_CLIP_LIMIT,
                normalize_face=self.config.NORMALIZE_FACE
            )
            
            # Initialize recognition model
            self.recognition_model = HybridRecognitionModel()
            
            # Check if model exists and load it
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.recognition_model.load(self.model_path)
            else:
                logger.warning(f"Model not found at {self.model_path}. Will need to train before testing.")
                
            # Initialize real-time recognition system
            self.recognition_system = RealtimeFaceRecognition(
                face_detector=self.detector,
                face_preprocessor=self.preprocessor,
                recognition_model=self.recognition_model,
                config=self.config
            )
            
            logger.info("System components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
            
    def run_benchmark_on_video(self, video_path=None, duration=30, show_display=True):
        """
        Run benchmark on video file or webcam
        
        Parameters:
        -----------
        video_path : str, optional
            Path to video file (uses webcam if None)
        duration : int, optional
            Duration in seconds for the test (default: 30)
        show_display : bool, optional
            Whether to show display during benchmark
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        try:
            # Use webcam if video_path is None
            cap = cv2.VideoCapture(video_path if video_path else 0)
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {video_path}")
                return None
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if video_path is None:  # Webcam
                fps = 30  # Assume 30 fps for webcam
                
            # Calculate number of frames to process
            max_frames = int(duration * fps)
            
            logger.info(f"Running benchmark on {'webcam' if video_path is None else video_path} "
                       f"for {duration} seconds ({max_frames} frames)")
                       
            # Process frames
            frame_count = 0
            start_time = time.time()
            
            with tqdm(total=max_frames, desc="Processing frames") as pbar:
                while frame_count < max_frames:
                    # Record memory usage
                    self.metrics['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
                    
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        if video_path:  # End of video file
                            logger.info("End of video file reached")
                            break
                        else:  # Webcam error
                            logger.error("Error reading from webcam")
                            break
                            
                    # Process frame and measure time
                    t0 = time.time()
                    
                    # Detect faces
                    t1 = time.time()
                    faces = self.detector.detect_faces(frame)
                    t2 = time.time()
                    self.metrics['detection_times'].append((t2 - t1) * 1000)  # ms
                    
                    # Process each face
                    for face_rect in faces:
                        # Extract face ROI
                        face_roi = self.detector.extract_face_roi(frame, face_rect)
                        
                        # Preprocess face
                        t3 = time.time()
                        preprocessed_face = self.preprocessor.preprocess(face_roi)
                        t4 = time.time()
                        self.metrics['preprocessing_times'].append((t4 - t3) * 1000)  # ms
                        
                        # Recognize face
                        t5 = time.time()
                        if self.recognition_model.is_trained:
                            identity, confidence = self.recognition_model.predict(preprocessed_face)
                        else:
                            identity, confidence = "Unknown", 0.0
                        t6 = time.time()
                        self.metrics['recognition_times'].append((t6 - t5) * 1000)  # ms
                        
                        # Draw results on frame if showing display
                        if show_display:
                            x, y, w, h = face_rect
                            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, f"{identity} ({confidence:.2f})", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Calculate total processing time
                    t7 = time.time()
                    self.metrics['total_times'].append((t7 - t0) * 1000)  # ms
                    
                    # Display frame
                    if show_display:
                        cv2.putText(frame, f"FPS: {1/(t7-t0):.2f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("Benchmark", frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Benchmark stopped by user")
                            break
                    
                    frame_count += 1
                    pbar.update(1)
                    
            # Clean up
            cap.release()
            if show_display:
                cv2.destroyAllWindows()
                
            # Calculate summary metrics
            end_time = time.time()
            actual_duration = end_time - start_time
            actual_fps = frame_count / actual_duration
            
            summary = {
                'total_frames': frame_count,
                'actual_duration': actual_duration,
                'average_fps': actual_fps,
                'detection_time_avg': np.mean(self.metrics['detection_times']),
                'detection_time_std': np.std(self.metrics['detection_times']),
                'preprocessing_time_avg': np.mean(self.metrics['preprocessing_times']) if self.metrics['preprocessing_times'] else 0,
                'preprocessing_time_std': np.std(self.metrics['preprocessing_times']) if self.metrics['preprocessing_times'] else 0,
                'recognition_time_avg': np.mean(self.metrics['recognition_times']) if self.metrics['recognition_times'] else 0,
                'recognition_time_std': np.std(self.metrics['recognition_times']) if self.metrics['recognition_times'] else 0,
                'total_time_avg': np.mean(self.metrics['total_times']),
                'total_time_std': np.std(self.metrics['total_times']),
                'memory_usage_avg': np.mean(self.metrics['memory_usage']),
                'memory_usage_max': np.max(self.metrics['memory_usage'])
            }
            
            logger.info(f"Benchmark results: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error during benchmark: {e}")
            return None
            
    def run_accuracy_test(self, test_dataset_path):
        """
        Run accuracy test on a labeled test dataset
        
        Parameters:
        -----------
        test_dataset_path : str
            Path to test dataset directory
            
        Returns:
        --------
        dict
            Dictionary of accuracy metrics
        """
        if not self.recognition_model.is_trained:
            logger.error("Recognition model is not trained. Cannot run accuracy test.")
            return None
            
        try:
            logger.info(f"Running accuracy test on {test_dataset_path}")
            
            # Load test dataset
            subjects = []
            face_images = []
            true_labels = []
            
            # Iterate through dataset directories (each directory = one subject)
            for subject_dir in os.listdir(test_dataset_path):
                subject_path = os.path.join(test_dataset_path, subject_dir)
                
                if os.path.isdir(subject_path):
                    subject_name = subject_dir
                    subjects.append(subject_name)
                    
                    # Load all images for this subject
                    for img_file in os.listdir(subject_path):
                        if img_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(subject_path, img_file)
                            img = cv2.imread(img_path)
                            
                            if img is not None:
                                # Detect face
                                faces = self.detector.detect_faces(img)
                                
                                if faces:
                                    # Use the first detected face
                                    face_roi = self.detector.extract_face_roi(img, faces[0])
                                    
                                    # Preprocess face
                                    preprocessed_face = self.preprocessor.preprocess(face_roi)
                                    
                                    # Add to dataset
                                    face_images.append(preprocessed_face)
                                    true_labels.append(subject_name)
                                else:
                                    logger.warning(f"No face detected in {img_path}")
                            else:
                                logger.warning(f"Could not read image {img_path}")
            
            # Check if we have data
            if not face_images:
                logger.error("No valid face images found in test dataset")
                return None
                
            logger.info(f"Loaded {len(face_images)} test images for {len(subjects)} subjects")
            
            # Predict each face and calculate accuracy
            predictions = []
            confidences = []
            correct = 0
            
            for i, face in enumerate(tqdm(face_images, desc="Testing recognition")):
                predicted_label, confidence = self.recognition_model.predict(face)
                predictions.append(predicted_label)
                confidences.append(confidence)
                
                if predicted_label == true_labels[i]:
                    correct += 1
                    
            # Calculate metrics
            accuracy = correct / len(face_images)
            
            # Calculate confusion matrix
            confusion = defaultdict(lambda: defaultdict(int))
            for i in range(len(true_labels)):
                confusion[true_labels[i]][predictions[i]] += 1
                
            # Calculate per-subject accuracy
            subject_accuracy = {}
            for subject in subjects:
                subject_indices = [i for i, label in enumerate(true_labels) if label == subject]
                if subject_indices:
                    correct_subject = sum(1 for i in subject_indices if predictions[i] == subject)
                    subject_accuracy[subject] = correct_subject / len(subject_indices)
                    
            # Prepare results
            results = {
                'accuracy': accuracy,
                'num_samples': len(face_images),
                'num_subjects': len(subjects),
                'subject_accuracy': subject_accuracy,
                'confusion_matrix': confusion
            }
            
            logger.info(f"Accuracy test results: accuracy={accuracy:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error during accuracy test: {e}")
            return None
            
    def plot_benchmark_results(self, results, output_path="benchmark_results"):
        """
        Plot benchmark results
        
        Parameters:
        -----------
        results : dict
            Dictionary of benchmark results
        output_path : str, optional
            Path to save plots
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Plot processing times
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = ['Detection', 'Preprocessing', 'Recognition', 'Total']
            times = [
                results['detection_time_avg'],
                results['preprocessing_time_avg'],
                results['recognition_time_avg'],
                results['total_time_avg']
            ]
            std_devs = [
                results['detection_time_std'],
                results['preprocessing_time_std'],
                results['recognition_time_std'],
                results['total_time_std']
            ]
            
            ax.bar(categories, times, yerr=std_devs, capsize=10)
            ax.set_ylabel('Time (ms)')
            ax.set_title('Average Processing Times')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add target latency line
            if hasattr(self.config, 'TARGET_LATENCY_MS'):
                ax.axhline(y=self.config.TARGET_LATENCY_MS, color='r', linestyle='-', label=f'Target ({self.config.TARGET_LATENCY_MS} ms)')
                ax.legend()
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'processing_times.png'))
            
            # Plot FPS
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(['Achieved FPS'], [results['average_fps']], color='green')
            
            # Add target FPS line
            if hasattr(self.config, 'TARGET_FPS'):
                ax.axhline(y=self.config.TARGET_FPS, color='r', linestyle='-', label=f'Target ({self.config.TARGET_FPS} FPS)')
                ax.legend()
                
            ax.set_ylabel('Frames Per Second')
            ax.set_title('System Performance')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'fps.png'))
            
            # Save summary as text file
            with open(os.path.join(output_path, 'benchmark_summary.txt'), 'w') as f:
                f.write("Face Recognition System Benchmark Summary\n")
                f.write("==========================================\n\n")
                f.write(f"Total frames processed: {results['total_frames']}\n")
                f.write(f"Actual duration: {results['actual_duration']:.2f} seconds\n")
                f.write(f"Average FPS: {results['average_fps']:.2f}\n\n")
                
                f.write("Processing Times (ms):\n")
                f.write(f"  Detection: {results['detection_time_avg']:.2f} ± {results['detection_time_std']:.2f}\n")
                f.write(f"  Preprocessing: {results['preprocessing_time_avg']:.2f} ± {results['preprocessing_time_std']:.2f}\n")
                f.write(f"  Recognition: {results['recognition_time_avg']:.2f} ± {results['recognition_time_std']:.2f}\n")
                f.write(f"  Total: {results['total_time_avg']:.2f} ± {results['total_time_std']:.2f}\n\n")
                
                f.write("Memory Usage (MB):\n")
                f.write(f"  Average: {results['memory_usage_avg']:.2f}\n")
                f.write(f"  Maximum: {results['memory_usage_max']:.2f}\n")
            
            logger.info(f"Benchmark plots and summary saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error plotting benchmark results: {e}")
            
    def plot_accuracy_results(self, results, output_path="accuracy_results"):
        """
        Plot accuracy test results
        
        Parameters:
        -----------
        results : dict
            Dictionary of accuracy test results
        output_path : str, optional
            Path to save plots
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Plot overall accuracy
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(['Overall Accuracy'], [results['accuracy']], color='blue')
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Recognition Accuracy (N={results["num_samples"]})')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add target accuracy line
            if hasattr(self.config, 'TARGET_ACCURACY'):
                ax.axhline(y=self.config.TARGET_ACCURACY, color='r', linestyle='-', 
                          label=f'Target ({self.config.TARGET_ACCURACY*100:.0f}%)')
                ax.legend()
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'overall_accuracy.png'))
            
            # Plot per-subject accuracy
            if results['subject_accuracy']:
                subjects = list(results['subject_accuracy'].keys())
                accuracies = [results['subject_accuracy'][s] for s in subjects]
                
                # Sort by accuracy for better visualization
                sorted_indices = np.argsort(accuracies)
                subjects = [subjects[i] for i in sorted_indices]
                accuracies = [accuracies[i] for i in sorted_indices]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(subjects, accuracies, color='skyblue')
                ax.set_ylim(0, 1.0)
                ax.set_ylabel('Accuracy')
                ax.set_xlabel('Subject')
                ax.set_title('Per-Subject Recognition Accuracy')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, 'subject_accuracy.png'))
                
                # Save confusion matrix as CSV
                confusion_df = pd.DataFrame(results['confusion_matrix']).fillna(0)
                confusion_df.to_csv(os.path.join(output_path, 'confusion_matrix.csv'))
                
            # Save summary as text file
            with open(os.path.join(output_path, 'accuracy_summary.txt'), 'w') as f:
                f.write("Face Recognition System Accuracy Summary\n")
                f.write("=======================================\n\n")
                f.write(f"Overall accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Number of test samples: {results['num_samples']}\n")
                f.write(f"Number of subjects: {results['num_subjects']}\n\n")
                
                f.write("Per-Subject Accuracy:\n")
                for subject, acc in results['subject_accuracy'].items():
                    f.write(f"  {subject}: {acc:.4f}\n")
                    
            logger.info(f"Accuracy plots and summary saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error plotting accuracy results: {e}")
            
    def run_stress_test(self, iterations=100):
        """
        Run a stress test to check system stability
        
        Parameters:
        -----------
        iterations : int, optional
            Number of iterations to run
            
        Returns:
        --------
        bool
            True if test passes, False otherwise
        """
        try:
            logger.info(f"Running stress test with {iterations} iterations")
            
            # Load test image or use a generated one
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(test_image, (320, 240), 100, (255, 255, 255), -1)
            cv2.circle(test_image, (280, 220), 10, (0, 0, 0), -1)
            cv2.circle(test_image, (360, 220), 10, (0, 0, 0), -1)
            cv2.ellipse(test_image, (320, 280), (50, 20), 0, 0, 180, (0, 0, 0), -1)
            
            # Initialize metrics
            detection_success = 0
            memory_usage = []
            
            # Run test iterations
            for i in tqdm(range(iterations), desc="Running stress test"):
                # Record memory usage
                memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
                
                # Detect faces
                faces = self.detector.detect_faces(test_image)
                
                if faces:
                    detection_success += 1
                    face_roi = self.detector.extract_face_roi(test_image, faces[0])
                    
                    # Preprocess face
                    preprocessed_face = self.preprocessor.preprocess(face_roi)
                    
                    # Recognize face if model is trained
                    if self.recognition_model.is_trained:
                        identity, confidence = self.recognition_model.predict(preprocessed_face)
                        
            # Calculate metrics
            success_rate = detection_success / iterations
            memory_growth = (memory_usage[-1] - memory_usage[0]) if len(memory_usage) > 1 else 0
            
            # Results
            results = {
                'success_rate': success_rate,
                'iterations': iterations,
                'initial_memory': memory_usage[0],
                'final_memory': memory_usage[-1],
                'memory_growth': memory_growth
            }
            
            logger.info(f"Stress test results: {results}")
            
            # Test passes if success rate is high and memory growth is minimal
            test_passed = success_rate > 0.95 and memory_growth < 50  # Less than 50MB growth
            
            logger.info(f"Stress test {'PASSED' if test_passed else 'FAILED'}")
            return test_passed, results
            
        except Exception as e:
            logger.error(f"Error during stress test: {e}")
            return False, None
            
def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test and benchmark real-time face recognition system")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, default="trained_hybrid_model.pkl", help="Path to trained model")
    parser.add_argument("--test-data", type=str, help="Path to test data directory")
    parser.add_argument("--video", type=str, help="Path to video file for benchmark (uses webcam if not specified)")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds for video benchmark")
    parser.add_argument("--no-display", action="store_true", help="Hide display during benchmark")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Directory for output results")
    parser.add_argument("--test-type", type=str, choices=["all", "benchmark", "accuracy", "stress"], 
                        default="all", help="Type of test to run")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize tester
        tester = SystemTester(
            config_file=args.config,
            model_path=args.model,
            test_data_path=args.test_data
        )
        
        # Run requested tests
        if args.test_type in ["all", "benchmark"]:
            logger.info("Running benchmark test")
            benchmark_results = tester.run_benchmark_on_video(
                video_path=args.video,
                duration=args.duration,
                show_display=not args.no_display
            )
            
            if benchmark_results:
                benchmark_output_dir = os.path.join(args.output_dir, "benchmark")
                tester.plot_benchmark_results(benchmark_results, benchmark_output_dir)
                
        if args.test_type in ["all", "accuracy"] and args.test_data:
            logger.info("Running accuracy test")
            accuracy_results = tester.run_accuracy_test(args.test_data)
            
            if accuracy_results:
                accuracy_output_dir = os.path.join(args.output_dir, "accuracy")
                tester.plot_accuracy_results(accuracy_results, accuracy_output_dir)
                
        if args.test_type in ["all", "stress"]:
            logger.info("Running stress test")
            stress_passed, stress_results = tester.run_stress_test(iterations=100)
            
            if stress_results:
                # Save stress test results
                with open(os.path.join(args.output_dir, "stress_test_results.txt"), "w") as f:
                    f.write("Stress Test Results\n")
                    f.write("==================\n\n")
                    f.write(f"Test {'PASSED' if stress_passed else 'FAILED'}\n\n")
                    for key, value in stress_results.items():
                        f.write(f"{key}: {value}\n")
        
        logger.info(f"All tests completed. Results saved to {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return 1
        
if __name__ == "__main__":
    sys.exit(main()) 