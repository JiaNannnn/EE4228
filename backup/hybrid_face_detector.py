"""
Hybrid Face Detector

Combines Viola-Jones and LBP detectors for improved detection
of both frontal and profile faces, including multi-scale detection.
"""

import cv2
import numpy as np
from collections import deque
import time
import logging
from config import DETECTION_CONFIG

logger = logging.getLogger("hybrid_face_detector")

class HybridFaceDetector:
    """
    A hybrid face detector that combines Viola-Jones (Haar) and LBP classifiers
    for improved face detection in various poses.
    """
    
    def __init__(self, 
                 min_face_size=DETECTION_CONFIG["min_face_size"],
                 scale_factor=DETECTION_CONFIG["scale_factor"],
                 min_neighbors=DETECTION_CONFIG["min_neighbors"],
                 scale_pyramid=DETECTION_CONFIG["scale_pyramid"],
                 confidence_threshold=DETECTION_CONFIG["confidence_threshold"]):
        """
        Initialize the hybrid face detector
        
        Parameters:
        -----------
        min_face_size : tuple
            Minimum face size to detect (width, height)
        scale_factor : float
            Scale factor for the detection algorithm
        min_neighbors : int
            Minimum number of neighbors for detection
        scale_pyramid : list
            List of scales to use for multi-scale detection
        confidence_threshold : float
            Confidence threshold for detection
        """
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.scale_pyramid = scale_pyramid
        self.confidence_threshold = confidence_threshold
        
        # Initialize detectors
        self.frontal_detector = cv2.CascadeClassifier(DETECTION_CONFIG["frontal_cascade"])
        self.profile_detector = cv2.CascadeClassifier(DETECTION_CONFIG["profile_cascade"])
        self.lbp_detector = cv2.CascadeClassifier(DETECTION_CONFIG["lbp_cascade"])
        
        # Check if classifiers loaded successfully
        if self.frontal_detector.empty():
            logger.error(f"Error: Failed to load frontal face cascade from {DETECTION_CONFIG['frontal_cascade']}")
            raise ValueError(f"Failed to load frontal face cascade")
            
        if self.profile_detector.empty():
            logger.warning(f"Warning: Failed to load profile face cascade from {DETECTION_CONFIG['profile_cascade']}")
            
        if self.lbp_detector.empty():
            logger.warning(f"Warning: Failed to load LBP face cascade from {DETECTION_CONFIG['lbp_cascade']}")
            
        # Face detection history for temporal smoothing
        self.detection_history = deque(maxlen=5)
        
    def detect_faces(self, frame):
        """
        Detect faces in the input frame using the hybrid approach
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input image/frame
        
        Returns:
        --------
        list
            List of (x, y, w, h, confidence) tuples for detected faces
        """
        if frame is None:
            logger.warning("Received None frame for face detection")
            return []
            
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        all_faces = []
        
        # Multi-scale detection
        for scale in self.scale_pyramid:
            if scale != 1.0:
                scaled_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
            else:
                scaled_gray = gray
                
            # Apply brightness and contrast normalization to improve detection
            normalized_gray = self._normalize_brightness_contrast(scaled_gray)
            
            # Detect with Viola-Jones frontal
            frontal_faces = self.frontal_detector.detectMultiScale(
                normalized_gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Add frontal faces with confidence and scale adjustment
            for (x, y, w, h) in frontal_faces:
                # Adjust coordinates back to original scale
                if scale != 1.0:
                    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                all_faces.append((x, y, w, h, 0.9))  # High confidence for frontal
            
            # Detect with LBP (faster but less accurate)
            if not self.lbp_detector.empty():
                lbp_faces = self.lbp_detector.detectMultiScale(
                    normalized_gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors - 1,  # More permissive
                    minSize=self.min_face_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Add LBP faces with confidence and scale adjustment
                for (x, y, w, h) in lbp_faces:
                    # Adjust coordinates back to original scale
                    if scale != 1.0:
                        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    all_faces.append((x, y, w, h, 0.8))  # Medium confidence for LBP
            
            # Detect profile faces
            if not self.profile_detector.empty():
                # Detect right profiles
                right_profile_faces = self.profile_detector.detectMultiScale(
                    normalized_gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors - 2,  # Even more permissive for profiles
                    minSize=self.min_face_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Detect left profiles (flip image)
                flipped_gray = cv2.flip(normalized_gray, 1)
                left_profile_faces = self.profile_detector.detectMultiScale(
                    flipped_gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors - 2,
                    minSize=self.min_face_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Adjust left profile coordinates
                width = normalized_gray.shape[1]
                left_profile_faces = [(width - x - w, y, w, h) for (x, y, w, h) in left_profile_faces]
                
                # Add profile faces with confidence and scale adjustment
                for (x, y, w, h) in right_profile_faces:
                    # Adjust coordinates back to original scale
                    if scale != 1.0:
                        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    all_faces.append((x, y, w, h, 0.7))  # Lower confidence for profiles
                    
                for (x, y, w, h) in left_profile_faces:
                    # Adjust coordinates back to original scale
                    if scale != 1.0:
                        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    all_faces.append((x, y, w, h, 0.7))  # Lower confidence for profiles
        
        # Apply non-maximum suppression to remove overlapping detections
        filtered_faces = self._apply_nms(all_faces)
        
        # Update detection history for temporal smoothing
        self.detection_history.append(filtered_faces)
        
        # Apply temporal smoothing
        smoothed_faces = self._apply_temporal_smoothing()
        
        # Filter by confidence threshold
        final_faces = [face for face in smoothed_faces if face[4] >= self.confidence_threshold]
        
        detection_time = time.time() - start_time
        if detection_time > 0.1:  # Log only if detection takes longer than 100ms
            logger.debug(f"Face detection took {detection_time*1000:.2f}ms, found {len(final_faces)} faces")
        
        return final_faces
    
    def _normalize_brightness_contrast(self, image):
        """
        Normalize brightness and contrast to improve detection
        """
        # CLAHE for adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _apply_nms(self, faces, overlap_threshold=0.3):
        """
        Apply non-maximum suppression to remove overlapping detections
        """
        if len(faces) == 0:
            return []
            
        # Convert to numpy arrays
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h, _) in faces])
        confidences = np.array([conf for (_, _, _, _, conf) in faces])
        
        # Compute areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by confidence
        idxs = np.argsort(confidences)
        
        pick = []
        while len(idxs) > 0:
            # Pick the face with highest confidence
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Compute overlap with all other boxes
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[:last], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[:last], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[:last], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[:last], 3])
            
            # Compute overlap area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / areas[idxs[:last]]
            
            # Remove overlapping boxes
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
        
        # Return filtered faces
        return [(int(boxes[i][0]), int(boxes[i][1]), 
                 int(boxes[i][2] - boxes[i][0]), int(boxes[i][3] - boxes[i][1]), 
                 confidences[i]) for i in pick]
    
    def _apply_temporal_smoothing(self):
        """
        Apply temporal smoothing to face detections
        """
        if len(self.detection_history) < 2:
            return self.detection_history[-1] if self.detection_history else []
            
        # Get current and previous detections
        current_faces = self.detection_history[-1]
        
        # If no current faces, return empty list
        if not current_faces:
            return []
            
        # If only one frame in history, return current faces
        if len(self.detection_history) == 1:
            return current_faces
            
        # Apply smoothing to reduce jitter
        smoothed_faces = []
        
        # Weighting factors for temporal smoothing (current detection has higher weight)
        alpha = 0.7  # Weight for current detection
        
        for i, (x, y, w, h, conf) in enumerate(current_faces):
            # Find matching faces in previous frames
            matched_faces = []
            
            for prev_faces in list(self.detection_history)[:-1]:
                best_match = None
                best_iou = 0
                
                for prev_x, prev_y, prev_w, prev_h, prev_conf in prev_faces:
                    # Compute intersection over union
                    x_intersect = max(0, min(x + w, prev_x + prev_w) - max(x, prev_x))
                    y_intersect = max(0, min(y + h, prev_y + prev_h) - max(y, prev_y))
                    intersection = x_intersect * y_intersect
                    union = w * h + prev_w * prev_h - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.5 and iou > best_iou:  # Match threshold
                        best_match = (prev_x, prev_y, prev_w, prev_h, prev_conf)
                        best_iou = iou
                
                if best_match:
                    matched_faces.append(best_match)
            
            # If matches found, apply smoothing
            if matched_faces:
                # Calculate weighted average of coordinates
                sum_x, sum_y, sum_w, sum_h, sum_conf = 0, 0, 0, 0, 0
                total_weight = alpha
                
                # Current detection (higher weight)
                sum_x += alpha * x
                sum_y += alpha * y
                sum_w += alpha * w
                sum_h += alpha * h
                sum_conf += alpha * conf
                
                # Previous detections (distribute remaining weight evenly)
                beta = (1 - alpha) / len(matched_faces) if matched_faces else 0
                
                for prev_x, prev_y, prev_w, prev_h, prev_conf in matched_faces:
                    sum_x += beta * prev_x
                    sum_y += beta * prev_y
                    sum_w += beta * prev_w
                    sum_h += beta * prev_h
                    sum_conf += beta * prev_conf
                    total_weight += beta
                
                # Normalize by total weight
                smooth_x = int(sum_x / total_weight)
                smooth_y = int(sum_y / total_weight)
                smooth_w = int(sum_w / total_weight)
                smooth_h = int(sum_h / total_weight)
                smooth_conf = sum_conf / total_weight
                
                smoothed_faces.append((smooth_x, smooth_y, smooth_w, smooth_h, smooth_conf))
            else:
                # No matches, use current detection
                smoothed_faces.append((x, y, w, h, conf))
        
        return smoothed_faces
    
    def get_face_roi(self, frame, face_rect):
        """
        Extract face ROI from frame using detected rectangle
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input image/frame
        face_rect : tuple
            (x, y, w, h, conf) or (x, y, w, h) face rectangle
        
        Returns:
        --------
        numpy.ndarray
            Extracted face ROI
        """
        if len(face_rect) >= 4:
            x, y, w, h = face_rect[:4]
            # Add a margin to the face ROI (20% on each side)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            # Calculate expanded ROI with margins
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            
            return frame[y1:y2, x1:x2]
        else:
            logger.warning(f"Invalid face rectangle: {face_rect}")
            return None 