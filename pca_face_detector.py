import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

class PCAFaceDetector:
    def __init__(self, n_components=0.95, target_size=(100, 100)):
        self.n_components = n_components
        self.target_size = target_size
        self.pca = PCA(n_components=self.n_components)
        self.scaler = StandardScaler()
        self.threshold = None
        self.trained = False
        self.mean_face = None
        
    def _prepare_data(self, X):
        """Prepare image data for PCA"""
        if len(X.shape) == 3:  # Multiple images
            n_samples, height, width = X.shape
            return X.reshape(n_samples, height * width)
        elif len(X.shape) == 2:  # Single image
            height, width = X.shape
            return X.reshape(1, height * width)
            
    def train(self, face_images, non_face_images):
        """Train PCA face detector with face and non-face images"""
        try:
            # Prepare face data
            X_faces = self._prepare_data(face_images)
            X_non_faces = self._prepare_data(non_face_images)
            
            # Check for NaN values
            if np.isnan(X_faces).any():
                print("Warning: NaN values detected in face images. Replacing with zeros.")
                X_faces = np.nan_to_num(X_faces, nan=0.0)
                
            if np.isnan(X_non_faces).any():
                print("Warning: NaN values detected in non-face images. Replacing with zeros.")
                X_non_faces = np.nan_to_num(X_non_faces, nan=0.0)
            
            # Combine datasets
            X = np.vstack([X_faces, X_non_faces])
            y = np.hstack([np.ones(len(X_faces)), np.zeros(len(X_non_faces))])
            
            # Scale data
            X_scaled = self.scaler.fit_transform(X)
            
            # Check for NaN values after scaling
            if np.isnan(X_scaled).any():
                print("Warning: NaN values detected after scaling. Replacing with zeros.")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            
            # Train PCA
            self.pca.fit(X_scaled)
            
            # Project data to PCA space
            X_pca = self.pca.transform(X_scaled)
            
            # Compute reconstruction errors
            X_rec = self.pca.inverse_transform(X_pca)
            rec_errors = np.mean(np.square(X_scaled - X_rec), axis=1)
            
            # Compute threshold using face samples
            face_errors = rec_errors[:len(X_faces)]
            self.threshold = np.mean(face_errors) + 2 * np.std(face_errors)
            
            # Store mean face
            self.mean_face = np.mean(X_faces, axis=0).reshape(self.target_size)
            
            self.trained = True
            return True
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False
            
    def detect_faces(self, image, min_size=(30, 30), scale_factor=1.1, stride_factor=0.1):
        """
        Detect faces in image using sliding window and PCA reconstruction error
        Returns: List of (x, y, w, h) face rectangles
        """
        if not self.trained:
            raise ValueError("Detector not trained yet!")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = []
        
        # Multi-scale detection
        current_scale = 1.0
        while True:
            # Calculate current window size
            window_w = int(self.target_size[1] * current_scale)
            window_h = int(self.target_size[0] * current_scale)
            
            # Break if window is too large
            if window_w > image.shape[1] or window_h > image.shape[0]:
                break
                
            # Calculate stride
            stride_w = max(1, int(window_w * stride_factor))
            stride_h = max(1, int(window_h * stride_factor))
            
            # Slide window over image
            for y in range(0, image.shape[0] - window_h, stride_h):
                for x in range(0, image.shape[1] - window_w, stride_w):
                    # Extract window
                    window = gray[y:y+window_h, x:x+window_w]
                    
                    # Resize window to target size
                    window_resized = cv2.resize(window, self.target_size)
                    
                    try:
                        # Prepare for PCA
                        X = self._prepare_data(window_resized)
                        
                        # Check for NaN values
                        if np.isnan(X).any():
                            continue  # Skip this window
                            
                        X_scaled = self.scaler.transform(X)
                        
                        # Check for NaN values after scaling
                        if np.isnan(X_scaled).any():
                            continue  # Skip this window
                        
                        # Project and reconstruct
                        X_pca = self.pca.transform(X_scaled)
                        
                        # Check for NaN values after PCA
                        if np.isnan(X_pca).any():
                            continue  # Skip this window
                            
                        X_rec = self.pca.inverse_transform(X_pca)
                        
                        # Compute reconstruction error
                        rec_error = np.mean(np.square(X_scaled - X_rec))
                        
                        # If error is below threshold, it's a face
                        if rec_error < self.threshold:
                            faces.append((x, y, window_w, window_h))
                    except Exception as e:
                        # Skip this window if any error occurs
                        continue
            
            # Increase scale
            current_scale *= scale_factor
            
        # Apply non-maximum suppression
        faces = self._non_max_suppression(faces)
        return faces
        
    def _non_max_suppression(self, boxes, overlap_thresh=0.3):
        """Apply non-maximum suppression to remove overlapping detections"""
        if len(boxes) == 0:
            return []
            
        boxes = np.array(boxes)
        pick = []
        
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,0] + boxes[:,2]
        y2 = boxes[:,1] + boxes[:,3]
        
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
        return boxes[pick].astype("int").tolist()
        
    def save_model(self, model_path):
        """Save trained model"""
        if not self.trained:
            raise ValueError("Model not trained yet!")
            
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'mean_face': self.mean_face,
            'target_size': self.target_size
        }
        joblib.dump(model_data, model_path)
        
    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
            
        model_data = joblib.load(model_path)
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.mean_face = model_data['mean_face']
        self.target_size = model_data['target_size']
        self.trained = True 