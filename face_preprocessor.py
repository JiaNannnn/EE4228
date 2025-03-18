import cv2
import numpy as np
import os
import math
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    print("Warning: dlib not available. Using basic face alignment.")
    DLIB_AVAILABLE = False

class FacePreprocessor:
    def __init__(self, target_size=(100, 100)):
        self.target_size = target_size
        # Load facial landmark detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize dlib face detector and shape predictor if available
        if DLIB_AVAILABLE:
            try:
                # Check if shape predictor file exists, if not we'll need to download it
                shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
                if not os.path.exists(shape_predictor_path):
                    print("Facial landmark predictor file not found. Will use basic alignment.")
                    self.shape_predictor = None
                else:
                    self.detector = dlib.get_frontal_face_detector()
                    self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
                    print("Using dlib for advanced face alignment")
            except Exception as e:
                print(f"Error initializing dlib: {e}")
                self.shape_predictor = None
        else:
            self.shape_predictor = None
        
        # Create histogram equalizer for more consistent illumination
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # For debug visualization
        self.debug_dir = "debug_preprocess"
        os.makedirs(self.debug_dir, exist_ok=True)
        self.debug_counter = 0
        self.save_debug_images = False  # Set to True to save preprocessor stages for debugging

    def save_debug_image(self, img, stage_name):
        """Save image for debugging"""
        if not self.save_debug_images:
            return
            
        filename = f"{self.debug_dir}/{self.debug_counter:04d}_{stage_name}.jpg"
        # Convert to uint8 if float
        if img.dtype != np.uint8:
            save_img = (img * 255).astype(np.uint8)
        else:
            save_img = img
        cv2.imwrite(filename, save_img)

    def align_face_with_landmarks(self, image):
        """Align face using facial landmarks (much more accurate than eye detection)"""
        if not DLIB_AVAILABLE or self.shape_predictor is None:
            return self.align_face(image)  # Fall back to basic alignment
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        try:
            # Detect faces using dlib
            dlib_faces = self.detector(gray, 1)
            
            if len(dlib_faces) > 0:
                # Get the largest face
                if len(dlib_faces) > 1:
                    areas = [(face.right() - face.left()) * (face.bottom() - face.top()) for face in dlib_faces]
                    largest_idx = np.argmax(areas)
                    face = dlib_faces[largest_idx]
                else:
                    face = dlib_faces[0]
                
                # Get facial landmarks
                landmarks = self.shape_predictor(gray, face)
                
                # Convert landmarks to numpy array
                landmarks_points = []
                for i in range(68):  # 68 landmarks
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    landmarks_points.append((x, y))
                
                # Left eye is the average of landmarks 36-41
                left_eye_pts = landmarks_points[36:42]
                left_eye_center = np.mean(left_eye_pts, axis=0).astype(int)
                
                # Right eye is the average of landmarks 42-47
                right_eye_pts = landmarks_points[42:48]
                right_eye_center = np.mean(right_eye_pts, axis=0).astype(int)
                
                # Calculate angle for alignment
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                
                if abs(dx) < 1:  # Avoid division by zero
                    angle = 0
                else:
                    angle = np.degrees(np.arctan2(dy, dx))
                
                # Limit extreme angles (probably misdetections)
                angle = min(15, max(-15, angle))
                
                # Additional reference points for scaling/cropping
                nose_tip = landmarks_points[30]  # Nose tip
                chin = landmarks_points[8]  # Chin
                left_face = landmarks_points[0]  # Left face boundary
                right_face = landmarks_points[16]  # Right face boundary
                
                # Calculate desired eye positions
                desired_left_eye = (0.35, 0.4)  # 35% from left, 40% from top
                desired_right_eye = (0.65, 0.4)  # 65% from left, 40% from top
                
                # Calculate scale to set eyes at fixed positions
                desired_dist = (desired_right_eye[0] - desired_left_eye[0]) * image.shape[1]
                actual_dist = np.sqrt((right_eye_center[0] - left_eye_center[0])**2 + 
                                    (right_eye_center[1] - left_eye_center[1])**2)
                scale = desired_dist / max(actual_dist, 1)  # Avoid division by zero
                
                # Calculate rotation matrix
                center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                        (left_eye_center[1] + right_eye_center[1]) // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
                
                # Update translation component of matrix
                tx = image.shape[1] * 0.5
                ty = image.shape[0] * desired_left_eye[1]
                rotation_matrix[0, 2] += (tx - center[0])
                rotation_matrix[1, 2] += (ty - center[1])
                
                # Apply transformation
                aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), 
                                            borderMode=cv2.BORDER_REPLICATE)
                
                self.save_debug_image(aligned_image, "landmark_aligned")
                return aligned_image
        except Exception as e:
            print(f"Error in landmark alignment: {e}")
        
        # Fall back to basic alignment if anything fails
        return self.align_face(image)

    def align_face(self, image):
        """Align face based on eye positions with better handling"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Try to detect eyes with multiple parameters to ensure consistency
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # If first attempt fails, try with more lenient parameters
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3)
            
        # If still not enough eyes, try with even more lenient parameters
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate to get left and right eye
            eyes = sorted(eyes, key=lambda x: x[0])
            
            # Filter eyes by y-coordinate to avoid misdetections (eyes should be roughly on the same level)
            y_values = [eye[1] for eye in eyes]
            median_y = np.median(y_values)
            valid_eyes = [eye for eye in eyes if abs(eye[1] - median_y) < gray.shape[0] * 0.15]  # Within 15% of height
            
            if len(valid_eyes) >= 2:
                left_eye, right_eye = valid_eyes[:2]
                
                # Calculate eye centers
                left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
                right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
                
                # Calculate angle to align eyes horizontally
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                
                # Protect against division by zero or very small values
                if abs(dx) < 1:
                    angle = 0
                else:
                    angle = np.degrees(np.arctan2(dy, dx))
                
                # Limit extreme angles (probably misdetections)
                angle = min(15, max(-15, angle))
                
                # Calculate desired eye positions
                desired_left_eye = (0.35, 0.4)  # 35% from left, 40% from top
                desired_right_eye = (0.65, 0.4)  # 65% from left, 40% from top
                
                # Calculate scale to set eyes at fixed positions
                desired_dist = (desired_right_eye[0] - desired_left_eye[0]) * gray.shape[1]
                actual_dist = np.sqrt((right_eye_center[0] - left_eye_center[0])**2 + 
                                    (right_eye_center[1] - left_eye_center[1])**2)
                scale = desired_dist / max(actual_dist, 1)  # Avoid division by zero
                
                # Calculate rotation matrix
                center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                        (left_eye_center[1] + right_eye_center[1]) // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
                
                # Update translation component of matrix
                tx = gray.shape[1] * 0.5
                ty = gray.shape[0] * desired_left_eye[1]
                rotation_matrix[0, 2] += (tx - center[0])
                rotation_matrix[1, 2] += (ty - center[1])
                
                # Rotate image
                aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), 
                                              borderMode=cv2.BORDER_REPLICATE)
                self.save_debug_image(aligned_image, "aligned")
                return aligned_image
        
        # If eye detection fails, return original image
        self.save_debug_image(image, "not_aligned")
        return image

    def enhance_image(self, image):
        """Enhance image quality with more robust methods"""
        # Make sure the image is uint8 before applying CLAHE
        if image.dtype != np.uint8:
            if np.max(image) <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
        else:
            image_uint8 = image
        
        # Apply histogram equalization for better contrast
        enhanced = self.clahe.apply(image_uint8)
        self.save_debug_image(enhanced, "clahe")
        
        # Apply bilateral filter to reduce noise while preserving edges
        # Use smaller values for more consistent results
        enhanced = cv2.bilateralFilter(enhanced, 5, 35, 35)  # Less aggressive filtering
        self.save_debug_image(enhanced, "bilateral")
        
        # Apply additional Gaussian blur to reduce high-frequency noise
        # This makes the face representation more stable
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        self.save_debug_image(enhanced, "gaussian")
        
        return enhanced

    def normalize_illumination(self, image):
        """Normalize illumination using simpler, more robust method"""
        # Convert to float32
        float_img = image.astype(np.float32)
        
        # Apply gamma correction first (helps with different lighting conditions)
        # Convert to range [0,1] first
        if np.max(float_img) > 1.0:
            float_img = float_img / 255.0
            
        gamma = 0.8
        gamma_corrected = np.power(float_img, gamma)
        self.save_debug_image(gamma_corrected, "gamma")
        
        # Simple global normalization - more stable than local normalization
        mean = np.mean(gamma_corrected)
        std = np.std(gamma_corrected) + 1e-6  # Avoid division by zero
        
        # Normalize globally
        normalized = (gamma_corrected - mean) / std
        
        # Clip values to avoid extreme outliers
        normalized = np.clip(normalized, -2.5, 2.5)  # Less aggressive clipping
        
        # Scale to [0, 1] range
        normalized = (normalized + 2.5) / 5.0
        
        # Handle any remaining NaN or Inf values
        normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
        
        self.save_debug_image(normalized, "normalized")
        return normalized

    def standardize_face(self, face):
        """Apply standard face cropping to ensure consistency"""
        h, w = face.shape[:2]
        
        # Define standard face landmarks (rough approximations)
        # These are ratios of the image dimensions
        eye_level = int(h * 0.35)  # Eyes at 35% from top
        mouth_level = int(h * 0.75)  # Mouth at 75% from top
        
        # Apply very simple face standardization
        # This helps when eye detection fails
        face_standardized = face.copy()
        
        # Center the face (assuming it's already roughly centered)
        # Apply contrast stretching to ensure consistent intensity range
        min_val = np.min(face_standardized)
        max_val = np.max(face_standardized)
        if max_val > min_val:  # Avoid division by zero
            # Make sure we have uint8 output for next steps
            if face_standardized.dtype != np.uint8:
                # FIX: Check for NaN or Inf values before type casting
                face_standardized_float = ((face_standardized - min_val) / (max_val - min_val)) * 255
                
                # Replace any NaN or Inf values before casting to uint8
                face_standardized_float = np.nan_to_num(face_standardized_float, nan=127, posinf=255, neginf=0)
                
                # Ensure values are in valid range for uint8
                face_standardized_float = np.clip(face_standardized_float, 0, 255)
                
                # Now it's safe to cast to uint8
                face_standardized = face_standardized_float.astype(np.uint8)
            else:
                # Similarly handle the case when it's already uint8
                face_standardized_float = ((face_standardized.astype(float) - min_val) / (max_val - min_val)) * 255
                face_standardized_float = np.nan_to_num(face_standardized_float, nan=127, posinf=255, neginf=0)
                face_standardized_float = np.clip(face_standardized_float, 0, 255)
                face_standardized = face_standardized_float.astype(np.uint8)
        
        self.save_debug_image(face_standardized, "standardized")
        return face_standardized

    def preprocess(self, image):
        """
        Preprocess face image with more consistent preprocessing
        """
        self.debug_counter += 1
        
        try:
            # Input validation
            if image is None or image.size == 0:
                print("Empty or invalid image input")
                return np.zeros(self.target_size[::-1], dtype=np.float32)
                
            # Check for NaN values in input
            if np.isnan(image).any():
                print("Warning: NaN values in input image. Replacing with zeros.")
                image = np.nan_to_num(image, nan=0.0)

            # Also check for inf values
            if np.isinf(image).any():
                print("Warning: Inf values in input image. Replacing with appropriate values.")
                image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
                
            # Save original image for debugging
            self.save_debug_image(image, "original")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()  # Make a copy to avoid modifying original
            
            self.save_debug_image(gray, "grayscale")
                
            # Make sure we're working with uint8 images for OpenCV functions
            if gray.dtype != np.uint8:
                if np.max(gray) <= 1.0:
                    # FIXED: Handle NaN/Inf values before type conversion
                    gray_float = gray * 255.0
                    gray_float = np.nan_to_num(gray_float, nan=127.0, posinf=255.0, neginf=0.0)
                    gray_float = np.clip(gray_float, 0, 255)
                    gray = gray_float.astype(np.uint8)
                else:
                    # FIXED: Check for NaN/Inf values
                    gray = np.nan_to_num(gray, nan=127.0, posinf=255.0, neginf=0.0)
                    gray = np.clip(gray, 0, 255)
                    gray = gray.astype(np.uint8)
            
            # Resize first to make eye detection more consistent
            initial_resize = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_CUBIC)
            self.save_debug_image(initial_resize, "initial_resize")
            
            # Try to align face using landmarks (falls back to basic alignment if needed)
            aligned = self.align_face_with_landmarks(initial_resize)
            
            # Resize to target size
            resized = cv2.resize(aligned, self.target_size, interpolation=cv2.INTER_CUBIC)
            self.save_debug_image(resized, "resized")
            
            # Explicitly ensure it's uint8 for CLAHE
            if resized.dtype != np.uint8:
                # FIXED: Check for NaN/Inf values
                resized = np.nan_to_num(resized, nan=127.0, posinf=255.0, neginf=0.0)
                resized = np.clip(resized, 0, 255)
                resized = resized.astype(np.uint8)
                
            # Apply image enhancement
            enhanced = self.enhance_image(resized)
            
            # Apply illumination normalization
            normalized = self.normalize_illumination(enhanced)
            
            # Apply face standardization as a final step
            standardized = self.standardize_face(normalized)
            
            return standardized
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            # Return zeros in case of error
            return np.zeros(self.target_size, dtype=np.float32) 