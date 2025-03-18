"""
Face Preprocessor for Face Recognition

A clean implementation of face preprocessing techniques for recognition:
1. Color to grayscale conversion
2. Illumination normalization 
3. Face detection and cropping
4. Rotation alignment using eye detection
5. Image enhancement
6. Scale normalization
"""

import os
import cv2
import numpy as np
import logging
import math
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_preprocessor")

class FacePreprocessor:
    """
    Class for preprocessing face images with normalization
    and enhancement techniques for face recognition
    """
    
    def __init__(self, target_size=(224, 224), force_grayscale=True):
        """
        Initialize the face preprocessor
        
        Parameters:
        -----------
        target_size : tuple
            Output size for normalized face images (width, height)
        force_grayscale : bool
            Whether to always convert images to grayscale (recommended for recognition)
        """
        self.target_size = target_size
        self.force_grayscale = force_grayscale
        
        # Initialize Haar cascade face detector
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if self.face_cascade.empty():
            logger.warning("Warning: Face cascade file not found or invalid")
        
        # Initialize eye detector for rotation normalization
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if self.eye_cascade.empty():
            logger.warning("Warning: Eye cascade file not found or invalid")
        
        # Initialize HOG detector
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("HOG detector initialized")
        
        # Initialize DNN detector if model files are available
        try:
            os.makedirs('models', exist_ok=True)
            prototxt_path = 'models/deploy.prototxt'
            caffemodel_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
            
            if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                self.dnn_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                logger.info("DNN face detector loaded successfully")
            else:
                logger.warning("DNN model files not found. DNN detector will not be used.")
                self.dnn_detector = None
        except Exception as e:
            logger.warning(f"Error initializing DNN detector: {str(e)}")
            self.dnn_detector = None
        
        logger.info(f"Face preprocessor initialized with target_size={target_size}, force_grayscale={force_grayscale}")
    
    def detect_face(self, image):
        """
        Detect the largest face in the image using multiple detection methods
        Returns: (x, y, w, h) or None if no face detected
        """
        # Ensure grayscale for Haar
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = []
        
        # Try Haar cascade with multiple scale factors for better results
        for scale in [1.05, 1.1, 1.15, 1.2]:
            faces_haar = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=5,
                minSize=(30, 30)
            )
            if len(faces_haar) > 0:
                for face in faces_haar:
                    faces.append(tuple(face))  # Convert to tuple to avoid numpy array issues
        
        # Try HOG detector if no faces found with Haar
        if len(faces) == 0:
            # HOG works better with full image rather than just grayscale
            if len(image.shape) == 2:
                img_hog = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                img_hog = image
                
            # Resize for better HOG detection if image is too large
            original_dims = img_hog.shape[:2]
            resize_factor = 1.0
            if original_dims[0] > 400 or original_dims[1] > 400:
                resize_factor = 400 / max(original_dims)
                img_hog = cv2.resize(img_hog, None, fx=resize_factor, fy=resize_factor)
            
            # Detect with HOG
            hog_detections, _ = self.hog_detector.detectMultiScale(
                img_hog, 
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.05
            )
            
            # Convert HOG detections to same format as Haar
            for (x, y, w, h) in hog_detections:
                # Scale back to original size if resized
                if resize_factor != 1.0:
                    x, y, w, h = int(x/resize_factor), int(y/resize_factor), int(w/resize_factor), int(h/resize_factor)
                faces.append((x, y, w, h))
        
        # Try DNN detector if still no faces found
        if len(faces) == 0 and hasattr(self, 'dnn_detector') and self.dnn_detector is not None:
            # Prepare image for DNN
            if len(image.shape) == 2:
                img_dnn = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                img_dnn = image
            
            # DNN detection requires specific input
            blob = cv2.dnn.blobFromImage(img_dnn, 1.0, (300, 300), [104, 117, 123], False, False)
            self.dnn_detector.setInput(blob)
            detections = self.dnn_detector.forward()
            
            # Process DNN detections
            img_height, img_width = img_dnn.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # Confidence threshold
                    x1 = int(detections[0, 0, i, 3] * img_width)
                    y1 = int(detections[0, 0, i, 4] * img_height)
                    x2 = int(detections[0, 0, i, 5] * img_width)
                    y2 = int(detections[0, 0, i, 6] * img_height)
                    
                    # Ensure coordinates are within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_width, x2), min(img_height, y2)
                    
                    # Add to faces list
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:
                        faces.append((x1, y1, w, h))
        
        # Return largest face if any found
        if len(faces) > 0:
            # Find the largest face by area (w*h)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return largest_face
        else:
            logger.warning("No face detected in image")
            return None
    
    def detect_eyes(self, face_image):
        """
        Detect eyes in a face image for rotation normalization
        
        Args:
            face_image: Face region image
            
        Returns:
            List of eye rectangles (x, y, w, h) or empty list if no eyes detected
        """
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Apply histogram equalization to improve eye detection
        gray = cv2.equalizeHist(gray)
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        return eyes
    
    def align_face(self, image, face_rect):
        """
        Align face based on eye positions
        
        Args:
            image: Input image
            face_rect: Face rectangle (x, y, w, h)
            
        Returns:
            Aligned face image
        """
        # Extract face region
        x, y, w, h = face_rect
        face_img = image[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = self.detect_eyes(face_img)
        
        # If we found at least two eyes, align the face
        if len(eyes) >= 2:
            try:
                # Sort eyes by x-coordinate (left to right)
                sorted_eyes = sorted(eyes, key=lambda e: e[0])
                
                # If we have more than 2 eyes detected, take the first two
                left_eye, right_eye = sorted_eyes[0], sorted_eyes[1]
                
                # Get eye centers
                left_eye_center = (int(left_eye[0] + left_eye[2] // 2), int(left_eye[1] + left_eye[3] // 2))
                right_eye_center = (int(right_eye[0] + right_eye[2] // 2), int(right_eye[1] + right_eye[3] // 2))
                
                # Calculate angle between eye centers
                dx = right_eye_center[0] - left_eye_center[0]
                dy = right_eye_center[1] - left_eye_center[1]
                angle = math.degrees(math.atan2(dy, dx))
                
                # Get center of rotation
                center_x = int((left_eye_center[0] + right_eye_center[0]) // 2)
                center_y = int((left_eye_center[1] + right_eye_center[1]) // 2)
                
                # Get rotation matrix
                M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
                
                # Apply rotation
                aligned_face = cv2.warpAffine(face_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                logger.info("Face aligned using eye detection")
                return aligned_face
            except Exception as e:
                logger.warning(f"Error during face alignment: {str(e)}")
                logger.info("Face alignment failed, returning original face")
                return face_img
        else:
            logger.info("Not enough eyes detected for alignment, returning original face")
            return face_img
    
    def normalize_illumination(self, image, method='clahe'):
        """
        Apply illumination normalization to the image
        
        Args:
            image: Input image (grayscale)
            method: Normalization method ('histogram', 'clahe', 'gamma', 'dog', 'combined')
            
        Returns:
            Illumination normalized image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply selected normalization method
        if method == 'histogram':
            # Simple histogram equalization
            normalized = cv2.equalizeHist(gray)
        
        elif method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(gray)
        
        elif method == 'gamma':
            # Gamma correction
            # Estimate optimal gamma value based on image mean
            mean = np.mean(gray) / 255.0
            gamma = math.log(0.5) / math.log(mean)
            gamma = min(max(gamma, 0.5), 2.0)  # Limit gamma to reasonable range
            
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            normalized = cv2.LUT(gray, table)
        
        elif method == 'dog':
            # Difference of Gaussians
            g1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
            g2 = cv2.GaussianBlur(gray, (9, 9), 2.0)
            normalized = cv2.absdiff(g1, g2)
            # Normalize to 0-255 range
            normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        elif method == 'combined':
            # Combine multiple methods for best results
            # First apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
            
            # Then apply DoG to enhance edges
            g1 = cv2.GaussianBlur(clahe_img, (3, 3), 0.5)
            g2 = cv2.GaussianBlur(clahe_img, (7, 7), 1.5)
            dog_img = cv2.absdiff(g1, g2)
            
            # Normalize and blend with CLAHE result
            dog_normalized = cv2.normalize(dog_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            normalized = cv2.addWeighted(clahe_img, 0.8, dog_normalized, 0.2, 0)
        
        else:
            # Default to CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(gray)
        
        return normalized
    
    def enhance_image(self, image, method='adaptive'):
        """
        Enhance image quality using various techniques
        
        Args:
            image: Input image (grayscale)
            method: Enhancement method ('basic', 'denoising', 'sharpening', 'adaptive')
            
        Returns:
            Enhanced image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply selected enhancement method
        if method == 'basic':
            # Basic enhancement with histogram equalization
            enhanced = cv2.equalizeHist(gray)
        
        elif method == 'denoising':
            # Apply non-local means denoising
            enhanced = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            # Then apply histogram equalization
            enhanced = cv2.equalizeHist(enhanced)
        
        elif method == 'sharpening':
            # Apply sharpening filter
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            enhanced = cv2.filter2D(gray, -1, kernel)
            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        elif method == 'adaptive':
            # Adaptive enhancement based on image properties
            # First measure image properties
            mean_val = np.mean(gray)
            std_dev = np.std(gray)
            
            # Apply different enhancement strategies based on image quality
            if std_dev < 30:  # Low contrast image
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Apply mild sharpening
                kernel = np.array([[-0.5, -0.5, -0.5],
                                  [-0.5,  5.0, -0.5],
                                  [-0.5, -0.5, -0.5]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            elif mean_val < 80:  # Dark image
                # Increase brightness with gamma correction
                gamma = 0.7
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                enhanced = cv2.LUT(gray, table)
                
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(enhanced)
            elif mean_val > 180:  # Bright image
                # Reduce brightness with gamma correction
                gamma = 1.3
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                enhanced = cv2.LUT(gray, table)
                
                # Apply denoising
                enhanced = cv2.fastNlMeansDenoising(enhanced, None, 5, 7, 21)
            else:  # Normal image
                # Apply denoising
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                
                # Apply mild sharpening
                kernel = np.array([[-0.5, -0.5, -0.5],
                                  [-0.5,  5.0, -0.5],
                                  [-0.5, -0.5, -0.5]])
                enhanced = cv2.filter2D(denoised, -1, kernel)
            
            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        else:
            # Default to basic enhancement
            enhanced = cv2.equalizeHist(gray)
        
        return enhanced
    
    def resize_image(self, image, preserve_aspect_ratio=True):
        """
        Resize image to target size with optional aspect ratio preservation
        
        Args:
            image: Input image
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        if preserve_aspect_ratio:
            # Calculate target size while preserving aspect ratio
            if w/h > target_w/target_h:
                # Image is wider than target, fit to width
                new_w = target_w
                new_h = int(h * (target_w / w))
            else:
                # Image is taller than target, fit to height
                new_h = target_h
                new_w = int(w * (target_h / h))
            
            # Resize with aspect ratio preserved
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Create target-sized image with padding
            result = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Calculate position to paste the resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Paste resized image onto target-sized canvas
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return result
        else:
            # Simple resize to target dimensions
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
    
    def preprocess_image(self, image, save_path=None, return_intermediates=False,
                         illumination_method='clahe', enhance_method='adaptive',
                         preserve_aspect_ratio=True):
        """
        Preprocess an image for face recognition:
        1. Convert to grayscale
        2. Detect face
        3. Crop to face region with margin
        4. Align face using eye detection
        5. Normalize illumination
        6. Enhance image quality
        7. Normalize scale (resize to target size)
        
        Args:
            image: Input image or path to image
            save_path: Path to save the preprocessed image (optional)
            return_intermediates: Whether to return intermediate results
            illumination_method: Method for illumination normalization
            enhance_method: Method for image quality enhancement
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            
        Returns:
            Preprocessed face image and optionally intermediate results
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                logger.error(f"Failed to load image from {image}")
                return None
        
        # Store intermediate results if requested
        intermediates = {}
        if return_intermediates:
            intermediates['original'] = image.copy()
        
        # Step 1: Convert to grayscale if needed/requested
        if len(image.shape) == 3 and self.force_grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if return_intermediates:
                intermediates['grayscale'] = gray.copy()
        else:
            gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Detect face
        face_rect = self.detect_face(image)
        if face_rect is None:
            logger.warning("No face detected in image")
            # If no face detected, process the whole image
            normalized = self.normalize_illumination(gray, method=illumination_method)
            enhanced = self.enhance_image(normalized, method=enhance_method)
            resized = self.resize_image(enhanced, preserve_aspect_ratio=preserve_aspect_ratio)
            
            if return_intermediates:
                intermediates['final'] = resized.copy()
            
            # Save result if requested
            if save_path:
                cv2.imwrite(save_path, resized)
                logger.info(f"Saved processed image (no face) to {save_path}")
            
            if return_intermediates:
                return resized, intermediates
            return resized
        
        # Mark face in original image if returning intermediates
        x, y, w, h = face_rect
        if return_intermediates:
            face_marked = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(face_marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
            intermediates['face_detected'] = face_marked
        
        # Step 3: Crop to face region with margin
        margin = 0.3  # 30% margin around face
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Ensure coordinates are within image bounds
        img_h, img_w = image.shape[:2]
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(img_w, x + w + margin_x)
        bottom = min(img_h, y + h + margin_y)
        
        face_region = image[top:bottom, left:right]
        if return_intermediates:
            intermediates['face_cropped'] = face_region.copy()
        
        # Step 4: Align face using eye detection
        # Get face rect within the cropped region
        local_face_rect = (x - left, y - top, w, h)
        aligned_face = self.align_face(face_region, local_face_rect)
        
        # Convert to grayscale if not already and if forcing grayscale
        if len(aligned_face.shape) == 3 and self.force_grayscale:
            aligned_face_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        else:
            aligned_face_gray = aligned_face.copy() if len(aligned_face.shape) == 2 else cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        
        if return_intermediates:
            intermediates['aligned'] = aligned_face_gray.copy()
        
        # Step 5: Normalize illumination
        normalized = self.normalize_illumination(aligned_face_gray, method=illumination_method)
        if return_intermediates:
            intermediates['illumination_normalized'] = normalized.copy()
        
        # Step 6: Enhance image quality
        enhanced = self.enhance_image(normalized, method=enhance_method)
        if return_intermediates:
            intermediates['enhanced'] = enhanced.copy()
        
        # Step 7: Normalize scale (resize to target size)
        resized = self.resize_image(enhanced, preserve_aspect_ratio=preserve_aspect_ratio)
        if return_intermediates:
            intermediates['final'] = resized.copy()
        
        # Save result if requested
        if save_path:
            cv2.imwrite(save_path, resized)
            logger.info(f"Saved preprocessed face image to {save_path}")
        
        if return_intermediates:
            return resized, intermediates
        return resized


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess face images for recognition")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--target-size", type=str, default="224,224", help="Target size (width,height)")
    parser.add_argument("--illumination", type=str, default="clahe", 
                        choices=["histogram", "clahe", "gamma", "dog", "combined"],
                        help="Illumination normalization method")
    parser.add_argument("--enhance", type=str, default="adaptive",
                        choices=["basic", "denoising", "sharpening", "adaptive"],
                        help="Image enhancement method")
    parser.add_argument("--color", action="store_true", help="Keep color information (default: convert to grayscale)")
    parser.add_argument("--preserve-ratio", action="store_true", help="Preserve aspect ratio when resizing")
    parser.add_argument("--visualize", action="store_true", help="Save visualization of intermediate steps")
    
    args = parser.parse_args()
    
    # Parse target size
    target_width, target_height = map(int, args.target_size.split(','))
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(
        target_size=(target_width, target_height),
        force_grayscale=not args.color
    )
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process input (single image or directory)
    if os.path.isdir(args.input):
        # Process all images in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(Path(args.input).glob(f"*{ext}")))
            image_paths.extend(list(Path(args.input).glob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(image_paths)} images in {args.input}")
        
        for img_path in image_paths:
            output_path = os.path.join(args.output, f"{img_path.stem}_processed{img_path.suffix}")
            
            try:
                if args.visualize:
                    # Process with visualization
                    _, intermediates = preprocessor.preprocess_image(
                        str(img_path),
                        save_path=output_path,
                        return_intermediates=True,
                        illumination_method=args.illumination,
                        enhance_method=args.enhance,
                        preserve_aspect_ratio=args.preserve_ratio
                    )
                    
                    # Save visualization
                    vis_path = os.path.join(args.output, f"{img_path.stem}_visualization.jpg")
                    
                    # Create visualization image
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 8))
                    
                    # Determine grid size
                    n_plots = len(intermediates)
                    cols = min(3, n_plots)
                    rows = (n_plots + cols - 1) // cols
                    
                    # Plot each intermediate result
                    for i, (title, img) in enumerate(intermediates.items()):
                        plt.subplot(rows, cols, i+1)
                        
                        # Handle different image types
                        if len(img.shape) == 3:
                            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        else:
                            plt.imshow(img, cmap='gray')
                        
                        plt.title(title)
                        plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(vis_path)
                    plt.close()
                    logger.info(f"Saved visualization to {vis_path}")
                else:
                    # Process without visualization
                    preprocessor.preprocess_image(
                        str(img_path),
                        save_path=output_path,
                        illumination_method=args.illumination,
                        enhance_method=args.enhance,
                        preserve_aspect_ratio=args.preserve_ratio
                    )
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
    else:
        # Process single image
        output_path = os.path.join(args.output, f"processed.jpg")
        
        try:
            if args.visualize:
                # Process with visualization
                _, intermediates = preprocessor.preprocess_image(
                    args.input,
                    save_path=output_path,
                    return_intermediates=True,
                    illumination_method=args.illumination,
                    enhance_method=args.enhance,
                    preserve_aspect_ratio=args.preserve_ratio
                )
                
                # Save visualization
                vis_path = os.path.join(args.output, "visualization.jpg")
                
                # Create visualization image
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))
                
                # Determine grid size
                n_plots = len(intermediates)
                cols = min(3, n_plots)
                rows = (n_plots + cols - 1) // cols
                
                # Plot each intermediate result
                for i, (title, img) in enumerate(intermediates.items()):
                    plt.subplot(rows, cols, i+1)
                    
                    # Handle different image types
                    if len(img.shape) == 3:
                        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        plt.imshow(img, cmap='gray')
                    
                    plt.title(title)
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(vis_path)
                plt.close()
                logger.info(f"Saved visualization to {vis_path}")
            else:
                # Process without visualization
                preprocessor.preprocess_image(
                    args.input,
                    save_path=output_path,
                    illumination_method=args.illumination,
                    enhance_method=args.enhance,
                    preserve_aspect_ratio=args.preserve_ratio
                )
        except Exception as e:
            logger.error(f"Error processing {args.input}: {str(e)}")
    
    logger.info("Processing complete!") 