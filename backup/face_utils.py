import os
import cv2
import numpy as np
from datetime import datetime
import random

class FaceDetector:
    """Class for detecting faces in images"""
    
    def __init__(self, cascade_path=None, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Initialize face detector
        
        Parameters:
        -----------
        cascade_path : str, optional
            Path to Haar cascade XML file. If None, use OpenCV's default face cascade
        scale_factor : float
            Scale factor for face detection
        min_neighbors : int
            Minimum number of neighbors for face detection
        min_size : tuple
            Minimum face size (width, height)
        """
        if cascade_path is None:
            # Use OpenCV's default face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (BGR or grayscale)
            
        Returns:
        --------
        faces : list
            List of face rectangles (x, y, w, h)
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        return faces
    
    def get_face_roi(self, image, face_rect, margin=0):
        """
        Extract face region of interest (ROI) from image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        face_rect : tuple
            Face rectangle (x, y, w, h)
        margin : int
            Margin to add around face
            
        Returns:
        --------
        face_roi : numpy.ndarray
            Face region of interest
        """
        x, y, w, h = face_rect
        
        # Apply margin (with bounds checking)
        y1 = max(0, y - margin)
        y2 = min(image.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(image.shape[1], x + w + margin)
        
        # Extract face ROI
        face_roi = image[y1:y2, x1:x2]
        
        return face_roi
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles around detected faces
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        faces : list
            List of face rectangles (x, y, w, h)
        color : tuple
            Rectangle color (B, G, R)
        thickness : int
            Rectangle thickness
            
        Returns:
        --------
        annotated_image : numpy.ndarray
            Image with face rectangles drawn
        """
        # Make a copy to avoid modifying the original
        annotated_image = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, thickness)
        
        return annotated_image

class FacePreprocessor:
    """Class for preprocessing face images"""
    
    def __init__(self, target_size=(100, 100), equalize_hist=True, normalize=True):
        """
        Initialize face preprocessor
        
        Parameters:
        -----------
        target_size : tuple
            Target size of face images (width, height)
        equalize_hist : bool
            Whether to apply histogram equalization
        normalize : bool
            Whether to normalize pixel values to [0, 1]
        """
        self.target_size = target_size
        self.equalize_hist = equalize_hist
        self.normalize = normalize
    
    def preprocess(self, image):
        """
        Preprocess a face image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input face image
            
        Returns:
        --------
        processed_image : numpy.ndarray
            Preprocessed face image
        """
        # Make a copy to avoid modifying the original
        img = image.copy()
        
        # Convert to grayscale if needed
        if len(img.shape) > 2 and img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        img = cv2.resize(img, self.target_size)
        
        # Apply histogram equalization
        if self.equalize_hist:
            img = cv2.equalizeHist(img)
        
        # Normalize pixel values
        if self.normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def batch_preprocess(self, images):
        """
        Preprocess a batch of face images
        
        Parameters:
        -----------
        images : list or numpy.ndarray
            List or array of face images
            
        Returns:
        --------
        processed_images : numpy.ndarray
            Preprocessed face images
        """
        return np.array([self.preprocess(img) for img in images])

def capture_training_faces(person_name, camera, detector, preprocessor, num_images=20, output_dir='gallery'):
    """
    Capture face images for training
    
    Parameters:
    -----------
    person_name : str
        Name of the person
    camera : cv2.VideoCapture
        OpenCV camera object
    detector : FaceDetector
        Face detector object
    preprocessor : FacePreprocessor
        Face preprocessor object
    num_images : int
        Number of face images to capture
    output_dir : str
        Directory to save face images
        
    Returns:
    --------
    captured_faces : numpy.ndarray
        Array of captured and preprocessed face images
    """
    # Create directory for person
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Capture face images
    captured_faces = []
    capture_count = 0
    
    # Instructions for pose variety
    pose_instructions = [
        "Look straight at the camera",
        "Turn slightly to the left",
        "Turn slightly to the right",
        "Tilt your head up slightly",
        "Tilt your head down slightly",
        "Show a neutral expression",
        "Show a smiling expression",
        "Adjust lighting slightly (if possible)"
    ]
    
    print(f"Capturing {num_images} face images for {person_name}...")
    
    while capture_count < num_images:
        # Show instruction
        current_instruction = pose_instructions[capture_count % len(pose_instructions)]
        print(f"Please: {current_instruction} ({capture_count+1}/{num_images})")
        
        # Read frame from camera
        ret, frame = camera.read()
        if not ret:
            print("Error reading from camera")
            continue
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        if len(faces) == 1:
            # Extract face ROI
            x, y, w, h = faces[0]
            face_roi = detector.get_face_roi(frame, (x, y, w, h), margin=10)
            
            # Preprocess face
            processed_face = preprocessor.preprocess(face_roi)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.jpg"
            
            # Save image
            if preprocessor.normalize:
                # Convert back to uint8 for saving
                save_image = (processed_face * 255).astype(np.uint8)
            else:
                save_image = processed_face
                
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, save_image)
            
            # Add to captured faces
            captured_faces.append(processed_face)
            capture_count += 1
            
            print(f"Captured image {capture_count}/{num_images}")
            
            # Add delay to avoid duplicate frames
            cv2.waitKey(500)
    
    return np.array(captured_faces)

def augment_face_images(faces, num_augmentations=5):
    """
    Apply data augmentation to face images
    
    Parameters:
    -----------
    faces : numpy.ndarray
        Array of face images
    num_augmentations : int
        Number of augmentations per face
        
    Returns:
    --------
    augmented_faces : numpy.ndarray
        Array of augmented face images
    """
    augmented_faces = []
    
    # Add original faces
    for face in faces:
        augmented_faces.append(face.copy())
    
    # Create augmentations
    for face in faces:
        for i in range(num_augmentations):
            # Make a copy of the face
            aug_face = face.copy()
            
            # Random rotation (-10 to 10 degrees)
            angle = np.random.uniform(-10, 10)
            h, w = face.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_face = cv2.warpAffine(aug_face, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Random brightness/contrast adjustment
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # Contrast
                beta = np.random.uniform(-10, 10)    # Brightness
                aug_face = cv2.convertScaleAbs(aug_face, alpha=alpha, beta=beta)
            
            # Random crop and resize back
            if np.random.random() > 0.5:
                crop_percent = np.random.uniform(0.9, 1.0)
                crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
                start_y = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
                start_x = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
                aug_face = aug_face[start_y:start_y+crop_h, start_x:start_x+crop_w]
                aug_face = cv2.resize(aug_face, (w, h))
            
            # Add random noise
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 0.01, aug_face.shape).astype(np.float32)
                aug_face = np.clip(aug_face.astype(np.float32) + noise, 0, 1).astype(aug_face.dtype)
            
            # Horizontal flip
            if np.random.random() > 0.5:
                aug_face = cv2.flip(aug_face, 1)
            
            augmented_faces.append(aug_face)
    
    return np.array(augmented_faces)

def load_external_databases(att_dir='data/att_faces', yale_dir='data/yale_faces', target_size=(100, 100)):
    """
    Load external face databases (AT&T and Yale)
    
    Parameters:
    -----------
    att_dir : str
        Directory containing AT&T face database
    yale_dir : str
        Directory containing Yale face database
    target_size : tuple
        Target size of face images (width, height)
        
    Returns:
    --------
    faces : numpy.ndarray
        Array of face images
    labels : numpy.ndarray
        Array of face labels
    """
    all_faces = []
    all_labels = []
    
    # Load AT&T faces if available
    if os.path.exists(att_dir):
        print("Loading AT&T face database...")
        att_faces, att_labels = load_att_faces(att_dir, target_size)
        if len(att_faces) > 0:
            all_faces.append(att_faces)
            all_labels.append(att_labels)
            print(f"Loaded {len(att_faces)} images from AT&T database")
    
    # Load Yale faces if available
    if os.path.exists(yale_dir):
        print("Loading Yale face database...")
        yale_faces, yale_labels = load_yale_faces(yale_dir, target_size)
        if len(yale_faces) > 0:
            all_faces.append(yale_faces)
            all_labels.append(yale_labels)
            print(f"Loaded {len(yale_faces)} images from Yale database")
    
    # Combine datasets
    if len(all_faces) > 0:
        faces = np.vstack(all_faces)
        labels = np.concatenate(all_labels)
        print(f"Combined dataset: {len(faces)} images from {len(np.unique(labels))} subjects")
        return faces, labels
    else:
        print("No external datasets found")
        return np.array([]), np.array([])

def load_att_faces(att_dir, target_size=(100, 100)):
    """
    Load AT&T face database
    
    Parameters:
    -----------
    att_dir : str
        Directory containing AT&T face database
    target_size : tuple
        Target size of face images (width, height)
        
    Returns:
    --------
    faces : numpy.ndarray
        Array of face images
    labels : numpy.ndarray
        Array of face labels
    """
    faces = []
    labels = []
    
    # Check alternate paths if the directory doesn't exist
    if not os.path.exists(att_dir):
        alt_paths = ["external_databases/att_faces", "att_faces", "orl_faces"]
        for path in alt_paths:
            if os.path.exists(path):
                att_dir = path
                break
    
    # Try to load AT&T faces with standard structure
    try:
        # Standard structure: s1, s2, ..., s40 directories
        for person_id in range(1, 41):
            person_dir = os.path.join(att_dir, f"s{person_id}")
            
            if not os.path.exists(person_dir):
                # Try alternate structure
                person_dir = os.path.join(att_dir, "orl_faces", f"s{person_id}")
                if not os.path.exists(person_dir):
                    continue
            
            for image_id in range(1, 11):
                image_path = os.path.join(person_dir, f"{image_id}.pgm")
                
                if not os.path.exists(image_path):
                    # Try alternate extension
                    image_path = os.path.join(person_dir, f"{image_id}.jpg")
                    if not os.path.exists(image_path):
                        continue
                
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize to target size
                    img = cv2.resize(img, target_size)
                    
                    # Normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    faces.append(img)
                    labels.append(f"att_s{person_id}")
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
    
        return np.array(faces), np.array(labels)
    
    except Exception as e:
        print(f"Error loading AT&T database: {str(e)}")
        return np.array([]), np.array([])

def load_yale_faces(yale_dir, target_size=(100, 100)):
    """
    Load Yale face database
    
    Parameters:
    -----------
    yale_dir : str
        Directory containing Yale face database
    target_size : tuple
        Target size of face images (width, height)
        
    Returns:
    --------
    faces : numpy.ndarray
        Array of face images
    labels : numpy.ndarray
        Array of face labels
    """
    faces = []
    labels = []
    
    # Check alternate paths if the directory doesn't exist
    if not os.path.exists(yale_dir):
        alt_paths = ["external_databases/yale_faces", "yale_faces", "yalefaces"]
        for path in alt_paths:
            if os.path.exists(path):
                yale_dir = path
                break
    
    try:
        # Try to find structure: subject directories or flat structure
        subject_dirs = [d for d in os.listdir(yale_dir) 
                       if os.path.isdir(os.path.join(yale_dir, d)) and 'subject' in d.lower()]
        
        if subject_dirs:
            # We have subject directories
            for subject_dir in subject_dirs:
                full_path = os.path.join(yale_dir, subject_dir)
                
                # Extract subject ID
                subject_id = ''.join(filter(str.isdigit, subject_dir))
                if not subject_id:
                    subject_id = subject_dir
                
                # Get image files
                image_files = [f for f in os.listdir(full_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.pgm', '.png', '.gif'))]
                
                for image_file in image_files:
                    image_path = os.path.join(full_path, image_file)
                    try:
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        
                        # Resize to target size
                        img = cv2.resize(img, target_size)
                        
                        # Normalize to [0, 1]
                        img = img.astype(np.float32) / 255.0
                        
                        faces.append(img)
                        labels.append(f"yale_{subject_id}")
                    except Exception as e:
                        print(f"Error loading image {image_path}: {str(e)}")
        else:
            # Try flat structure
            image_files = [f for f in os.listdir(yale_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.pgm', '.png', '.gif'))]
            
            for image_file in image_files:
                # Try to extract subject from filename (e.g., "subject01.sad.gif")
                parts = image_file.split('.')
                if len(parts) >= 2:
                    subject_part = parts[0]
                    subject_id = ''.join(filter(str.isdigit, subject_part))
                    if not subject_id:
                        subject_id = subject_part
                else:
                    subject_id = "unknown"
                
                image_path = os.path.join(yale_dir, image_file)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize to target size
                    img = cv2.resize(img, target_size)
                    
                    # Normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    faces.append(img)
                    labels.append(f"yale_{subject_id}")
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
        
        return np.array(faces), np.array(labels)
    
    except Exception as e:
        print(f"Error loading Yale database: {str(e)}")
        return np.array([]), np.array([])

if __name__ == "__main__":
    print("This module provides face detection, preprocessing, and augmentation functions.")
    print("Import this module and use its classes and functions in your application.") 