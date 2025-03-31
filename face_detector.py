import cv2
import numpy as np
import os

# Try to import dlib for HOG-based detection
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Dlib not available. Will use Viola-Jones (Haar Cascade) detector only.")

class FaceDetector:
    """
    Face detector class supporting multiple detection methods:
    
    1. Viola-Jones (Haar Cascade) - OpenCV's built-in face detector
       - Fast but can be less accurate in challenging conditions
       - Works using Haar-like features and AdaBoost classifiers
       
    2. HOG-based detector (requires dlib) - More robust face detector
       - Histogram of Oriented Gradients (HOG) based detection:
         a) Computes gradient magnitude and direction for each pixel
         b) Divides image into cells and blocks
         c) Creates HOG feature descriptor for each block
         d) Uses a linear SVM to classify face vs non-face
       - Generally more accurate than Haar in varying conditions
       - Slightly slower but still real-time capable
       
    See the use_hog parameter in the constructor to choose between methods.
    """
    
    def __init__(self, use_hog=True):
        """
        Initialize face detector.
        
        Args:
            use_hog (bool): Whether to use HOG-based detection when available.
                           Falls back to Viola-Jones if dlib is not available.
        """
        # Store detector preference
        self.use_hog = use_hog and DLIB_AVAILABLE
        
        # Load Viola-Jones detectors (always load as fallback)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Initialize HOG detector if dlib is available
        if DLIB_AVAILABLE:
            self.hog_detector = dlib.get_frontal_face_detector()
            
            # Try to load face landmarks predictor if available
            self.shape_predictor = None
            shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(shape_predictor_path):
                try:
                    self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
                    print("Loaded dlib shape predictor for facial landmarks")
                except Exception as e:
                    print(f"Could not load shape predictor: {e}")
            else:
                print("Shape predictor file not found. Facial landmarks not available.")
        
        # Default parameters - can be adjusted via methods
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_face_size = (30, 30)
        self.max_face_size = None  # No maximum by default
        
        # HOG detector parameters
        self.hog_upsample_num_times = 1  # Higher values find smaller faces but slower

    def set_parameters(self, scale_factor=None, min_neighbors=None, min_face_size=None, max_face_size=None, hog_upsample=None):
        """
        Set face detection parameters.
        
        Args:
            scale_factor (float): Factor to scale image between detections for Viola-Jones
            min_neighbors (int): Minimum neighbors for Viola-Jones
            min_face_size (tuple): Minimum face size (width, height)
            max_face_size (tuple): Maximum face size (width, height)
            hog_upsample (int): Number of times to upsample image for HOG detection
        
        Returns:
            FaceDetector: self for method chaining
        """
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if min_neighbors is not None:
            self.min_neighbors = min_neighbors
        if min_face_size is not None:
            self.min_face_size = min_face_size
        if max_face_size is not None:
            self.max_face_size = max_face_size
        if hog_upsample is not None and 0 <= hog_upsample <= 2:
            self.hog_upsample_num_times = hog_upsample
        
        return self
    
    def _detect_faces_viola_jones(self, image):
        """
        Detect faces using Viola-Jones (Haar Cascade) method.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of (x, y, w, h) face rectangles
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Normalize image brightness and contrast for better detection
        equalized = cv2.equalizeHist(gray)
        
        # Try initially with standard parameters
        faces = self.face_cascade.detectMultiScale(
            equalized,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
            maxSize=self.max_face_size
        )
        
        # If no faces detected, try with more lenient parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                equalized,
                scaleFactor=self.scale_factor + 0.05,  # Make slightly higher
                minNeighbors=max(2, self.min_neighbors - 2),  # Make slightly lower
                minSize=tuple(int(dim * 0.8) for dim in self.min_face_size),  # Smaller min size
                maxSize=self.max_face_size
            )
            
        # If still no faces, try with even more lenient parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,  # Use original grayscale image
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        # Apply face validation to remove false positives
        valid_faces = []
        for (x, y, w, h) in faces:
            # Check face aspect ratio - faces should be roughly square
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.4:  # Reasonable face aspect ratios
                valid_faces.append((x, y, w, h))
        
        # Return the original detection if validation filtered all faces
        return valid_faces if valid_faces else faces
    
    def _detect_faces_hog(self, image):
        """
        Detect faces using HOG-based detector from dlib.
        
        The HOG-based detector works as follows:
        1. Computes the gradient (magnitude and orientation) at each pixel
        2. Divides the image into cells (typically 8x8 pixels)
        3. For each cell, creates a histogram of gradient orientations
        4. Normalizes histograms within larger blocks to improve invariance
        5. The resulting HOG feature descriptor is fed to a linear SVM
           that has been trained to distinguish faces from non-faces
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of (x, y, w, h) face rectangles
        """
        if not DLIB_AVAILABLE:
            return self._detect_faces_viola_jones(image)
            
        # Convert color format if needed (dlib uses RGB)
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Convert BGR to RGB for dlib
                dlib_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Handle other channel counts
                dlib_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Convert grayscale to RGB for consistent processing
            dlib_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Detect faces using HOG detector
        # upsample_num_times: Higher values find smaller faces but slower
        dlib_faces = self.hog_detector(dlib_image, self.hog_upsample_num_times)
        
        # Convert dlib rectangles to OpenCV format (x, y, w, h)
        faces = []
        for face in dlib_faces:
            x = face.left()
            y = face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            faces.append((x, y, w, h))
            
        return faces

    def detect_faces(self, image):
        """
        Detect faces in the input image using the preferred method.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of (x, y, w, h) face rectangles
        """
        if image is None or image.size == 0:
            print("Warning: Empty image in face detection")
            return []
            
        # Use the appropriate detection method
        if self.use_hog and DLIB_AVAILABLE:
            return self._detect_faces_hog(image)
        else:
            return self._detect_faces_viola_jones(image)

    def get_face_roi(self, image, face_rect):
        """
        Extract face ROI from image using detected rectangle with better cropping.
        
        Args:
            image (numpy.ndarray): Input image
            face_rect (tuple): Face rectangle (x, y, w, h)
            
        Returns:
            numpy.ndarray: Face region of interest
        """
        x, y, w, h = face_rect
        
        # Apply a dynamic margin based on face size
        margin_x = int(w * 0.15)  # 15% horizontal margin
        margin_y = int(h * 0.15)  # 15% vertical margin
        
        # Get image dimensions
        img_h, img_w = image.shape[:2]
        
        # Calculate new coordinates with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)
        
        # Extract ROI with margin
        face_roi = image[y1:y2, x1:x2]
        
        # If the ROI is empty, return the original cropping
        if face_roi.size == 0:
            face_roi = image[y:y+h, x:x+w]
        
        return face_roi
        
    def detect_and_process_face(self, image):
        """
        Detect face and process it in one step, returning the best face ROI.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Detected face ROI or None if no face detected
        """
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return None
            
        # Find the largest face (most likely to be the main face)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        
        # Extract face ROI
        face_roi = self.get_face_roi(image, largest_face)
        
        return face_roi
        
    def detect_facial_landmarks(self, image, face_rect=None):
        """
        Detect facial landmarks using dlib's shape predictor.
        
        Args:
            image (numpy.ndarray): Input image
            face_rect (tuple, optional): Face rectangle (x, y, w, h). 
                                        If None, detect face first.
        
        Returns:
            list or None: List of (x, y) landmark points or None if detection fails
        """
        if not DLIB_AVAILABLE or self.shape_predictor is None:
            return None
            
        # Ensure image is in the right format for dlib
        if len(image.shape) == 3:
            dlib_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            dlib_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # If no face rectangle provided, detect face first
        if face_rect is None:
            faces = self.detect_faces(image)
            if not faces:
                return None
            face_rect = faces[0]
            
        # Convert OpenCV rect to dlib rect
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
        
        # Get facial landmarks
        shape = self.shape_predictor(dlib_image, dlib_rect)
        
        # Convert to list of (x, y) tuples
        landmarks = []
        for i in range(68):  # 68 landmarks
            landmarks.append((shape.part(i).x, shape.part(i).y))
            
        return landmarks 