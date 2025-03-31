import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
import time
import json
import glob
import sys
import subprocess

from face_detector import FaceDetector
from face_preprocessor import FacePreprocessor
from face_recognizer import FaceRecognizer
from pca_face_detector import PCAFaceDetector
from face_database_loader import FaceDatabaseLoader

# Try to import dlib to check if it's available
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

# Initialize session state variables
if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 0
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = FaceRecognizer()
if 'current_person' not in st.session_state:
    st.session_state.current_person = "Unknown"
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = ""
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'detector_type' not in st.session_state:
    st.session_state.detector_type = 'viola_jones'
if 'pca_detector' not in st.session_state:
    st.session_state.pca_detector = None
if 'database_loader' not in st.session_state:
    st.session_state.database_loader = FaceDatabaseLoader()
if 'model_path' not in st.session_state:
    # Check for specialized model first
    if os.path.exists("jia_nan_specialized_model.joblib"):
        st.session_state.model_path = "jia_nan_specialized_model.joblib"
        st.session_state.using_specialized_model = True
    else:
        st.session_state.model_path = "face_recognition_model.joblib"
        st.session_state.using_specialized_model = False
if 'use_att_database' not in st.session_state:
    st.session_state.use_att_database = True  # Default to using AT&T
if 'att_data_weight' not in st.session_state:
    st.session_state.att_data_weight = 0.5  # Default weight of AT&T data vs user data
if 'face_threshold' not in st.session_state:
    st.session_state.face_threshold = 0.1  # Lower threshold for better recognition
if 'use_face_alignment' not in st.session_state:
    # Default to True if dlib is available, False otherwise
    st.session_state.use_face_alignment = DLIB_AVAILABLE

# Initialize components
viola_jones_detector = FaceDetector()
preprocessor = FacePreprocessor(target_size=(100, 100))

# Create necessary directories
gallery_dir = "gallery"
os.makedirs(gallery_dir, exist_ok=True)

# Load registered users from file
users_file = "registered_users.json"
if os.path.exists(users_file):
    with open(users_file, 'r') as f:
        st.session_state.registered_users = json.load(f)

def save_registered_users():
    """Save registered users to file"""
    with open(users_file, 'w') as f:
        json.dump(st.session_state.registered_users, f)

def reset_capture_count():
    """Reset capture count"""
    st.session_state.capture_count = 0

def download_dlib_model():
    """Download dlib face landmark model"""
    try:
        from download_dlib_model import download_dlib_model as dl_model
        success = dl_model()
        return success
    except Exception as e:
        st.error(f"Error downloading dlib model: {e}")
        return False

def get_current_detector():
    """Get the current face detector based on selection"""
    if st.session_state.detector_type == 'pca':
        if st.session_state.pca_detector is None or not st.session_state.pca_detector.trained:
            st.warning("PCA detector not trained. Using Viola-Jones as fallback.")
            return viola_jones_detector
        return st.session_state.pca_detector
    return viola_jones_detector

def train_pca_detector():
    """Train PCA-based face detector"""
    try:
        # Get face samples from external databases
        face_samples = st.session_state.database_loader.get_face_samples(num_samples=1000)
        non_face_samples = st.session_state.database_loader.get_non_face_samples(num_samples=1000)
        
        if face_samples is None or non_face_samples is None:
            st.error("Could not load training samples. Please add background images.")
            return False
        
        # Initialize and train PCA detector
        st.session_state.pca_detector = PCAFaceDetector()
        if st.session_state.pca_detector.train(face_samples, non_face_samples):
            st.success("PCA face detector trained successfully!")
            return True
        else:
            st.error("PCA detector training failed.")
            return False
    except Exception as e:
        st.error(f"Error training PCA detector: {str(e)}")
        return False

def enhance_training_dataset(person_name, person_dir):
    """Enhance training dataset with variations"""
    images = []
    image_paths = glob.glob(os.path.join(person_dir, "*.jpg"))
    
    st.session_state.debug_info += f"\nEnhancing dataset for {person_name} with {len(image_paths)} base images"
    
    for img_path in image_paths:
        # Load original image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            continue
        
        # Add original image
        images.append(img)
        
        # Create variations with small rotations
        for angle in [-5, 5]:  # Small rotations
            h, w = img.shape
            center = (w/2, h/2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            images.append(rotated)
            
        # Create variations with small brightness changes
        for factor in [0.9, 1.1]:  # Slightly darker and brighter
            brightness_var = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            images.append(brightness_var)
    
    st.session_state.debug_info += f"\nCreated {len(images)} variations for {person_name}"
    return images

def capture_and_save(frame, person_name):
    """Capture and save face images for training"""
    try:
        detector = get_current_detector()
        faces = detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Get face with margin directly from detector
            face_roi = detector.get_face_roi(frame, (x, y, w, h))
            
            # Debug information
            st.session_state.debug_info = f"ROI shape: {face_roi.shape}"
            
            # Apply preprocessing with or without advanced alignment based on setting
            # The preprocessor will automatically use landmarks if available and enabled
            processed_face = preprocessor.preprocess(face_roi)
            
            # Check for NaN values and fix them
            if np.isnan(processed_face).any():
                st.session_state.debug_info += "\nFound NaN values in processed face, fixing..."
                processed_face = np.nan_to_num(processed_face, nan=0.5)
                
            # Also check for inf values
            if np.isinf(processed_face).any():
                st.session_state.debug_info += "\nFound inf values in processed face, fixing..."
                processed_face = np.clip(np.nan_to_num(processed_face, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
                
            # Verify processed face dimensions
            if processed_face is None or processed_face.size == 0:
                st.session_state.debug_info += "\nProcessed face is empty"
                return False
                
            st.session_state.debug_info += f"\nProcessed face shape: {processed_face.shape}"
            
            # Save processed face
            person_dir = os.path.join(gallery_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(person_dir, f"{timestamp}.jpg")
            
            # Save as uint8 image
            save_image = (processed_face * 255).astype(np.uint8)
            cv2.imwrite(save_path, save_image)
            
            st.session_state.capture_count += 1
            # Update user's image count
            st.session_state.registered_users[person_name]['images_captured'] = st.session_state.capture_count
            save_registered_users()
            return True
            
        return False
    except Exception as e:
        st.session_state.debug_info += f"\nError in capture_and_save: {str(e)}"
        return False

def train_model():
    """Train the face recognition model with improved data handling"""
    try:
        X = []
        y = []
        users_images = {}  # Track images per user
        
        # First collect all user data
        for person_name in os.listdir(gallery_dir):
            person_dir = os.path.join(gallery_dir, person_name)
            if os.path.isdir(person_dir):
                # Generate enhanced training set with variations
                person_images = enhance_training_dataset(person_name, person_dir)
                
                # Check and fix any NaN values in the images
                fixed_images = []
                for img in person_images:
                    if np.isnan(img).any():
                        st.session_state.debug_info += f"\nFixing NaN values in image for {person_name}"
                        img = np.nan_to_num(img, nan=0.5)
                    if np.isinf(img).any():
                        st.session_state.debug_info += f"\nFixing inf values in image for {person_name}"
                        img = np.clip(np.nan_to_num(img, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
                    fixed_images.append(img)
                
                # Store images by user if we have enough
                if len(fixed_images) >= 20:
                    users_images[person_name] = fixed_images
                    
                    # Update user status
                    if person_name in st.session_state.registered_users:
                        st.session_state.registered_users[person_name]['trained'] = True
                        save_registered_users()
        
        if not users_images:
            st.session_state.debug_info = "No valid training images found"
            return False
        
        # Add AT&T database to training if enabled
        att_images = []
        att_labels = []
        
        if st.session_state.use_att_database:
            try:
                att_path = "external_databases/AT&T"
                if os.path.exists(att_path):
                    st.session_state.debug_info += "\nLoading AT&T database for training..."
                    att_subjects = [d for d in os.listdir(att_path) if os.path.isdir(os.path.join(att_path, d))]
                    
                    for subject in att_subjects:
                        subject_dir = os.path.join(att_path, subject)
                        subject_images = []
                        
                        # Load subject images
                        for img_file in os.listdir(subject_dir):
                            if img_file.endswith('.pgm'):
                                img_path = os.path.join(subject_dir, img_file)
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    # Resize to match our target size
                                    img = cv2.resize(img, (100, 100))
                                    # Preprocess with same pipeline as user images
                                    img = preprocessor.preprocess(img)
                                    # Fix any NaN values
                                    if np.isnan(img).any() or np.isinf(img).any():
                                        img = np.clip(np.nan_to_num(img, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
                                    subject_images.append(img)
                        
                        # Add to AT&T dataset
                        if subject_images:
                            att_images.extend(subject_images)
                            att_labels.extend([subject] * len(subject_images))
                    
                    st.session_state.debug_info += f"\nAdded {len(att_subjects)} subjects from AT&T database"
            except Exception as e:
                st.session_state.debug_info += f"\nWarning: Error loading AT&T database: {str(e)}"
        
        # Prepare final training data
        for person_name, images in users_images.items():
            X.extend(images)
            y.extend([person_name] * len(images))
        
        # Add AT&T data if available
        if att_images and st.session_state.use_att_database:
            # Balance AT&T data with user data
            att_weight = st.session_state.att_data_weight
            max_att_samples = int(len(X) * att_weight / (1 - att_weight))
            
            if len(att_images) > max_att_samples:
                # Randomly select subset of AT&T data
                indices = np.random.choice(len(att_images), max_att_samples, replace=False)
                att_images = [att_images[i] for i in indices]
                att_labels = [att_labels[i] for i in indices]
            
            X.extend(att_images)
            y.extend(att_labels)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Initialize recognizer with appropriate settings
        recognizer = FaceRecognizer()
        
        # Set appropriate thresholds based on dataset
        if len(users_images) == 1:
            # Single user mode
            recognizer.face_threshold = 0.1
            user_name = list(users_images.keys())[0]
            if user_name == "Jia Nan":
                # Special handling for Jia Nan
                recognizer.reconstruction_threshold = 0.75
                st.session_state.debug_info += "\nUsing specialized settings for Jia Nan"
        else:
            # Multi-user mode
            recognizer.face_threshold = 0.2
        
        # Train the model
        if recognizer.train(X, y):
            # Save the model
            model_path = "jia_nan_specialized_model.joblib" if "Jia Nan" in users_images else "face_recognition_model.joblib"
            recognizer.save_model(model_path)
            
            # Update session state
            st.session_state.recognizer = recognizer
            st.session_state.model_trained = True
            st.session_state.model_path = model_path
            
            # Save model state
            model_state = {
                'model_trained': True,
                'trained_users': list(users_images.keys()),
                'model_path': model_path
            }
            
            with open("model_state.json", 'w') as f:
                json.dump(model_state, f)
            
            return True
        else:
            st.session_state.debug_info += "\nModel training failed"
            return False
            
    except Exception as e:
        st.session_state.debug_info += f"\nError in model training: {str(e)}"
        return False

def load_model_state():
    """Load model state and retrain if necessary"""
    try:
        # First try to load saved model
        if os.path.exists(st.session_state.model_path):
            try:
                st.session_state.debug_info += "\nLoading saved model from disk..."
                if st.session_state.recognizer.load_model(st.session_state.model_path):
                    st.session_state.model_trained = True
                    st.session_state.debug_info += "\nModel loaded successfully!"
                    return True
            except Exception as e:
                st.session_state.debug_info += f"\nError loading model: {str(e)}"
        
        # Check model state file
        if os.path.exists('model_state.json'):
            with open('model_state.json', 'r') as f:
                model_state = json.load(f)
            
            # Load AT&T database settings
            if 'att_data_weight' in model_state:
                st.session_state.att_data_weight = model_state['att_data_weight']
            if 'use_att_database' in model_state:
                st.session_state.use_att_database = model_state['use_att_database']
                
            # Check if we need to retrain
            current_trained_users = [user for user, info in st.session_state.registered_users.items() 
                                   if info['trained']]
            saved_trained_users = set(model_state.get('trained_users', []))
            
            if set(current_trained_users) != saved_trained_users:
                st.session_state.model_trained = False
                return False
            
            st.session_state.model_trained = model_state.get('model_trained', False)
            return True
    except Exception as e:
        st.session_state.debug_info = f"Error loading model state: {str(e)}"
        st.session_state.model_trained = False
        return False

def detector_settings_page():
    st.title("Face Detector Settings")
    
    # Select detector type
    detector_type = st.radio(
        "Select Face Detection Method",
        ['viola_jones', 'pca'],
        index=0 if st.session_state.detector_type == 'viola_jones' else 1
    )
    
    if detector_type != st.session_state.detector_type:
        st.session_state.detector_type = detector_type
    
    # Face Alignment Settings
    st.subheader("Face Alignment Settings")
    
    # Check for dlib availability
    if not DLIB_AVAILABLE:
        st.warning("dlib is not installed. Advanced face alignment is not available.")
        st.info("To enable advanced face alignment, install dlib with: pip install dlib")
        dlib_status = "Not installed"
    else:
        dlib_status = "Installed"
        # Check for landmark model
        if os.path.exists("shape_predictor_68_face_landmarks.dat"):
            landmark_status = "Available"
        else:
            landmark_status = "Not found"
            if st.button("Download Landmark Model"):
                with st.spinner("Downloading facial landmark model..."):
                    if download_dlib_model():
                        st.success("Landmark model downloaded successfully!")
                        landmark_status = "Available"
                    else:
                        st.error("Failed to download landmark model")
        
        st.write(f"dlib status: {dlib_status}")
        st.write(f"Facial landmark model: {landmark_status}")
        
        # Only enable checkbox if dlib is available
        use_alignment = st.checkbox(
            "Use Advanced Face Alignment",
            value=st.session_state.use_face_alignment,
            help="Uses facial landmarks for better face alignment"
        )
        
        if use_alignment != st.session_state.use_face_alignment:
            st.session_state.use_face_alignment = use_alignment
            if use_alignment and landmark_status != "Available":
                st.warning("Landmark model not available. Please download it first.")
    
    # AT&T database integration settings
    st.subheader("AT&T Database Integration")
    
    use_att = st.checkbox("Use AT&T Database", 
                          value=st.session_state.use_att_database,
                          help="Include AT&T database faces in training")
    
    if use_att != st.session_state.use_att_database:
        st.session_state.use_att_database = use_att
        
    att_weight = st.slider("AT&T Data Weight", 
                          min_value=0.0, 
                          max_value=1.0, 
                          value=st.session_state.att_data_weight,
                          step=0.1,
                          help="Weight of AT&T data vs your own face data (0=only your data, 1=equal weights)")
    
    if att_weight != st.session_state.att_data_weight:
        st.session_state.att_data_weight = att_weight
        
    # PCA detector training
    if detector_type == 'pca':
        st.subheader("PCA Face Detector Training")
        
        # Database management
        st.write("### External Face Databases")
        if st.button("List Available Databases"):
            st.session_state.database_loader.list_available_databases()
            
        # Download databases
        selected_db = st.selectbox(
            "Select database to download",
            list(st.session_state.database_loader.DATABASES.keys())
        )
        
        if st.button("Download Selected Database"):
            with st.spinner(f"Downloading {selected_db} database..."):
                try:
                    st.session_state.database_loader.download_database(selected_db)
                    st.success(f"Successfully downloaded {selected_db} database!")
                except Exception as e:
                    st.error(f"Error downloading database: {str(e)}")
        
        # Train PCA detector
        if st.button("Train PCA Detector"):
            with st.spinner("Training PCA face detector..."):
                train_pca_detector()
    
    # Viola-Jones detector parameters
    if detector_type == 'viola_jones':
        st.subheader("Viola-Jones Detector Parameters")
        
        scale_factor = st.slider(
            "Scale Factor",
            min_value=1.05,
            max_value=1.5,
            value=viola_jones_detector.scale_factor,
            step=0.05,
            help="How much the image size is reduced at each scale. Smaller values detect more faces but are slower."
        )
        
        min_neighbors = st.slider(
            "Min Neighbors",
            min_value=1,
            max_value=10,
            value=viola_jones_detector.min_neighbors,
            step=1,
            help="How many neighbors each candidate rectangle should have. Higher values give fewer false positives."
        )
        
        min_face_size = st.slider(
            "Min Face Size",
            min_value=20,
            max_value=100,
            value=viola_jones_detector.min_face_size[0],
            step=5,
            help="Minimum possible face size in pixels."
        )
        
        # Update detector parameters
        viola_jones_detector.set_parameters(
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_face_size=(min_face_size, min_face_size)
        )
        
        if st.button("Reset to Default Parameters"):
            viola_jones_detector.set_parameters(
                scale_factor=1.1,
                min_neighbors=5,
                min_face_size=(30, 30)
            )
            st.rerun()

def user_management_page():
    st.title("User Management")
    
    # Add new user
    with st.expander("Add New User"):
        new_user = st.text_input("Enter new user name")
        if st.button("Add User"):
            if new_user and new_user not in st.session_state.registered_users:
                st.session_state.registered_users[new_user] = {
                    'trained': False,
                    'images_captured': 0,
                    'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                save_registered_users()
                st.success(f"User {new_user} added successfully!")
            else:
                st.error("Invalid username or user already exists!")
    
    # List and manage users
    st.subheader("Registered Users")
    if st.session_state.registered_users:
        for user, info in st.session_state.registered_users.items():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"**{user}**")
            with col2:
                st.write("Status: " + ("✅ Trained" if info['trained'] else "❌ Not trained"))
            with col3:
                if st.button("Retrain", key=f"retrain_{user}"):
                    # Mark as untrained to force retraining
                    st.session_state.registered_users[user]['trained'] = False
                    save_registered_users()
                    st.rerun()
            with col4:
                if st.button("Delete", key=f"del_{user}"):
                    # Delete user data
                    user_dir = os.path.join(gallery_dir, user)
                    if os.path.exists(user_dir):
                        import shutil
                        shutil.rmtree(user_dir)
                    del st.session_state.registered_users[user]
                    save_registered_users()
                    st.rerun()
    else:
        st.info("No users registered yet.")

def enrollment_page():
    """Page for enrolling new users by capturing multiple face images"""
    st.subheader("🔐 User Enrollment")
    
    # Add enrollment explanation
    st.markdown("""
    This page helps you enroll new users into the face recognition system.
    For best results, capture at least 10 different images of the person's face:
    - Try different head poses (looking slightly left, right, up, down)
    - Try different facial expressions (neutral, smiling, etc.)
    - Try different distances from the camera
    - Try with and without glasses if applicable
    """)
    
    # Enter the person's name
    person_name = st.text_input("Enter person's name", key="enrollment_name")
    
    # Show video feed for enrollment
    if person_name:
        # Create person directory if needed
        person_dir = os.path.join(gallery_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Count existing images
        existing_images = len(glob.glob(os.path.join(person_dir, "*.jpg")))
        st.write(f"Current images for {person_name}: {existing_images}/10+")
        
        # Progress bar
        progress_bar = st.progress(min(1.0, existing_images / 10))
        
        col1, col2 = st.columns([3, 1])
        
        # Camera feed in the left column
        with col1:
            # Container for the video feed
            video_feed = st.empty()
            
            # Get webcam feed and detect faces
            if 'camera' not in st.session_state:
                st.session_state.camera = cv2.VideoCapture(0)
            
            # Check if camera is working
            if not st.session_state.camera.isOpened():
                st.error("Could not access webcam. Please check your camera.")
                return
                
            # Get the current frame from the webcam
            ret, frame = st.session_state.camera.read()
            
            if not ret:
                st.error("Failed to get image from webcam.")
                return
            
            # Get detector based on settings
            detector = get_current_detector()
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw a rectangle around each face
            frame_with_faces = frame.copy()
            for (x, y, w, h) in faces:
                # Draw a green rectangle around the face
                cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add status text
                if len(faces) == 1:
                    cv2.putText(
                        frame_with_faces,
                        "Ready to capture",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        frame_with_faces,
                        "Multiple faces detected",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )
            
            # Convert to RGB for display (OpenCV uses BGR)
            rgb_frame = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            video_feed.image(rgb_frame, channels="RGB", use_column_width=True)
        
        # Controls in the right column
        with col2:
            # Show the most recently captured face if available
            latest_images = glob.glob(os.path.join(person_dir, "*.jpg"))
            latest_images.sort(reverse=True)  # Most recent first
            
            if latest_images:
                st.write("Latest capture:")
                latest_image = Image.open(latest_images[0])
                st.image(latest_image, width=150)
            
            # Capture button
            col_a, col_b = st.columns(2)
            with col_a:
                capture_button = st.button("📸 Capture", key="enrollment_capture")
            with col_b:
                done_button = st.button("✅ Done", key="enrollment_done")
            
            # Capture logic
            if capture_button:
                # Only proceed if at least one face is detected
                if faces:
                    # Use the biggest face if multiple detected
                    if len(faces) > 1:
                        areas = [(f[2] * f[3]) for f in faces]
                        face_rect = faces[np.argmax(areas)]
                    else:
                        face_rect = faces[0]
                    
                    # Extract the face and preprocess
                    success = capture_and_save(frame, person_name)
                    
                    if success:
                        st.success(f"Face captured for {person_name}!")
                        
                        # Update image count and progress
                        existing_images = len(glob.glob(os.path.join(person_dir, "*.jpg")))
                        progress_bar.progress(min(1.0, existing_images / 10))
                    else:
                        st.error("Failed to capture face. Please try again.")
                else:
                    st.warning("No face detected! Please make sure your face is visible.")
            
            if done_button:
                existing_images = len(glob.glob(os.path.join(person_dir, "*.jpg")))
                if existing_images < 10:
                    st.warning(f"Only {existing_images} images captured. It's recommended to have at least 10 images.")
                    if not st.button("Continue anyway"):
                        return
                
                st.success(f"Enrollment complete for {person_name}! {existing_images} images captured.")
                st.session_state.registered_users[person_name] = {
                    "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "images": existing_images
                }
                save_registered_users()
                st.info("You can now train the model to recognize this person.")
                st.balloons()

def training_page():
    st.title("Face Recognition System - Training")
    
    # User selection
    users_to_train = [user for user, info in st.session_state.registered_users.items() 
                     if not info['trained']]
    
    if not users_to_train:
        st.warning("No users need training. Please add new users in User Management.")
        return
        
    selected_user = st.selectbox("Select user to train", users_to_train)
    
    if selected_user:
        # Reset capture count when selecting a new user
        if 'last_trained_user' not in st.session_state or st.session_state.last_trained_user != selected_user:
            reset_capture_count()
            st.session_state.last_trained_user = selected_user
        
        st.write(f"""
        ### Training Mode for {selected_user}
        Please follow the instructions to capture your face from different angles.
        We'll capture 20 images to ensure better recognition accuracy.
        Look directly at the camera and slowly rotate your head as instructed.
        """)

        # Camera feed for training
        cam_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        debug_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        detector = get_current_detector()  # Get the current detector
        
        try:
            while not st.session_state.registered_users[selected_user]['trained']:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera!")
                    break
                
                # Detect faces
                faces = detector.detect_faces(frame)
                
                # Draw rectangle and instructions
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display current instruction based on capture progress
                total_images = st.session_state.capture_count
                if total_images < 4:
                    instruction = "Look straight at the camera"
                elif total_images < 8:
                    instruction = "Slowly turn your head to the left"
                elif total_images < 12:
                    instruction = "Slowly turn your head to the right"
                elif total_images < 16:
                    instruction = "Tilt your head up and down slightly"
                elif total_images < 20:
                    instruction = "Make different expressions (smile, neutral, etc.)"
                else:
                    instruction = "Training data collection complete!"
                
                # Update status
                status_placeholder.write(f"### {instruction}")
                progress_bar.progress(min(1.0, total_images / 20))
                debug_placeholder.text(st.session_state.debug_info)
                
                # Capture images
                if total_images < 20:
                    if len(faces) > 0:
                        if capture_and_save(frame, selected_user):
                            time.sleep(0.5)  # Delay between captures
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                if total_images >= 20:
                    break
                
                time.sleep(0.1)
            
            # Train model
            if st.session_state.capture_count >= 20:
                with st.spinner("Training model..."):
                    if train_model():
                        st.success(f"Training completed successfully for {selected_user}!")
                        debug_placeholder.text(st.session_state.debug_info)
                    else:
                        st.error("Training failed. Please check debug information and try again.")
                        debug_placeholder.text(st.session_state.debug_info)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.debug_info += f"\nError in training_page: {str(e)}"
        finally:
            cap.release()

def recognize_video():
    """Display webcam video with face recognition (enhanced version)"""
    # Container for the video feed
    video_feed = st.empty()
    
    # Container for multiple face results
    results_container = st.empty()
    
    # Status info
    status_container = st.empty()
    
    # Confidence display
    metrics_container = st.container()
    
    # Get webcam feed
    if 'camera' not in st.session_state:
        st.session_state.camera = cv2.VideoCapture(0)
    
    # Check if camera is working
    if not st.session_state.camera.isOpened():
        st.error("Could not access webcam. Please check your camera.")
        return
        
    # Get the current frame from the webcam
    ret, frame = st.session_state.camera.read()
    
    if not ret or frame is None or frame.size == 0:
        st.error("Failed to get valid image from webcam.")
        return
    
    # Get detector based on settings
    detector = get_current_detector()
    
    # Display frame dimensions for debugging
    h, w = frame.shape[:2]
    status = f"Frame dimensions: {w}x{h}"
    
    # Detect faces in the frame
    start_time = time.time()
    faces = detector.detect_faces(frame)
    detection_time = time.time() - start_time
    
    # Update status
    status += f" | Detection time: {detection_time:.3f}s"
    
    # Create a copy of the frame for drawing
    frame_with_faces = frame.copy()
    
    # Results for all detected faces
    all_results = []
    
    # Process each detected face
    if faces:
        status += f" | Faces detected: {len(faces)}"
        
        # For metrics display
        metrics_vals = []
        metrics_cols = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Use face_preprocessor to process the face (ensures consistent processing)
            face_roi = detector.get_face_roi(frame, (x, y, w, h))
            
            # Skip if face extraction failed
            if face_roi is None or face_roi.size == 0:
                continue
                
            # Time preprocessing
            preprocess_start = time.time()
            processed_face = preprocessor.preprocess(face_roi)
            preprocess_time = time.time() - preprocess_start
            
            # Skip if preprocessing failed
            if processed_face is None or processed_face.size == 0:
                continue
                
            # Time recognition
            recognition_start = time.time()
            prediction = st.session_state.recognizer.predict(processed_face)
            
            # Unpack prediction
            if len(prediction) == 2:  # Old model format
                name, confidence = prediction
                is_known = confidence > st.session_state.face_threshold
            else:  # New model format with is_known flag
                name, confidence, is_known = prediction
                
            recognition_time = time.time() - recognition_start
            
            # Determine color based on confidence
            if is_known and name != "Unknown":
                if confidence > 0.7:
                    rect_color = (0, 255, 0)  # Green for high confidence
                    text_color = (0, 255, 0)
                elif confidence > 0.5:
                    rect_color = (0, 255, 255)  # Yellow for medium confidence
                    text_color = (0, 255, 255)
                else:
                    rect_color = (0, 165, 255)  # Orange for low confidence
                    text_color = (0, 165, 255)
            else:
                rect_color = (0, 0, 255)  # Red for unknown
                text_color = (0, 0, 255)
                name = "Unknown"
            
            # Draw rectangle around face
            cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), rect_color, 2)
            
            # Draw filled rectangle for text background
            cv2.rectangle(frame_with_faces, (x, y-25), (x+w, y), rect_color, -1)
            
            # Add name and confidence
            confidence_pct = int(confidence * 100)
            label = f"{name} ({confidence_pct}%)"
            cv2.putText(
                frame_with_faces,
                label,
                (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1
            )
            
            # Add to metrics for display
            metrics_cols.append(name)
            metrics_vals.append(confidence)
            
            # Store result
            result_info = {
                "name": name,
                "confidence": confidence,
                "is_known": is_known,
                "rect": (x, y, w, h),
                "processing_time": preprocess_time + recognition_time
            }
            all_results.append(result_info)
            
        # Display metrics if we have results
        if metrics_cols:
            with metrics_container:
                cols = st.columns(len(metrics_cols))
                for i, col in enumerate(cols):
                    if i < len(metrics_cols):
                        col.metric(
                            label=metrics_cols[i], 
                            value=f"{metrics_vals[i]:.2f}"
                        )
    else:
        status += " | No faces detected"
    
    # Convert to RGB for display (OpenCV uses BGR)
    rgb_frame = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    video_feed.image(rgb_frame, channels="RGB", use_column_width=True)
    
    # Display status
    status_container.text(status)
    
    # Display results for each face
    if all_results:
        results_text = "## Recognition Results\n\n"
        for i, result in enumerate(all_results):
            results_text += f"**Face {i+1}:** {result['name']} - Confidence: {result['confidence']:.2f} - "
            results_text += f"Processing time: {result['processing_time']*1000:.1f}ms\n\n"
        
        results_container.markdown(results_text)
    else:
        results_container.markdown("No faces recognized in this frame.")

# Main app
def main():
    # Set page config
    st.set_page_config(
        page_title="Face Recognition App",
        page_icon="👤",
        layout="wide"
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = {
        "🏠 Home": "home",
        "👤 Recognition": "recognition",
        "🔐 Enrollment": "enrollment",
        "🧠 Training": "training",
        "👥 User Management": "users",
        "⚙️ Detector Settings": "detector"
    }
    
    # Default to Home
    default_page = "home"
    
    # Get the page selection
    page = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Main page display
    st.title("Face Recognition System")
    
    # Show appropriate page based on selection
    if page == "🏠 Home":
        home_page()
    elif page == "👤 Recognition":
        recognition_page()
    elif page == "🔐 Enrollment":
        enrollment_page()
    elif page == "🧠 Training":
        training_page()
    elif page == "👥 User Management":
        user_management_page()
    elif page == "⚙️ Detector Settings":
        detector_settings_page()
        
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.text("Made with ❤️ by Team EE4228")
    
    # Display debug info for development
    if st.sidebar.checkbox("Show Debug Info", False):
        st.sidebar.text(st.session_state.debug_info)

if __name__ == "__main__":
    main()

# Display additional information
st.sidebar.markdown("---")
st.sidebar.write("""
### Instructions
1. Start with User Management to add users
2. Train each user in Training mode
3. Use Recognition mode to identify people
4. Configure face detection in Settings
""")

st.sidebar.markdown("---")
st.sidebar.write("""
### System Status
- Total Users: {}
- Trained Users: {}
- AT&T Integration: {}
- AT&T Weight: {:.1f}
- Current Detector: {}
""".format(
    len(st.session_state.registered_users),
    sum(1 for info in st.session_state.registered_users.values() if info['trained']),
    "Enabled" if st.session_state.use_att_database else "Disabled",
    st.session_state.att_data_weight,
    st.session_state.detector_type.upper()
))

def home_page():
    """Display the home page with application overview"""
    st.markdown("""
    ## Welcome to Face Recognition System
    
    This application demonstrates a complete face recognition pipeline, using traditional computer vision techniques:
    
    ### Key Features:
    
    - **Face Detection**: Using HOG or Viola-Jones (Haar Cascade) method
    - **Face Preprocessing**: Alignment, illumination normalization, and standardization
    - **Face Recognition**: Using PCA (Eigenfaces) and KNN classifier
    
    ### How to Use:
    
    1. **🔐 Enrollment**: Capture 10+ images of each person you want to recognize
    2. **🧠 Training**: Train the recognition model on the captured images
    3. **👤 Recognition**: Use live camera to recognize faces
    4. **⚙️ Settings**: Configure detector parameters and other options
    
    ### Implementation Details:
    
    - **HOG-based Face Detector**: The Histogram of Oriented Gradients (HOG) detector computes gradients at each pixel, groups them into cells, and creates feature descriptors that are fed to a linear SVM for face/non-face classification.
    
    - **PCA + KNN Recognition (Eigenfaces)**: Principal Component Analysis reduces dimensionality by finding the principal components (eigenfaces) of the face dataset. K-Nearest Neighbors classification is then performed in this reduced space.
    
    - **Preprocessing Pipeline**: Includes grayscale conversion, face alignment using facial landmarks, illumination normalization (CLAHE, histogram equalization, or Tan & Triggs), and standardization.
    """)
    
    # Model status information
    st.markdown("## System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Information")
        if st.session_state.model_trained:
            st.success("✅ Model Status: Trained")
            if st.session_state.using_specialized_model:
                st.info("Using specialized model")
        else:
            st.warning("⚠️ Model Status: Not Trained")
            st.markdown("Go to the **Training** page to train a model.")
    
    with col2:
        st.markdown("### User Information")
        if st.session_state.registered_users:
            st.success(f"✅ Registered Users: {len(st.session_state.registered_users)}")
            for name in st.session_state.registered_users:
                if "images" in st.session_state.registered_users[name]:
                    img_count = st.session_state.registered_users[name]["images"]
                    st.markdown(f"- **{name}**: {img_count} images")
                else:
                    st.markdown(f"- **{name}**")
        else:
            st.warning("⚠️ No users registered")
            st.markdown("Go to the **Enrollment** page to register users.")
    
    # Add quick action buttons
    st.markdown("## Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚡ Go to Recognition"):
            st.session_state.current_page = "recognition"
            st.rerun()
    
    with col2:
        if st.button("👤 Enroll New User"):
            st.session_state.current_page = "enrollment"
            st.rerun()
    
    with col3:
        if st.button("🧠 Train Model"):
            st.session_state.current_page = "training"
            st.rerun() 

def recognition_page():
    """Face recognition page with enhanced UI and multiple face recognition"""
    st.subheader("👤 Face Recognition")
    
    # Load model if not loaded
    if not hasattr(st.session_state.recognizer, 'trained') or not st.session_state.recognizer.trained:
        try:
            if os.path.exists(st.session_state.model_path):
                success = st.session_state.recognizer.load_model(st.session_state.model_path)
                if success:
                    st.session_state.model_trained = True
                    
                    # Set appropriate threshold based on model type
                    if st.session_state.using_specialized_model:
                        st.session_state.face_threshold = 0.1  # Lower threshold for specialized model
                        st.info("Using specialized model with adjusted threshold")
                    else:
                        st.session_state.face_threshold = 0.2  # Standard threshold
                else:
                    st.warning("Failed to load the model. Try training a new one.")
                    return
            else:
                st.warning("No trained model found. Please train the model first.")
                return
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    
    # Advanced face alignment section
    with st.expander("Face Alignment Settings", expanded=False):
        # Check for dlib
        if not DLIB_AVAILABLE:
            st.warning("dlib is not installed. Advanced face alignment is not available.")
            st.info("To enable advanced face alignment, install dlib with: pip install dlib")
            st.session_state.use_face_alignment = False
        else:
            st.session_state.use_face_alignment = st.checkbox(
                "Use Advanced Face Alignment",
                value=st.session_state.use_face_alignment,
                help="Uses facial landmarks for better face alignment"
            )
            
            # Check if landmark model exists
            landmark_file = "shape_predictor_68_face_landmarks.dat"
            if st.session_state.use_face_alignment and not os.path.exists(landmark_file):
                st.warning("Facial landmark model file not found")
                if st.button("Download Landmark Model"):
                    with st.spinner("Downloading facial landmark model..."):
                        if download_dlib_model():
                            st.success("Landmark model downloaded successfully!")
                        else:
                            st.error("Failed to download landmark model")
    
    # Advanced recognition settings
    with st.expander("Recognition Settings", expanded=False):
        # Confidence threshold adjustment
        new_threshold = st.slider(
            "Recognition Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.face_threshold,
            step=0.05,
            help="Lower values make recognition more sensitive but may increase false positives"
        )
        if new_threshold != st.session_state.face_threshold:
            st.session_state.face_threshold = new_threshold
        
        # Select detector type
        detector_type = st.radio(
            "Face Detector",
            ["Viola-Jones (Haar)", "HOG (dlib)"],
            index=1 if st.session_state.detector_type == "dlib" else 0,
            help="HOG is generally more accurate but may be slower"
        )
        st.session_state.detector_type = "dlib" if "HOG" in detector_type else "viola_jones"
    
    # Start recognition section
    st.markdown("### Live Recognition")
    
    # Initialize streaming state if needed
    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False
    
    # Start/stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.is_streaming:
            if st.button("▶️ Start Recognition"):
                st.session_state.is_streaming = True
                st.experimental_rerun()
    with col2:
        if st.session_state.is_streaming:
            if st.button("⏹️ Stop Recognition"):
                st.session_state.is_streaming = False
                # Release camera resource
                if 'camera' in st.session_state and st.session_state.camera is not None:
                    st.session_state.camera.release()
                    st.session_state.camera = None
                st.experimental_rerun()
    
    # Run recognition if streaming is enabled
    if st.session_state.is_streaming:
        recognition_container = st.container()
        with recognition_container:
            recognize_video() 