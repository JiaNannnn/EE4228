import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
import time
import json

from face_detector import FaceDetector
from face_preprocessor import FacePreprocessor
from face_recognizer import FaceRecognizer
from pca_face_detector import PCAFaceDetector
from face_database_loader import FaceDatabaseLoader

# Initialize session state variables
if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 0
if 'is_capturing' not in st.session_state:
    st.session_state.is_capturing = False
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

def capture_and_save(frame, person_name):
    """Capture and save face images for training"""
    try:
        detector = get_current_detector()
        faces = detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face_roi = detector.get_face_roi(frame, (x, y, w, h))
            
            # Debug information
            st.session_state.debug_info = f"ROI shape: {face_roi.shape}"
            
            processed_face = preprocessor.preprocess(face_roi)
            
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
    """Train the face recognition model"""
    try:
        X = []
        y = []
        
        # Collect training data
        for person_name in os.listdir(gallery_dir):
            person_dir = os.path.join(gallery_dir, person_name)
            if os.path.isdir(person_dir):
                person_images = []
                for img_file in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.size > 0:
                        person_images.append(img)
                
                # Only include person if they have enough images
                if len(person_images) >= 20:
                    X.extend(person_images)
                    y.extend([person_name] * len(person_images))
                    
                    # Update user status
                    if person_name in st.session_state.registered_users:
                        st.session_state.registered_users[person_name]['trained'] = True
                        save_registered_users()
        
        if len(X) == 0:
            st.session_state.debug_info = "No valid training images found"
            return False
            
        # Debug information
        st.session_state.debug_info = f"Training with {len(X)} images for {len(set(y))} users"
        st.session_state.debug_info += f"\nImage shape: {X[0].shape}"
        
        # Train the model
        st.session_state.recognizer.train(np.array(X), np.array(y))
        st.session_state.model_trained = True
        
        # Save training state
        model_state = {
            'model_trained': True,
            'trained_users': list(set(y))
        }
        with open('model_state.json', 'w') as f:
            json.dump(model_state, f)
            
        return True
    except Exception as e:
        st.session_state.debug_info += f"\nError in train_model: {str(e)}"
        return False

def load_model_state():
    """Load model state and retrain if necessary"""
    try:
        if os.path.exists('model_state.json'):
            with open('model_state.json', 'r') as f:
                model_state = json.load(f)
            
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
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{user}**")
            with col2:
                st.write("Status: " + ("✅ Trained" if info['trained'] else "❌ Not trained"))
            with col3:
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

def recognition_page():
    st.title("Face Recognition System - Recognition")
    
    # Load or check model state
    load_model_state()
    
    # Check if we need to retrain
    if not st.session_state.model_trained and any(info['trained'] for info in st.session_state.registered_users.values()):
        st.info("Retraining model with all users...")
        with st.spinner("Training in progress..."):
            if train_model():
                st.success("Model retrained successfully!")
            else:
                st.error("Training failed. Please check debug information.")
                return
    
    if not st.session_state.model_trained:
        st.warning("Model not trained yet. Please complete training first!")
        return
    
    if not any(info['trained'] for info in st.session_state.registered_users.values()):
        st.warning("No trained users found. Please complete training first!")
        return
    
    st.write("### Recognition Mode")
    cam_placeholder = st.empty()
    status_placeholder = st.empty()
    debug_placeholder = st.empty()
    
    # Add confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Adjust this value to control recognition sensitivity"
    )
    
    # Display current detector
    st.sidebar.write(f"Using {st.session_state.detector_type.upper()} detector")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    detector = get_current_detector()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera!")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            status_text = []
            
            # Process faces
            for (x, y, w, h) in faces:
                try:
                    face_roi = detector.get_face_roi(frame, (x, y, w, h))
                    processed_face = preprocessor.preprocess(face_roi)
                    
                    prediction, confidence = st.session_state.recognizer.predict(processed_face)
                    
                    rect_color = (0, 0, 255)  # Red for unknown
                    text_color = (0, 0, 255)
                    name_text = "Unknown"
                    
                    if confidence > confidence_threshold:
                        rect_color = (0, 255, 0)  # Green for recognized
                        text_color = (0, 255, 0)
                        name_text = f"{prediction} ({confidence:.2f})"
                        status_text.append(f"Recognized: {name_text}")
                    else:
                        status_text.append(f"Unknown Person (conf: {confidence:.2f})")
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                    cv2.putText(frame, name_text,
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.9, text_color, 2)
                    
                except Exception as e:
                    st.session_state.debug_info = f"Recognition error: {str(e)}"
                    status_text.append("Recognition error")
            
            if status_text:
                status_placeholder.write("### Recognition Status")
                for status in status_text:
                    status_placeholder.write(status)
            else:
                status_placeholder.write("No faces detected")
            
            debug_placeholder.text(st.session_state.debug_info)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            time.sleep(0.1)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        cap.release()

# Main app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Mode", ["User Management", "Training", "Recognition", "Detector Settings"])

if page == "User Management":
    user_management_page()
elif page == "Training":
    training_page()
elif page == "Recognition":
    recognition_page()
else:
    detector_settings_page()

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
- Images Required: 20
- Current Detector: {}
""".format(
    len(st.session_state.registered_users),
    sum(1 for info in st.session_state.registered_users.values() if info['trained']),
    st.session_state.detector_type.upper()
)) 