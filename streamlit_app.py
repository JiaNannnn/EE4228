"""
Facial Recognition System - Streamlit Application

A web application for managing users, training models, and performing facial recognition.
"""

import os
import cv2
import numpy as np
import streamlit as st
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import logging
import shutil
import glob
from datetime import datetime
from config import (
    DATA_DIR, MODELS_DIR, FACE_SIZE, 
    MODEL_PATH, CONFIDENCE_THRESHOLD
)

# Import facial recognition components
from face_detector import FaceDetector
from face_preprocessor import FacePreprocessor
from face_recognizer import FaceRecognizer

# Configure logging
logging.basicConfig(
    filename=os.path.join("logs", f"streamlit_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streamlit_app")

# Create directories if they don't exist
USERS_DIR = Path(DATA_DIR) / "users"
USERS_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path(MODELS_DIR)
MODELS_DIR.mkdir(exist_ok=True)

# Initialize facial recognition components
detector = FaceDetector()
preprocessor = FacePreprocessor()
recognizer = FaceRecognizer()

# Try to load the model if it exists
model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        model_loaded = recognizer.load(MODEL_PATH)
        if model_loaded:
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Failed to load model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def preprocess_face(face):
    """
    Apply consistent preprocessing to a face image, used in both training and recognition
    
    Parameters:
    -----------
    face : numpy.ndarray
        Input face image
        
    Returns:
    --------
    numpy.ndarray
        Processed face image
    """
    try:
        # Ensure grayscale
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray = face
            
        # Resize to standard size
        resized = cv2.resize(gray, (FACE_SIZE, FACE_SIZE))
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(resized)
        
        return equalized
    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        return None

# Helper functions
def get_registered_users():
    """
    Get list of registered users
    
    Returns:
    --------
    list
        List of user IDs
    """
    users = []
    for user_dir in USERS_DIR.glob("*"):
        if user_dir.is_dir():
            users.append(user_dir.name)
    return sorted(users)

def get_user_images(user_id):
    """
    Get list of images for a user
    
    Parameters:
    -----------
    user_id : str
        User ID
        
    Returns:
    --------
    list
        List of image paths
    """
    user_dir = USERS_DIR / user_id
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(list(user_dir.glob(ext)))
    return image_paths

def count_user_images():
    """
    Count images for each user
    
    Returns:
    --------
    dict
        Dictionary with user IDs as keys and image counts as values
    """
    user_counts = {}
    for user_id in get_registered_users():
        user_counts[user_id] = len(get_user_images(user_id))
    return user_counts

def capture_and_save_face(user_id, min_faces=10):
    """
    Capture and save face images for a user
    
    Parameters:
    -----------
    user_id : str
        User ID
    min_faces : int
        Minimum number of face images to capture
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Create user directory if it doesn't exist
    user_dir = USERS_DIR / user_id
    user_dir.mkdir(exist_ok=True)
    
    # Get existing images to avoid overwriting
    existing_images = len(list(user_dir.glob("*.jpg")))
    
    # Create a placeholder for the webcam feed
    video_placeholder = st.empty()
    
    # Status information
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Start webcam
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            status_text.error("Error: Could not open webcam")
            return False
            
        # Counter for captured faces
        captured_count = 0
        frames_since_last_capture = 0
        
        # Continue until we have enough faces
        while captured_count < min_faces:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                status_text.error("Error: Failed to read frame from webcam")
                break
                
            # Mirror the frame (selfie mode)
            frame = cv2.flip(frame, 1)
                
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw rectangles around faces
            frame_with_faces = detector.draw_faces(frame, faces)
            
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            video_placeholder.image(frame_rgb, caption="Webcam Feed", use_container_width=True)
            
            # If a face is detected and it's time to capture
            if len(faces) == 1 and frames_since_last_capture > 5:  # Wait a few frames between captures
                face_rect = faces[0]
                face_img = detector.get_face_roi(frame, face_rect)
                
                # Save the face image
                image_path = user_dir / f"face_{existing_images + captured_count + 1}.jpg"
                cv2.imwrite(str(image_path), face_img)
                
                # Update counter
                captured_count += 1
                frames_since_last_capture = 0
                
                # Update progress
                progress = captured_count / min_faces
                progress_bar.progress(progress)
                status_text.info(f"Captured {captured_count}/{min_faces} faces")
            else:
                frames_since_last_capture += 1
                
            # Check for stop
            if st.button("Stop Capture"):
                break
                
            # Small delay to control frame rate
            time.sleep(0.1)
        
        # Release webcam
        cap.release()
        
        # Final update
        if captured_count >= min_faces:
            progress_bar.progress(1.0)
            status_text.success(f"Successfully captured {captured_count} faces for user {user_id}")
            return True
        else:
            status_text.warning(f"Capture interrupted. Only {captured_count}/{min_faces} faces captured.")
            return False
            
    except Exception as e:
        status_text.error(f"Error capturing faces: {e}")
        logger.error(f"Error capturing faces: {e}")
        return False

def train_model():
    """
    Train the face recognition model with both AT&T dataset and registered users
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Show progress
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Step 1: Get registered users
        progress_text.text("Loading user data...")
        users = get_registered_users()
        
        # Step 2: Load AT&T dataset
        progress_text.text("Loading AT&T dataset...")
        progress_bar.progress(0.1)
        
        att_images = []
        att_labels = []
        att_dataset_path = os.path.join(DATA_DIR, "AT&T")
        
        if not os.path.exists(att_dataset_path):
            st.warning("AT&T dataset not found. Training with user data only.")
        else:
            # Load AT&T dataset - consists of folders s1, s2, ..., s40
            att_subjects = [d for d in os.listdir(att_dataset_path) 
                           if os.path.isdir(os.path.join(att_dataset_path, d))]
            
            for subject_id in att_subjects:
                subject_path = os.path.join(att_dataset_path, subject_id)
                image_files = [f for f in os.listdir(subject_path) 
                              if f.endswith(('.pgm', '.jpg', '.png', '.jpeg'))]
                
                for img_file in image_files:
                    img_path = os.path.join(subject_path, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        # Process the face using the consistent preprocessing function
                        processed_face = preprocess_face(img)
                        if processed_face is not None:
                            # Add to training data
                            att_images.append(processed_face.flatten())
                            # Label as "ATT_s1", "ATT_s2", etc.
                            att_labels.append(f"ATT_{subject_id}")
            
            logger.info(f"Loaded {len(att_images)} images from AT&T dataset")
            progress_text.text(f"Loaded {len(att_images)} images from AT&T dataset")
        
        # Step 3: Load user images - directly use pre-processed images
        progress_text.text("Loading pre-processed user images...")
        progress_bar.progress(0.4)
        
        user_images = []
        user_labels = []
        
        if users:
            for user_id in users:
                # Get user images
                image_paths = get_user_images(user_id)
                
                # Load pre-processed images directly
                for img_path in image_paths:
                    try:
                        # Load image
                        img = cv2.imread(str(img_path))
                        
                        if img is not None:
                            # Process or verify preprocessing
                            processed_face = preprocess_face(img)
                            if processed_face is not None:
                                # Add to training data
                                user_images.append(processed_face.flatten())
                                user_labels.append(user_id)
                    except Exception as e:
                        logger.error(f"Error loading image {img_path}: {e}")
            
            logger.info(f"Loaded {len(user_images)} pre-processed images from {len(users)} users")
            progress_text.text(f"Loaded {len(user_images)} pre-processed images from {len(users)} users")
        
        # Step 4: Combine datasets
        progress_text.text("Combining datasets...")
        progress_bar.progress(0.6)
        
        all_images = []
        all_labels = []
        
        # First add AT&T images (if any)
        if att_images:
            all_images.extend(att_images)
            all_labels.extend(att_labels)
        
        # Then add user images (if any)
        if user_images:
            all_images.extend(user_images)
            all_labels.extend(user_labels)
        
        # Check if we have enough images
        if len(all_images) < 5:
            st.error("Not enough images for training. At least 5 images are required.")
            return False
        
        # Step 5: Train the model
        progress_text.text("Training model...")
        progress_bar.progress(0.8)
        
        # Convert to numpy arrays
        X = np.array(all_images)
        y = np.array(all_labels)
        
        # Get unique labels and create numerical labels
        unique_labels = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y_numeric = np.array([label_map[label] for label in y])
        
        # Store the label mapping for later use
        recognizer.label_map = label_map
        recognizer.reverse_map = {i: label for label, i in label_map.items()}
        
        # Train the model
        result = recognizer.train(X, y_numeric, test_size=0.2)
        accuracy = result.get('accuracy', 0.0)
        
        # Save the model and label mapping
        recognizer.save(MODEL_PATH)
        
        # Final update
        progress_bar.progress(1.0)
        progress_text.text(f"Model trained successfully with accuracy: {accuracy:.2%}")
        
        # Display summary
        att_count = len(att_images)
        user_count = len(user_images)
        total_count = att_count + user_count
        
        st.success(f"Model trained with {total_count} images:")
        st.info(f"- AT&T dataset: {att_count} images")
        st.info(f"- User images: {user_count} images ({len(users)} users)")
        st.info(f"- Total classes: {len(unique_labels)}")
        st.info(f"- Accuracy: {accuracy:.2%}")
        
        # Set model_loaded flag
        global model_loaded
        model_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        logger.error(f"Error training model: {e}")
        return False

def recognize_face_realtime():
    """
    Perform real-time face recognition using the webcam
    """
    global model_loaded
    
    if not model_loaded and not os.path.exists(MODEL_PATH):
        st.error("No trained model available. Please train the model first.")
        return
    
    # Load the model if not already loaded
    if not model_loaded:
        try:
            model_loaded = recognizer.load(MODEL_PATH)
            if not model_loaded:
                st.error(f"Failed to load model from {MODEL_PATH}")
                return
        except Exception as e:
            st.error(f"Error loading model: {e}")
            logger.error(f"Error loading model: {e}")
            return
    
    # Create a placeholder for the webcam feed
    video_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Start webcam
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera connection.")
            return
            
        # Add a stop button
        stop_button = st.button("Stop Recognition")
        
        while not stop_button:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to get frame from camera")
                break
                
            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detector.detect_faces(gray)
            
            # Process each face
            results = []
            for face_rect in faces:
                # Extract face ROI
                x, y, w, h = face_rect
                face_roi = gray[y:y+h, x:x+w]
                
                # Preprocess face
                try:
                    # Apply the same preprocessing used in training
                    processed_face = preprocess_face(face_roi)
                    
                    if processed_face is None:
                        continue
                        
                    # Predict identity
                    label, confidence = recognizer.predict(processed_face)
                    
                    # Determine color based on confidence (green=high, yellow=medium, red=low)
                    if label == "Unknown":
                        color = (0, 0, 255)  # Red for unknown
                    else:
                        if confidence > 0.7:
                            color = (0, 255, 0)  # Green
                        elif confidence > 0.5:
                            color = (0, 255, 255)  # Yellow
                        else:
                            color = (0, 128, 255)  # Orange
                    
                    # Display label text (strip 'ATT_' prefix for AT&T subjects)
                    display_label = label
                    if isinstance(label, str) and label.startswith("ATT_"):
                        display_label = label[4:]  # Remove "ATT_" prefix
                    
                    label_text = f"{display_label} ({confidence:.2f})"
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Add filled background for text
                    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x, y-25), (x+label_size[0], y), color, cv2.FILLED)
                    cv2.putText(frame, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # Add to results
                    if label != "Unknown":
                        results.append({
                            "ID": display_label,
                            "Confidence": f"{confidence:.2f}"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue
            
            # Display the processed frame
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            # Display results table
            if results:
                result_placeholder.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True
                )
            else:
                result_placeholder.info("No faces recognized")
            
            # Check if stop button was clicked
            if stop_button:
                break
                
            # Small delay to control frame rate
            time.sleep(0.1)
        
        # Release webcam
        cap.release()
        
    except Exception as e:
        st.error(f"Error during real-time recognition: {e}")
        logger.error(f"Error during real-time recognition: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def delete_user(user_id):
    """
    Delete a user and their images
    
    Parameters:
    -----------
    user_id : str
        User ID to delete
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        user_dir = USERS_DIR / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)
            logger.info(f"Deleted user {user_id}")
            return True
        else:
            logger.warning(f"User directory {user_id} does not exist")
            return False
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        return False

def display_user_samples(user_id, max_samples=5):
    """
    Display sample images for a user
    
    Parameters:
    -----------
    user_id : str
        User ID
    max_samples : int
        Maximum number of samples to display
    """
    image_paths = get_user_images(user_id)
    if not image_paths:
        st.warning(f"No images found for user {user_id}")
        return
        
    # Pick a few samples
    samples = image_paths[:min(max_samples, len(image_paths))]
    
    # Create columns for images
    cols = st.columns(len(samples))
    
    # Display images
    for i, img_path in enumerate(samples):
        img = Image.open(img_path)
        cols[i].image(img, caption=f"Sample {i+1}", use_container_width=True)

# Streamlit Application
def main():
    global model_loaded
    
    st.set_page_config(
        page_title="Facial Recognition System",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("Facial Recognition System")
    
    # Sidebar menu
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Choose a page", [
            "Dashboard", 
            "Manage Users", 
            "Add New User", 
            "Train Model", 
            "Live Recognition"
        ])
    
    # Dashboard page
    if page == "Dashboard":
        st.header("System Dashboard")
        
        # Display system status
        st.subheader("System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            users = get_registered_users()
            st.metric("Registered Users", len(users))
        
        with col2:
            user_counts = count_user_images()
            total_images = sum(user_counts.values())
            st.metric("Total Face Images", total_images)
        
        with col3:
            if model_loaded or os.path.exists(MODEL_PATH):
                st.success("Model Trained ✓")
            else:
                st.warning("Model Not Trained ✗")
        
        # User statistics
        if users:
            st.subheader("User Statistics")
            
            # Create a DataFrame for user stats
            user_data = []
            for user_id in users:
                user_data.append({
                    "User ID": user_id,
                    "Images": user_counts.get(user_id, 0)
                })
            
            user_df = pd.DataFrame(user_data)
            
            # Display stats
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.dataframe(user_df, use_container_width=True)
                
            with col2:
                # Create a bar chart of image counts
                fig, ax = plt.subplots()
                ax.bar(user_df["User ID"], user_df["Images"])
                ax.set_ylabel("Number of Images")
                ax.set_xlabel("User ID")
                ax.set_title("Face Images per User")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    # Manage Users page
    elif page == "Manage Users":
        st.header("Manage Users")
        
        # Get all users
        users = get_registered_users()
        
        if not users:
            st.warning("No users registered. Please add users first.")
        else:
            # Create tabs for user management
            user_tabs = st.tabs(users)
            
            for i, user_id in enumerate(users):
                with user_tabs[i]:
                    st.subheader(f"User: {user_id}")
                    
                    # Display user samples
                    st.write("Sample Images:")
                    display_user_samples(user_id)
                    
                    # User stats
                    image_count = len(get_user_images(user_id))
                    st.info(f"Total images: {image_count}")
                    
                    # Actions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"Add More Images", key=f"add_{user_id}"):
                            st.session_state['add_user_id'] = user_id
                            st.session_state['page'] = "Add New User"
                            st.experimental_rerun()
                            
                    with col2:
                        if st.button(f"Delete User", key=f"delete_{user_id}"):
                            if delete_user(user_id):
                                st.success(f"User {user_id} deleted successfully")
                                st.experimental_rerun()
                            else:
                                st.error(f"Failed to delete user {user_id}")
    
    # Add New User page
    elif page == "Add New User":
        st.header("Add New User")
        
        # Check if we're adding images to an existing user
        user_id = st.session_state.get('add_user_id', "")
        
        if user_id:
            st.subheader(f"Adding images for existing user: {user_id}")
        else:
            # Get user ID
            user_id = st.text_input("Enter User ID (name or identifier):")
            
            # Validate input
            if user_id:
                # Check if user already exists
                if (USERS_DIR / user_id).exists() and not st.session_state.get('add_user_id'):
                    st.warning(f"User {user_id} already exists. Adding more images.")
        
        # If we have a valid user ID, show the image capture interface
        if user_id:
            st.subheader("Capture Face Images")
            st.write("Please position your face in front of the camera. Images will be captured automatically.")
            
            # Number of images to capture
            min_faces = st.slider("Number of images to capture:", min_value=5, max_value=50, value=10)
            
            # Start capture button
            if st.button("Start Capture"):
                with st.spinner("Capturing face images..."):
                    success = capture_and_save_face(user_id, min_faces)
                    
                    if success:
                        st.session_state.pop('add_user_id', None)  # Clear the session state
                        # Show option to train model
                        if st.button("Train Model Now"):
                            st.session_state['page'] = "Train Model"
                            st.experimental_rerun()
    
    # Train Model page
    elif page == "Train Model":
        st.header("Train Face Recognition Model")
        
        # Check if there are registered users
        users = get_registered_users()
        if not users:
            st.warning("No users registered. Please add users first.")
        else:
            # Display user statistics
            user_counts = count_user_images()
            user_data = []
            for user_id in users:
                user_data.append({
                    "User ID": user_id,
                    "Images": user_counts.get(user_id, 0)
                })
            
            user_df = pd.DataFrame(user_data)
            st.dataframe(user_df, use_container_width=True)
            
            # Check if we have enough images
            if sum(user_counts.values()) < 5:
                st.warning("Not enough images for training. At least 5 images are required.")
            else:
                # Train model button
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        success = train_model()
                        
                        if success:
                            st.success("Model trained successfully!")
                            model_loaded = True
                            
                            # Show option to start recognition
                            if st.button("Start Recognition"):
                                st.session_state['page'] = "Live Recognition"
                                st.experimental_rerun()
                        else:
                            st.error("Failed to train model. Check logs for details.")
    
    # Live Recognition page
    elif page == "Live Recognition":
        st.header("Live Face Recognition")
        
        # Check if model is trained
        if not model_loaded and not os.path.exists(MODEL_PATH):
            st.warning("No trained model available. Please train the model first.")
            
            # Show option to go to training
            if st.button("Go to Training"):
                st.session_state['page'] = "Train Model"
                st.experimental_rerun()
        else:
            st.write("Press the button below to start real-time face recognition.")
            
            # Start recognition button
            if st.button("Start Recognition"):
                with st.spinner("Initializing camera..."):
                    recognize_face_realtime()

if __name__ == "__main__":
    main() 