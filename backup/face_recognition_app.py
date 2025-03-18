"""
Face Recognition Web Application

A Streamlit web application to demonstrate the face recognition system.
The app allows users to:
1. Upload images to test face recognition
2. View preprocessed faces from the AT&T database
3. Explore eigenfaces and recognition results
4. Train models with different parameters
"""

import os
import sys
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import glob
from pathlib import Path
import logging

# Set page configuration at the beginning
st.set_page_config(
    page_title="Advanced Face Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from face_preprocessor import FacePreprocessor
    from face_recognition_att import FaceRecognizer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Make sure face_preprocessor.py and face_recognition_att.py are in the current directory.")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_recognition_app")

# Global variables
MODEL_DIR = "face_models"
PROCESSED_DIR = "processed_att"

def home_page():
    """Home page of the application"""
    st.title("Advanced Face Recognition System")
    
    st.markdown("""
    ### Welcome to the Advanced Face Recognition System!
    
    This application demonstrates a state-of-the-art face recognition system using:
    - Advanced face preprocessing techniques
    - PCA (eigenfaces) and LDA for feature extraction
    - SVM/KNN for classification
    
    The system has been trained on the AT&T Database of Faces, which contains 
    400 images of 40 individuals (10 images per person).
    
    ### Features
    
    **Advanced Face Preprocessing:**
    - Multiple face detection methods (DNN, HOG, Haar Cascade, MediaPipe)
    - Precise facial landmark detection using Dlib and MediaPipe
    - Smart face alignment based on facial landmarks
    - Multiple illumination normalization techniques
    - Attention mechanisms focusing on important facial regions
    - Smart and edge-aware cropping
    
    **Face Recognition:**
    - PCA for dimensionality reduction (eigenfaces)
    - Optional LDA for improved class separation
    - SVM and KNN classifiers
    
    ### Get Started
    
    Use the sidebar to navigate through the application:
    - **Recognize Face**: Upload an image and see if the system can recognize the face
    - **Explore Database**: View preprocessed faces from the AT&T database
    - **Visualize Eigenfaces**: Explore the eigenfaces used for recognition
    - **Model Performance**: View model performance metrics
    
    """)
    
    # Display sample images
    st.subheader("Sample Images from AT&T Database")
    
    # Get a few sample images
    sample_images = []
    sample_dirs = sorted(glob.glob(os.path.join(PROCESSED_DIR, "s*")))[:5]
    
    if sample_dirs:
        cols = st.columns(len(sample_dirs))
        
        for i, subject_dir in enumerate(sample_dirs):
            sample_files = sorted(glob.glob(os.path.join(subject_dir, "*_processed.jpg")))
            
            if sample_files:
                sample_file = sample_files[0]
                img = cv2.imread(sample_file)
                
                if img is not None:
                    # Convert from BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Display in column
                    with cols[i]:
                        st.image(img_rgb, caption=f"Subject {os.path.basename(subject_dir)}")
    else:
        st.info("No processed images found. Please process the AT&T database first.")
        
        # Show placeholder images
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.image("https://via.placeholder.com/92x112.png?text=Face+Image", caption=f"Subject {i+1}")

def recognize_face_page():
    """Page for uploading an image and recognizing a face"""
    st.title("Recognize Face")
    
    st.markdown("""
    Upload an image containing a face, and the system will:
    1. Detect the face using advanced detection methods
    2. Preprocess the face with alignment and normalization
    3. Extract features using trained eigenfaces
    4. Compare with known faces to find the closest match
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    # Preprocessing options
    with st.expander("Advanced Preprocessing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            detector_type = st.selectbox("Face Detector", 
                                        ["dnn", "hog", "haar", "mediapipe", "auto"],
                                        index=0)
            enable_landmarks = st.checkbox("Enable Landmarks", value=True)
            use_mediapipe = st.checkbox("Use MediaPipe", value=True)
        
        with col2:
            illumination_method = st.selectbox("Illumination Normalization", 
                                              ["clahe", "gamma", "dog", "multi"],
                                              index=3)
            enable_attention = st.checkbox("Enable Attention Mechanism", value=True)
            smart_crop = st.checkbox("Smart Cropping", value=True)
    
    # Recognition options
    with st.expander("Recognition Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            use_lda = st.checkbox("Use LDA after PCA", value=True)
            classifier = st.selectbox("Classifier", ["svm", "knn"], index=0)
        
        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    if uploaded_file is not None:
        # Display original image
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        img_array = np.array(pil_image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:  # Grayscale
            img = img_array
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(pil_image, use_column_width=True)
        
        # Process with progress indicator
        with st.spinner("Processing image..."):
            try:
                # Initialize face preprocessor
                preprocessor = FacePreprocessor(
                    target_size=(112, 92),  # Standard size for AT&T database
                    detector_type=detector_type,
                    enable_landmarks=enable_landmarks,
                    illumination_method=illumination_method,
                    use_mediapipe=use_mediapipe,
                    enable_attention=enable_attention,
                    smart_crop=smart_crop
                )
                
                # Detect faces
                faces = preprocessor.detect_faces(img)
                
                if not faces:
                    st.error("No faces detected in the uploaded image. Please try another image.")
                    return
                
                # Preprocess the largest face
                face_img = preprocessor.preprocess_face(img, faces[0])
                
                if face_img is None:
                    st.error("Failed to preprocess face. Please try another image.")
                    return
                
                # Initialize face recognizer
                recognizer = FaceRecognizer(processed_dir="", model_dir=MODEL_DIR)
                
                # Load models
                if not recognizer.load_models():
                    st.error(f"Failed to load models from {MODEL_DIR}")
                    st.info("Please make sure you have trained the model first.")
                    return
                
                # Recognize face
                predicted_label, confidence, subject_name = recognizer.recognize(face_img)
                
                # Display preprocessed face and recognition result
                with col2:
                    st.subheader("Preprocessed Face")
                    
                    # Convert to RGB for display
                    if len(face_img.shape) == 2:  # Grayscale
                        display_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                    else:  # BGR
                        display_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    st.image(display_img, use_column_width=True)
                
                # Display recognition result
                st.subheader("Recognition Result")
                
                if confidence >= confidence_threshold:
                    st.success(f"✅ Recognized as subject **{subject_name}** with {confidence:.2%} confidence")
                    
                    # Try to display original images of the recognized subject
                    subject_imgs = sorted(glob.glob(os.path.join(PROCESSED_DIR, subject_name, "*_processed.jpg")))
                    
                    if subject_imgs:
                        st.subheader("Reference Images for Subject " + subject_name)
                        
                        # Display up to 5 reference images
                        ref_imgs = []
                        for img_path in subject_imgs[:5]:
                            ref_img = cv2.imread(img_path)
                            if ref_img is not None:
                                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                                ref_imgs.append(ref_img)
                        
                        if ref_imgs:
                            st.image(ref_imgs, width=80)
                else:
                    st.warning(f"⚠️ Low confidence match: subject **{subject_name}** with {confidence:.2%} confidence")
                    st.info("The confidence is below the threshold. This might not be a reliable match.")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.exception("Error in face recognition")
    else:
        st.info("Please upload an image to start face recognition.")

def explore_database_page():
    """Page for exploring the processed AT&T database"""
    st.title("Explore Processed Database")
    
    st.markdown("""
    This page allows you to explore the processed faces from the AT&T database.
    Each subject has 10 different images with varying expressions and lighting.
    
    All images have been preprocessed with the advanced face preprocessing pipeline.
    """)
    
    # Check if processed directory exists
    if not os.path.isdir(PROCESSED_DIR):
        st.error(f"Processed directory not found: {PROCESSED_DIR}")
        st.info("Please process the AT&T database first using the process_and_train.py script.")
        return
    
    # Get all subject directories
    subject_dirs = sorted(glob.glob(os.path.join(PROCESSED_DIR, "s*")))
    
    if not subject_dirs:
        st.error(f"No subject directories found in {PROCESSED_DIR}")
        st.info("Please process the AT&T database first using the process_and_train.py script.")
        return
    
    # Create subject selector
    selected_subject = st.selectbox(
        "Select Subject",
        [os.path.basename(d) for d in subject_dirs]
    )
    
    # Get images for selected subject
    subject_path = os.path.join(PROCESSED_DIR, selected_subject)
    image_files = sorted(glob.glob(os.path.join(subject_path, "*_processed.jpg")))
    
    if not image_files:
        st.error(f"No processed images found for subject {selected_subject}")
        return
    
    # Display subject images
    st.subheader(f"Images for Subject {selected_subject}")
    
    # Calculate number of columns and rows needed
    num_cols = 5
    num_rows = (len(image_files) + num_cols - 1) // num_cols
    
    # Display images in grid
    for row in range(num_rows):
        cols = st.columns(num_cols)
        
        for col in range(num_cols):
            idx = row * num_cols + col
            
            if idx < len(image_files):
                img_path = image_files[idx]
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Convert from BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Display in column
                    with cols[col]:
                        st.image(img_rgb, caption=os.path.basename(img_path))

def visualize_eigenfaces_page():
    """Page for visualizing eigenfaces and model components"""
    st.title("Visualize Eigenfaces")
    
    st.markdown("""
    Eigenfaces are the principal components derived from the set of faces in the training data.
    They represent the major variations in the face dataset and are used for dimension reduction.
    
    The first few eigenfaces capture the most significant variations in the dataset.
    """)
    
    # Check if model directory exists
    if not os.path.isdir(MODEL_DIR):
        st.error(f"Model directory not found: {MODEL_DIR}")
        st.info("Please train the model first using the process_and_train.py script.")
        return
    
    # Check if PCA model exists
    pca_path = os.path.join(MODEL_DIR, "pca_model.pkl")
    if not os.path.isfile(pca_path):
        st.error(f"PCA model not found: {pca_path}")
        st.info("Please train the model first using the process_and_train.py script.")
        return
    
    # Display eigenfaces visualization if it exists
    eigenfaces_path = os.path.join(MODEL_DIR, "eigenfaces.png")
    if os.path.isfile(eigenfaces_path):
        st.image(eigenfaces_path, caption="Top Eigenfaces", use_column_width=True)
    else:
        st.warning("Eigenfaces visualization not found. Run the model training to generate it.")
    
    # Display number of eigenfaces to show
    num_eigenfaces = st.slider("Number of Eigenfaces to Display", 5, 50, 10, 5)
    
    # Load Face Recognizer
    try:
        recognizer = FaceRecognizer(processed_dir="", model_dir=MODEL_DIR)
        if recognizer.load_models():
            # Generate and display more eigenfaces
            recognizer.visualize_eigenfaces(n_eigenfaces=num_eigenfaces)
            
            # Display newly generated visualization
            if os.path.isfile(eigenfaces_path):
                st.image(eigenfaces_path, caption=f"Top {num_eigenfaces} Eigenfaces", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        logger.exception("Error visualizing eigenfaces")

def model_performance_page():
    """Page for displaying model performance"""
    st.title("Model Performance")
    
    st.markdown("""
    This page shows the performance metrics of the face recognition model.
    The model is evaluated using cross-validation to ensure reliable performance estimates.
    """)
    
    # Check if model directory exists
    if not os.path.isdir(MODEL_DIR):
        st.error(f"Model directory not found: {MODEL_DIR}")
        st.info("Please train the model first using the process_and_train.py script.")
        return
    
    # Display confusion matrix if it exists
    confusion_matrix_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    if os.path.isfile(confusion_matrix_path):
        st.subheader("Confusion Matrix")
        st.image(confusion_matrix_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.warning("Confusion matrix visualization not found. Run the model training to generate it.")
    
    # Display cross-validation accuracy if it exists
    cv_accuracy_path = os.path.join(MODEL_DIR, "cv_accuracy.png")
    if os.path.isfile(cv_accuracy_path):
        st.subheader("Cross-Validation Accuracy")
        st.image(cv_accuracy_path, caption="Cross-Validation Accuracy by Fold", use_column_width=True)
    else:
        st.warning("Cross-validation accuracy visualization not found. Run the model training to generate it.")
    
    # Option to retrain model with different parameters
    st.subheader("Retrain Model")
    
    st.markdown("""
    You can retrain the model with different parameters to compare performance.
    Note that this will overwrite the existing model files.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_lda = st.checkbox("Use LDA after PCA", value=True)
        classifier_type = st.selectbox("Classifier", ["svm", "knn"], index=0)
    
    with col2:
        k_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
        components = st.slider("PCA Components", 10, 100, 40)
    
    if st.button("Retrain Model"):
        try:
            with st.spinner("Retraining model... This may take a few minutes"):
                # Initialize face recognizer
                recognizer = FaceRecognizer(
                    processed_dir=PROCESSED_DIR,
                    model_dir=MODEL_DIR,
                    use_lda=use_lda,
                    n_components=components
                )
                
                # Load data
                num_images = recognizer.load_data()
                
                if num_images == 0:
                    st.error("No images loaded. Please check the processed directory.")
                    return
                
                st.info(f"Loaded {num_images} images for training")
                
                # Train model
                recognizer.train(classifier_type=classifier_type)
                
                # Evaluate performance
                results = recognizer.evaluate(k_folds=k_folds)
                
                if results is None:
                    st.error("Evaluation failed. Check the logs.")
                    return
                
                # Display results
                st.success("Model retrained successfully!")
                
                if k_folds > 1:
                    st.metric("Cross-validation accuracy", 
                             f"{results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
                else:
                    st.metric("Test accuracy", f"{results['mean_accuracy']:.4f}")
                
                # Visualize results
                recognizer.visualize_eigenfaces()
                recognizer.visualize_results(results)
                
                # Refresh page to show new visualizations
                st.experimental_rerun()
        
        except Exception as e:
            st.error(f"Error retraining model: {str(e)}")
            logger.exception("Error retraining model")

def main():
    """Main function for the Streamlit app"""
    # Set sidebar
    st.sidebar.title("Navigation")
    
    pages = {
        "Home": home_page,
        "Recognize Face": recognize_face_page,
        "Explore Database": explore_database_page,
        "Visualize Eigenfaces": visualize_eigenfaces_page,
        "Model Performance": model_performance_page
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Add information about the dataset
    st.sidebar.markdown("---")
    st.sidebar.subheader("About AT&T Database")
    st.sidebar.info(
        "The AT&T Database of Faces contains 400 images of 40 subjects, "
        "with 10 images per person. The images show variations in lighting, "
        "facial expressions, and facial details."
    )
    
    # Information about the model
    st.sidebar.subheader("Model Information")
    
    # Check if model exists
    model_exists = os.path.isdir(MODEL_DIR) and os.path.isfile(os.path.join(MODEL_DIR, "pca_model.pkl"))
    
    if model_exists:
        st.sidebar.success("✅ Model trained and ready")
    else:
        st.sidebar.warning("⚠️ Model not found")
        st.sidebar.info("Please run process_and_train.py to train the model")
    
    # Run the selected page
    pages[selection]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ using Streamlit")
    st.sidebar.markdown("© 2023 Advanced Face Recognition")

if __name__ == "__main__":
    main() 