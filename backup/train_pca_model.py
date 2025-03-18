import os
import numpy as np
import cv2
import streamlit as st
import json
from face_database_loader import FaceDatabaseLoader
from pca_face_detector import PCAFaceDetector
from face_preprocessor import FacePreprocessor
from face_recognizer import FaceRecognizer

# Initialize components
database_loader = FaceDatabaseLoader()
preprocessor = FacePreprocessor(target_size=(100, 100))
pca_detector = PCAFaceDetector()
recognizer = FaceRecognizer()

def download_and_train_pca():
    """Download face databases and train PCA model"""
    print("Starting download and training process...")
    
    # Step 1: Download databases
    databases_to_download = ['yale', 'att', 'georgia_tech']
    successful_downloads = []
    
    for db in databases_to_download:
        print(f"Attempting to download {db} database...")
        try:
            if database_loader.download_database(db):
                successful_downloads.append(db)
                print(f"Successfully downloaded {db} database!")
            else:
                print(f"Failed to download {db} database.")
        except Exception as e:
            print(f"Error downloading {db} database: {str(e)}")
    
    print(f"Successfully downloaded {len(successful_downloads)} out of {len(databases_to_download)} databases.")
    
    if not successful_downloads:
        print("No databases downloaded successfully. Falling back to LFW and synthetic databases.")
        successful_downloads = ['lfw', 'synthetic']
    
    # Step 2: Gather face samples for training
    print("Gathering face samples for training...")
    all_face_samples = []
    all_non_face_samples = []
    
    # Get face samples from successful databases
    for db in successful_downloads:
        try:
            faces, _ = database_loader.load_database(db, target_size=(100, 100))
            if len(faces) > 0:
                all_face_samples.append(faces)
                print(f"Loaded {len(faces)} face samples from {db}")
        except Exception as e:
            print(f"Error loading face samples from {db}: {str(e)}")
    
    # Get additional face samples and non-face samples
    try:
        # Use the built-in methods to get more face samples and non-face samples
        more_faces = database_loader.get_face_samples(num_samples=3000)
        non_faces = database_loader.get_non_face_samples(num_samples=3000)
        
        if len(more_faces) > 0:
            all_face_samples.append(more_faces)
            print(f"Loaded {len(more_faces)} additional face samples")
        
        print(f"Loaded {len(non_faces)} non-face samples")
        all_non_face_samples = non_faces
    except Exception as e:
        print(f"Error loading additional samples: {str(e)}")
    
    # Combine all face samples
    if all_face_samples:
        combined_faces = np.concatenate(all_face_samples)
        num_faces = len(combined_faces)
        print(f"Total face samples for training: {num_faces}")
    else:
        print("No face samples found for training. Exiting.")
        return False
    
    if len(all_non_face_samples) == 0:
        print("No non-face samples found. Generating synthetic non-faces.")
        all_non_face_samples = database_loader._generate_synthetic_non_faces(num_samples=3000)
    
    # Step 3: Train PCA face detector
    print("Training PCA face detector...")
    try:
        if pca_detector.train(combined_faces, all_non_face_samples):
            print("PCA face detector trained successfully!")
            
            # Save the trained model
            os.makedirs('models', exist_ok=True)
            pca_detector.save_model('models/pca_detector.joblib')
            print("PCA detector model saved to models/pca_detector.joblib")
            
            # Update model state
            model_state = {
                'model_trained': True,
                'pca_trained': True,
                'face_samples': num_faces,
                'non_face_samples': len(all_non_face_samples),
                'databases_used': successful_downloads
            }
            
            # Load existing model state if it exists
            try:
                if os.path.exists('model_state.json'):
                    with open('model_state.json', 'r') as f:
                        existing_state = json.load(f)
                    # Merge with existing state
                    model_state.update(existing_state)
            except:
                pass
                
            with open('model_state.json', 'w') as f:
                json.dump(model_state, f)
            
            print("Model state updated.")
            return True
        else:
            print("PCA detector training failed.")
            return False
    except Exception as e:
        print(f"Error during PCA training: {str(e)}")
        return False

# Step 4: Optional - Train face recognizer with PCA dimensionality reduction
def train_face_recognizer_with_pca():
    """Train face recognizer using PCA for dimensionality reduction"""
    print("Training face recognizer with PCA dimensionality reduction...")
    
    # Check if we have a gallery directory with face images
    gallery_dir = "gallery"
    if not os.path.exists(gallery_dir) or not os.listdir(gallery_dir):
        print("No face images found in gallery directory. Skipping recognizer training.")
        return False
    
    try:
        # Configure recognizer to use PCA
        recognizer.use_dimensionality_reduction = True
        recognizer.n_components = 0.95  # Retain 95% of variance
        
        # Get training data from gallery
        X = []
        y = []
        
        for person_dir in os.listdir(gallery_dir):
            person_path = os.path.join(gallery_dir, person_dir)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            processed_img = preprocessor.preprocess(img)
                            if processed_img is not None:
                                X.append(processed_img.flatten())
                                y.append(person_dir)
        
        if not X:
            print("No valid face images found in gallery. Skipping recognizer training.")
            return False
            
        X = np.array(X)
        y = np.array(y)
        
        # Train the recognizer
        if recognizer.train(X, y):
            print(f"Face recognizer trained successfully with {len(X)} images of {len(set(y))} people!")
            
            # Save the trained model
            os.makedirs('models', exist_ok=True)
            recognizer.save_model('models/face_recognizer.joblib')
            print("Face recognizer model saved to models/face_recognizer.joblib")
            
            # Update model state
            if os.path.exists('model_state.json'):
                with open('model_state.json', 'r') as f:
                    model_state = json.load(f)
            else:
                model_state = {}
                
            model_state.update({
                'recognizer_trained': True,
                'trained_users': list(set(y)),
                'pca_dim_reduction': True,
                'training_samples': len(X)
            })
            
            with open('model_state.json', 'w') as f:
                json.dump(model_state, f)
                
            print("Model state updated.")
            return True
        else:
            print("Face recognizer training failed.")
            return False
    except Exception as e:
        print(f"Error during face recognizer training: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Face Recognition System Training ===")
    print("1. Downloading face databases and training PCA face detector")
    success = download_and_train_pca()
    
    if success:
        print("\n2. Training face recognizer with PCA dimensionality reduction")
        train_face_recognizer_with_pca()
    
    print("\nTraining process completed.") 