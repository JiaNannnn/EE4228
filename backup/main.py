import os
import cv2
import numpy as np
import time
import argparse
import logging
from pathlib import Path

# Import custom modules
from face_utils import FaceDetector, FacePreprocessor, capture_training_faces, augment_face_images, load_external_databases
from user_manager import UserManager, get_user_summary
from pca_model import train_and_save_pca_model, load_pca_model, create_hybrid_pca_lda_model, load_hybrid_model
from recognition import FaceRecognizer, draw_recognition_results

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_recognition')

# Constants
TARGET_SIZE = (100, 100)
CAMERA_ID = 0
MODEL_DIR = "models"
GALLERY_DIR = "gallery"
LOG_DIR = "logs"
DATA_DIR = "data"
CONFIDENCE_THRESHOLD = 0.5

def setup_directories():
    """Create necessary directories"""
    dirs = [MODEL_DIR, GALLERY_DIR, LOG_DIR, DATA_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    logger.info("Directories created")

def initialize_camera():
    """Initialize camera with fallback options"""
    # Try different backends
    if os.name == 'nt':  # Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:  # Linux/Mac
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends:
        camera = cv2.VideoCapture(CAMERA_ID, backend)
        if camera.isOpened():
            # Set camera properties
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize delay
            
            logger.info(f"Camera initialized using backend {backend}")
            return camera
    
    # Try different camera IDs as a last resort
    for cam_id in range(3):
        if cam_id == CAMERA_ID:
            continue
        camera = cv2.VideoCapture(cam_id)
        if camera.isOpened():
            logger.info(f"Camera initialized using camera ID {cam_id}")
            return camera
    
    logger.error("Failed to initialize camera")
    return None

def train_pca_model_with_external_databases():
    """Train a PCA model using external face databases or user images if no external data found"""
    # Load external databases
    faces, labels = load_external_databases(target_size=TARGET_SIZE)
    
    # Check if external databases were found
    if len(faces) == 0:
        logger.warning("No external face data found. Attempting to use registered user data instead.")
        
        # Use user data instead
        user_manager = UserManager(gallery_dir=GALLERY_DIR)
        trained_users = user_manager.get_trained_users()
        untrained_users = user_manager.get_untrained_users()
        
        if not trained_users and not untrained_users:
            logger.error("No user data found. Please register users with face images first.")
            return None
        
        # Collect face images from user galleries
        all_faces = []
        all_labels = []
        preprocessor = FacePreprocessor(target_size=TARGET_SIZE)
        
        # Process trained users first
        for username in trained_users:
            logger.info(f"Loading gallery images for user: {username}")
            user_faces = user_manager.load_gallery_images(username, preprocessor)
            
            if len(user_faces) > 0:
                all_faces.append(user_faces)
                all_labels.extend([username] * len(user_faces))
                logger.info(f"  Loaded {len(user_faces)} images")
        
        # Also process untrained users that may have images
        for username in untrained_users:
            image_count = user_manager.count_gallery_images(username)
            if image_count > 0:
                logger.info(f"Loading gallery images for untrained user: {username}")
                user_faces = user_manager.load_gallery_images(username, preprocessor)
                
                if len(user_faces) > 0:
                    all_faces.append(user_faces)
                    all_labels.extend([username] * len(user_faces))
                    logger.info(f"  Loaded {len(user_faces)} images")
        
        # Check if we found any user faces
        if not all_faces:
            logger.error("No face images found in user galleries. Please capture training images first.")
            return None
        
        # Combine user faces
        faces = np.vstack(all_faces)
        labels = np.array(all_labels)
        logger.info(f"Using {len(faces)} user face images for training from {len(np.unique(labels))} users")
        
        # Augment data to improve model robustness
        logger.info("Augmenting face images...")
        augmented_faces = []
        augmented_labels = []
        
        for user in np.unique(labels):
            user_indices = np.where(labels == user)[0]
            user_faces = faces[user_indices]
            
            # Apply augmentation (create 3 variations per face)
            user_faces_augmented = augment_face_images(user_faces, num_augmentations=3)
            
            augmented_faces.append(user_faces_augmented)
            augmented_labels.extend([user] * len(user_faces_augmented))
        
        faces = np.vstack(augmented_faces)
        labels = np.array(augmented_labels)
        logger.info(f"After augmentation: {len(faces)} face images")
    
    # Train PCA model
    model_path = os.path.join(MODEL_DIR, 'pca_face_model.joblib')
    model_data = train_and_save_pca_model(
        faces, 
        labels, 
        target_size=TARGET_SIZE, 
        variance_to_retain=0.95, 
        output_path=model_path,
        visualize=True
    )
    
    logger.info(f"PCA model trained with {len(faces)} face images")
    
    # After training PCA model, convert all user embeddings
    if model_data is not None:
        logger.info("Updating user embeddings with new PCA model...")
        user_manager = UserManager(gallery_dir=GALLERY_DIR)
        preprocessor = FacePreprocessor(target_size=TARGET_SIZE)
        
        # Process all registered users
        for username, user_data in user_manager.get_all_users().items():
            # Skip users with no gallery images
            if user_manager.count_gallery_images(username) == 0:
                continue
                
            # Load user images
            user_faces = user_manager.load_gallery_images(username, preprocessor)
            
            if len(user_faces) > 0:
                # Process face images with PCA model
                face_flat = user_faces.reshape(len(user_faces), -1)
                face_std = model_data['scaler'].transform(face_flat)
                face_pca = model_data['pca'].transform(face_std)
                
                # Use average embedding as user reference
                user_embedding = np.mean(face_pca, axis=0)
                
                # Store updated embedding
                user_manager.store_user_embedding(username, user_embedding)
                logger.info(f"Updated embedding for user: {username}")
    
    return model_data

def register_and_train_user(username, full_name=None, num_images=15):
    """Register a new user and train with their face"""
    # Initialize components
    user_manager = UserManager(gallery_dir=GALLERY_DIR)
    detector = FaceDetector()
    preprocessor = FacePreprocessor(target_size=TARGET_SIZE)
    
    # Register user
    if not user_manager.is_user_registered(username):
        user_manager.register_user(username, full_name)
    elif user_manager.is_user_trained(username):
        logger.info(f"User '{username}' is already registered and trained")
        return True
    
    # Initialize camera
    camera = initialize_camera()
    if camera is None:
        logger.error("Cannot capture training faces: Camera initialization failed")
        return False
    
    try:
        # Capture training faces
        logger.info(f"Capturing {num_images} training faces for user '{username}'")
        captured_faces = capture_training_faces(
            username, camera, detector, preprocessor, 
            num_images=num_images, output_dir=GALLERY_DIR
        )
        
        if len(captured_faces) == 0:
            logger.error(f"Failed to capture any faces for user '{username}'")
            return False
        
        # Augment face images
        logger.info("Augmenting face images")
        augmented_faces = augment_face_images(captured_faces, num_augmentations=3)
        
        # Load PCA model or train it if it doesn't exist
        pca_model_path = os.path.join(MODEL_DIR, 'pca_face_model.joblib')
        pca_model_exists = os.path.exists(pca_model_path)
        
        if pca_model_exists:
            logger.info("Loading existing PCA model")
            model_data = load_pca_model(pca_model_path)
        else:
            logger.info("No existing PCA model found, training with user faces only")
            # Train a minimal PCA model just with this user's faces
            model_data = train_and_save_pca_model(
                augmented_faces, 
                [username] * len(augmented_faces),
                target_size=TARGET_SIZE,
                output_path=pca_model_path
            )
        
        # Process user's face with PCA model
        face_flat = augmented_faces.reshape(len(augmented_faces), -1)
        face_std = model_data['scaler'].transform(face_flat)
        face_pca = model_data['pca'].transform(face_std)
        
        # Use average embedding as the user's reference
        user_embedding = np.mean(face_pca, axis=0)
        
        # Store user embedding
        user_manager.store_user_embedding(username, user_embedding)
        
        logger.info(f"User '{username}' trained successfully")
        
        # If this is the first user and we trained a minimal model, suggest retraining
        if not pca_model_exists:
            logger.info("A minimal PCA model was created using only this user's faces.")
            logger.info("For better recognition, train a more robust model with:")
            logger.info("  python main.py train-model")
        
        return True
    
    except Exception as e:
        logger.error(f"Error training user '{username}': {str(e)}")
        return False
    
    finally:
        # Release camera
        if camera is not None and camera.isOpened():
            camera.release()

def recognize_faces_realtime():
    """Perform real-time face recognition"""
    # Initialize components
    recognizer = FaceRecognizer(
        pca_model_path=os.path.join(MODEL_DIR, 'pca_face_model.joblib'),
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    detector = FaceDetector()
    
    # Check if model is trained
    if not recognizer.is_model_trained():
        logger.error("Cannot perform recognition: PCA model not trained")
        return
    
    # Initialize camera
    camera = initialize_camera()
    if camera is None:
        logger.error("Cannot perform recognition: Camera initialization failed")
        return
    
    try:
        logger.info("Starting real-time face recognition")
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            # Read frame
            ret, frame = camera.read()
            if not ret:
                logger.warning("Error reading frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Detect and recognize faces
            results = recognizer.recognize_faces_in_image(frame)
            
            # Draw results
            if results:
                frame = draw_recognition_results(frame, results)
            
            # Calculate FPS
            fps_counter += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1.0:
                fps = fps_counter / elapsed_time
                fps_counter = 0
                fps_start_time = time.time()
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Face Recognition", frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        logger.error(f"Error during recognition: {str(e)}")
    
    finally:
        # Release resources
        if camera is not None and camera.isOpened():
            camera.release()
        cv2.destroyAllWindows()

def run_command_line_interface():
    """Run command-line interface for face recognition system"""
    parser = argparse.ArgumentParser(description="Face Recognition System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train PCA model
    train_parser = subparsers.add_parser("train-model", help="Train PCA model using external databases")
    
    # Register and train user
    register_parser = subparsers.add_parser("register", help="Register and train a new user")
    register_parser.add_argument("username", help="Username for registration")
    register_parser.add_argument("--full-name", help="Full name of the user")
    register_parser.add_argument("--images", type=int, default=15, help="Number of images to capture")
    
    # Recognize faces
    recognize_parser = subparsers.add_parser("recognize", help="Perform real-time face recognition")
    
    # List users
    list_parser = subparsers.add_parser("list-users", help="List registered users")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create directories
    setup_directories()
    
    # Execute command
    if args.command == "train-model":
        train_pca_model_with_external_databases()
    
    elif args.command == "register":
        register_and_train_user(args.username, args.full_name, args.images)
    
    elif args.command == "recognize":
        recognize_faces_realtime()
    
    elif args.command == "list-users":
        user_summary = get_user_summary()
        print("\n--- User Summary ---")
        print(f"Registered Users: {user_summary['registered_users']}")
        print(f"Trained Users: {user_summary['trained_users']}")
        print(f"Untrained Users: {user_summary['untrained_users']}")
        
        print("\n--- User Details ---")
        for username, data in user_summary['users'].items():
            trained = data.get('trained', False)
            image_count = user_summary['gallery_counts'].get(username, 0)
            print(f"Username: {username}")
            print(f"  Full Name: {data.get('full_name', username)}")
            print(f"  Trained: {trained}")
            print(f"  Gallery Images: {image_count}")
            print(f"  Registered At: {data.get('registered_at', 'Unknown')}")
            print()
    
    else:
        parser.print_help()

def run_interactive_mode():
    """Run interactive mode for face recognition system"""
    setup_directories()
    
    while True:
        print("\n=== Face Recognition System ===")
        print("1. Train PCA model using external databases")
        print("2. Register and train a new user")
        print("3. Perform real-time face recognition")
        print("4. List registered users")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            train_pca_model_with_external_databases()
        
        elif choice == "2":
            username = input("Enter username: ")
            full_name = input("Enter full name (optional): ")
            num_images = input("Enter number of images to capture (default: 15): ")
            
            try:
                num_images = int(num_images) if num_images else 15
            except ValueError:
                num_images = 15
            
            register_and_train_user(username, full_name or None, num_images)
        
        elif choice == "3":
            recognize_faces_realtime()
        
        elif choice == "4":
            user_summary = get_user_summary()
            print("\n--- User Summary ---")
            print(f"Registered Users: {user_summary['registered_users']}")
            print(f"Trained Users: {user_summary['trained_users']}")
            print(f"Untrained Users: {user_summary['untrained_users']}")
            
            print("\n--- User Details ---")
            for username, data in user_summary['users'].items():
                trained = data.get('trained', False)
                image_count = user_summary['gallery_counts'].get(username, 0)
                print(f"Username: {username}")
                print(f"  Full Name: {data.get('full_name', username)}")
                print(f"  Trained: {trained}")
                print(f"  Gallery Images: {image_count}")
                print(f"  Registered At: {data.get('registered_at', 'Unknown')}")
                print()
        
        elif choice == "5":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    # Check if any command-line arguments are provided
    if len(os.sys.argv) > 1:
        run_command_line_interface()
    else:
        run_interactive_mode() 