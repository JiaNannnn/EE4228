import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from datetime import datetime

from face_database_loader import FaceDatabaseLoader
from face_detector import FaceDetector
from face_preprocessor import FacePreprocessor
from face_recognizer import FaceRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')  # And to file
    ]
)
logger = logging.getLogger(__name__)

class FaceRecognizerTrainer:
    """
    Trainer class that orchestrates the face recognition training pipeline:
    1. Load gallery images
    2. Detect faces
    3. Preprocess faces
    4. Train recognizer
    5. Save model
    """
    def __init__(self, gallery_dir, target_size=(100, 100), min_face_size=(30, 30)):
        self.gallery_dir = Path(gallery_dir)
        self.target_size = target_size
        self.min_face_size = min_face_size
        
        # Initialize components
        self.detector = FaceDetector()
        self.preprocessor = FacePreprocessor(target_size=target_size)
        self.recognizer = FaceRecognizer(
            n_components=0.95,  # Keep 95% of variance
            n_neighbors=5,      # Use 5 neighbors for KNN
            reconstruction_error_threshold=0.02  # Threshold for unknown face detection
        )
        self.db_loader = FaceDatabaseLoader()
        
        # Create output directory for processed faces
        self.processed_dir = self.gallery_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
    def load_gallery_images(self):
        """Load and organize gallery images by person"""
        logger.info(f"Loading gallery images from {self.gallery_dir}")
        
        images_by_person = {}
        total_images = 0
        
        # Iterate through person directories
        for person_dir in self.gallery_dir.iterdir():
            if not person_dir.is_dir() or person_dir.name == 'processed':
                continue
                
            person_name = person_dir.name
            images_by_person[person_name] = []
            
            # Load all images for this person
            for img_path in person_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            images_by_person[person_name].append((img, img_path.name))
                            total_images += 1
                    except Exception as e:
                        logger.warning(f"Error loading image {img_path}: {e}")
        
        logger.info(f"Loaded {total_images} images for {len(images_by_person)} people")
        return images_by_person
        
    def process_gallery_images(self, images_by_person):
        """Detect faces and preprocess them"""
        logger.info("Processing gallery images...")
        
        processed_faces = []
        labels = []
        skipped = 0
        
        for person_name, images in images_by_person.items():
            person_dir = self.processed_dir / person_name
            person_dir.mkdir(exist_ok=True)
            
            logger.info(f"Processing images for {person_name}")
            for img, img_name in tqdm(images, desc=f"Processing {person_name}"):
                try:
                    # 1. Detect face
                    faces = self.detector.detect_faces(img)
                    if not faces:
                        logger.warning(f"No face detected in {img_name}")
                        skipped += 1
                        continue
                    
                    # Use the largest face if multiple detected
                    if len(faces) > 1:
                        logger.warning(f"Multiple faces detected in {img_name}, using largest")
                        areas = [(f[2] * f[3]) for f in faces]
                        face_rect = faces[np.argmax(areas)]
                    else:
                        face_rect = faces[0]
                    
                    # Extract face region
                    x, y, w, h = face_rect
                    face_img = img[y:y+h, x:x+w]
                    
                    # 2. Preprocess face
                    # Convert to grayscale if color
                    if len(face_img.shape) == 3:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    
                    # Align and normalize
                    face_img = self.preprocessor.align_face_with_landmarks(face_img)
                    face_img = self.preprocessor.enhance_image(face_img)
                    face_img = cv2.resize(face_img, self.target_size)
                    
                    # Save processed face
                    processed_path = person_dir / f"processed_{img_name}"
                    cv2.imwrite(str(processed_path), face_img)
                    
                    # Add to training data
                    processed_faces.append(face_img)
                    labels.append(person_name)
                    
                except Exception as e:
                    logger.error(f"Error processing {img_name}: {e}")
                    skipped += 1
                    continue
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} images due to errors or no face detected")
        
        return np.array(processed_faces), np.array(labels)
        
    def augment_with_external_data(self, faces, labels, use_att=True, use_lfw=True):
        """Optionally augment training data with external databases"""
        if not (use_att or use_lfw):
            return faces, labels
            
        logger.info("Augmenting training data with external databases...")
        
        if use_att:
            try:
                # Download and load AT&T database
                self.db_loader.download_database('att')
                att_faces, att_labels = self.db_loader.load_database('att', target_size=self.target_size)
                if att_faces is not None:
                    faces = np.concatenate([faces, att_faces])
                    labels = np.concatenate([labels, att_labels])
                    logger.info(f"Added {len(att_faces)} faces from AT&T database")
            except Exception as e:
                logger.warning(f"Failed to load AT&T database: {e}")
        
        if use_lfw:
            try:
                # Load subset of LFW database
                lfw_faces, lfw_labels = self.db_loader.load_database('lfw', target_size=self.target_size)
                if lfw_faces is not None:
                    # Only use a subset to avoid overwhelming gallery images
                    max_per_person = 20
                    unique_people = np.unique(lfw_labels)
                    subset_faces = []
                    subset_labels = []
                    for person in unique_people:
                        mask = lfw_labels == person
                        person_faces = lfw_faces[mask][:max_per_person]
                        subset_faces.extend(person_faces)
                        subset_labels.extend([person] * len(person_faces))
                    
                    faces = np.concatenate([faces, subset_faces])
                    labels = np.concatenate([labels, subset_labels])
                    logger.info(f"Added {len(subset_faces)} faces from LFW database")
            except Exception as e:
                logger.warning(f"Failed to load LFW database: {e}")
        
        return faces, labels
        
    def train(self, use_external_data=True, model_dir='models'):
        """Run the complete training pipeline"""
        try:
            # 1. Load gallery images
            images_by_person = self.load_gallery_images()
            if not images_by_person:
                raise ValueError("No gallery images found!")
            
            # 2. Process gallery images
            faces, labels = self.process_gallery_images(images_by_person)
            if len(faces) == 0:
                raise ValueError("No valid faces detected in gallery images!")
            
            logger.info(f"Processed {len(faces)} faces for {len(np.unique(labels))} people")
            
            # 3. Optionally augment with external data
            if use_external_data:
                faces, labels = self.augment_with_external_data(faces, labels)
            
            # 4. Train the recognizer
            logger.info("Training face recognizer...")
            self.recognizer.train(faces, labels)
            
            # 5. Save the trained model
            model_dir = Path(model_dir)
            model_dir.mkdir(exist_ok=True)
            self.recognizer.save_model(str(model_dir))
            logger.info(f"Model saved to {model_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train face recognition model on gallery images')
    parser.add_argument('gallery_dir', type=str, help='Directory containing person-specific subdirectories with face images')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save trained model')
    parser.add_argument('--no-external-data', action='store_true', help='Do not use external databases for training')
    parser.add_argument('--target-size', type=int, nargs=2, default=[100, 100], help='Target size for face images (width height)')
    args = parser.parse_args()
    
    trainer = FaceRecognizerTrainer(
        gallery_dir=args.gallery_dir,
        target_size=tuple(args.target_size)
    )
    
    success = trainer.train(
        use_external_data=not args.no_external_data,
        model_dir=args.model_dir
    )
    
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")
        exit(1)

if __name__ == '__main__':
    main() 