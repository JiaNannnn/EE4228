import os
import requests
import zipfile
import tarfile
import numpy as np
import cv2
from tqdm import tqdm
import streamlit as st
import time
from sklearn.datasets import fetch_lfw_people
import random

class FaceDatabaseLoader:
    """Utility class to download and prepare external face databases"""
    
    DATABASES = {
        'att': {
            'url': 'https://cs.nyu.edu/~roweis/data/olivettifaces.mat',
            'description': 'Olivetti/AT&T Database: 400 images of 40 subjects',
            'type': 'mat'
        },
        'synthetic': {
            'url': None,  # Generated internally
            'description': 'Synthetic Face Database: Programmatically generated faces',
            'type': 'internal'
        },
        'lfw': {
            'url': None,  # Using scikit-learn's built-in fetcher
            'description': 'Labeled Faces in the Wild: Sample of face images',
            'type': 'internal'
        },
        'yale': {
            'url': 'http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip',
            'description': 'Yale Face Database B: 5760 images of 10 subjects under different lighting conditions',
            'type': 'zip'
        },
        'georgia_tech': {
            'url': 'http://www.anefian.com/research/face_reco.zip',
            'description': 'Georgia Tech Face Database: 750 images of 50 subjects with different expressions and lighting',
            'type': 'zip'
        }
    }
    
    def __init__(self, data_dir='external_databases', target_size=(100, 100)):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.background_dir = os.path.join(self.data_dir, 'background')
        os.makedirs(self.background_dir, exist_ok=True)
        self.target_size = target_size
        self.database_paths = {
            'att': ['data/att_faces', 'external_databases/att_faces', 'att_faces'],
            'yale': ['data/yale_faces', 'external_databases/yale_faces', 'yale_faces'],
            'backgrounds': ['backgrounds', 'data/backgrounds']
        }
        
    def list_available_databases(self):
        """List all available databases with descriptions"""
        st.write("### Available Face Databases:")
        for name, info in self.DATABASES.items():
            st.write(f"\n**{name.upper()}**:")
            st.write(info['description'])
            if name == 'synthetic' or name == 'lfw':
                st.write("✅ Always available (generated on-demand)")
            elif os.path.exists(os.path.join(self.data_dir, name)):
                st.write("✅ Downloaded")
            else:
                st.write("❌ Not downloaded")
        
        # Check background images
        if not os.listdir(self.background_dir):
            st.write("\n### Background Images (for non-face samples):")
            st.warning("No background images found. Will use synthetic non-face images.")
            st.write("Auto-generating background images...")
            self._generate_synthetic_backgrounds(10)
    
    def _generate_synthetic_backgrounds(self, count=10):
        """Generate synthetic background images for training"""
        for i in range(count):
            # Create random noise image
            noise = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            
            # Apply some filtering to make it look more natural
            noise = cv2.GaussianBlur(noise, (15, 15), 0)
            
            # Create gradient image
            gradient = np.zeros((512, 512), dtype=np.uint8)
            for y in range(512):
                gradient[y, :] = y // 2
            
            # Mix noise and gradient
            bg_image = cv2.addWeighted(noise, 0.7, gradient, 0.3, 0)
            
            # Add some random shapes
            for _ in range(np.random.randint(5, 15)):
                # Random rectangle
                pt1 = (np.random.randint(0, 400), np.random.randint(0, 400))
                pt2 = (pt1[0] + np.random.randint(50, 150), pt1[1] + np.random.randint(50, 150))
                color = np.random.randint(0, 255)
                thickness = np.random.randint(1, 5)
                cv2.rectangle(bg_image, pt1, pt2, color, thickness)
            
            # Save the image
            save_path = os.path.join(self.background_dir, f"synthetic_bg_{i}.jpg")
            cv2.imwrite(save_path, bg_image)
        
        st.success(f"Generated {count} synthetic background images")
        
    def _generate_synthetic_faces(self, num_samples=100, target_size=(100, 100)):
        """Generate synthetic face-like patterns for training"""
        st.info("Generating synthetic face patterns...")
        faces = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        for i in range(num_samples):
            # Create base oval face shape
            face = np.zeros(target_size, dtype=np.uint8)
            center = (target_size[0] // 2, target_size[1] // 2)
            axes = (target_size[0] // 3, target_size[1] // 2)
            cv2.ellipse(face, center, axes, 0, 0, 360, 200, -1)
            
            # Add eyes (two small circles)
            eye_y = target_size[1] // 3
            left_eye_x = target_size[0] // 3
            right_eye_x = target_size[0] * 2 // 3
            eye_radius = target_size[0] // 10
            
            cv2.circle(face, (left_eye_x, eye_y), eye_radius, 100, -1)
            cv2.circle(face, (right_eye_x, eye_y), eye_radius, 100, -1)
            
            # Add mouth (curved line)
            mouth_y = target_size[1] * 7 // 10
            mouth_width = target_size[0] // 2
            
            mouth_start = (target_size[0] // 4, mouth_y)
            mouth_end = (target_size[0] * 3 // 4, mouth_y)
            mouth_ctrl = (target_size[0] // 2, mouth_y + 10)
            
            for t in np.linspace(0, 1, 20):
                # Quadratic Bezier curve
                x = int((1-t)**2 * mouth_start[0] + 2*(1-t)*t*mouth_ctrl[0] + t**2*mouth_end[0])
                y = int((1-t)**2 * mouth_start[1] + 2*(1-t)*t*mouth_ctrl[1] + t**2*mouth_end[1])
                cv2.circle(face, (x, y), 1, 150, 2)
            
            # Add some noise and blur to make it realistic
            noise = np.random.randint(0, 30, target_size, dtype=np.uint8)
            face = cv2.add(face, noise)
            face = cv2.GaussianBlur(face, (5, 5), 0)
            
            faces.append(face)
            progress_bar.progress(min(1.0, (i + 1) / num_samples))
        
        return np.array(faces)
        
    def download_database(self, database_name):
        """Download or generate a specific face database"""
        if database_name not in self.DATABASES:
            raise ValueError(f"Unknown database: {database_name}")
            
        info = self.DATABASES[database_name]
        target_dir = os.path.join(self.data_dir, database_name)
        
        # Handle synthetic database
        if database_name == 'synthetic':
            st.info("Generating synthetic face database...")
            return True
            
        # Handle LFW through scikit-learn
        if database_name == 'lfw':
            st.info("Fetching LFW dataset through scikit-learn...")
            try:
                # This will download LFW if not already downloaded
                fetch_lfw_people(min_faces_per_person=10, resize=0.5)
                return True
            except Exception as e:
                st.error(f"Failed to fetch LFW dataset: {str(e)}")
                return False
                
        # Handle standard downloads
        archive_path = os.path.join(self.data_dir, f"{database_name}.{info['type']}")
        
        try:
            if os.path.exists(target_dir):
                st.info(f"Database {database_name} already exists in {target_dir}")
                return True
                
            st.info(f"Downloading {database_name} database...")
            
            # Download with progress bar
            response = requests.get(info['url'], stream=True, timeout=30)
            response.raise_for_status()  # Raise an error for bad status codes
            
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            downloaded_size = 0
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size:
                            progress = min(1.0, downloaded_size / total_size)
                            progress_bar.progress(progress)
            
            # Handle different file types
            os.makedirs(target_dir, exist_ok=True)
            
            if info['type'] == 'zip':
                # Verify archive
                if not zipfile.is_zipfile(archive_path):
                    raise ValueError("Downloaded file is not a valid ZIP archive")
                    
                st.info("Extracting files...")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                    
            elif info['type'] == 'tar':
                st.info("Extracting files...")
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(target_dir)
            
            elif info['type'] == 'mat':
                # Just copy the file to the target directory
                import shutil
                shutil.copy(archive_path, os.path.join(target_dir, os.path.basename(archive_path)))
                    
            # Clean up
            if os.path.exists(archive_path):
                os.remove(archive_path)
                
            st.success(f"Successfully downloaded {database_name} database!")
            return True
            
        except requests.exceptions.RequestException as e:
            st.error(f"Download failed: {str(e)}")
            if os.path.exists(archive_path):
                os.remove(archive_path)
            if os.path.exists(target_dir) and not os.listdir(target_dir):
                import shutil
                shutil.rmtree(target_dir)
            return False
            
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            if os.path.exists(archive_path):
                os.remove(archive_path)
            return False
        
    def load_database(self, database_name, target_size=(100, 100)):
        """Load and preprocess images from a database"""
        # Handle synthetic database
        if database_name == 'synthetic':
            faces = self._generate_synthetic_faces(num_samples=100, target_size=target_size)
            labels = np.array(['synthetic_face'] * len(faces))
            return faces, labels
            
        # Handle LFW through scikit-learn
        if database_name == 'lfw':
            try:
                st.info("Loading LFW dataset...")
                lfw = fetch_lfw_people(min_faces_per_person=10, resize=0.5)
                faces = []
                labels = []
                
                # Create progress bar
                progress_bar = st.progress(0)
                
                for i in range(len(lfw.images)):
                    face = cv2.resize(lfw.images[i], target_size)
                    faces.append(face)
                    labels.append(lfw.target_names[lfw.target[i]])
                    progress_bar.progress(min(1.0, (i + 1) / len(lfw.images)))
                
                return np.array(faces), np.array(labels)
            except Exception as e:
                st.error(f"Failed to load LFW dataset: {str(e)}")
                return np.array([]), np.array([])
        
        # Handle standard databases
        target_dir = os.path.join(self.data_dir, database_name)
        if not os.path.exists(target_dir):
            if not self.download_database(database_name):
                return np.array([]), np.array([])
            
        images = []
        labels = []
        
        st.info(f"Loading {database_name} database...")
        progress_bar = st.progress(0)
        
        # Special handling for .mat files (Olivetti/AT&T)
        if database_name == 'att':
            try:
                from scipy.io import loadmat
                mat_file = os.path.join(target_dir, 'olivettifaces.mat')
                if os.path.exists(mat_file):
                    mat_data = loadmat(mat_file)
                    faces = mat_data['faces'].T
                    for i in range(faces.shape[0]):
                        face = faces[i].reshape(64, 64)
                        face = cv2.resize(face, target_size)
                        images.append(face)
                        # Use subject ID as label (10 images per subject, 40 subjects)
                        label = f"subject_{i // 10 + 1}"
                        labels.append(label)
                        progress_bar.progress(min(1.0, (i + 1) / faces.shape[0]))
                    return np.array(images), np.array(labels)
            except Exception as e:
                st.error(f"Error loading AT&T database: {str(e)}")
                return np.array([]), np.array([])
        
        # Special handling for Yale Database
        elif database_name == 'yale':
            try:
                # Yale database has a specific directory structure
                count = 0
                total_files = 0
                
                # First count total files for progress bar
                for root, dirs, files in os.walk(target_dir):
                    total_files += len([f for f in files if f.endswith('.pgm')])
                
                if total_files == 0:
                    st.warning("No .pgm files found in Yale database directory")
                    return np.array([]), np.array([])
                
                # Then process files
                for root, dirs, files in os.walk(target_dir):
                    for file in files:
                        if file.endswith('.pgm'):
                            file_path = os.path.join(root, file)
                            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                            
                            if img is not None:
                                # Resize to target size
                                img = cv2.resize(img, target_size)
                                images.append(img)
                                
                                # Extract subject from directory path
                                # Directory format in Yale is usually yaleB##, where ## is the subject number
                                dir_name = os.path.basename(root)
                                if dir_name.startswith('yaleB'):
                                    subject = dir_name
                                else:
                                    subject = os.path.basename(os.path.dirname(root))
                                
                                labels.append(subject)
                                count += 1
                                progress_bar.progress(min(1.0, count / total_files))
                
                st.success(f"Loaded {len(images)} images from Yale Face Database")
                return np.array(images), np.array(labels)
            except Exception as e:
                st.error(f"Error loading Yale database: {str(e)}")
                return np.array([]), np.array([])
        
        # Special handling for Georgia Tech Face Database
        elif database_name == 'georgia_tech':
            try:
                # Georgia Tech database has a specific directory structure
                count = 0
                total_files = 0
                
                # First count total files for progress bar
                for root, dirs, files in os.walk(target_dir):
                    total_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                if total_files == 0:
                    st.warning("No image files found in Georgia Tech database directory")
                    return np.array([]), np.array([])
                
                # Then process files
                for root, dirs, files in os.walk(target_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            file_path = os.path.join(root, file)
                            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                            
                            if img is not None:
                                # Resize to target size
                                img = cv2.resize(img, target_size)
                                images.append(img)
                                
                                # Extract subject from directory path
                                subject = os.path.basename(root)
                                labels.append(subject)
                                count += 1
                                progress_bar.progress(min(1.0, count / total_files))
                
                st.success(f"Loaded {len(images)} images from Georgia Tech Face Database")
                return np.array(images), np.array(labels)
            except Exception as e:
                st.error(f"Error loading Georgia Tech database: {str(e)}")
                return np.array([]), np.array([])
        
        # Default handling for other databases (directory of images)
        else:
            # Standard image file handling
            file_list = []
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pgm')):
                        file_list.append((root, file))
            
            if not file_list:
                st.warning(f"No valid images found in {database_name} database")
                return np.array([]), np.array([])
            
            # Process files with progress bar
            for i, (root, file) in enumerate(file_list):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    try:
                        # Resize image
                        img = cv2.resize(img, target_size)
                        images.append(img)
                        
                        # Extract label from directory name
                        label = os.path.basename(root)
                        labels.append(label)
                    except Exception as e:
                        st.warning(f"Error processing {img_path}: {str(e)}")
                
                progress_bar.progress(min(1.0, (i + 1) / len(file_list)))
            
            if not images:
                st.warning(f"No valid images could be processed from {database_name} database")
                return np.array([]), np.array([])
            
            return np.array(images), np.array(labels)
        
    def _find_database_dir(self, database_name):
        """Find the directory for a specific database"""
        if database_name not in self.database_paths:
            return None
            
        for path in self.database_paths[database_name]:
            if os.path.exists(path) and os.path.isdir(path):
                return path
                
        return None
        
    def get_face_samples(self, database_names=None, num_samples=500):
        """Get face samples from specified databases"""
        if database_names is None:
            database_names = ['att', 'yale']
            
        face_samples = []
        
        for db_name in database_names:
            db_dir = self._find_database_dir(db_name)
            if db_dir:
                faces = self._load_faces_from_dir(db_dir)
                face_samples.extend(faces)
                print(f"Loaded {len(faces)} samples from {db_name} database")
            
        # If we don't have enough face samples, try to generate synthetic ones
        if len(face_samples) < num_samples:
            # TODO: Add synthetic face generation if needed
            pass
            
        # Limit to the requested number of samples
        if num_samples > 0 and len(face_samples) > num_samples:
            random.shuffle(face_samples)
            face_samples = face_samples[:num_samples]
            
        return np.array(face_samples) if face_samples else None
        
    def _load_faces_from_dir(self, directory):
        """Load face images from a directory"""
        faces = []
        
        # If it's a flat directory, just load all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.pgm', '.gif', '.bmp']
        
        # Walk through the directory tree
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    try:
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        # Skip invalid images
                        if img is None or img.size == 0:
                            continue
                            
                        # Resize to target size
                        img = cv2.resize(img, self.target_size)
                        faces.append(img)
                    except Exception as e:
                        print(f"Error loading image {file}: {str(e)}")
                        
        return faces
        
    def get_non_face_samples(self, num_samples=500):
        """Get non-face samples from backgrounds or generate random ones"""
        non_faces = []
        
        # Try to load background images first
        bg_dir = self._find_database_dir('backgrounds')
        if bg_dir:
            # Load background images
            bg_images = self._load_faces_from_dir(bg_dir)
            
            # Extract random patches from background images
            for bg in bg_images:
                if bg.shape[0] > self.target_size[1] and bg.shape[1] > self.target_size[0]:
                    # Extract 10 random patches per background
                    for _ in range(10):
                        x = random.randint(0, bg.shape[1] - self.target_size[0])
                        y = random.randint(0, bg.shape[0] - self.target_size[1])
                        patch = bg[y:y+self.target_size[1], x:x+self.target_size[0]]
                        non_faces.append(patch)
        
        # If we don't have enough samples, generate random ones
        if len(non_faces) < num_samples:
            needed = num_samples - len(non_faces)
            for _ in range(needed):
                random_img = np.random.randint(0, 255, size=(self.target_size[1], self.target_size[0]), dtype=np.uint8)
                non_faces.append(random_img)
                
        # Limit to the requested number of samples
        if num_samples > 0 and len(non_faces) > num_samples:
            random.shuffle(non_faces)
            non_faces = non_faces[:num_samples]
            
        return np.array(non_faces) if non_faces else None 