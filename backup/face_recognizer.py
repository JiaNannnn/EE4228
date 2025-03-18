import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics.pairwise import cosine_similarity

# Add function to detect NaN values
def check_for_nans(data, stage_name):
    """Check if data contains NaN values and log if found"""
    if isinstance(data, np.ndarray):
        if np.isnan(data).any():
            nan_count = np.isnan(data).sum()
            nan_indices = np.where(np.isnan(data))
            print(f"WARNING: NaN values detected in {stage_name}! Count: {nan_count}")
            print(f"NaN indices: {nan_indices}")
            return True
    return False

class FaceRecognizer:
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, whiten=True)  # Added whitening for better performance
        self.lda = None
        self.scaler = StandardScaler()
        
        # Create multiple classifiers
        self.knn = KNeighborsClassifier(n_neighbors=1, weights='distance')  # Changed from 3 to 1 for better recognition
        self.svm = SVC(kernel='rbf', probability=True, C=10.0)  # Increased C for less regularization
        
        # Create ensemble classifier
        self.ensemble = VotingClassifier(
            estimators=[
                ('knn', self.knn),
                ('svm', self.svm)
            ],
            voting='soft'
        )
        
        self.trained = False
        self.single_user_mode = False
        self.single_user_name = None
        self.mean_face = None
        self.face_threshold = None
        self.training_faces = None
        self.reconstruction_threshold = None
        self.class_thresholds = {}  # Store thresholds for each class
        
        # Additional PCA parameters
        self.use_dimensionality_reduction = True  # Enable PCA by default
        self.eigenfaces = None
        self.mean_image = None
        
    def _prepare_data(self, X):
        """Flatten 2D images into 1D vectors"""
        print(f"Input shape before prepare_data: {X.shape}")
        
        # Check for NaNs in input
        if check_for_nans(X, "input to _prepare_data"):
            print("Fixing NaN values in input data...")
            X = np.nan_to_num(X, nan=0.0)
        
        if len(X.shape) == 3:
            n_samples, height, width = X.shape
            X_flat = X.reshape(n_samples, height * width)
        elif len(X.shape) == 2:
            height, width = X.shape
            X_flat = X.reshape(1, height * width)
        else:
            raise ValueError(f"Invalid input shape: {X.shape}")
        
        # Check for NaNs after reshaping
        if check_for_nans(X_flat, "output of _prepare_data"):
            print("Fixing NaN values after reshaping...")
            X_flat = np.nan_to_num(X_flat, nan=0.0)
        
        print(f"Output shape after prepare_data: {X_flat.shape}")
        return X_flat
    
    def _extract_features(self, X):
        """Extract features using PCA and LDA if available"""
        # Print input statistics for debugging
        print(f"Input statistics - Min: {np.min(X)}, Max: {np.max(X)}, Mean: {np.mean(X)}")
        
        # Check for NaNs before PCA
        if check_for_nans(X, "input to _extract_features"):
            print("Fixing NaN values before feature extraction...")
            X = np.nan_to_num(X, nan=0.0)
        
        # Apply PCA
        try:
            X_pca = self.pca.transform(X)
            # Check for NaNs after PCA
            if check_for_nans(X_pca, "PCA output"):
                print("Fixing NaN values in PCA output...")
                X_pca = np.nan_to_num(X_pca, nan=0.0)
                
            # Store eigenfaces and mean image if not already stored
            if self.eigenfaces is None and hasattr(self.pca, 'components_'):
                self.eigenfaces = self.pca.components_
            if self.mean_image is None and hasattr(self.pca, 'mean_'):
                self.mean_image = self.pca.mean_
                
        except Exception as e:
            print(f"Error in PCA transform: {str(e)}")
            # Examine the input for potential issues
            print(f"Input statistics - Min: {np.min(X)}, Max: {np.max(X)}, Mean: {np.mean(X)}")
            print(f"Input contains NaN: {np.isnan(X).any()}, Inf: {np.isinf(X).any()}")
            # Try to fix the input and retry
            X = np.nan_to_num(X, nan=0.0)
            X = np.clip(X, -1e6, 1e6)  # Clip extreme values
            try:
                X_pca = self.pca.transform(X)
            except:
                # If all else fails, return zeros
                X_pca = np.zeros((X.shape[0], min(10, X.shape[1])))
        
        # Apply LDA only if we have multiple classes and LDA is initialized
        if not self.single_user_mode and self.lda is not None:
            try:
                X_lda = self.lda.transform(X_pca)
                # Check for NaNs after LDA
                if check_for_nans(X_lda, "LDA output"):
                    print("Fixing NaN values in LDA output...")
                    X_lda = np.nan_to_num(X_lda, nan=0.0)
                return np.hstack([X_pca, X_lda])
            except Exception as e:
                print(f"Error in LDA transform: {str(e)}")
                # Return just PCA features if LDA fails
                return X_pca
        
        return X_pca
    
    def _compute_reconstruction_error(self, X):
        """Compute PCA reconstruction error"""
        # Check for NaNs in input
        if check_for_nans(X, "input to _compute_reconstruction_error"):
            print("Fixing NaN values before computing reconstruction error...")
            X = np.nan_to_num(X, nan=0.0)
        
        try:
            X_proj = self.pca.transform(X)
            # Check for NaNs after PCA transform
            if check_for_nans(X_proj, "PCA projection"):
                print("Fixing NaN values in PCA projection...")
                X_proj = np.nan_to_num(X_proj, nan=0.0)
            
            X_rec = self.pca.inverse_transform(X_proj)
            # Check for NaNs after PCA inverse transform
            if check_for_nans(X_rec, "PCA reconstruction"):
                print("Fixing NaN values in PCA reconstruction...")
                X_rec = np.nan_to_num(X_rec, nan=0.0)
            
            error = np.mean(np.square(X - X_rec), axis=1)
            # Check for NaNs in error
            if check_for_nans(error, "reconstruction error"):
                print("Fixing NaN values in reconstruction error...")
                error = np.nan_to_num(error, nan=1.0)  # Use a high error value for NaNs
            
            return error
        except Exception as e:
            print(f"Error computing reconstruction error: {str(e)}")
            # Return a high error value if computation fails
            return np.ones(X.shape[0])
    
    def _compute_similarity_scores(self, X_pca, class_name=None):
        """Compute multiple similarity metrics"""
        # Check for NaNs in input
        if check_for_nans(X_pca, "input to _compute_similarity_scores"):
            print("Fixing NaN values in input to similarity computation...")
            X_pca = np.nan_to_num(X_pca, nan=0.0)
        
        # Use class-specific mean face if available
        mean_face = self.class_mean_faces.get(class_name, self.mean_face) if hasattr(self, 'class_mean_faces') else self.mean_face
        
        if check_for_nans(mean_face, "mean_face"):
            print("Fixing NaN values in mean face...")
            mean_face = np.nan_to_num(mean_face, nan=0.0)
        
        # Ensure X_pca is properly shaped for comparison
        X_pca_flat = X_pca.reshape(1, -1) if len(X_pca.shape) == 1 else X_pca
        mean_face_flat = mean_face.reshape(1, -1) if len(mean_face.shape) == 1 else mean_face
        
        try:
            # Euclidean distance to mean face
            euclidean_dist = np.linalg.norm(X_pca_flat - mean_face_flat)
            euclidean_score = 1.0 / (1.0 + euclidean_dist)
            
            # Cosine similarity with mean face
            cosine_score = cosine_similarity(X_pca_flat, mean_face_flat)[0][0]
            
            # Minimum distance to training samples
            min_dist = float('inf')
            
            # Use class-specific training faces if available
            training_faces = self.class_training_faces.get(class_name, self.training_faces) if hasattr(self, 'class_training_faces') else self.training_faces
            
            for face in training_faces:
                if check_for_nans(face, "training face"):
                    print("Fixing NaN values in training face...")
                    face = np.nan_to_num(face, nan=0.0)
                
                face_flat = face.reshape(1, -1) if len(face.shape) == 1 else face
                dist = np.linalg.norm(X_pca_flat - face_flat)
                min_dist = min(min_dist, dist)
            
            # Convert inf to large value if no faces were compared
            if min_dist == float('inf'):
                min_dist = 1000.0
                
            min_dist_score = 1.0 / (1.0 + min_dist)
        except Exception as e:
            print(f"Error computing similarity scores: {str(e)}")
            # Return default scores if computation fails
            euclidean_score = 0.0
            cosine_score = 0.0
            min_dist_score = 0.0
        
        # Check for NaNs in output scores
        if np.isnan(euclidean_score) or np.isnan(cosine_score) or np.isnan(min_dist_score):
            print(f"WARNING: NaN values detected in similarity scores!")
            print(f"euclidean_score: {euclidean_score}, cosine_score: {cosine_score}, min_dist_score: {min_dist_score}")
            # Replace NaN scores with zeros
            euclidean_score = 0.0 if np.isnan(euclidean_score) else euclidean_score
            cosine_score = 0.0 if np.isnan(cosine_score) else cosine_score
            min_dist_score = 0.0 if np.isnan(min_dist_score) else min_dist_score
        
        return euclidean_score, cosine_score, min_dist_score
        
    def train(self, X, y):
        """Train the face recognition model"""
        if len(X) == 0:
            raise ValueError("Empty training set")
            
        X = np.array(X)
        y = np.array(y)
        
        # Check for NaNs in input training data
        if check_for_nans(X, "input training data"):
            print("WARNING: Training data contains NaN values. Attempting to fix...")
            # Replace NaNs with mean value of non-NaN elements
            mask = np.isnan(X)
            mean_vals = np.nanmean(X, axis=0)
            X = np.where(mask, mean_vals, X)
            if check_for_nans(X, "fixed training data"):
                print("Failed to fix NaN values with mean. Using zeros instead.")
                X = np.nan_to_num(X, nan=0.0)
        
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D array (n_samples, height, width), got shape {X.shape}")
            
        unique_users = np.unique(y)
        self.n_classes_ = len(unique_users)
        
        self.single_user_mode = (self.n_classes_ == 1)
        if self.single_user_mode:
            self.single_user_name = unique_users[0]
            print(f"Training in single-user mode for user: {self.single_user_name}")
        
        X_flat = self._prepare_data(X)
        try:
            X_scaled = self.scaler.fit_transform(X_flat)
            # Check for NaNs after scaling
            if check_for_nans(X_scaled, "scaled data"):
                print("Fixing NaN values after scaling...")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        except Exception as e:
            print(f"Error in scaling: {str(e)}")
            # Examine the input for potential issues
            print(f"Input statistics - Min: {np.min(X_flat)}, Max: {np.max(X_flat)}, Mean: {np.mean(X_flat)}")
            print(f"Input contains NaN: {np.isnan(X_flat).any()}, Inf: {np.isinf(X_flat).any()}")
            # Try to fix the input and retry
            X_flat = np.nan_to_num(X_flat, nan=0.0)
            X_flat = np.clip(X_flat, -1e6, 1e6)  # Clip extreme values
            X_scaled = self.scaler.fit_transform(X_flat)
        
        # Compute optimal PCA components
        n_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
        self.pca.n_components = min(n_components, int(X_scaled.shape[1] * 0.95))
        
        try:
            X_pca = self.pca.fit_transform(X_scaled)
            # Check for NaNs after PCA
            if check_for_nans(X_pca, "PCA output"):
                print("Fixing NaN values in PCA output...")
                X_pca = np.nan_to_num(X_pca, nan=0.0)
        except Exception as e:
            print(f"Error in PCA: {str(e)}")
            # Examine the input for potential issues
            print(f"Input statistics - Min: {np.min(X_scaled)}, Max: {np.max(X_scaled)}, Mean: {np.mean(X_scaled)}")
            print(f"Input contains NaN: {np.isnan(X_scaled).any()}, Inf: {np.isinf(X_scaled).any()}")
            # Try to fix the input and retry
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            X_scaled = np.clip(X_scaled, -1e6, 1e6)  # Clip extreme values
            X_pca = self.pca.fit_transform(X_scaled)
        
        if self.single_user_mode:
            # Store training data for reference
            self.training_faces = X_pca
            self.mean_face = np.mean(X_pca, axis=0)
            
            # Compute reconstruction error threshold
            rec_errors = self._compute_reconstruction_error(X_scaled)
            self.reconstruction_threshold = np.mean(rec_errors) + 3 * np.std(rec_errors)  # Increased from 2 to 3 std
            
            # Compute similarity thresholds
            distances = np.linalg.norm(X_pca - self.mean_face, axis=1)
            self.face_threshold = np.mean(distances) + 3 * np.std(distances)  # Increased from 2 to 3 std
            
            print(f"Single-user mode thresholds - Face: {self.face_threshold:.4f}, Reconstruction: {self.reconstruction_threshold:.4f}")
            self.trained = True
        else:
            try:
                # Try multi-class approach first
                n_components_lda = min(self.n_classes_ - 1, X_pca.shape[1])
                self.lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
                X_lda = self.lda.fit_transform(X_pca, y)
                # Check for NaNs after LDA
                if check_for_nans(X_lda, "LDA output"):
                    print("Fixing NaN values in LDA output...")
                    X_lda = np.nan_to_num(X_lda, nan=0.0)
                
                X_features = np.hstack([X_pca, X_lda])
                self.ensemble.fit(X_features, y)
                self.trained = True
                
                # Also compute class-specific thresholds for one-vs-all approach
                self.class_mean_faces = {}
                self.class_training_faces = {}
                self.class_thresholds = {}
                
                for class_name in unique_users:
                    # Get samples for this class
                    mask = (y == class_name)
                    X_pca_class = X_pca[mask]
                    X_scaled_class = X_scaled[mask]
                    
                    # Store class-specific training data
                    self.class_training_faces[class_name] = X_pca_class
                    self.class_mean_faces[class_name] = np.mean(X_pca_class, axis=0)
                    
                    # Compute class-specific thresholds
                    rec_errors = self._compute_reconstruction_error(X_scaled_class)
                    rec_threshold = np.mean(rec_errors) + 3 * np.std(rec_errors)
                    
                    distances = np.linalg.norm(X_pca_class - self.class_mean_faces[class_name], axis=1)
                    face_threshold = np.mean(distances) + 3 * np.std(distances)
                    
                    self.class_thresholds[class_name] = {
                        'reconstruction': rec_threshold,
                        'face': face_threshold
                    }
                    
                    print(f"Class {class_name} thresholds - Face: {face_threshold:.4f}, Reconstruction: {rec_threshold:.4f}")
                
            except Exception as e:
                print(f"Error in LDA: {str(e)}")
                # Fall back to one-vs-all approach
                print("Falling back to one-vs-all approach...")
                
                self.class_mean_faces = {}
                self.class_training_faces = {}
                self.class_thresholds = {}
                
                for class_name in unique_users:
                    # Get samples for this class
                    mask = (y == class_name)
                    X_pca_class = X_pca[mask]
                    X_scaled_class = X_scaled[mask]
                    
                    # Store class-specific training data
                    self.class_training_faces[class_name] = X_pca_class
                    self.class_mean_faces[class_name] = np.mean(X_pca_class, axis=0)
                    
                    # Compute class-specific thresholds
                    rec_errors = self._compute_reconstruction_error(X_scaled_class)
                    rec_threshold = np.mean(rec_errors) + 3 * np.std(rec_errors)
                    
                    distances = np.linalg.norm(X_pca_class - self.class_mean_faces[class_name], axis=1)
                    face_threshold = np.mean(distances) + 3 * np.std(distances)
                    
                    self.class_thresholds[class_name] = {
                        'reconstruction': rec_threshold,
                        'face': face_threshold
                    }
                    
                    print(f"Class {class_name} thresholds - Face: {face_threshold:.4f}, Reconstruction: {rec_threshold:.4f}")
                
                # Use the first class as default
                self.single_user_mode = True
                self.single_user_name = unique_users[0]
                self.training_faces = self.class_training_faces[self.single_user_name]
                self.mean_face = self.class_mean_faces[self.single_user_name]
                self.reconstruction_threshold = self.class_thresholds[self.single_user_name]['reconstruction']
                self.face_threshold = self.class_thresholds[self.single_user_name]['face']
                
                self.trained = True
        
    def predict(self, face_image):
        """Predict identity of a face"""
        if not self.trained:
            return "Unknown", 0.0, (0.0, 0.0, 0.0)
            
        if len(face_image.shape) != 2:
            raise ValueError(f"Expected 2D array (height, width), got shape {face_image.shape}")
            
        try:
            # Preprocess input
            X = self._prepare_data(np.array([face_image]))
            
            # Scale the data
            X_scaled = self.scaler.transform(X)
            
            # Check for NaNs after scaling
            if check_for_nans(X_scaled, "scaled input"):
                print("Fixing NaN values in scaled input...")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
                
            # Extract features
            X_feat = self._extract_features(X_scaled)
            
            # Single-user mode
            if self.single_user_mode:
                # Compute reconstruction error
                rec_error = self._compute_reconstruction_error(X_scaled)[0]
                
                # Compute similarity scores
                euclidean_score, cosine_score, min_dist_score = self._compute_similarity_scores(X_feat)
                
                # Compute confidence
                rec_confidence = max(0, 1 - (rec_error / self.reconstruction_threshold))
                sim_confidence = max(0, 1 - (euclidean_score / self.face_threshold))
                
                # Final confidence score
                confidence = (rec_confidence + sim_confidence) / 2
                
                # Print debug info
                print(f"Single user mode - rec_conf: {rec_confidence:.4f}, sim_conf: {sim_confidence:.4f}")
                
                # If confidence is high enough, return the identity
                if confidence > 0.5:
                    return self.single_user_name, confidence, (euclidean_score, cosine_score, min_dist_score)
                else:
                    return "Unknown", confidence, (euclidean_score, cosine_score, min_dist_score)
            
            # Multi-user mode
            try:
                # Get probability predictions
                proba = self.ensemble.predict_proba(X_feat)[0]
                
                # Get the highest probability class
                best_idx = np.argmax(proba)
                best_class = self.ensemble.classes_[best_idx]
                confidence = proba[best_idx]
                
                # Get similarity scores for the predicted class
                euclidean_score, cosine_score, min_dist_score = self._compute_similarity_scores(X_feat, best_class)
                
                # Apply class-specific threshold if available
                if best_class in self.class_thresholds:
                    threshold = self.class_thresholds[best_class]
                    if min_dist_score > threshold:
                        return "Unknown", confidence, (euclidean_score, cosine_score, min_dist_score)
                
                # Make final decision
                if confidence > 0.5:  # Confidence threshold
                    return best_class, confidence, (euclidean_score, cosine_score, min_dist_score)
                else:
                    return "Unknown", confidence, (euclidean_score, cosine_score, min_dist_score)
            except Exception as e:
                print(f"Error in ensemble prediction: {str(e)}")
                # Fallback prediction
                try:
                    prediction = self.knn.predict(X_feat)[0]
                    distances, _ = self.knn.kneighbors(X_feat)
                    dist = distances[0][0]
                    confidence = max(0, 1 - (dist / 10))  # Simple conversion of distance to confidence
                    
                    # Get similarity scores
                    euclidean_score, cosine_score, min_dist_score = self._compute_similarity_scores(X_feat, prediction)
                    
                    if confidence > 0.3:  # Lower threshold for KNN fallback
                        return prediction, confidence, (euclidean_score, cosine_score, min_dist_score)
                    else:
                        return "Unknown", confidence, (euclidean_score, cosine_score, min_dist_score)
                except Exception as e2:
                    print(f"Error in KNN fallback: {str(e2)}")
                    return "Unknown", 0.0, (0.0, 0.0, 0.0)
        except Exception as e:
            print(f"Unexpected error in prediction: {str(e)}")
            return "Unknown", 0.0, (0.0, 0.0, 0.0) 