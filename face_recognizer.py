import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2  # For visualization if needed

class FaceRecognizer:
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.lda = None
        self.scaler = StandardScaler()
        
        # Create multiple classifiers with better parameters
        self.knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine')
        self.svm = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2)
        
        # Create ensemble classifier with more models
        self.ensemble = VotingClassifier(
            estimators=[
                ('knn', self.knn),
                ('svm', self.svm),
                ('rf', self.rf)
            ],
            voting='soft',
            weights=[1, 2, 1]  # Give more weight to SVM
        )
        
        self.trained = False
        self.single_user_mode = False
        self.single_user_name = None
        self.mean_face = None
        self.face_threshold = None
        self.training_faces = None
        self.reconstruction_threshold = 0.01  # Set a reasonable default
        self.user_names = []  # Store user names from training data
        self.last_predictions = []  # Store last few predictions for temporal consistency
        self.last_confidences = []  # Store last few confidences
        self.registered_user_faces = {}  # Store faces for each registered user
        
        # Add minimum number of components to avoid empty feature vectors
        self.pca.min_components = 5  # Ensure at least 5 components
        
    def _prepare_data(self, X):
        """Flatten 2D images into 1D vectors"""
        if len(X.shape) == 3:
            n_samples, height, width = X.shape
            return X.reshape(n_samples, height * width)
        elif len(X.shape) == 2:
            height, width = X.shape
            return X.reshape(1, height * width)
        else:
            raise ValueError("Invalid input shape")
    
    def _extract_features(self, X):
        """Extract features using PCA and LDA if available"""
        # Apply PCA
        X_pca = self.pca.transform(X)
        
        # Apply LDA only if we have multiple classes and LDA is initialized
        if not self.single_user_mode and self.lda is not None:
            X_lda = self.lda.transform(X_pca)
            return np.hstack([X_pca, X_lda])
        
        return X_pca
    
    def _compute_reconstruction_error(self, X):
        """Compute PCA reconstruction error"""
        X_proj = self.pca.transform(X)
        X_rec = self.pca.inverse_transform(X_proj)
        return np.mean(np.square(X - X_rec), axis=1)
    
    def _compute_similarity_scores(self, X_pca, user_name=None):
        """Compute multiple similarity metrics"""
        # Ensure X_pca is properly shaped for comparison
        X_pca_flat = X_pca.reshape(1, -1) if len(X_pca.shape) == 1 else X_pca
        
        if user_name and user_name in self.registered_user_faces:
            # If we know which user to compare against, use their specific data
            user_faces = self.registered_user_faces[user_name]
            user_mean_face = np.mean(user_faces, axis=0)
            
            # Cosine similarity with user's mean face
            cosine_score = cosine_similarity(X_pca_flat, user_mean_face.reshape(1, -1))[0][0]
            # Ensure cosine_score is in [0,1] range by normalizing from [-1,1] to [0,1]
            cosine_score = (cosine_score + 1) / 2.0
            
            # Minimum distance to user's training samples
            min_dist = float('inf')
            for face in user_faces:
                face_flat = face.reshape(1, -1) if len(face.shape) == 1 else face
                # Use cosine distance for better results with normalized data
                similarity = cosine_similarity(X_pca_flat, face_flat)[0][0]
                # Normalize similarity from [-1,1] to [0,1]
                similarity = (similarity + 1) / 2.0
                dist = 1.0 - similarity  # Convert to distance (0 = identical, 1 = completely different)
                min_dist = min(min_dist, dist)
            
            # Convert inf to large value if no faces were compared
            if min_dist == float('inf'):
                min_dist = 1.0
                
            min_dist_score = 1.0 - min_dist  # Convert back to similarity (0-1)
            
            # For single-user mode, weight direct comparisons more heavily
            return 0.7 * cosine_score + 0.3 * min_dist_score
            
        else:
            # General comparison with all data
            mean_face_flat = self.mean_face.reshape(1, -1) if len(self.mean_face.shape) == 1 else self.mean_face
            
            # Euclidean distance to mean face
            euclidean_dist = np.linalg.norm(X_pca_flat - mean_face_flat)
            euclidean_score = 1.0 / (1.0 + euclidean_dist)
            
            # Cosine similarity with mean face
            cosine_score = cosine_similarity(X_pca_flat, mean_face_flat)[0][0]
            # Ensure cosine_score is in [0,1] range
            cosine_score = (cosine_score + 1) / 2.0
            
            # Minimum distance to training samples
            min_dist = float('inf')
            for face in self.training_faces:
                face_flat = face.reshape(1, -1) if len(face.shape) == 1 else face
                dist = np.linalg.norm(X_pca_flat - face_flat)
                min_dist = min(min_dist, dist)
            
            # Convert inf to large value if no faces were compared
            if min_dist == float('inf'):
                min_dist = 1000.0
                
            min_dist_score = 1.0 / (1.0 + min_dist)
            
            return 0.3 * euclidean_score + 0.5 * cosine_score + 0.2 * min_dist_score
    
    def _is_user_name(self, name):
        """Check if the name looks like a registered user vs. AT&T database subject"""
        # AT&T subjects typically have names like 's1', 's2', etc.
        if isinstance(name, str) and name.startswith('s') and name[1:].isdigit():
            return False
        return True
    
    def _analyze_training_data(self, X, y):
        """Analyze training data to determine quality and class balance"""
        unique_users, counts = np.unique(y, return_counts=True)
        user_counts = dict(zip(unique_users, counts))
        total_images = len(X)
        
        registered_users = [name for name in unique_users if self._is_user_name(name)]
        att_subjects = [name for name in unique_users if not self._is_user_name(name)]
        
        print(f"Training data analysis:")
        print(f"Total images: {total_images}")
        print(f"Registered users: {len(registered_users)} ({sum(counts[np.isin(unique_users, registered_users)])/total_images*100:.1f}% of data)")
        print(f"AT&T subjects: {len(att_subjects)} ({sum(counts[np.isin(unique_users, att_subjects)])/total_images*100:.1f}% of data)")
        
        # Check class imbalance
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        # If significant imbalance, print warning
        if imbalance_ratio > 3:
            print("Warning: Significant class imbalance detected. This may affect recognition accuracy.")
            
        return user_counts
        
    def train(self, X, y):
        """Train the face recognition model"""
        if len(X) == 0:
            raise ValueError("Empty training set")
            
        X = np.array(X)
        y = np.array(y)
        
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D array (n_samples, height, width), got shape {X.shape}")
            
        unique_users = np.unique(y)
        self.n_classes_ = len(unique_users)
        
        # Analyze training data
        user_counts = self._analyze_training_data(X, y)
        
        # Store user names for later, filtering out AT&T subjects
        self.user_names = [name for name in unique_users if self._is_user_name(name)]
        print(f"Registered users found in training data: {self.user_names}")
        
        # Check if we have at least one registered user
        if not self.user_names:
            print("Warning: No registered users found in training data")
            self.single_user_mode = False
        else:
            self.single_user_mode = (len(self.user_names) == 1)
            if self.single_user_mode:
                self.single_user_name = self.user_names[0]
                print(f"Training in single-user mode for user: {self.single_user_name}")
        
        X_flat = self._prepare_data(X)
        
        # Check for NaN values
        if np.isnan(X_flat).any():
            print("Warning: NaN values detected in training data. Replacing with zeros.")
            X_flat = np.nan_to_num(X_flat, nan=0.0)
            
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Check for NaN values after scaling
        if np.isnan(X_scaled).any():
            print("Warning: NaN values detected after scaling. Replacing with zeros.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Compute optimal PCA components - but ensure we capture enough variance
        n_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
        
        # Calculate PCA components based on variance ratio or minimum count
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Variance ratio approach
            pca_components = min(n_components, int(X_scaled.shape[1] * self.n_components))
        else:
            # Fixed number approach
            pca_components = min(n_components, self.n_components)
        
        # Ensure minimum number of components is used
        min_components = getattr(self.pca, 'min_components', 5)
        self.pca.n_components = max(pca_components, min_components)
        
        # Make sure n_components is at least 1 (even for small datasets)
        self.pca.n_components = max(self.pca.n_components, 1)
        
        # Check if n_components is valid
        if self.pca.n_components >= min(X_scaled.shape):
            # If we have more components than samples or features, reduce it
            self.pca.n_components = max(1, min(X_scaled.shape) - 1)
        
        try:
            X_pca = self.pca.fit_transform(X_scaled)
            
            # Verify output has at least one feature
            if X_pca.shape[1] == 0:
                print("Warning: PCA produced empty feature vectors, using 5 components instead")
                self.pca.n_components = 5
                X_pca = self.pca.fit_transform(X_scaled)
                
        except Exception as e:
            print(f"Error in PCA: {e}. Using 5 components instead.")
            self.pca.n_components = 5
            X_pca = self.pca.fit_transform(X_scaled)
        
        # Print explained variance
        explained_variance = sum(self.pca.explained_variance_ratio_)
        print(f"PCA: Using {self.pca.n_components_} components, explaining {explained_variance*100:.2f}% of variance")
        
        # Store training faces for each registered user
        self.registered_user_faces = {}
        for user_name in self.user_names:
            user_indices = np.where(y == user_name)[0]
            if len(user_indices) > 0:
                user_X_pca = X_pca[user_indices]
                self.registered_user_faces[user_name] = user_X_pca
                print(f"Stored {len(user_X_pca)} face representations for {user_name}")
        
        if self.single_user_mode:
            # Filter to only use the single user's images for training
            user_indices = np.where(y == self.single_user_name)[0]
            user_X_pca = X_pca[user_indices]
            
            # Store training data for reference
            self.training_faces = user_X_pca
            self.mean_face = np.mean(user_X_pca, axis=0)
            
            # Compute reconstruction error threshold
            user_X_scaled = X_scaled[user_indices]
            rec_errors = self._compute_reconstruction_error(user_X_scaled)
            
            # Set a reasonable threshold - don't let it be too small
            mean_error = np.mean(rec_errors)
            std_error = np.std(rec_errors)
            self.reconstruction_threshold = max(0.01, mean_error + 2 * std_error)
            print(f"Reconstruction errors - Mean: {mean_error}, Std: {std_error}")
            
            # Compute similarity thresholds using cosine metric
            similarities = []
            for face in user_X_pca:
                sim = cosine_similarity(face.reshape(1, -1), self.mean_face.reshape(1, -1))[0][0]
                similarities.append(sim)
            
            sim_mean = np.mean(similarities)
            sim_std = np.std(similarities)
            self.face_threshold = sim_mean - 2 * sim_std  # Lower bound for similarity
            
            print(f"Single-user mode thresholds - Face similarity: {self.face_threshold:.4f}, Reconstruction: {self.reconstruction_threshold:.4f}")
            self.trained = True
        else:
            # For multi-user mode, we need to balance the training data if using AT&T
            
            # Check if we have both registered users and AT&T subjects
            registered_indices = np.array([i for i, name in enumerate(y) if self._is_user_name(name)])
            att_indices = np.array([i for i, name in enumerate(y) if not self._is_user_name(name)])
            
            # Store overall training statistics
            self.training_faces = X_pca
            self.mean_face = np.mean(X_pca, axis=0)
            
            # Set a reasonable reconstruction threshold
            rec_errors = self._compute_reconstruction_error(X_scaled)
            mean_error = np.mean(rec_errors)
            std_error = np.std(rec_errors)
            self.reconstruction_threshold = max(0.01, mean_error + 2 * std_error)
            print(f"Multi-user reconstruction threshold: {self.reconstruction_threshold:.4f}")
            
            if len(registered_indices) > 0 and len(att_indices) > 0:
                print("Using data balancing for better recognition of registered users...")
                
                # Determine how many AT&T samples to use per registered user
                # This balances the weight between AT&T and registered users
                registered_user_names = np.unique(y[registered_indices])
                samples_per_registered = min(30, len(att_indices) // len(registered_user_names))
                
                # Select subset of AT&T samples - reduced to 30 per registered user (was 50)
                np.random.seed(42)  # For reproducibility
                selected_att = np.random.choice(att_indices, 
                                               size=min(samples_per_registered * len(registered_user_names), 
                                                       len(att_indices)),
                                               replace=False)
                
                # Combine indices with more weight given to registered users
                balanced_indices = np.concatenate([
                    np.repeat(registered_indices, 3),  # Repeat registered user samples 3 times
                    selected_att
                ])
                print(f"Using {len(registered_indices)*3} registered user samples (repeated 3x) and {len(selected_att)} AT&T samples")
                
                # Use the balanced dataset
                X_pca_balanced = X_pca[balanced_indices]
                y_balanced = np.array([y[i] for i in balanced_indices])
                
                # Apply LDA for dimensionality reduction
                n_components_lda = min(len(np.unique(y_balanced)) - 1, X_pca_balanced.shape[1])
                self.lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
                X_lda = self.lda.fit_transform(X_pca_balanced, y_balanced)
                X_features = np.hstack([X_pca_balanced, X_lda])
                self.ensemble.fit(X_features, y_balanced)
            else:
                # Standard approach if we don't have both types
                n_components_lda = min(self.n_classes_ - 1, X_pca.shape[1])
                self.lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
                X_lda = self.lda.fit_transform(X_pca, y)
                X_features = np.hstack([X_pca, X_lda])
                self.ensemble.fit(X_features, y)
            
            self.trained = True
        
        # Reset temporal consistency lists
        self.last_predictions = []
        self.last_confidences = []
        
        return self.trained
        
    def predict(self, face_image):
        """Predict the identity of a face image"""
        # Check if we're trained
        if not hasattr(self, 'pca') or not hasattr(self.pca, 'components_'):
            raise RuntimeError("Model not trained yet")
            
        # Prepare the input image
        if face_image is None:
            raise ValueError("Input image is None")
            
        # Check dimensions
        if len(face_image.shape) != 2:
            raise ValueError(f"Expected grayscale image with shape (height, width), got {face_image.shape}")
            
        # Check for NaN values in input
        if np.isnan(face_image).any():
            print("Warning: NaN values in input image. Replacing with zeros.")
            face_image = np.nan_to_num(face_image, nan=0.0)
            
        # Prepare data (flatten the image)
        X_flat = self._prepare_data(face_image)
        
        try:
            # Scale
            X_scaled = self.scaler.transform(X_flat)
            
            # Check for NaN values after scaling
            if np.isnan(X_scaled).any():
                print("Warning: NaN values after scaling. Replacing with zeros.")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            
            # Transform with PCA
            try:
                X_pca = self.pca.transform(X_scaled)
                
                # Handle the case where PCA returns empty feature vectors
                if X_pca.size == 0 or X_pca.shape[1] == 0:
                    print("Error: PCA returned empty feature vectors. Using scaled data directly.")
                    # Use a simple projection as a fallback
                    n_features = min(5, X_scaled.shape[1])
                    X_pca = X_scaled[:, :n_features]
            except Exception as e:
                print(f"PCA transform error: {e}. Using scaled data directly.")
                # Use a simple projection as a fallback
                n_features = min(5, X_scaled.shape[1])
                X_pca = X_scaled[:, :n_features]
            
            if self.single_user_mode:
                try:
                    # Check reconstruction error with more reasonable threshold
                    rec_error = self._compute_reconstruction_error(X_scaled)[0]
                    print(f"Reconstruction error: {rec_error:.6f}, threshold: {self.reconstruction_threshold:.6f}")
                    
                    # Skip this check if threshold is unreasonably small
                    if self.reconstruction_threshold > 0.001 and rec_error > self.reconstruction_threshold * 3.0:
                        print(f"Reconstruction error too high: {rec_error} > {self.reconstruction_threshold * 3.0}")
                        return "Unknown", 0.0
                    
                    # Compute similarity score directly with registered user
                    confidence = self._compute_similarity_scores(X_pca, self.single_user_name)
                    
                    # Ensure confidence is in the range [0,1]
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # Even lower threshold for single-user mode
                    threshold = 0.2  # Very low threshold
                    
                    # Add temporal consistency - if we've seen this face before
                    if len(self.last_confidences) > 0:
                        # Compute average recent confidence
                        avg_confidence = sum(self.last_confidences) / len(self.last_confidences)
                        
                        # If current is close to average, boost confidence slightly
                        if abs(confidence - avg_confidence) < 0.15:
                            confidence = min(confidence * 1.2, 1.0)  # Boost by 20%, but cap at 1.0
                    
                    # Update history (keep last 5)
                    self.last_confidences.append(confidence)
                    if len(self.last_confidences) > 5:
                        self.last_confidences.pop(0)
                    
                    print(f"Single-user confidence: {confidence}, threshold: {threshold}")
                    if confidence > threshold:
                        return self.single_user_name, confidence
                    return "Unknown", confidence
                except Exception as e:
                    print(f"Error in single-user prediction: {str(e)}")
                    return "Unknown", 0.0
            
            # Multi-user mode
            try:
                X_features = self._extract_features(X_scaled)
                probas = self.ensemble.predict_proba(X_features)
                
                # For debugging - list all confidences
                class_confidences = {cls: prob for cls, prob in zip(self.ensemble.classes_, probas[0])}
                print(f"Class confidences: {class_confidences}")
                
                # Get prediction and confidence
                prediction = self.ensemble.predict(X_features)[0]
                confidence = probas[0][list(self.ensemble.classes_).index(prediction)]
                
                # Get the predicted class and confidence
                pred_class = prediction
                
                # Much lower confidence threshold for registered users
                base_threshold = 0.05  # Extremely low (was 0.10)
                
                # If prediction is a registered user, try direct similarity comparison
                if self._is_user_name(pred_class) and pred_class in self.registered_user_faces:
                    # Do additional direct similarity check
                    direct_sim = self._compute_similarity_scores(X_pca, pred_class)
                    
                    # Blend the ensemble probability with direct similarity for registered users
                    confidence = 0.7 * confidence + 0.3 * direct_sim  # Weight ensemble more
                    
                    # Ensure confidence is in the range [0,1]
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # Use a very low threshold for registered users
                    threshold = base_threshold
                    
                    # Add temporal consistency for registered users
                    if len(self.last_predictions) > 0:
                        # If same user predicted multiple times in a row
                        if pred_class in self.last_predictions:
                            # Count occurrences
                            occurrences = self.last_predictions.count(pred_class)
                            
                            # Boost confidence based on consistency
                            confidence = min(confidence * (1.0 + 0.1 * occurrences), 1.0)  # Conservative boost, cap at 1.0
                else:
                    # For AT&T subjects, use higher threshold
                    threshold = 0.40  # Much higher threshold for AT&T subjects
                    
                    # If confidence is above threshold but below a higher threshold,
                    # and we have registered users, check if any registered user has close confidence
                    if confidence > threshold and confidence < 0.50 and self.user_names:
                        # Get probability scores for registered users
                        user_indices = [list(self.ensemble.classes_).index(name) 
                                       for name in self.user_names if name in self.ensemble.classes_]
                        if user_indices:
                            user_probs = [probas[0][idx] for idx in user_indices]
                            max_user_prob = max(user_probs)
                            max_user_idx = user_indices[user_probs.index(max_user_prob)]
                            
                            # If probability is close enough, prefer the registered user
                            # (Very aggressive preference for registered users)
                            if max_user_prob > confidence * 0.4:
                                pred_class = self.ensemble.classes_[max_user_idx]
                                confidence = max_user_prob
                
                # Ensure final confidence is in the range [0,1]
                confidence = max(0.0, min(1.0, confidence))
                
                # Update prediction history (keep last 5)
                self.last_predictions.append(pred_class)
                if len(self.last_predictions) > 5:
                    self.last_predictions.pop(0)
                
                # Print diagnostics
                print(f"Prediction: {pred_class}, Confidence: {confidence}, Threshold: {threshold}")
                if confidence > threshold:
                    return pred_class, confidence
                return "Unknown", confidence
            except Exception as e:
                print(f"Error in multi-user prediction: {str(e)}")
                return "Unknown", 0.0
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "Unknown", 0.0
    
    def save_model(self, filepath):
        """Save model to disk"""
        if not self.trained:
            raise ValueError("Model not trained yet!")
            
        import joblib
        model_data = {
            'pca': self.pca,
            'lda': self.lda,
            'scaler': self.scaler,
            'ensemble': self.ensemble,
            'single_user_mode': self.single_user_mode,
            'single_user_name': self.single_user_name,
            'mean_face': self.mean_face,
            'face_threshold': self.face_threshold,
            'training_faces': self.training_faces,
            'reconstruction_threshold': self.reconstruction_threshold,
            'user_names': self.user_names,
            'trained': self.trained,
            'registered_user_faces': self.registered_user_faces
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load model from disk"""
        import joblib
        
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        
        self.pca = model_data['pca']
        self.lda = model_data['lda']
        self.scaler = model_data['scaler']
        self.ensemble = model_data['ensemble']
        self.single_user_mode = model_data['single_user_mode']
        self.single_user_name = model_data['single_user_name']
        self.mean_face = model_data['mean_face']
        self.face_threshold = model_data['face_threshold']
        self.training_faces = model_data['training_faces']
        self.reconstruction_threshold = model_data['reconstruction_threshold']
        self.user_names = model_data['user_names']
        self.trained = model_data['trained']
        
        # Load registered user faces if available in saved model
        if 'registered_user_faces' in model_data:
            self.registered_user_faces = model_data['registered_user_faces']
        else:
            self.registered_user_faces = {}
        
        # Reset temporal consistency lists
        self.last_predictions = []
        self.last_confidences = []
        
        print(f"Model loaded from {filepath}")
        return True 