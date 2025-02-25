import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.lda = None
        self.scaler = StandardScaler()
        
        # Create multiple classifiers
        self.knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
        self.svm = SVC(kernel='rbf', probability=True)
        
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
    
    def _compute_similarity_scores(self, X_pca):
        """Compute multiple similarity metrics"""
        # Ensure X_pca is properly shaped for comparison
        X_pca_flat = X_pca.reshape(1, -1) if len(X_pca.shape) == 1 else X_pca
        mean_face_flat = self.mean_face.reshape(1, -1) if len(self.mean_face.shape) == 1 else self.mean_face
        
        # Euclidean distance to mean face
        euclidean_dist = np.linalg.norm(X_pca_flat - mean_face_flat)
        euclidean_score = 1.0 / (1.0 + euclidean_dist)
        
        # Cosine similarity with mean face
        cosine_score = cosine_similarity(X_pca_flat, mean_face_flat)[0][0]
        
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
        
        return euclidean_score, cosine_score, min_dist_score
        
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
        
        self.single_user_mode = (self.n_classes_ == 1)
        if self.single_user_mode:
            self.single_user_name = unique_users[0]
            print(f"Training in single-user mode for user: {self.single_user_name}")
        
        X_flat = self._prepare_data(X)
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Compute optimal PCA components
        n_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
        self.pca.n_components = min(n_components, int(X_scaled.shape[1] * 0.95))
        X_pca = self.pca.fit_transform(X_scaled)
        
        if self.single_user_mode:
            # Store training data for reference
            self.training_faces = X_pca
            self.mean_face = np.mean(X_pca, axis=0)
            
            # Compute reconstruction error threshold
            rec_errors = self._compute_reconstruction_error(X_scaled)
            self.reconstruction_threshold = np.mean(rec_errors) + 2 * np.std(rec_errors)
            
            # Compute similarity thresholds
            distances = np.linalg.norm(X_pca - self.mean_face, axis=1)
            self.face_threshold = np.mean(distances) + 2 * np.std(distances)
            
            print(f"Single-user mode thresholds - Face: {self.face_threshold:.4f}, Reconstruction: {self.reconstruction_threshold:.4f}")
            self.trained = True
        else:
            n_components_lda = min(self.n_classes_ - 1, X_pca.shape[1])
            self.lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
            X_lda = self.lda.fit_transform(X_pca, y)
            X_features = np.hstack([X_pca, X_lda])
            self.ensemble.fit(X_features, y)
            self.trained = True
        
    def predict(self, face_image):
        """Recognize a face image"""
        if not self.trained:
            raise ValueError("Model not trained yet!")
            
        X_flat = self._prepare_data(face_image)
        X_scaled = self.scaler.transform(X_flat)
        X_pca = self.pca.transform(X_scaled)
        
        if self.single_user_mode:
            try:
                # Check reconstruction error with more relaxed threshold
                rec_error = self._compute_reconstruction_error(X_scaled)[0]
                if rec_error > self.reconstruction_threshold * 1.5:  # Allow 50% more error
                    return "Unknown", 0.0
                
                # Compute multiple similarity scores
                euclidean_score, cosine_score, min_dist_score = self._compute_similarity_scores(X_pca)
                
                # Weighted combination of scores
                confidence = (0.4 * euclidean_score + 
                            0.4 * cosine_score + 
                            0.2 * min_dist_score)
                
                # Lower threshold for single-user mode
                threshold = 0.5  # Reduced from 0.65
                if confidence > threshold:
                    return self.single_user_name, confidence
                return "Unknown", confidence
            except Exception as e:
                print(f"Error in single-user prediction: {str(e)}")
                return "Unknown", 0.0
        
        # Multi-user mode
        X_features = self._extract_features(X_scaled)
        probas = self.ensemble.predict_proba(X_features)
        prediction = self.ensemble.predict(X_features)
        confidence = np.max(probas)
        
        return prediction[0], confidence 