"""
Image Preprocessor Core Module

This module provides the core image preprocessing functionality:
1. Grayscale conversion
2. Illumination normalization
3. Image enhancement
4. Scale normalization
"""

import os
import cv2
import numpy as np
import logging
import math

# Configure module logger
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Class for preprocessing images with normalization
    and enhancement techniques
    """
    
    def __init__(self, target_size=(224, 224), force_grayscale=True):
        """
        Initialize the image preprocessor
        
        Parameters:
        -----------
        target_size : tuple
            Output size for normalized images (width, height)
        force_grayscale : bool
            Whether to always convert images to grayscale
        """
        self.target_size = target_size
        self.force_grayscale = force_grayscale
        
        logger.info(f"Image preprocessor initialized with target_size={target_size}, force_grayscale={force_grayscale}")
    
    def convert_to_grayscale(self, image):
        """
        Convert an image to grayscale if it's not already
        
        Args:
            image: Input image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def normalize_illumination(self, image, method='clahe'):
        """
        Apply illumination normalization to the image
        
        Args:
            image: Input image (grayscale)
            method: Normalization method 
                    ('hist_eq', 'clahe', 'gamma', 'dog', 'tantriggs', 'all')
            
        Returns:
            Illumination normalized image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply selected normalization method
        if method == 'hist_eq':
            # Simple histogram equalization
            normalized = cv2.equalizeHist(gray)
            logger.debug("Applied histogram equalization")
        
        elif method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(gray)
            logger.debug("Applied CLAHE normalization")
        
        elif method == 'gamma':
            # Gamma correction
            # Estimate optimal gamma value based on image mean
            mean = np.mean(gray) / 255.0
            gamma = math.log(0.5) / math.log(mean) if mean > 0 else 1.0
            gamma = min(max(gamma, 0.5), 2.0)  # Limit gamma to reasonable range
            
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            normalized = cv2.LUT(gray, table)
            logger.debug(f"Applied gamma correction with gamma={gamma:.2f}")
        
        elif method == 'dog':
            # Difference of Gaussians
            g1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
            g2 = cv2.GaussianBlur(gray, (9, 9), 2.0)
            normalized = cv2.absdiff(g1, g2)
            # Normalize to 0-255 range
            normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            logger.debug("Applied Difference of Gaussians normalization")
        
        elif method == 'tantriggs':
            # Tan-Triggs preprocessing
            # Step 1: Gamma correction
            gamma = 0.2
            gray_gamma = np.power(gray / 255.0, gamma) * 255.0
            gray_gamma = gray_gamma.astype(np.uint8)
            
            # Step 2: DoG filtering
            g1 = cv2.GaussianBlur(gray_gamma, (3, 3), 1.0)
            g2 = cv2.GaussianBlur(gray_gamma, (7, 7), 2.0)
            dog = cv2.absdiff(g1, g2)
            
            # Step 3: Contrast equalization
            # Simple version: use histogram equalization
            normalized = cv2.equalizeHist(dog)
            logger.debug("Applied Tan-Triggs normalization")
        
        elif method == 'all':
            # Apply multiple methods and combine results
            results = []
            
            # Apply each method
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            results.append(clahe.apply(gray))  # CLAHE
            
            # Gamma correction
            mean = np.mean(gray) / 255.0
            gamma = math.log(0.5) / math.log(mean) if mean > 0 else 1.0
            gamma = min(max(gamma, 0.5), 2.0)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            results.append(cv2.LUT(gray, table))  # Gamma
            
            # DoG
            g1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
            g2 = cv2.GaussianBlur(gray, (9, 9), 2.0)
            dog = cv2.absdiff(g1, g2)
            results.append(cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))  # DoG
            
            # Average the results
            normalized = np.zeros_like(gray)
            for result in results:
                normalized = cv2.addWeighted(normalized, 0.5, result, 0.5, 0)
            
            logger.debug("Applied combined illumination normalization (all methods)")
        
        else:
            # Default to CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(gray)
            logger.debug(f"Applied default CLAHE normalization (unknown method: {method})")
        
        return normalized
    
    def enhance_image(self, image, method='adaptive'):
        """
        Enhance image quality using various techniques
        
        Args:
            image: Input image (grayscale)
            method: Enhancement method ('none', 'sharpen', 'contrast', 'adaptive')
            
        Returns:
            Enhanced image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply selected enhancement method
        if method == 'none':
            # No enhancement
            enhanced = gray
            logger.debug("No enhancement applied")
            
        elif method == 'sharpen':
            # Apply sharpening filter
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            enhanced = cv2.filter2D(gray, -1, kernel)
            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            logger.debug("Applied sharpening filter")
        
        elif method == 'contrast':
            # Global contrast enhancement with histogram equalization
            enhanced = cv2.equalizeHist(gray)
            logger.debug("Applied contrast enhancement with histogram equalization")
        
        elif method == 'adaptive':
            # Adaptive enhancement based on image properties
            # First measure image properties
            mean_val = np.mean(gray)
            std_dev = np.std(gray)
            
            # Apply different enhancement strategies based on image quality
            if std_dev < 30:  # Low contrast image
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Apply mild sharpening
                kernel = np.array([[-0.5, -0.5, -0.5],
                                  [-0.5,  5.0, -0.5],
                                  [-0.5, -0.5, -0.5]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                logger.debug("Applied adaptive enhancement for low contrast image")
                
            elif mean_val < 80:  # Dark image
                # Increase brightness with gamma correction
                gamma = 0.7
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                enhanced = cv2.LUT(gray, table)
                
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(enhanced)
                logger.debug("Applied adaptive enhancement for dark image")
                
            elif mean_val > 180:  # Bright image
                # Reduce brightness with gamma correction
                gamma = 1.3
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                enhanced = cv2.LUT(gray, table)
                
                # Apply denoising
                enhanced = cv2.fastNlMeansDenoising(enhanced, None, 5, 7, 21)
                logger.debug("Applied adaptive enhancement for bright image")
                
            else:  # Normal image
                # Apply denoising
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                
                # Apply mild sharpening
                kernel = np.array([[-0.5, -0.5, -0.5],
                                  [-0.5,  5.0, -0.5],
                                  [-0.5, -0.5, -0.5]])
                enhanced = cv2.filter2D(denoised, -1, kernel)
                logger.debug("Applied adaptive enhancement for normal image")
            
            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        else:
            # Default to no enhancement
            enhanced = gray
            logger.debug(f"No enhancement applied (unknown method: {method})")
        
        return enhanced
    
    def resize_image(self, image, target_size=None, preserve_aspect_ratio=True):
        """
        Resize image to target size with optional aspect ratio preservation
        
        Args:
            image: Input image
            target_size: Target size (width, height) or None to use default target_size
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
            
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        if preserve_aspect_ratio:
            # Calculate target size while preserving aspect ratio
            if w/h > target_w/target_h:
                # Image is wider than target, fit to width
                new_w = target_w
                new_h = int(h * (target_w / w))
            else:
                # Image is taller than target, fit to height
                new_h = target_h
                new_w = int(w * (target_h / h))
            
            # Resize with aspect ratio preserved
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Create target-sized image with padding
                result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # Calculate position to paste the resized image
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                
                # Paste resized image onto target-sized canvas
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            else:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Create target-sized image with padding
                result = np.zeros((target_h, target_w), dtype=np.uint8)
                
                # Calculate position to paste the resized image
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                
                # Paste resized image onto target-sized canvas
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            logger.debug(f"Resized image to {new_w}x{new_h} with padding to {target_w}x{target_h} (aspect ratio preserved)")
            return result
        else:
            # Simple resize to target dimensions
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Resized image to {target_w}x{target_h}")
            return resized
    
    def preprocess(self, image, illumination_method='clahe', enhancement_method='adaptive',
                  preserve_aspect_ratio=True, return_intermediates=False):
        """
        Apply the full preprocessing pipeline to an image:
        1. Convert to grayscale (if needed)
        2. Normalize illumination
        3. Enhance image quality
        4. Normalize scale (resize to target size)
        
        Args:
            image: Input image (numpy array) or path to image
            illumination_method: Method for illumination normalization
            enhancement_method: Method for image quality enhancement
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Preprocessed image and optionally intermediate results
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                logger.error(f"Failed to load image from {image}")
                return None if not return_intermediates else (None, {})
        
        # Store intermediate results if requested
        intermediates = {}
        if return_intermediates:
            intermediates['original'] = image.copy()
        
        # Step 1: Convert to grayscale if needed/requested
        if self.force_grayscale and len(image.shape) == 3:
            gray = self.convert_to_grayscale(image)
            if return_intermediates:
                intermediates['grayscale'] = gray.copy()
        else:
            gray = image.copy() if len(image.shape) == 2 else self.convert_to_grayscale(image)
        
        # Step 2: Normalize illumination
        normalized = self.normalize_illumination(gray, method=illumination_method)
        if return_intermediates:
            intermediates['illumination_normalized'] = normalized.copy()
        
        # Step 3: Enhance image quality
        enhanced = self.enhance_image(normalized, method=enhancement_method)
        if return_intermediates:
            intermediates['enhanced'] = enhanced.copy()
        
        # Step 4: Normalize scale (resize to target size)
        resized = self.resize_image(enhanced, preserve_aspect_ratio=preserve_aspect_ratio)
        if return_intermediates:
            intermediates['final'] = resized.copy()
        
        if return_intermediates:
            return resized, intermediates
        return resized 