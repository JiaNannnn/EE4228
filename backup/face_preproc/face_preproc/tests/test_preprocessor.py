#!/usr/bin/env python
"""
Test script for FacePreprocessor

This module tests the core functionality of the FacePreprocessor class.
"""

import os
import sys
import cv2
import numpy as np
import unittest
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(parent_dir))
    __package__ = "face_preproc.tests"

# Import from package
from face_preproc.core.preprocessor import FacePreprocessor

class TestFacePreprocessor(unittest.TestCase):
    """Test cases for the FacePreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = FacePreprocessor(
            target_size=(224, 224),
            force_grayscale=True
        )
        
        # Create a simple test image (black square with white square inside)
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        self.test_image[100:200, 100:200] = 255  # White square in the middle
    
    def test_initialization(self):
        """Test that preprocessor initializes properly"""
        self.assertEqual(self.preprocessor.target_size, (224, 224))
        self.assertTrue(self.preprocessor.force_grayscale)
        self.assertIsNotNone(self.preprocessor.face_cascade)
        self.assertIsNotNone(self.preprocessor.eye_cascade)
    
    def test_detect_faces_empty(self):
        """Test face detection on an empty image"""
        # Create empty image
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Detect faces
        faces = self.preprocessor.detect_faces(empty_image)
        
        # Check results
        self.assertEqual(len(faces), 0)
    
    def test_resize_image(self):
        """Test image resizing functionality"""
        # Resize image
        resized = self.preprocessor.resize_image(
            self.test_image, 
            target_size=(150, 150),
            preserve_aspect_ratio=False
        )
        
        # Check dimensions
        self.assertEqual(resized.shape[:2], (150, 150))
    
    def test_resize_image_preserve_ratio(self):
        """Test image resizing with aspect ratio preservation"""
        # Create non-square image
        rect_image = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Resize image with aspect ratio preservation
        resized = self.preprocessor.resize_image(
            rect_image,
            target_size=(100, 100),
            preserve_aspect_ratio=True
        )
        
        # Check dimensions (should be padded to square)
        self.assertEqual(resized.shape[:2], (100, 100))
        
        # Check that aspect ratio was preserved (center should be filled)
        # The resized image should have a rectangle of zeros (padding) on top and bottom
        self.assertTrue(np.all(resized[0, 50] == 0))  # Top center should be padded
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion"""
        # Color image to grayscale
        color_img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        gray_img = self.preprocessor.convert_to_grayscale(color_img)
        
        # Check dimensions and type
        self.assertEqual(len(gray_img.shape), 2)
        self.assertEqual(gray_img.shape, (100, 100))
    
    def test_illumination_normalization(self):
        """Test illumination normalization methods"""
        # Create test grayscale image
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        gray_img[25:75, 25:75] = 200  # Bright region
        
        # Test each illumination method
        methods = ["hist_eq", "clahe", "gamma", "dog", "tantriggs"]
        for method in methods:
            normalized = self.preprocessor.normalize_illumination(gray_img, method)
            
            # Check that output is valid
            self.assertEqual(normalized.shape, gray_img.shape)
            self.assertTrue(normalized.dtype == np.uint8)
    
    def test_enhancement(self):
        """Test image enhancement methods"""
        # Create test grayscale image
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        gray_img[25:75, 25:75] = 200  # Bright region
        
        # Test each enhancement method
        methods = ["sharpen", "contrast", "adaptive"]
        for method in methods:
            enhanced = self.preprocessor.enhance_image(gray_img, method)
            
            # Check that output is valid
            self.assertEqual(enhanced.shape, gray_img.shape)
            self.assertTrue(enhanced.dtype == np.uint8)

if __name__ == '__main__':
    unittest.main() 