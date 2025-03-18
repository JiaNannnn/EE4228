#!/usr/bin/env python
"""
Setup script for the face_preproc package
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="face_preproc",
    version="0.1.0",
    author="AI Team",
    author_email="ai@example.com",
    description="Image preprocessing tools with normalization and enhancement techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/face_preproc",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'process-gallery=face_preproc.scripts.process_gallery:main',
            'process-images=face_preproc.scripts.process_images:main',
        ],
    },
) 