# EE4228 Face Recognition System

A face recognition system with advanced face alignment and specialized handling for recognition challenges.

## Features

- **Advanced Face Detection**: Support for both Viola-Jones and PCA-based face detection
- **Face Alignment**: Improved face alignment using facial landmarks (via dlib)
- **Robust Preprocessing**: Enhanced preprocessing pipeline with NaN/Inf value handling
- **Specialized Models**: Support for specialized recognition models for challenging cases
- **User Management**: Add and manage multiple users in the system
- **Interactive UI**: Streamlit-based UI for easy interaction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JiaNannnn/EE4228.git
cd EE4228
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Optional: Install dlib for advanced face alignment:
```bash
pip install dlib
```

4. Download the facial landmark model (if using dlib):
```bash
python download_dlib_model.py
```

## Usage

1. Start the application:
```bash
streamlit run streamlit_app.py
```

2. Using the application:
   - Add users in the User Management page
   - Train the model in the Training page
   - Use the Recognition page for real-time face recognition
   - Configure detector settings in the Detector Settings page

## Advanced Features

### Face Alignment

The system uses dlib's facial landmark detection for improved face alignment, which helps with:
- Consistent eye positioning
- Better handling of head rotations
- Improved recognition accuracy

### Specialized Models

For users with recognition challenges, the system can create specialized models:
- Lower confidence thresholds
- Improved handling of edge cases
- Better temporal consistency

## System Requirements

- Python 3.8+
- Webcam
- 4GB RAM minimum (8GB recommended)
- Windows/Mac/Linux

## Troubleshooting

### NaN Value Issues

If you encounter NaN value errors:
1. Check that face preprocessing is working correctly
2. Verify that face alignment is enabled
3. Use the specialized model for challenging cases

### Camera Not Working

1. Ensure webcam permissions are granted
2. Try restarting the application
3. Check that no other application is using the webcam

## License

MIT License 