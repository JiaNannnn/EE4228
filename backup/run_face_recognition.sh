#!/bin/bash

# Function to display header
function show_header {
    clear
    echo "==================================="
    echo "Real-Time Face Recognition System"
    echo "==================================="
    echo ""
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check for virtual environment and create if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if [ ! -d "venv/lib/python3"*/site-packages/cv2 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error installing dependencies. Please check requirements.txt."
        deactivate
        exit 1
    fi
    echo "Dependencies installed successfully."
fi

# Check if shape predictor model exists
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    echo "Downloading required models..."
    python download_models.py
    if [ $? -ne 0 ]; then
        echo "Error downloading models."
        echo "Please manually download dlib's shape_predictor_68_face_landmarks.dat model."
        deactivate
        exit 1
    fi
fi

# Main menu
function main_menu {
    show_header
    echo "Choose an option:"
    echo "1. Run real-time face recognition"
    echo "2. Train recognition model"
    echo "3. Run benchmark tests"
    echo "4. Exit"
    echo ""
    read -p "Enter option (1-4): " option
    
    case $option in
        1) run_recognition ;;
        2) train_model ;;
        3) run_tests ;;
        4) 
            deactivate
            echo "Exiting. Goodbye!"
            exit 0
            ;;
        *) 
            echo "Invalid option. Please try again."
            sleep 2
            main_menu
            ;;
    esac
}

# Run recognition function
function run_recognition {
    show_header
    echo "Running real-time face recognition..."
    echo "Press 'q' to quit, 's' to save a face, 'd' to delete last saved face"
    echo ""
    
    # Check if model exists
    if [ ! -f "trained_hybrid_model.pkl" ]; then
        echo "Warning: No trained model found."
        echo "You will need to train the system by adding faces during runtime."
        sleep 3
    fi
    
    python realtime_face_recognition.py
    echo ""
    echo "Recognition session ended."
    sleep 2
    main_menu
}

# Train model function
function train_model {
    show_header
    echo "Training face recognition model..."
    echo ""
    
    # Ask for dataset path
    read -p "Enter dataset path (or press Enter for default AT&T dataset): " dataset_path
    
    if [ -z "$dataset_path" ]; then
        echo "Using default AT&T dataset..."
        python train_hybrid_model.py --visualize
    else
        echo "Using custom dataset at $dataset_path..."
        python train_hybrid_model.py --dataset "$dataset_path" --visualize
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error during model training."
    else
        echo "Model training completed successfully."
    fi
    
    sleep 3
    main_menu
}

# Run tests function
function run_tests {
    show_header
    echo "Running benchmark tests..."
    echo ""
    
    echo "Choose test type:"
    echo "1. Performance benchmark"
    echo "2. Accuracy test"
    echo "3. Stress test"
    echo "4. All tests"
    echo "5. Back to main menu"
    echo ""
    read -p "Enter option (1-5): " test_option
    
    case $test_option in
        1)
            echo "Running performance benchmark..."
            python test_realtime_recognition.py --test-type benchmark --duration 20
            ;;
        2)
            read -p "Enter test dataset path: " test_data
            echo "Running accuracy test..."
            python test_realtime_recognition.py --test-type accuracy --test-data "$test_data"
            ;;
        3)
            echo "Running stress test..."
            python test_realtime_recognition.py --test-type stress
            ;;
        4)
            read -p "Enter test dataset path (optional, press Enter to skip accuracy test): " test_data
            echo "Running all tests..."
            if [ -z "$test_data" ]; then
                python test_realtime_recognition.py --test-type benchmark --duration 20
                python test_realtime_recognition.py --test-type stress
            else
                python test_realtime_recognition.py --test-type benchmark --duration 20
                python test_realtime_recognition.py --test-type accuracy --test-data "$test_data"
                python test_realtime_recognition.py --test-type stress
            fi
            ;;
        5)
            main_menu
            ;;
        *)
            echo "Invalid option."
            sleep 2
            run_tests
            ;;
    esac
    
    echo ""
    echo "Tests completed. Results saved in test_results directory."
    sleep 3
    main_menu
}

# Start the menu
main_menu 