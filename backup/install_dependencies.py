"""
Dependencies Installer for Face Recognition System

This script checks and installs all required dependencies for the face recognition system,
handling platform-specific requirements and providing detailed guidance for any issues.
"""

import os
import sys
import platform
import subprocess
import pkg_resources
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("install.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Installer")

# Define required packages
REQUIRED_PACKAGES = [
    "numpy>=1.19.0",
    "opencv-contrib-python>=4.5.0",
    "dlib>=19.21.0",
    "scikit-learn>=0.24.0",
    "matplotlib>=3.3.0",
    "scipy>=1.6.0",
    "pandas>=1.2.0",
    "tqdm>=4.50.0",
    "flask>=2.0.0",
    "Pillow>=8.0.0",
    "psutil>=5.8.0",
    "requests>=2.25.0"
]

# Platform-specific packages
if platform.system() == "Windows":
    PLATFORM_PACKAGES = [
        "pywin32>=300"
    ]
else:
    PLATFORM_PACKAGES = []

# Development packages (optional)
DEV_PACKAGES = [
    "pytest>=6.0.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.800"
]

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error(f"Python 3.8+ is required, but found {python_version.major}.{python_version.minor}")
        return False
    logger.info(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                             stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        logger.error("pip is not available")
        return False

def check_installed_packages(package_list):
    """
    Check which packages from the list are installed
    
    Returns:
    --------
    tuple
        (installed_packages, missing_packages)
    """
    installed = []
    missing = []
    
    for package_spec in package_list:
        package_name = package_spec.split('>=')[0]
        try:
            pkg_resources.get_distribution(package_name)
            installed.append(package_spec)
        except pkg_resources.DistributionNotFound:
            missing.append(package_spec)
            
    return installed, missing

def install_packages(packages, dev=False):
    """
    Install packages using pip
    
    Parameters:
    -----------
    packages : list
        List of packages to install
    dev : bool, optional
        Whether these are development packages
        
    Returns:
    --------
    bool
        True if all packages installed successfully, False otherwise
    """
    if not packages:
        return True
        
    # Common pip arguments
    pip_args = [sys.executable, "-m", "pip", "install"]
    
    # Add user flag if not in virtual environment
    in_virtualenv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_virtualenv:
        pip_args.append("--user")
        
    # Log what we're installing
    package_type = "development" if dev else "required"
    logger.info(f"Installing {len(packages)} {package_type} packages: {', '.join(packages)}")
    
    # Add packages to command
    pip_args.extend(packages)
    
    # Run pip
    try:
        result = subprocess.run(pip_args, check=False, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error installing packages: {result.stderr}")
            return False
            
        logger.info("Packages installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Exception during package installation: {e}")
        return False

def check_dlib_prerequisites():
    """
    Check if prerequisites for dlib are installed
    
    Returns:
    --------
    bool
        True if prerequisites are met, False otherwise
    """
    system = platform.system()
    
    if system == "Windows":
        # On Windows, check for Visual Studio
        try:
            # Check if CMake is installed
            cmake_result = subprocess.run(["cmake", "--version"], 
                                         capture_output=True, text=True, check=False)
            
            if cmake_result.returncode != 0:
                logger.warning("CMake not found, which is required for dlib installation")
                logger.warning("Please install CMake from https://cmake.org/download/")
                return False
                
            # Try to detect Visual Studio
            vs_paths = [
                "C:/Program Files (x86)/Microsoft Visual Studio",
                "C:/Program Files/Microsoft Visual Studio"
            ]
            
            vs_found = False
            for path in vs_paths:
                if os.path.exists(path):
                    vs_found = True
                    break
                    
            if not vs_found:
                logger.warning("Microsoft Visual Studio not found in standard paths")
                logger.warning("Visual Studio with C++ toolchain is required for dlib installation")
                logger.warning("Please install Visual Studio Community from https://visualstudio.microsoft.com/downloads/")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking build prerequisites: {e}")
            return False
            
    elif system == "Linux":
        # On Linux, check for build tools
        try:
            # Check for GCC
            gcc_result = subprocess.run(["gcc", "--version"], 
                                      capture_output=True, text=True, check=False)
            
            if gcc_result.returncode != 0:
                logger.warning("GCC not found, which is required for dlib installation")
                logger.warning("Please install build-essential package using your package manager")
                return False
                
            # Check for CMake
            cmake_result = subprocess.run(["cmake", "--version"], 
                                        capture_output=True, text=True, check=False)
            
            if cmake_result.returncode != 0:
                logger.warning("CMake not found, which is required for dlib installation")
                logger.warning("Please install cmake package using your package manager")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking build prerequisites: {e}")
            return False
            
    elif system == "Darwin":  # macOS
        # On macOS, check for build tools
        try:
            # Check for Command Line Tools
            xcode_result = subprocess.run(["xcode-select", "-p"], 
                                        capture_output=True, text=True, check=False)
            
            if xcode_result.returncode != 0:
                logger.warning("Xcode Command Line Tools not found")
                logger.warning("Please install them using: xcode-select --install")
                return False
                
            # Check for CMake
            cmake_result = subprocess.run(["cmake", "--version"], 
                                        capture_output=True, text=True, check=False)
            
            if cmake_result.returncode != 0:
                logger.warning("CMake not found, which is required for dlib installation")
                logger.warning("Please install CMake using Homebrew: brew install cmake")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking build prerequisites: {e}")
            return False
            
    else:
        logger.warning(f"Unsupported platform: {system}")
        return False

def print_manual_dlib_instructions():
    """Print instructions for manual dlib installation"""
    system = platform.system()
    
    logger.info("Instructions for manual dlib installation:")
    
    if system == "Windows":
        logger.info("1. Ensure you have Visual Studio with C++ toolchain installed")
        logger.info("2. Ensure you have CMake installed")
        logger.info("3. Run: pip install dlib")
        logger.info("   If that fails, try: pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl")
        logger.info("   (Replace cp310 with your Python version, e.g., cp39 for Python 3.9)")
        
    elif system == "Linux":
        logger.info("1. Install dependencies: sudo apt-get install build-essential cmake")
        logger.info("2. Install Python dev: sudo apt-get install python3-dev")
        logger.info("3. Run: pip install dlib")
        
    elif system == "Darwin":  # macOS
        logger.info("1. Install Command Line Tools: xcode-select --install")
        logger.info("2. Install CMake: brew install cmake")
        logger.info("3. Run: pip install dlib")

def create_virtual_environment():
    """
    Create virtual environment if it doesn't exist
    
    Returns:
    --------
    bool
        True if virtual environment exists or was created, False otherwise
    """
    venv_dir = "venv"
    
    # Check if venv already exists
    if os.path.exists(venv_dir) and os.path.isdir(venv_dir):
        logger.info(f"Virtual environment already exists at {venv_dir}")
        return True
        
    # Create virtual environment
    logger.info(f"Creating virtual environment at {venv_dir}")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        logger.info("Virtual environment created successfully")
        
        # Print activation instructions
        if platform.system() == "Windows":
            logger.info("To activate the virtual environment, run:")
            logger.info(f"    {venv_dir}\\Scripts\\activate.bat")
        else:
            logger.info("To activate the virtual environment, run:")
            logger.info(f"    source {venv_dir}/bin/activate")
            
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating virtual environment: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating virtual environment: {e}")
        return False

def download_models():
    """
    Download required models
    
    Returns:
    --------
    bool
        True if models were downloaded successfully, False otherwise
    """
    logger.info("Downloading required models")
    try:
        # Run the download_models.py script
        if os.path.exists("download_models.py"):
            subprocess.run([sys.executable, "download_models.py"], check=True)
            logger.info("Models downloaded successfully")
            return True
        else:
            logger.error("download_models.py script not found")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading models: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading models: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting Face Recognition System dependency installation")
    
    # Get operating system information
    system = platform.system()
    logger.info(f"Detected operating system: {system} {platform.release()}")
    
    # Check Python version
    if not check_python_version():
        logger.error("Please install Python 3.8 or newer")
        return 1
        
    # Check pip
    if not check_pip():
        logger.error("pip is not available. Please install pip before continuing")
        return 1
        
    # Create virtual environment
    if not create_virtual_environment():
        logger.error("Failed to create virtual environment")
        return 1
        
    # Combine required and platform-specific packages
    all_required_packages = REQUIRED_PACKAGES + PLATFORM_PACKAGES
    
    # Check installed packages
    installed_required, missing_required = check_installed_packages(all_required_packages)
    installed_dev, missing_dev = check_installed_packages(DEV_PACKAGES)
    
    logger.info(f"{len(installed_required)}/{len(all_required_packages)} required packages already installed")
    logger.info(f"{len(installed_dev)}/{len(DEV_PACKAGES)} development packages already installed")
    
    # Check dlib prerequisites if dlib is missing
    dlib_missing = any(pkg.startswith("dlib") for pkg in missing_required)
    if dlib_missing:
        if not check_dlib_prerequisites():
            logger.warning("dlib prerequisites not met")
            print_manual_dlib_instructions()
            
            # Still try to install other packages
            missing_required = [pkg for pkg in missing_required if not pkg.startswith("dlib")]
            logger.info("Continuing installation without dlib")
    
    # Install missing packages
    if missing_required:
        if not install_packages(missing_required):
            logger.error("Failed to install required packages")
            return 1
    else:
        logger.info("All required packages are already installed")
        
    # Ask if user wants to install development packages
    if missing_dev:
        install_dev = input("Do you want to install development packages? [y/N] ").strip().lower()
        if install_dev == 'y':
            if not install_packages(missing_dev, dev=True):
                logger.warning("Failed to install development packages")
                # This is not critical, so continue
                
    # Download models
    if not download_models():
        logger.warning("Failed to download models")
        logger.warning("You will need to download the models manually before using the system")
        # This is not critical, so continue
        
    logger.info("Installation completed successfully")
    
    # Print next steps
    logger.info("\nNext steps:")
    if platform.system() == "Windows":
        logger.info("1. Activate the virtual environment: venv\\Scripts\\activate.bat")
    else:
        logger.info("1. Activate the virtual environment: source venv/bin/activate")
    logger.info("2. Run the face recognition system: python realtime_face_recognition.py")
    logger.info("   - or - Run the web interface: python web_interface.py")
    
    return 0
    
if __name__ == "__main__":
    sys.exit(main()) 