"""
Logging Utilities for Face Preprocessing

This module provides utilities for configuring and managing logging
throughout the face preprocessing system.
"""

import os
import logging
import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Set up logging configuration for face preprocessing
    
    Args:
        log_file: Path to log file (if None, logging goes to console only)
        console_level: Logging level for console output
        file_level: Logging level for file output
    
    Returns:
        Configured logger instance
    """
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers (to avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Set up file handler if log file specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler (max 5MB per file, 3 backup files)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=3
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger

def log_process_start(logger, config=None):
    """
    Log the start of a processing operation with configuration details
    
    Args:
        logger: Logger instance
        config: Dictionary of configuration settings (optional)
    """
    logger.info(f"Processing started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if config:
        logger.info("Configuration settings:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

def log_process_complete(logger, stats=None):
    """
    Log the completion of a processing operation with statistics
    
    Args:
        logger: Logger instance
        stats: Dictionary of processing statistics (optional)
    """
    logger.info(f"Processing completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if stats:
        logger.info("Processing statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}") 