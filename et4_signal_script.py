# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:55:30 2024

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:04:17 2024

Author: Saarit
"""

import time
import sys
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Sets up logging for the signals script.
    Logs are written to both a file and the console.
    """
    logger = logging.getLogger('signals_script_logger')
    logger.setLevel(logging.INFO)  # Set to INFO or DEBUG as needed

    # File handler with rotation
    file_handler = RotatingFileHandler('signals_script.log', maxBytes=5*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Prevent adding multiple handlers if they already exist
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logging()

def process_signals():
    """
    Placeholder for your signal processing logic.
    Replace this function with your actual signal operations.
    """
    logger.info("Processing signals...")
    # Example operation: Simulate signal processing with a sleep
    time.sleep(1)  # Simulate signal processing
    logger.debug("Signals processed successfully.")

def main():
    """
    Main execution function for the signals script.
    Runs continuously with a 15-second sleep between iterations.
    """
    logger.info("Signals script execution started.")
    try:
        while True:
            process_signals()
            logger.debug("Waiting for 15 seconds before next iteration.")
            time.sleep(15)  # Sleep for 15 seconds before next iteration
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down gracefully.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        logger.info("Signals script has been terminated.")

if __name__ == "__main__":
    main()
