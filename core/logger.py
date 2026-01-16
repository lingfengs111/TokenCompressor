import logging
import os
import sys
import time

# Configure console logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)


def setup_logger(
    name: str = "semantic-id", level: int = logging.INFO, log_to_file: bool = False, log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up a logger with console output and optional file logging.

    Automatically detects if running in a DataLoader worker process and
    disables file logging to prevent multiple log files.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_to_file: Whether to log to file (default: False)
        log_dir: Directory for log files (default: "logs")

    Returns:
        Configured logger instance
    """

    # Also check if we're in a subprocess by checking process name
    try:
        import torch.multiprocessing as mp

        is_main_process = mp.current_process().name == "MainProcess"
    except ImportError:
        is_main_process = True

    # Disable file logging if we're in a worker process or not the main process
    if not is_main_process and log_to_file:
        log_to_file = False

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent propagation to root logger

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Add file handler if requested (and not in worker process)
    if log_to_file:
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"log-{timestamp}.txt"
        file_path = os.path.join(log_dir, log_filename)

        # Create file handler
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)

        # Add file handler to logger
        logger.addHandler(file_handler)

    return logger