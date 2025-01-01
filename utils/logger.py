import logging

def setup_logger(name: str, level: int = logging.DEBUG, log_file: str = None) -> logging.Logger:
    """
    Setup a logger with optional file output.

    :param name: Logger name
    :param level: Logging level
    :param log_file: Optional log file path
    :return: Configured logger
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(level)
    return logger

# Example usage
if __name__ == "__main__":
    log = setup_logger("example_logger", log_file="example.log")
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.critical("This is a critical message.")
