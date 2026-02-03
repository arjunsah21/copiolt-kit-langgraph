import logging
import sys
from datetime import datetime
from typing import Any, Dict

# Configure logging
def setup_logging():
    """Configure logging for the application"""
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Configure src logger
    src_logger = logging.getLogger("src")
    src_logger.setLevel(logging.INFO)
    src_logger.propagate = False  # Don't propagate to root if we handle it
    if src_logger.handlers:
        src_logger.handlers.clear()
    src_logger.addHandler(console_handler)
    
    # Configure Uvicorn loggers explicitly
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        u_logger = logging.getLogger(logger_name)
        u_logger.setLevel(logging.INFO)
        u_logger.propagate = False # Uvicorn handles its own propagation usually, but we want to control it
        if u_logger.handlers:
            u_logger.handlers.clear()
        u_logger.addHandler(console_handler)

# Apply configuration immediately
setup_logging()
logger = logging.getLogger("src")


def log_request(endpoint: str, data: dict):
    """Log incoming API request"""
    try:
        logger.info(
            f"Request to {endpoint} | timestamp={datetime.utcnow().isoformat()} | "
            f"data_keys={list(data.keys()) if data else []}"
        )
    except Exception as e:
        logger.error(f"Error logging request: {e}")


def log_agent_start(user_message: str, latitude: float, longitude: float):
    """Log agent execution start"""
    try:
        logger.info(
            f"Agent started | message_length={len(user_message)} | "
            f"location=({latitude}, {longitude})"
        )
    except Exception as e:
        logger.error(f"Error logging agent start: {e}")


def log_tool_call(tool_name: str, params: Dict[str, Any]):
    """Log tool invocation"""
    try:
        logger.info(
            f"Tool call: {tool_name} | params={params}"
        )
    except Exception as e:
        logger.error(f"Error logging tool call: {e}")


def log_error(error: Exception, context: str = ""):
    """Log error with context"""
    try:
        logger.error(
            f"Error in {context}: {type(error).__name__} - {str(error)}",
            exc_info=True
        )
    except Exception as e:
        logger.error(f"Error logging error: {e}")
