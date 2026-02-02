import logging
import sys
from datetime import datetime
from typing import Any, Dict

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


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
