import pytest


from utils.logger.logger import Logger, LogEntry, LogFile


@pytest.fixture
def logger():
    return Logger("test_logger")

# 1. Test message logging

def test_message_logging(logger):
    logger.info("Test info message")

# 2. Test error logging

# 3. Test log file generation

