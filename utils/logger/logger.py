
import logging
import os
from pathlib import Path
import sys
import time
from typing import Callable, Optional, TypeAlias


from pydantic import BaseModel, Field


def _pretty_format(message: str) -> str:
    """
    Format the message with a line of asterisks above and below it.
    The number of asterisks will have the same length as the input message, 
    with a maximum character length of 100.
    """
    asterisk = '*' * len(message)
    # Cut off the asterisk string at 50 characters to prevent wasted log space.
    if len(message) > 100:
        asterisk = asterisk[:100]
    return f"\n{asterisk}\n{message}\n{asterisk}\n"


class LogEntry(BaseModel):
    """
    A log entry with a message, level, line number, and a timestamp.
    """
    message: str = Field(..., description="The log message")
    level: int = Field(..., description="The log level")
    lineno: int = Field(..., description="The line number where the log entry was created")
    timestamp: float = Field(..., description="The timestamp of the log entry")

    def __init__(self, **data):
        super().__init__(**data)


class LogFile(BaseModel):
    filename: str = Field(..., description="The name of the log file")
    log_entries: list[LogEntry] = Field(default_factory=list, description="The log entries in the log file")


Configs = TypeAlias("Configs", BaseModel)


class Logger:

    def __init__(self, 
                 name: str, 
                 level: int = logging.INFO, 
                 log_folder: Path = Path("logs")
                ):
        self.name = name
        self.level = level
        self.stack_level = 2
        self.log_folder = log_folder if isinstance(log_folder, Path) else Path(log_folder)
        self.logger = None

        self._setup_logs()
        self._setup_logger()


    def _setup_logs(self):

        # Initialize LogFile instance to store structured logs
        self.log_file = LogFile(
            filename=f"{self.name}.log",
            log_entries=[]
        )

       # Ensure the log folder and file exist and are writable
        if not self.log_folder.exists():
            self.log_folder.mkdir(parents=True, exist_ok=True)

        if not (self.log_folder / self.log_file.filename).exists():
            (self.log_folder / self.log_file.filename).touch()


    def _setup_logger(self):

        # Create the logger itself.
        self.logger = logging.getLogger(self.name)

         # Set the default log level.
        self.logger.setLevel(self.level)
        self.logger.propagate = False # Prevent logs from being handled by parent loggers

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d - %(message)s')

        if not self.logger.handlers:
            # Create handlers (file and console)
            #self.filepath = self.log_folder / f"{self.name}.log"
            #file_handler = logging.FileHandler(self.filepath)
            console_handler = logging.StreamHandler()

            # Set level for handlers
            #file_handler.setLevel(logging.DEBUG) # We want to log everything to the file.
            console_handler.setLevel(self.level)

            # Create formatters and add it to handlers
            #file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to the logger
            #self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)


    def _add_log_entry(self, message: str, level: int, lineno: int) -> None:
        """
        Add a structured log entry to the LogFile model.
        """
        entry = LogEntry(
            message=message,
            level=level,
            lineno=lineno,
            timestamp=time.time()
        )
        self.log_file.log_entries.append(entry)


    def _message_template(self, message: str, method: Callable, f: bool, t: float, off: bool) -> None:
        """
        Args:
            f: Format the message with asterisk for easy viewing. These will not be saved to the log file.
            t: Pause the program by a specified number of seconds after the message has been printed to console.
            off: Turns off the logger for this message. Logs will still be saved to the log file
        """
        if not off:

            # Get the caller's line number
            frame = sys._getframe(2)  # Adjust frame level to get the correct caller
            lineno = frame.f_lineno

            # Add structured log entry
            level_mapping = {
                self.logger.info: logging.INFO,
                self.logger.debug: logging.DEBUG,
                self.logger.warning: logging.WARNING,
                self.logger.error: logging.ERROR,
                self.logger.critical: logging.CRITICAL
            }
            self._add_log_entry(message, level_mapping[method], lineno)

            if not f: # We move up the stack by 1 because it's a nested method.
                method(message, stacklevel=self.stacklevel+1)
            else:
                method(_pretty_format(message), stacklevel=self.stacklevel+1)
            if t:
                time.sleep(t)

    def info(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        Args:
            f: Format the message with asterisk for easy viewing. These will not be saved to the log file.
            t: Pause the program by a specified number of seconds after the message has been printed to console.
            off: Turns off the logger for this message. Logs will still be saved to the log file
        """
        self._message_template(message, self.logger.info, f, q, t, off)

    def debug(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        Args:
            f: Format the message with asterisk for easy viewing. These will not be saved to the log file.
            t: Pause the program by a specified number of seconds after the message has been printed to console.
            off: Turns off the logger for this message. Logs will still be saved to the log file
        """
        self._message_template(message, self.logger.debug, f, q, t, off)

    def warning(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        Args:
            f: Format the message with asterisk for easy viewing. These will not be saved to the log file.
            t: Pause the program by a specified number of seconds after the message has been printed to console.
            off: Turns off the logger for this message. Logs will still be saved to the log file
        """
        self._message_template(message, self.logger.warning, f, q, t, off)

    def error(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        Args:
            f: Format the message with asterisk for easy viewing. These will not be saved to the log file.
            t: Pause the program by a specified number of seconds after the message has been printed to console.
            off: Turns off the logger for this message. Logs will still be saved to the log file
        """
        self._message_template(message, self.logger.error, f, q, t, off)

    def critical(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        Args:
            f: Format the message with asterisk for easy viewing. These will not be saved to the log file.
            t: Pause the program by a specified number of seconds after the message has been printed to console.
            off: Turns off the logger for this message. Logs will still be saved to the log file
        """
        self._message_template(message, self.logger.critical, f, q, t, off)

    def exception(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        Args:
            f: Format the message with asterisk for easy viewing. These will not be saved to the log file.
            t: Pause the program by a specified number of seconds after the message has been printed to console.
            off: Turns off the logger for this message. Logs will still be saved to the log file
        """
        self._message_template(message, self.logger.exception, f, q, t, off)