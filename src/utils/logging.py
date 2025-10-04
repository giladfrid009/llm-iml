from __future__ import annotations
import os
import logging


LOG_FORMAT = "%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s"
LOG_DATEFMT = "%m-%d %H:%M:%S"
LOG_LEVEL_ENV = "IML_LOG_LEVEL"


def loglevel_names() -> list[str]:
    """Return a list of valid log level names."""
    return list(logging.getLevelNamesMapping().keys())


def parse_log_level(value: str, default: int = logging.INFO) -> int:
    name_to_level = logging.getLevelNamesMapping()
    value = value.strip().upper()

    level = int(value) if value.isdecimal() else None

    if value.isdecimal():
        level = int(value)
        if level not in name_to_level.values():
            logging.warning(f"Invalid log level: {level}. Using default: {default}.")
            return default
        return level

    if value not in name_to_level:
        logging.warning(f"Invalid log level name: {value}. Using default: {default}.")
        return default

    return name_to_level[value]


CURRENT_LEVEL: int = parse_log_level(os.getenv(LOG_LEVEL_ENV, "INFO"))
ACTIVE_LOGGERS: dict[str, logging.Logger] = {}


def _configure_logger(logger: logging.Logger, level: int, fmt: logging.Formatter) -> None:
    """
    Apply level, handlers, and formatter to a logger.
    """
    logger.setLevel(level)
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(fmt)
    else:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False


def setup_logging(level: int | str, *, is_global: bool = False) -> None:
    """
    Set the default logging level and reconfigure loggers.

    Args:
      level (int | str): the new default logging level (e.g., logging.DEBUG).
      is_global (bool): if True, also configure the root logger.
    """
    global CURRENT_LEVEL

    if isinstance(level, str):
        level = parse_log_level(level, default=CURRENT_LEVEL)

    CURRENT_LEVEL = level
    os.environ[LOG_LEVEL_ENV] = logging.getLevelName(level)

    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    if is_global:
        root = logging.getLogger()
        _configure_logger(root, level, fmt)

    # Update all tracked loggers
    for lg in ACTIVE_LOGGERS.values():
        _configure_logger(lg, level, fmt)


def create_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """
    Create or retrieve a logger and track it for future reconfiguration.

    Args:
      name (str): the logger name (commonly __name__ or any identifier).
      level (int | str | None): optional override; uses CURRENT_LEVEL if None.

    Returns:
        logging.Logger: the configured logger instance.
    """
    if isinstance(level, str):
        level = parse_log_level(level, default=CURRENT_LEVEL)

    level = level if level is not None else CURRENT_LEVEL
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    if name in ACTIVE_LOGGERS:
        logger = ACTIVE_LOGGERS[name]
        _configure_logger(logger, level, fmt)
        return logger

    logger = logging.getLogger(name)
    _configure_logger(logger, level, fmt)
    ACTIVE_LOGGERS[name] = logger
    return logger
