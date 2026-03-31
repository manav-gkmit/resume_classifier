import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger to output to stdout with a simple format.

    Call this once during application startup (app.main imports it and
    executes it). Subsequent imports will return the same logger
    configuration.
    """

    if logging.getLogger().handlers:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


logger = logging.getLogger("resume_classifier")
