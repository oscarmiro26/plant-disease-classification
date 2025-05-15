import os
import logging


def setup_logger(
    model_name: str,
    base_log_dir: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Create a run-unique logger in base_log_dir/model_name/.
    """
    # Makes sure the folder exists, important for the first time you train a model.
    model_dir = os.path.join(base_log_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Count existing log files
    existing = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]
    run_idx = len(existing)

    # Create new run folder
    run_name = f"{model_name}_{run_idx}"
    run_dir  = os.path.join(model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(run_dir, "train.log")
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )

    logger = logging.getLogger('train')
    logger.info(f"Created logger with run_dir: {run_dir}")
    return logger, run_dir
    