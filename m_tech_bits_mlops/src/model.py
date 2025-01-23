from loguru import logger
import time


def run_ml_model():
    logger.info("Starting ML model execution")
    time.sleep(1)  # Simulate model loading
    logger.info("Model loaded successfully")
    time.sleep(2)  # Simulate data preprocessing
    logger.info("Data preprocessing completed")
    time.sleep(3)  # Simulate model training/inference
    logger.info("ML model execution finished")
    return "Model Output"


if __name__ == "__main__":
    output = run_ml_model()
    logger.info(f"Model output: {output}")
