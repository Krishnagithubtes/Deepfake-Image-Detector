# Paths
TRAIN_DIR = 'data/Train'
VALIDATION_DIR = 'data/Validation'
TEST_DIR = 'data/Test'
MODEL_SAVE_PATH = 'model backup/deepfake_detector_model.keras'
LOG_DIR = 'logs'

# Training parameters
BATCH_SIZE = 128
IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 20
LEARNING_RATE = 0.0001

# Data augmentation parameters
DATA_AUGMENTATION = True

# Mixed precision training
MIXED_PRECISION = True