DEV_MODE: bool = False  # Imports less data: Only 500 articles, 50 dimensional embeddings, 400 tokens per line max
USE_D1: bool = False  # Use dummy vocab and articles
USE_D2 = True  # Use slightly larger dummy dataset with proper data loading (embeddings)

BATCH_SIZE: int = 3 if USE_D1 else 100
CUDA: bool = True

LEARNING_RATE = 0.001
NUM_EPOCHS = 2000
EPSILON = 0.000001
PRINT_SUMMARY_FREQUENCY = 7  # prints summary every x epochs
ARTICLES_ON_DEVICE = 1

INTRA_DECODER = True

CHEKPOINT_PATH = 'parameters/checkpoint'    # where to save checkpoint
