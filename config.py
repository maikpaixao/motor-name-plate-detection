from dotenv import load_dotenv
import os

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
TRAIN_STEPS = int(os.getenv("TRAIN_STEPS"))
VALIDATION_STEPS = int(os.getenv("VALIDATION_STEPS"))
EPOCHS = int(os.getenv("EPOCHS"))
INPUT_CHANNEL = int(os.getenv("INPUT_CHANNEL"))
WIDTH = int(os.getenv("WIDTH"))
HEIGHT = int(os.getenv("HEIGHT"))
TRAIN_FOLDER = os.getenv("TRAIN_FOLDER")
TRAIN_MASK_FOLDER = os.getenv("TRAIN_MASK_FOLDER")
VALIDATION_FOLDER = os.getenv("VALIDATION_FOLDER")
VALIDATION_MASK_FOLDER = os.getenv("VALIDATION_MASK_FOLDER")
TEST_FOLDER = os.getenv("TEST_FOLDER")
TEST_MASK_FOLDER = os.getenv("TEST_MASK_FOLDER")
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")
OUTPUT_IMAGES = os.getenv("OUTPUT_IMAGES")
BACKBONE = os.getenv("BACKBONE")
TYPE_OF_IMAGES = os.getenv("TYPE_OF_IMAGES")
LOAD_MODEL = os.getenv("LOAD_MODEL")
PLATE_MODEL = os.getenv("PLATE_MODEL")
TEXT_MODEL = os.getenv("TEXT_MODEL")
INITIAL_EPOCH = int(os.getenv("INITIAL_EPOCH"))
FONT_PATH = os.getenv("FONT_PATH")



