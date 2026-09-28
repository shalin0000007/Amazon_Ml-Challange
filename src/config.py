import os

# Project root paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# File paths
TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

OUTPUT_CANDIDATES = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
OUTPUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")

# Model and Pipeline Parameters
MAX_CANDIDATES_PER_ENTITY = 30
RANDOM_STATE = 42
F_BETA = 0.5
DEFAULT_THRESHOLD = 0.72
SINGLETON_THRESHOLD = 0.65

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
