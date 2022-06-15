import utils.models as models

DATASET_PATH = "datasets/MillionSongSubset"
OUTPUT_CSV_PATH = "datasets/songs.csv"
INPUT_70_PATH = "datasets/songs_70_percent_confidence.csv"
INPUT_70_TRAIN_PATH = "datasets/songs_70_percent_confidence_train.csv"
INPUT_70_TEST_PATH = "datasets/songs_70_percent_confidence_test.csv"
MODELS_PATH = "models"
MIN_K_IMPROVEMENT = 0.001
BEST_K = 10
EPOCHS = 60
CROSS_VAL_EPOCHS = 10
CROSS_VAL_EPOCHS_NN = 20
BATCH_SIZE = 64
BEST_TRAIN_MODEL = models.get_models_for_training()[8]
PATH_MATRIX = "images"
KEY_LABELS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
              'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
