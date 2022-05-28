import pandas as pd
from utils import test, prepare_data

#histogram = prepare_data.make_histogram("datasets/chosenFiles/TRAAAAW128F429D538.h5")
#print(prepare_data.normalize_histogram(histogram))


print(test.count_all_songs_with_key_over_given_percent("datasets/MillionSongSubset", 80))