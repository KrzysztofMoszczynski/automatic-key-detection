import pandas as pd
from utils import test, prepare_data, knn, validate_models

#histogram = prepare_data.make_histogram("datasets/chosenFiles/TRAAAAW128F429D538.h5")
#print(prepare_data.normalize_histogram(histogram))


#print(test.count_all_songs_with_key_over_given_percent("datasets/MillionSongSubset", 80))
#prepare_data.save_songs_with_key_confidence_over_given_percentage(0.8)

data = prepare_data.get_songs()
knn = knn.KNeighbours(data, 15)
knn.train_k()

#validate_models.validate_random_song(data)

#validate_models.validate_knn(data)