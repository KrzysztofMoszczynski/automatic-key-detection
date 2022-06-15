import pandas as pd
from utils import test, prepare_data, knn, validate_models, train, models
from constants import BEST_TRAIN_MODEL

#histogram = prepare_data.make_histogram("datasets/chosenFiles/TRAAAAW128F429D538.h5")
#print(prepare_data.normalize_histogram(histogram))


#print(test.count_all_songs_with_key_over_given_percent("datasets/MillionSongSubset", 80))
#prepare_data.save_songs_with_key_confidence_over_given_percentage(0.7)
#prepare_data.split_train_test()


train_data, test_data = prepare_data.get_train_test_sets()

train.train_model(BEST_TRAIN_MODEL, train_data, lr=0.01)
#train.find_the_best_model(train_data, 5, train_best_model=True)
#print(len(train_data), len(test_data))
#knn = knn.KNeighbours(train_data, 15)
#knn.train_k()

#validate_models.validate_random_song(data)

#validate_models.validate_knn(data)

#train.train_the_best_model(data, 5)