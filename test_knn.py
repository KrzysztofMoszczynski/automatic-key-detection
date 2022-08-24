from utils import test, prepare_data, validate_models

data = prepare_data.get_all_start_end_songs()
validate_models.test_knn(data)