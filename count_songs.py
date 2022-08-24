from utils import prepare_data

set= prepare_data.get_all_songs()

print(set['key'].value_counts())