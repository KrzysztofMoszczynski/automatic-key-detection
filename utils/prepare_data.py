import numpy
import numpy as np
from PythonSrc import hdf5_getters as getters
from constants import DATASET_PATH, OUTPUT_CSV_PATH, INPUT_70_PATH, INPUT_70_TRAIN_PATH, INPUT_70_TEST_PATH,\
    INPUT_70_START_END_TRAIN_PATH, INPUT_70_START_END_TEST_PATH, INPUT_70_START_END_PATH
import os
import glob
import pandas as pd
from collections import deque


#creates 12 transpositions from one song
def transpose_song(histogram, key, mode):
    histogram_size = len(histogram)
    transpositions = []
    keys = []
    for transposition_index in range(histogram_size):
        if transposition_index == 0:
            transpositions.append(histogram)
            keys.append(join_keys_and_modes(key, mode))
        else:
            transposition = deque(histogram)
            transposition.rotate(transposition_index)
            new_key = (key + transposition_index) % 12
            new_key = join_keys_and_modes(new_key, mode)
            transpositions.append(list(transposition))
            keys.append(new_key)
    return transpositions, keys


#leaves values 0-11 for minor mode and 12-23 for major mode
def join_keys_and_modes(key, mode):
    return key + mode * 12


def make_histogram(chromas):
    histogram = [0] * 12
    for chroma in chromas:
        for pitch_index in range(len(chroma)):
            histogram[pitch_index] += chroma[pitch_index]
    return histogram


def make_histogram_10_percent(chromas):
    histogram = [0] * 12
    percent_amount_10 = int(0.1 * len(chromas))
    new_chromas = chromas[:percent_amount_10] + chromas[len(chromas) - percent_amount_10:]
    for chroma in new_chromas:
        for pitch_index in range(len(chroma)):
            histogram[pitch_index] += chroma[pitch_index]
    return histogram


def make_histogram_from_file(file_name):
    file = getters.open_h5_file_read(file_name)
    chromas = getters.get_segments_pitches(file)
    histogram = [0] * 12
    for chroma in chromas:
        for pitch_index in range(len(chroma)):
            histogram[pitch_index] += chroma[pitch_index]
    file.close()
    return histogram

def normalize_histogram(histogram):
    sum = numpy.sum(histogram)
    for field_index in range(len(histogram)):
        histogram[field_index] /= sum
    return histogram


def save_songs_with_key_confidence_over_given_percentage(percentage, ext='.h5'):
    song_index = 0
    songs_to_save = pd.DataFrame(
        columns=['pitches', 'key', 'key_confidence'])
    for root, dirs, files in os.walk(DATASET_PATH):
        files = glob.glob(os.path.join(root, '*'+ext))
        for file in files:
            h5 = getters.open_h5_file_read(file)
            if getters.get_key_confidence(h5) > percentage and getters.get_mode_confidence(h5) > percentage\
                    and getters.get_mode(h5) != -1:
                #histogram = make_histogram(getters.get_segments_pitches(h5))
                histogram = make_histogram_10_percent(getters.get_segments_pitches(h5))
                normalized_histogram = normalize_histogram(histogram)
                key = getters.get_key(h5)
                mode = getters.get_mode(h5)
                key_confidence = getters.get_key_confidence(h5)
                transpositions, keys = transpose_song(normalized_histogram, key, mode)
                for index in range(len(keys)):
                    songs_to_save.loc[song_index + index] = [transpositions[index], keys[index], key_confidence]
                song_index = song_index + len(keys)
            h5.close()

    #songs_to_save.to_csv(f'datasets/songs_{int(percentage * 100)}_percent_confidence.csv', index=False)
    songs_to_save.to_csv(f'datasets/songs_{int(percentage * 100)}_percent_confidence_start_end.csv', index=False)


def get_all_songs():
    songs = pd.read_csv(INPUT_70_PATH)
    return songs


def get_all_start_end_songs():
    songs = pd.read_csv(INPUT_70_START_END_PATH)
    return songs


def get_train_test_sets():
    train = pd.read_csv(INPUT_70_TRAIN_PATH)
    test = pd.read_csv(INPUT_70_TEST_PATH)
    return train, test


def get_train_test_start_end_sets():
    train = pd.read_csv(INPUT_70_START_END_TRAIN_PATH)
    test = pd.read_csv(INPUT_70_START_END_TEST_PATH)
    return train, test


def split_train_test(ratio_test=0.2):
    songs = get_all_start_end_songs()
    msk = np.random.rand(len(songs)) < ratio_test
    songs_test = songs[msk]
    songs_train = songs[~msk]
    songs_test.to_csv('datasets/songs_70_percent_confidence_start_end_test.csv', index=False)
    songs_train.to_csv('datasets/songs_70_percent_confidence_start_end_train.csv', index=False)
