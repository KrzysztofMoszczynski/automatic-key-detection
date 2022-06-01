from PythonSrc import hdf5_getters as getters
import os
import glob
import shutil
import pandas as pd


def get_one_song():
    #f = h5.File('../datasets/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5', 'r')
    f = getters.open_h5_file_read('datasets/MillionSongSubset/A/A/A/TRAAABD128F429CF47.h5')
    print("segment pitches: ", getters.get_segments_pitches(f))
    print("key confidence: ", getters.get_key_confidence(f))
    f.close()


def count_all_files(basedir, ext='.h5'):
    count = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        count += len(files)
    return count


def count_all_songs_with_key_over_given_percent(basedir, percent, ext='.h5'):
    count = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        for file in files:
            h5 = getters.open_h5_file_read(file)
            if getters.get_key_confidence(h5) > percent:
                count += 1
            h5.close()
    return count


def save_all_songs_with_key_over_70_percent(basedir, ext='.h5'):
    count = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        for file in files:
            h5 = getters.open_h5_file_read(file)
            if getters.get_key_confidence(h5) > 0.7:
                count += 1
                shutil.copyfile(file, "datasets/chosenFiles/"+os.path.basename(file))
            h5.close()
    return count


def test_pandas():
    data_frame = pd.DataFrame(columns=['arr', 'key'])
    first_arr = [[1, 2], [3, 4]]
    keys = ['A', 'B']
    for index in range(len(first_arr)):
        data_frame.loc[index] = [first_arr[index], keys[index]]

    print(data_frame)