import numpy
import numpy as np
from PythonSrc import hdf5_getters as getters


def make_histogram(file_name):
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