import numpy


def get_array_subset_indices(set_arr, subset_arr):
    set_arr_sorted = numpy.argsort(set_arr)
    subset_arr_pos = numpy.searchsorted(set_arr[set_arr_sorted], subset_arr)
    indices = set_arr_sorted[subset_arr_pos]
    return indices


def inrange(val, range_tup):
    return range_tup[0] <= val <= range_tup[-1]


def inrange_it(vals, range_tups):
    for val, range_tup in zip(vals, range_tups):
        if not inrange(val, range_tup):
            return False
    return True
