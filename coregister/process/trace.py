import numpy
import scipy


def compare_traces(trace0, trace1):
    trace0 = trace0[:len(trace1)]
    trace1 = trace1[:len(trace0)]
    return numpy.corrcoef(trace0, trace1)[0, 1]


def smooth_trace(*args, **kwargs):
    return scipy.ndimage.gaussianfilter1d(
        *args, **kwargs)


def compare_traces_smooth(trace0, trace1,
                          smoothing_args=(), smoothing_kwargs=None):
    smoothing_kwargs = ({} if smoothing_kwargs is None else smoothing_kwargs)

    trace0 = smooth_trace(trace0, *smoothing_args, **smoothing_kwargs)
    trace1 = smooth_trace(trace1, *smoothing_args, **smoothing_kwargs)
    return compare_traces(trace0, trace1)
