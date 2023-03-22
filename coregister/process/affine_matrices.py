# affine transformations

# TODO:
#   reflection tform
#   scaling
#   translation
#   transform_pts
import numpy


def points_to_homogeneous(points):
    return numpy.insert(points, points.shape[1], 1, 1)


def points_from_homogeneous(points):
    return numpy.true_divide(points[:, :-1], points[:, [-1]])


def transform_pts(mtx, points):
    """Transform NxM point array using a KxK
    (K=M+1) homogeneous matrix
    """
    return points_from_homogeneous((
        mtx @ points_to_homogeneous(  # noqa
            numpy.array(points)).T).T)
