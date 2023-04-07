import cv2
import numpy
import scipy.ndimage
import shapely
import shapely.geometry

import edt


def mask_idx_to_array_offset(mask_idx_array, mask_buffer=(0, 0)):
    mask_buffer = numpy.array(mask_buffer)

    idx_offset = mask_idx_array.min(axis=1) - numpy.array(mask_buffer)
    mask_shape = mask_idx_array.ptp(axis=1) + 1

    zero_array = numpy.zeros(
        (mask_shape + (2 * numpy.array(mask_buffer)))[::-1],
        dtype=numpy.uint8)

    fill_idxs = (mask_idx_array.T - idx_offset).T

    zero_array[fill_idxs[1, :], fill_idxs[0, :]] = 1
    return zero_array, idx_offset


def filter_largest_cc(binary_img):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img)
    lcc_label = numpy.argmax(stats[1:, -1]) + 1
    return labels == lcc_label


def polygon_from_pix_mask(pix_mask, largest_cc=True, add_z=None):
    mask_array, offset = mask_idx_to_array_offset(pix_mask)
    if largest_cc:
        mask_array = filter_largest_cc(mask_array).astype(numpy.uint8)
    im, contours, hierarchy = cv2.findContours(
        mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    contour += offset
    contour_pts = contour.reshape(contour.shape[0], contour.shape[2])
    
    if add_z is not None:
        contour_pts = numpy.insert(
            contour_pts, contour_pts.shape[1], add_z, 1)
    return shapely.geometry.Polygon(contour_pts)


def _center_from_pix_mask_polygon(
        pix_mask, exterior_centroid=False, largest_cc=True, **kwargs):
    poly = polygon_from_pix_mask(pix_mask, largest_cc=largest_cc)
    return numpy.array(
        poly.exterior.centroid if exterior_centroid else poly.centroid)


def _center_from_pix_mask_edt(
        pix_mask, largest_cc=True, center_of_mass=True, **kwargs):
    arr, offsets = mask_idx_to_array_offset(pix_mask)
    if largest_cc:
        arr = filter_largest_cc(arr).astype(numpy.uint8)
    arr_edt = edt.edt(arr)
    if center_of_mass:
        return scipy.ndimage.center_of_mass(arr_edt)[::-1] + offsets
    else:
        return numpy.array(cv2.minMaxLoc(arr_edt)[-1]) + offsets


def center_from_pix_mask(pix_mask, method=None, **kwargs):
    method = (method or "edt")

    return {
        "edt": _center_from_pix_mask_edt,
        "polygon": _center_from_pix_mask_polygon
        }[method](pix_mask, **kwargs)


def minimum_axial_lengths_polygon(s):
    edge_arr = numpy.array(s.minimum_rotated_rectangle.exterior.coords)
    a1 = numpy.linalg.norm(edge_arr[0] - edge_arr[3])
    a0 = numpy.linalg.norm(edge_arr[0] - edge_arr[1])
    return sorted((a1, a0), reverse=True)


def center_point_from_roi_info(roi_info, **kwargs):
    return numpy.array(
        center_from_pix_mask(roi_info.pix_mask, **kwargs).tolist() +
        [roi_info.plane_z])
