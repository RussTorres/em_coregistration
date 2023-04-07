"""determine leave-one-out results of transformation"""
import copy

import numpy

import coregister.transform.transform


def leave_one_out_fitting(transform_dict, src_pts, dst_pts, weights_vec=None):
    # copy may not be necessary
    tform_dict = copy.deepcopy(transform_dict)

    def fit_transform(src, dst, weights_vec=None):
        tform = coregister.transform.Transform(json=tform_dict)
        tform.estimate(src, dst, weights_vec=weights_vec)
        return tform

    predicted_dsts = numpy.empty_like(dst_pts)

    for l1out_idx in range(src_pts.shape[0]):  # src_pts, dst_pts):
        pts_mask = numpy.zeros(src_pts.shape[0], dtype=bool)
        pts_mask[l1out_idx] = True

        masked_src_pts = src_pts[~pts_mask]
        masked_dst_pts = dst_pts[~pts_mask]
        masked_weights_vec = (
            None if weights_vec is None
            else weights_vec[~pts_mask]
        )

        l1out_tform = fit_transform(masked_src_pts, masked_dst_pts,
                                    weights_vec=masked_weights_vec)

        l1out_src_pts = numpy.array([src_pts[l1out_idx]])

        predicted_dst = l1out_tform.tform(l1out_src_pts)[0]

        predicted_dsts[l1out_idx] = predicted_dst

    residual = numpy.linalg.norm(dst_pts - predicted_dsts, axis=1)

    return residual, predicted_dsts
