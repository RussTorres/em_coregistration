import dataclasses
import os

import h5py
import numpy

from coregister.process.roi import (
    polygon_from_pix_mask,
    center_from_pix_mask
)


def ldel(s, lead):
    if s.startswith(lead):
        todel = len(lead)
    else:
        todel = 0
    return s[todel:]


@dataclasses.dataclass
class FunctionalRoiInfo:
    # pulled from file
    plane_name: str
    roi_name: str
    plane_z: float 
    pix_mask: numpy.ndarray
    trace: numpy.ndarray
    score: float

    # computed later
    polygon = None
    center = None

    @property
    def name(self):
        return f"{self.plane_name}_{self.roi_name}"

    def compute_values(self):
        # TODO: some wasted calculation here can be more efficient
        self.polygon = polygon_from_pix_mask(self.pix_mask)
        self.center = center_from_pix_mask(self.pix_mask)


class ROI_mask_nwb:
    def __init__(self, *args, **kwargs):
        self._prepare_h5(*args, **kwargs)
    
    def _prepare_h5(self, *args, **kwargs):
        self._h5_file = h5py.File(*args, **kwargs)
    
    def _close_file(self):
        self._h5_file.close()

    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self._close_file()
    
    @property
    def processing(self):
        return self._h5_file["processing"]

    @property
    def analysis(self):
        return self._h5_file["analysis"]
    
    @property
    def rois_and_traces(self):
        return [
            v for k, v in self.processing.items()
            if k.startswith("rois_and_traces")]
    
    @staticmethod
    def _group_name(group):
        return os.path.basename(group.name)

    @staticmethod
    def _plane_integer_from_name(plane_name):
        return int(ldel(plane_name, "plane"))
    
    def get_roi_groups_from_plane_name(self, plane_name):
        rat_name = f"rois_and_traces_{plane_name}"
        # TODO: make this a regex
        return [
            g for g_name, g
            in self.processing[rat_name][
                "ImageSegmentation"]["imaging_plane"].items()
            if g_name.startswith("roi_") and
            not g_name.endswith("_list")]
        
    def yield_plane_name_roi_name_roi_group_tups(self):
        for rat in self.rois_and_traces:
            rat_name = self._group_name(rat)
            plane_name = rat_name.split("_")[-1]
            for roi in self.get_roi_groups_from_plane_name(plane_name):
                roi_name = self._group_name(roi)
                yield plane_name, roi_name, roi

    def yield_roi_info(self, compute_roi_info=False):
        for rat in self.rois_and_traces:
            traces = rat["DfOverF"]["dff_raw"]["data"][...]
            rat_name = self._group_name(rat)
            plane_name = rat_name.split("_")[-1]
            plane_z = self._plane_integer_from_name(plane_name)
            pika_scores = self.analysis[f"roi_classification_pika/{plane_name}/score"][...]

            for i, roi in enumerate(self.get_roi_groups_from_plane_name(
                                        plane_name)):
                roi_name = self._group_name(roi)
                pix_mask = roi["pix_mask"][...]
                trace = traces[i]
                pika_score = pika_scores[i]
                roi_info = FunctionalRoiInfo(
                    plane_name=plane_name,
                    roi_name=roi_name,
                    plane_z=plane_z,
                    pix_mask=pix_mask,
                    trace=trace,
                    score=pika_score
                    )
                if compute_roi_info:
                    roi_info.compute_values()
                yield roi_info
