import dataclasses
import os

# import cv2
import h5py
import numpy
# import shapely


def ldel(s, lead):
    if s.startswith(lead):
        todel = len(lead)
    else:
        todel = 0
    return s[todel:]


# def roi_mask_to_contour_pts(mask_idx_array, mask_buffer=(0, 0)):
#     # TODO phase out buffer -- unnecessary
#     mask_buffer = numpy.array(mask_buffer)
# 
#     idx_offset = mask_idx_array.min(axis=1) - numpy.array(mask_buffer)
#     mask_shape = mask_idx_array.ptp(axis=1) + 1
# 
#     zero_array = numpy.zeros(
#         (mask_shape + (2 * numpy.array(mask_buffer)))[::-1],
#         dtype="uint8")
# 
#     fill_idxs = (mask_idx_array.T - idx_offset).T
# 
#     zero_array[fill_idxs[1, :], fill_idxs[0, :]] = 1
# 
#     im, contours, hierarchy = cv2.findContours(
#         zero_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     contour = contours[0]
#     contour += idx_offset
#     return contour.reshape(contour.shape[0], contour.shape[2])
# 
# 
# def roi_mask_to_polygon(*args, add_z=None, **kwargs):
#     contour_pts = roi_mask_to_contour_pts(*args, **kwargs)
#     if add_z is not None:
#         contour_pts = numpy.insert(
#             contour_pts, contour_pts.shape[1], add_z, 1)
#     return shapely.geometry.Polygon(contour_pts)


@dataclasses.dataclass
class FunctionalRoiInfo:
    # pulled from file
    plane_name: str
    roi_name: str
    plane_z: float 
    pix_mask: numpy.ndarray
    trace: numpy.ndarray

    # computed later
    polygon = None
    center = None

    @property
    def name(self):
        return f"{self.plane_name}_{self.roi_name}"


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

    def yield_roi_info(self):
        for rat in self.rois_and_traces:
            traces = rat["DfOverF"]["dff_raw"]["data"][...]
            rat_name = self._group_name(rat)
            plane_name = rat_name.split("_")[-1]
            plane_z = self._plane_integer_from_name(plane_name)

            for i, roi in enumerate(self.get_roi_groups_from_plane_name(
                                        plane_name)):
                roi_name = self._group_name(roi)
                pix_mask = roi["pix_mask"][...]
                trace = traces[i]
                yield FunctionalRoiInfo(
                    plane_name=plane_name,
                    roi_name=roi_name,
                    plane_z=plane_z,
                    pix_mask=pix_mask,
                    trace=trace,
                    )

    # def get_roi_plane_names_to_polygon(self, include_plane_z=True):
    #     # TODO namedtuple as keys
    #     plane_name_roi_name_to_polygon = {}
    #     for plane_name, roi_name, roi in self.yield_plane_name_roi_name_roi_group_tups():
    #         if include_plane_z:
    #             plane_z = self._plane_integer_from_name(plane_name)
    #         else:
    #             plane_z = None
    #         roi_polygon = roi_mask_to_polygon(roi["pix_mask"][:], add_z=plane_z)
    #         plane_name_roi_name_to_polygon[(plane_name, roi_name)] = roi_polygon
    #     return plane_name_roi_name_to_polygon