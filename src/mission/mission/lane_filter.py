"""
Reject lane lines that are far from the drivable area.

The lines bounding our lane touch the drivable area (YOLOPX removes lane pixels
from the drivable mask, so the gap is ~0). Lines of another lane / another part of
the track have non-drivable floor in between and are far from it.

Per lane path: median distance [m] from its window pixels to the nearest
drivable pixel (cv2.distanceTransform on the BEV ROI drivable mask). Larger than
lane_max_drivable_dist -> the line is dropped (path = None).

Skipped (lines kept) when the drivable mask of the frame is missing or covers less
than drivable_filter_min_ratio of the visible ROI. Pixels outside the camera view
count as "maybe drivable" so a line whose road is cut off by the view is kept.
"""
import cv2
import numpy as np

DEFAULT_FILTER_PARAMS = {
    'lane_max_drivable_dist': 0.10,     # m, farther than this from the drivable area -> drop
    'drivable_filter_min_ratio': 0.02,  # drivable share of the visible ROI to run the filter
}


class DrivableLaneFilter:

    def __init__(self, resolution, **params):
        p = dict(DEFAULT_FILTER_PARAMS)
        p.update({k: v for k, v in params.items() if k in DEFAULT_FILTER_PARAMS})
        self.p = p
        self.res = float(resolution)
        self.active = False         # whether the filter ran on the last frame

    def distance_map(self, drivable, visible):
        """Distance [m] from every ROI pixel to the nearest drivable (or unseen) pixel."""
        target = drivable | ~visible
        return cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 3) * self.res

    def apply(self, lanes, points, drivable, visible):
        """Drop far lines in place. Each SideResult gets .drivable_dist (m) when measured."""
        self.active = False
        if drivable is None or visible is None:
            return lanes
        n_visible = int(visible.sum())
        n_drivable = int((drivable & visible).sum())
        if n_visible == 0 or n_drivable < self.p['drivable_filter_min_ratio'] * n_visible:
            return lanes

        self.active = True
        dist = self.distance_map(drivable, visible)
        xs, ys = points
        for res in lanes.values():
            if res.path is None or res.pixel_idx is None or len(res.pixel_idx) == 0:
                continue
            idx = res.pixel_idx
            res.drivable_dist = float(np.median(dist[ys[idx], xs[idx]]))
            if res.drivable_dist > self.p['lane_max_drivable_dist']:
                res.rejected_path = res.path
                res.path = None
                res.reject = f'far from drivable ({res.drivable_dist * 100:.0f}cm)'
        return lanes
