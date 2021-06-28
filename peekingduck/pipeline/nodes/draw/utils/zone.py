from typing import List, Tuple, Any, Iterable, Union
import numpy as np
import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA
from peekingduck.pipeline.nodes.draw.utils.constants import \
    LEGEND_BOX, PRIMARY_PALETTE, PRIMARY_PALETTE_LENGTH, CHAMPAGNE, SMALL_FONTSCALE, THIN
from peekingduck.pipeline.nodes.draw.utils.general import \
    get_image_size, project_points_onto_original_image


def _draw_zone_area(frame: np.array, points: List[Tuple[int]],
                    zone_index: int) -> None:
    total_points = len(points)
    for i in range(total_points):
        if i == total_points-1:
            # for last point, link to first point
            cv2.line(frame, points[i], points[0],
                     PRIMARY_PALETTE[zone_index % PRIMARY_PALETTE_LENGTH], 3)
        else:
            # for all other points, link to next point in polygon
            cv2.line(frame, points[i], points[i+1],
                     PRIMARY_PALETTE[zone_index % PRIMARY_PALETTE_LENGTH], 3)


def draw_zones(frame: np.array, zones: List[Any]) -> None:
    """draw the boundaries of the zones used in zoning analytics

    Args:
        frame (np.array): image of current frame
        zones (Zone): zones used in the zoning analytics. possible
        classes are Area and Divider.
    """
    for i, zone_pts in enumerate(zones):
        _draw_zone_area(frame, zone_pts, i)


def draw_zone_count(frame: np.array, zone_count: List[int]) -> None:
    """draw pts of selected object onto frame

    Args:
        frame (np.array): image of current frame
        zone_count (List[float]): object count, likely people, of each zone used
        in the zone analytics
    """
    y_pos = 50
    text = '--ZONE COUNTS--'
    cv2.putText(frame, text, (25, y_pos), FONT_HERSHEY_SIMPLEX, SMALL_FONTSCALE,
                CHAMPAGNE, LEGEND_BOX['text_thickness'], LINE_AA)
    for i, count in enumerate(zone_count):
        y_pos += 25
        text = 'ZONE {0}: {1}'.format(i+1, count)
        cv2.putText(frame,
                    text,
                    (25, y_pos),
                    FONT_HERSHEY_SIMPLEX,
                    SMALL_FONTSCALE,
                    PRIMARY_PALETTE[i % PRIMARY_PALETTE_LENGTH],
                    THIN['thickness'],
                    LINE_AA)