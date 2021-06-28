from typing import List, Tuple, Any, Iterable, Union
import numpy as np
import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA
import peekingduck.pipeline.nodes.draw.utils.constants as constants
from peekingduck.pipeline.nodes.draw.utils.general import \
    get_image_size, project_points_onto_original_image


def draw_bboxes(frame: np.array,
                bboxes: List[List[float]],
                color: Tuple[int, int, int],
                thickness: int) -> None:
    """Draw bboxes onto an image frame.

    Args:
        frame (np.array): image of current frame
        bboxes (List[List[float]]): bounding box coordinates
        color (Tuple[int, int, int]): color of bounding box
        thickness (int): thickness of bounding box
    """
    image_size = get_image_size(frame)
    for bbox in bboxes:
        _draw_bbox(frame, bbox, image_size, color, thickness)


def _draw_bbox(frame: np.array,
               bbox: List[float],
               image_size: Tuple[int, int],
               color: Tuple[int, int, int],
               thickness: int) -> np.array:
    """ Draw a single bounding box """
    top_left, bottom_right = project_points_onto_original_image(
        bbox, image_size)
    cv2.rectangle(frame, (top_left[0], top_left[1]),
                  (bottom_right[0], bottom_right[1]),
                  color, thickness)


def draw_tags(frame: np.array,
              bboxes: List[List[float]],
              tags: List[str],
              color: Tuple[int, int, int]) -> None:
    """Draw tags above bboxes.

    Args:
        frame (np.array): image of current frame
        bboxes (List[List[float]]): bounding box coordinates
        tags (List[string]): tag associated with bounding box
        color (Tuple[int, int, int]): color of text
    """
    image_size = get_image_size(frame)
    for idx, bbox in enumerate(bboxes):
        _draw_tag(frame, bbox, tags[idx], image_size, color)


def _draw_tag(frame: np.array,
              bbox: np.array,
              tag: str,
              image_size: Tuple[int, int],
              color: Tuple[int, int, int]) -> None:
    """Draw a tag above a single bounding box.
    """
    top_left, _ = project_points_onto_original_image(bbox, image_size)
    position = int(top_left[0]), int(top_left[1]-25)
    cv2.putText(frame, tag, position, FONT_HERSHEY_SIMPLEX, 1, color, 2)


def draw_pts(frame: np.array, pts: List[Tuple[float]]) -> None:
    """draw pts of selected object onto frame

    Args:
        frame (np.array): image of current frame
        pts (List[Tuple[float]]): bottom midpoints of bboxes
    """
    for point in pts:
        cv2.circle(frame, point, 5, constants.CHAMPAGNE, -1)