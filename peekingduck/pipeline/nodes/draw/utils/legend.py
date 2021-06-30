"""Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from typing import Dict, List, Tuple, Any, Iterable, Union
import numpy as np
import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA
from peekingduck.pipeline.nodes.draw.utils.constants import \
    LEGEND_BOX, NORMAL_FONTSCALE
from peekingduck.pipeline.nodes.draw.utils.general import \
    get_image_size, project_points_onto_original_image


def draw_count(frame: np.array, count: int) -> None:
    """draw count of selected object onto frame

    Args:
        frame (np.array): image of current frame
        count (int): total count of selected object
            in current frame
    """
    text = 'COUNT: {0}'.format(count)
    cv2.putText(frame, text, (10, 50), FONT_HERSHEY_SIMPLEX,
                NORMAL_FONTSCALE, LEGEND_BOX['text_colour'], LEGEND_BOX['thickness'], LINE_AA)


def draw_fps(frame: np.array, current_fps: float) -> None:
    """ Draw FPS onto frame image

    Args:
        frame (np.array): image of current frame
        current_fps (float): value of the calculated FPS
    """
    text = "FPS: {:.05}".format(current_fps)
    text_location = (25, 25)

    cv2.putText(frame, text, text_location, FONT_HERSHEY_SIMPLEX, NORMAL_FONTSCALE,
                LEGEND_BOX['text_colour'], LEGEND_BOX['thickness'], LINE_AA)


def draw_legends(inputs: Dict[str, Any], choices: Dict[str, bool]) -> None:
    """ Draw legends onto image

    Args:
        frame (np.array): image of current frame
        choices (Dict[str, bool]): list of legends to be drawn
    """
    raise NotImplementedError()
