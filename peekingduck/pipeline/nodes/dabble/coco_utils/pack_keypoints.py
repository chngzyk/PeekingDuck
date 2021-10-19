# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pack pose estimation results into COCO format
"""

import logging
from typing import List, Dict, Any

import numpy as np

from peekingduck.pipeline.nodes.draw.utils.general import project_points_onto_original_image


class PackKeypoints:
    """
    Pack the outputs from PeekingDuck's pose estimation models into COCO's
    evaluation format.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def pack(self, model_predictions: List[Dict],
             filename_info: Dict[str, Any],
             inputs: Dict[str, Any]) -> List[Dict]:
        """Function to pack inputs from object detection model into COCO's
        evaluation format.

        Args:
            model_predictions (list): an empty list
            filename_info (dict): contains information on an image's ID and size
            inputs (dict): node's inputs containing an image's filename,
                           detected bounding boxes, labels, and scores.


        Returns:
            model_predictions (list): results packed into COCO's format
        """

        img_id = filename_info[inputs["filename"]]['id']
        img_size = filename_info[inputs["filename"]]['image_size']

        for keypoint, score in zip(inputs["keypoints"],
                                   inputs["keypoint_scores"]):

            keypoint = project_points_onto_original_image(keypoint, img_size)

            pred = np.append(keypoint, np.ones((len(keypoint), 1)), axis=1)
            pred = list(pred.flat)

            model_predictions.append({"image_id": int(img_id),
                                      "category_id": 1,
                                      "keypoints": pred,
                                      "score": sum(score)/len(score)})

        return model_predictions
