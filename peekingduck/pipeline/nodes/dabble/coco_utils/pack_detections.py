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
Pack object detection results into COCO format
"""

import logging
from typing import List, Dict, Any

from peekingduck.pipeline.nodes.dabble.coco_utils.constants import COCO_CATEGORY_DICTIONARY
from peekingduck.pipeline.nodes.draw.utils.general import project_points_onto_original_image


class PackDetections:
    """
    Pack the outputs from PeekingDuck's object detection models into COCO's
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

        for bbox, bbox_label, bbox_score in zip(inputs["bboxes"],
                                                inputs["bbox_labels"],
                                                inputs["bbox_scores"]):

            bbox_label_index = COCO_CATEGORY_DICTIONARY[bbox_label]

            bbox = project_points_onto_original_image(bbox, img_size)

            bbox = [bbox[0][0],
                    bbox[0][1],
                    bbox[1][0] - bbox[0][0],
                    bbox[1][1] - bbox[0][1]]

            model_predictions.append({"image_id": int(img_id),
                                      "category_id": int(bbox_label_index),
                                      "bbox": list(bbox),
                                      "score": bbox_score})

        return model_predictions
