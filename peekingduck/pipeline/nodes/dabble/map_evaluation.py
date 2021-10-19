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
MAP evaluation
"""

import os
import logging
from typing import Any, Dict, Type

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.input.utils.read import VideoNoThread
from peekingduck.pipeline.nodes.dabble.coco_utils.pack_detections import PackDetections
from peekingduck.pipeline.nodes.dabble.coco_utils.pack_keypoints import PackKeypoints


class Node(AbstractNode):
    """ MAP evaluation node class that evaluates the MAP of a model.

    This node evaluates a model using the MS COCO (val 2017) dataset. It uses
    the COCO API for loading the annotations and evaluating the outputs from
    the model.

    Inputs:
        |filename|

        |pipeline_end|

    Outputs:
        |img|

        |filename|

        |saved_video_fps|

        |pipeline_end|

    Configs:
        evaluation_task (:obj: `str`): **{"detection", "keypoints"}, default = 'detection'**

            evaluate model based on the specified task. "detection" for object
            detection models and "keypoints" for pose estimation models.

        evaluation_class (:obj: `str`): **default = ["all"]**

            evaluate images from the selected categories. Example:
            ["person", "dog", "spoon"]. The name of categories can be referenced
            from https://cocodataset.org/#explore. If ["all"] is specified,
            all images from all categories will be evaluated. When keypoints
            evaluation, evaluation task is automatically switched to "person".
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:

        self.logger = logging.getLogger(__name__)

        config = self._init_loader(config)

        super().__init__(config, node_path=__name__, **kwargs)

        self._allowed_extensions = ["jpg", "jpeg", "png"]
        self.file_end = False

        self._get_next_input()

        self.model_predictions = []

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        outputs = self._run_single_file()

        if self.file_end:
            self.logger.info(f"{len(self._filepaths)} images left to be processed")
            self._get_next_input()
            outputs = self._run_single_file()

            self.model_predictions = self.packer.pack(self.model_predictions,
                                                      self.filename_info,
                                                      inputs)

            if outputs["pipeline_end"] is True:
                self.logger.info("Evaluating model results...")
                coco_dt = self.coco.loadRes(self.model_predictions)
                eval_type = 'bbox' if self.evaluation_task == 'detection' else self.evaluation_task
                coco_eval = COCOeval(self.coco, coco_dt, eval_type)

                if self.evaluation_class[0] != "all":
                    coco_eval.params.catIds = [self.cat_ids]
                    img_ids = [img_info['id'] for img_info in self.filename_info.values()]
                    coco_eval.params.imgIds = img_ids

                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

        return outputs

    def _init_loader(self, config: Dict[str, Any]) -> Dict[str, Any]:

        self.evaluation_task = config["evaluation_task"]
        if self.evaluation_task == 'keypoints':
            config["input"] = config["input"] + ["keypoints", "keypoint_scores"]
            self.evaluation_class = ['person']
            self.coco = COCO(config['keypoints_dir'])
            self.packer = PackKeypoints()
        else:
            config["input"] = config["input"] + ['bboxes', 'bbox_labels', 'bbox_scores']
            self.evaluation_class = config['evaluation_class']
            self.coco = COCO(config['instances_dir'])
            self.packer = PackDetections()

        coco_instance = COCO(config['instances_dir'])
        self._load_images(config["images_dir"], coco_instance)

        return config

    def _load_images(self, images_dir: str, coco_instance: Type[COCO]) -> None:

        if self.evaluation_class[0] == 'all':
            self.logger.info("Using images from all the categories for evaluation.")
            img_ids = sorted(coco_instance.getImgIds())
        else:
            self.logger.info(f" Using images from: {self.evaluation_class}")
            cat_names = self.evaluation_class
            self.cat_ids = coco_instance.getCatIds(catNms=cat_names)
            img_ids = coco_instance.getImgIds(catIds=self.cat_ids)

        self.filename_info = {}
        self._filepaths = []
        prefix = os.path.join(os.getcwd(), images_dir)
        for img_id in img_ids[0:50]:
            img = coco_instance.loadImgs(img_id)[0]
            image_path = os.path.join(prefix, img['file_name'])
            self._filepaths.append(image_path)

            self.filename_info[img['file_name']] = {'id': img_id,
                                                    'image_size': (img['width'],
                                                                   img['height'])}

    def _get_next_input(self) -> None:

        if self._filepaths:
            file_path = self._filepaths.pop(0)
            self._file_name = os.path.basename(file_path)

            if self._is_valid_file_type(file_path):
                self.videocap = VideoNoThread(
                    file_path,
                    False
                )
                self._fps = self.videocap.fps
            else:
                self.logger.warning("Skipping '%s' as it is not an accepted file format %s",
                                    file_path,
                                    str(self._allowed_extensions)
                                    )
                self._get_next_input()

    def _run_single_file(self) -> Dict[str, Any]:
        success, img = self.videocap.read_frame()  # type: ignore

        self.file_end = True
        outputs = {"img": None,
                   "filename": self._file_name,
                   "saved_video_fps": self._fps,
                   "pipeline_end": True}
        if success:
            self.file_end = False
            outputs = {"img": img,
                       "filename": self._file_name,
                       "saved_video_fps": self._fps,
                       "pipeline_end": False}

        if self.file_end:
            self.counter = True

        return outputs

    def _is_valid_file_type(self, filepath: str) -> bool:

        if filepath.split(".")[-1] in self._allowed_extensions:
            return True
        return False
