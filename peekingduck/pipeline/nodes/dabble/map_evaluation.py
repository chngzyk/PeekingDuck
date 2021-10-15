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

"""

import os
import logging
from typing import Any, Dict, List

from pycocotools.coco import COCO

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.input.utils.read import VideoNoThread
from peekingduck.pipeline.nodes.dabble.coco_utils.load_coco_images import load_images
from peekingduck.pipeline.nodes.dabble.coco_utils.pack_detections import PackDetections


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.logger = logging.getLogger(__name__)

        self._allowed_extensions = ["jpg", "jpeg", "png"]
        self.file_end = False
        self.frame_counter = -1
        self.tens_counter = 10

        self._init_loader(config)

        self._get_next_input()

        self.packer = {"instances": PackDetections()}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        outputs = self._predict_images()

        if bool(inputs):
            if inputs['pipeline_end']:
                self.logger.info("Evaluating results...")
            else:
                self.logger.info(inputs['bboxes'])
                self.packer[self.evaluation_type].pack()

        return outputs

    def _init_loader(self, config) -> None:

        self.images_dir = config["images_dir"]

        self.evaluation_type = config["evaluation_type"]
        if self.evaluation_type == 'keypoints':
            self.evaluation_class = ['person']
        else:
            self.evaluation_class = config['evaluation_class']

        self.coco_instance = COCO(config['instances_dir'])

        self.images_info, self.filename_info, self._filepaths = load_images(self.coco_instance,
                                                                            self.evaluation_class,
                                                                            self.images_dir)

    def _predict_images(self):
        outputs = self._run_single_file()

        approx_processed = round((self.frame_counter/self.videocap.frame_count)*100)
        self.frame_counter += 1

        if approx_processed > self.tens_counter:
            self.logger.info('Approximately Processed: %s%%...', self.tens_counter)
            self.tens_counter += 10

        if self.file_end:
            self.logger.info('Completed processing file: %s', self._file_name)
            self._get_next_input()
            outputs = self._run_single_file()
            self.frame_counter = 0
            self.tens_counter = 10

        return outputs

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
                   "pipeline_end": True,
                   "filename": self._file_name,
                   "saved_video_fps": self._fps,
                   "image_id": self.filename_info[self._file_name]['id'],
                   "image_size": self.filename_info[self._file_name]['image_size'],
                   "coco_instance": self.coco_instance}
        if success:
            self.file_end = False
            outputs = {"img": img,
                       "pipeline_end": False,
                       "filename": self._file_name,
                       "saved_video_fps": self._fps,
                       "image_id": self.filename_info[self._file_name]['id'],
                       "image_size": self.filename_info[self._file_name]['image_size'],
                       "coco_instance": None}

        return outputs

    def _is_valid_file_type(self, filepath: str) -> bool:

        if filepath.split(".")[-1] in self._allowed_extensions:
            return True
        return False
