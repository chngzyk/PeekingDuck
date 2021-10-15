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

logger = logging.getLogger(__name__)


def load_images(coco_instance, evaluation_class: str, image_dir: str):

    if evaluation_class[0] == 'all':
        logger.info(" Using images from all the categories for evaluation.")
        img_ids = sorted(coco_instance.getImgIds())
    else:
        logger.info(f" Using images from: {evaluation_class}")
        cat_name = evaluation_class
        catIds = coco_instance.getCatIds(catNms=cat_name)
        img_ids = coco_instance.getImgIds(catIds=catIds)

    images_info = {}
    filename_info = {}
    images_path = []
    prefix = os.path.join(os.getcwd(), image_dir)
    for img_id in img_ids[0:5]:
        img = coco_instance.loadImgs(img_id)[0]
        image_dir = os.path.join(prefix, img['file_name'])
        profile = {'image_dir': image_dir,
                   'image_size': (img['width'], img['height'])}
        images_info[img_id] = profile

        filename_info[img['file_name']] = {'id': img_id,
                                           'image_size': (img['width'],
                                                          img['height'])}

        images_path.append(image_dir)

    return images_info, filename_info, images_path
