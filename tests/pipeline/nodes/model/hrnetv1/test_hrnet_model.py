"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
from unittest import TestCase, mock

import pytest
import yaml


from peekingduck.pipeline.nodes.model.hrnet import Node


@pytest.fixture
def hrnet_config():
    filepath = (
        Path.cwd()
        / "tests"
        / "pipeline"
        / "nodes"
        / "model"
        / "hrnetv1"
        / "test_hrnet.yml"
    )
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture
def hrnet(request, hrnet_config):
    node = Node(hrnet_config)
    return node


def replace_download_weights(model_dir, blob_file):
    return False


@pytest.mark.mlmodel
class TestHrnetModel:
    def test_no_weights(self, hrnet_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.yolov4.yolo_model.logger"
        ) as captured:
            hrnet = Node(config=hrnet_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "---no weights detected. proceeding to download...---"
            )
            assert "weights downloaded" in captured.records[1].getMessage()
            assert hrnet is not None
