import os
import yaml
import pytest
import cv2
import numpy as np
from unittest import mock, TestCase
from peekingduck.pipeline.nodes.model.yolo_license_plate import Node
from peekingduck.pipeline.nodes.model.yolov4_license_plate.licenseplate_files.detector import (
    Detector,
)


@pytest.fixture
def LP_config():
    filepath = os.path.join(
        os.getcwd(),
        "tests/pipeline/nodes/model/yolov4_license_plate/test_yolo_license_plate.yml",
    )
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = os.getcwd()

    return node_config


@pytest.fixture(params=["v4", "v4tiny"])
def LPyolo(request, LP_config):
    LP_config["model_type"] = request.param
    node = Node(LP_config)

    return node


@pytest.fixture()
def LPyolo_detector(LP_config):
    LP_config["model_type"] = "v4tiny"
    detector = Detector(LP_config)

    return detector


def replace_download_weights(root, blob_file):
    return False


@pytest.mark.mlmodel
class TestLPYolo:
    def test_no_LP_image(self, test_no_lp_images, LPyolo):

        blank_image = cv2.imread(test_no_lp_images)
        output = LPyolo.run({"img": blank_image})
        expected_output = {"bboxes": [], "bbox_labels": [], "bbox_scores": []}
        assert output.keys() == expected_output.keys()
        assert type(output["bboxes"]) == np.ndarray
        assert type(output["bbox_labels"]) == np.ndarray
        assert type(output["bbox_scores"]) == np.ndarray
        assert len(output["bboxes"]) == 0
        assert len(output["bbox_labels"]) == 0
        assert len(output["bbox_scores"]) == 0

    def test_at_least_one_LP_image(self, test_LP_images, LPyolo):
        test_img = cv2.imread(test_LP_images)
        output = LPyolo.run({"img": test_img})
        assert "bboxes" in output
        assert len(output["bboxes"]) != 0
        assert len(output["bboxes"]) == len(output["bbox_labels"])

    def test_no_weights(self, LP_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ):
            with mock.patch(
                "peekingduck.weights_utils.downloader.download_weights",
                wraps=replace_download_weights,
            ):
                with TestCase.assertLogs(
                    "peekingduck.pipeline.nodes.model.yolov4_license_plate.LP_detector_model.logger"
                ) as captured:

                    LPyolo = Node(config=LP_config)
                    # records 0 - 20 records are updates to configs
                    assert (
                        captured.records[0].getMessage()
                        == "---no LP weights detected. proceeding to download...---"
                    )
                    assert (
                        captured.records[1].getMessage()
                        == "---LP weights download complete.---"
                    )
                    assert LPyolo is not None

    def test_model_initialization(self, LP_config):
        detector = Detector(config=LP_config)
        model = detector.yolo
        assert model is not None
