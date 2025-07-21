import os

import onnxruntime as ort

from .module.arcface import ArcFaceONNX
from .module.attribute import Attribute
from .module.landmark import Landmark
from .module.retinaface import RetinaFace
from .utils.face import Face


class FaceAnalyzer:
    def __init__(self, weight_root):
        ort.set_default_logger_severity(3)
        self.models = {
            "landmark_3d_68": Landmark(model_file = os.path.join(weight_root, "1k3d68.onnx")),
            "landmark_2d_106": Landmark(model_file = os.path.join(weight_root, "2d106det.onnx")),
            "genderage": Attribute(model_file = os.path.join(weight_root, "genderage.onnx")),
            "recognition": ArcFaceONNX(model_file = os.path.join(weight_root, "w600k_r50.onnx")),
        }
        self.det_model = RetinaFace(model_file = os.path.join(weight_root, "det_10g.onnx"))
    
    def prepare(self, ctx_id, det_thresh = 0.5, det_size = (640, 640)):
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.det_model.prepare(ctx_id, input_size = det_size, det_thresh = det_thresh)
        for _, model in self.models.items():
            model.prepare(ctx_id)
        
    def get(self, img, max_num = 0):
        bboxes, keypoints = self.det_model.detect(img, 
                                                  max_num = max_num, 
                                                  metric = "default")
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx, : 4]
            det_score = bboxes[idx, 4]
            keypoint = None
            if keypoints is not None:
                keypoint = keypoints[idx]
            face = Face(bbox = bbox, kps = keypoint, det_score = det_score)
            for _, model in self.models.items():
                model.get(img, face)
            ret.append(face)
        return ret
