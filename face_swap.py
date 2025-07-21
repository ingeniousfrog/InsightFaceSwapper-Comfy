import os
import folder_paths
import numpy as np
import torch
from PIL import Image

from .face_analyzer.analyzer import FaceAnalyzer
from .face_analyzer.module.inswapper import INSwapper

class FaceSwapper:

    def __init__(self, ):
        super(FaceSwapper, self).__init__()
        self.app = FaceAnalyzer(weight_root = os.path.join(folder_paths.models_dir, "insightface"))
        self.app.prepare(ctx_id = 0, det_size = (640, 640))
        self.swapper = INSwapper(model_file = os.path.join(folder_paths.models_dir, "insightface", "inswapper_128.onnx"))

    def swap(self, src, tgt):
        src_face = self.app.get(src)[0]
        faces = self.app.get(tgt)
        faces = sorted(faces, key = lambda x : x.bbox[0])
        res = tgt.copy()
        res = self.swapper.get(
            res, faces[0], src_face, paste_back = True
        )
        return res

class InsightFaceSwapper:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("IMAGE", ),
                "target": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "swap_face"
    CATEGORY = "loader"

    def swap_face(self, source, target):
        source_img = Image.fromarray(np.clip(255. * source.squeeze(0).cpu().numpy(), 0, 255).astype(np.uint8))
        source_img = source_img.convert("RGB")
        src_img = np.array(source_img)[:, :, : : -1]
        target_img = Image.fromarray(np.clip(255. * target.squeeze(0).cpu().numpy(), 0, 255).astype(np.uint8))
        target_img = target_img.convert("RGB")
        tgt_img = np.array(target_img)[:, :, : : -1]
        model = FaceSwapper()
        image = model.swap(src_img, tgt_img)
        image = Image.fromarray(image[:, :, : : -1])
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image, )
    
NODE_CLASS_MAPPINGS = {
    "InsightFaceSwapper": InsightFaceSwapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsightFaceSwapper": "Load InsightFaceSwapper",
}
