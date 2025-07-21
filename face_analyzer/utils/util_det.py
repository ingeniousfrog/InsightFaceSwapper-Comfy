import cv2
import numpy as np

from .utils import _preprocess, decode, shift_coord


def det_face(model, img, score_thresh = 0.995):
    imh, imw = img.shape[: 2]
    image, scale = _preprocess(img)
    output = model.run(None, {'image': image})
    bboxes, scores, landmarks = decode(output[0], scale)

    # find the max box size
    max_width = 0
    max_height = 0
    for idx in range(bboxes.shape[0]):
        if scores[idx] >= score_thresh:
            bbox = bboxes[idx, :]
            max_width = max(max_width, bbox[2] - bbox[0])
            max_height = max(max_height, bbox[3] - bbox[1])
    
    if max_width / imw >= 0.3 or max_height / imh >= 0.3:
        image_ = cv2.copyMakeBorder(
            img, 
            top = 0, 
            bottom = imh, 
            left = 0, 
            right = imw,
            borderType = cv2.BORDER_CONSTANT, 
            value = [104, 117, 123]
        )
        image, scale = _preprocess(image_)
        output = model.run(None, {'image': image})
        bboxes, scores, landmarks = decode(output[0], scale)
    return bboxes, scores, landmarks


def get_landmarks(bbox, img, model):
    imh, imw = img.shape[: 2]
    x0, y0, x1, y1 = shift_coord(bbox, imh, imw)
    crop_height = y1 - y0 + 1
    crop_width = x1 - x0 + 1
    scale_x = 60 / crop_width
    scale_y = 60 / crop_height
    rois = cv2.cvtColor(cv2.resize(img[y0 : y1, x0 : x1, :], (60, 60)), cv2.COLOR_BGR2GRAY)
    rois_input = ((rois.astype(np.float32) - 127.5) / 128)[None, None, :, :]
    pts_ = model.run(None, {'data_input': rois_input})
    pts = []
    for i in range(len(pts_)):
        pts.append(pts_[i][0].reshape(-1, 2))
    pts = np.concatenate(pts)
    pts[:, 0] = pts[:, 0] / scale_x + x0
    pts[:, 1] = pts[:, 1] / scale_y + y0
    return pts


def preprocess(img):
    # resize to (256, 256, 3)
    image = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
    # bgr to rgb
    image = image[:, :, : : -1]
    # normalization
    mean = np.float64(np.array([123.675, 116.28, 103.53]).reshape(1, -1))
    stdinv = 1 / np.float64(np.array([58.395, 57.12, 57.375]).reshape(1, -1))
    image = (image - mean) * stdinv
    return image.transpose((2, 0, 1))[None, :, :, :].astype(np.float32)


def postprocess(model_out, img):
    # vis_parse = cv2.resize(model_out[0].copy().astype(np.uint8), 
    #                        None,
    #                        fx = 1, 
    #                        fy = 1,
    #                        interpolation = cv2.INTER_NEAREST)
    vis_parse = cv2.resize(model_out[0].copy().astype(np.uint8), (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
    return vis_parse


def parse(image, parser):
    input_image = preprocess(image)
    outputs = parser.run(None, {'image': input_image})
    mask = postprocess(outputs, image)
    return mask

def get_rect(box, image, increase_area = 0.3):
    imh, imw = image.shape[: 2]
    box_size = box[2 :] - box[: 2]
    width_increase = max(increase_area, ((1 + 2 * increase_area) * box_size[1] - box_size[0]) / (2 * box_size[0]))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * box_size[0] - box_size[1]) / (2 * box_size[1]))
    
    hx1 = max(0, box[0] - width_increase * box_size[0] * 0.8)
    hy1 = max(0, box[1] - height_increase * box_size[1])
    hx2 = max(min(imw, box[0] + box_size[0] + width_increase * box_size[0] * 0.8), 0)
    hy2 = max(min(imh, box[1] + box_size[1] + height_increase * box_size[1] * 0.8), 0)
    return np.int_(np.array([hx1, hy1, hx2, hy2]))
