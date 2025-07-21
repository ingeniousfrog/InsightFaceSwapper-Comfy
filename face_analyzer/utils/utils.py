import math
from itertools import product

import cv2
import numpy as np
import torch
import torchvision
from skimage import transform as trans


def _preprocess(img, mean = [104, 117, 123], std = 1, to_rgb = False):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = np.float32(img)
    img_shape = image.shape
    img_maxs = np.max(img_shape[: 2])
    scale = 640 / img_maxs
    w, h = int(scale * img_shape[1]), int(scale * img_shape[0])
    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_LINEAR)
    horizontal = (640 - w)
    vertical = (640 - h)
    image = cv2.copyMakeBorder(
        image,
        top = 0,
        bottom = vertical,
        left = 0,
        right = horizontal,
        borderType = cv2.BORDER_CONSTANT,
        value = mean
    )
   
    mean = np.float64(np.array(mean).reshape(1, -1))
    image = (image - mean) / std
    image_input = image.transpose((2, 0, 1))[None, :, :, :].astype(np.float32)
    return image_input, scale

def gen_priors(steps, img_size, min_sizes):
    feature_maps = [
        [
            math.ceil(img_size[0] / step),
            math.ceil(img_size[1] / step)
        ]
        for step in steps
    ]
    anchors = []
    for index, feat_map in enumerate(feature_maps):
        min_sizes_ = min_sizes[index]
        for i, j in product(range(feat_map[0]), range(feat_map[1])):
            for min_size in min_sizes_:
                sx = min_size / img_size[1]
                sy = min_size / img_size[0]
                dense_cx = [x * steps[index] / img_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[index] / img_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, sx, sy]
        
    priors = np.array(anchors).reshape(-1, 4)
    return priors


def decode(out,
           scale, 
           steps = [8, 16, 32], 
           img_size = [640, 640],
           min_sizes = [[10, 20], [32, 64], [128, 256]],
           variances = [0.1, 0.2],
           conf_thresh = 0.02,
           nms_thresh = 0.4,
           topK = 5000,
           keep_topK = 750):
    priors = gen_priors(steps, img_size, min_sizes)

    loc = out[0, :, : 4]
    bboxes = np.hstack((
        priors[:, : 2] + loc[:, : 2] * variances[0] * priors[:, 2 :],
        priors[:, 2 :] * np.exp(loc[:, 2 :] * variances[1])
    ))
    bboxes[:, : 2] -= bboxes[:, 2 :] / 2
    bboxes[:, 2 :] += bboxes[:, : 2]
    bboxes = bboxes.astype(np.float32)

    bboxes *= np.array([640, 640, 640, 640]) / scale
    scores = out[0, :, 4]
    lands = out[0, :, 5 :].reshape(-1, 5, 2)
    landmarks = priors[:, np.newaxis, : 2] + lands * variances[0] * priors[:, np.newaxis, 2 :]
    landmarks = landmarks.reshape(-1, 10)
    landmarks *= np.repeat(np.array([640, 640]), repeats = 5) / scale

    # ignore low scores
    inds = np.where(scores > conf_thresh)[0]
    bboxes = bboxes[inds]
    scores = scores[inds]
    landmarks = landmarks[inds]

    # keep top-K before NMS
    order = scores.argsort()[: : -1][: topK]
    bboxes = bboxes[order]
    scores = scores[order]
    landmarks = landmarks[order]

    # non-maximum suppression with torchvision
    keep = torchvision.ops.nms(torch.from_numpy(bboxes), torch.from_numpy(scores), nms_thresh)
    bboxes = bboxes[keep]
    scores = scores[keep]
    landmarks = landmarks[keep]

    # keep top-K after NMS
    if len(bboxes.shape) == 1:
        bboxes = bboxes[None, :]
    bboxes = bboxes[: keep_topK, :]
    if len(landmarks.shape) == 1:
        landmarks = landmarks[None, :]
    landmarks = landmarks[: keep_topK, :]
    if isinstance(scores, np.float32):
        scores = [scores]
    scores = scores[: keep_topK]
    return bboxes, scores, landmarks


def shift_coord(shape, image_height, image_width, enlarge_ratio = 0.0):
    dest = shape.copy()
    x0, y0, x1, y1 = shape
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    ew = w * enlarge_ratio
    eh = h * enlarge_ratio
    x0 -= ew / 2.
    y0 -= eh / 2.
    x1 += ew / 2.
    y1 += eh / 2.
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    if w > h:
        offset = (w - h) / 2.
        dest[:] = [x0, y0 - offset, x1, y1 + offset]
    else:
        offset = (h - w) / 2.
        dest[:] = [x0 - offset, y0, x1 + offset, y1]
    dest = map(int, dest)
    x0, y0, x1, y1 = dest
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 >= image_width:
        x1 = image_width - 1
    if y1 >= image_height:
        y1 = image_height - 1
    return [x0, y0, x1, y1]

def face_align(image, img_size, landmarks):
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]], dtype=np.float32)
    if img_size[0] == 96:
        src[:, 0] -= 8.0
        src[:, 1] -= 8.0
    dst = np.array(landmarks).astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[: 2, :]
    warped = cv2.warpAffine(image, M, (img_size[1], img_size[0]), borderValue = 0.0)
    return warped