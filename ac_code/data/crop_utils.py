"""
裁剪工具：把任务 A 的 mask 转成结节外接框，再从原图（letterbox 后）裁出结节 patch，
送给任务 C 分类器。train/evaluate/infer 共用，保证坐标空间一致（均在 img_size 空间内）。

流程：
    mask(0/1) -> mask_to_bbox -> (top,left,bottom,right) -> crop_patch(img, bbox) -> patch
"""
import numpy as np
from PIL import Image


def mask_to_bbox(mask_bin, margin=0.1, min_size=16):
    """mask_bin: HxW 二值(0/1 or bool)。返回 (top,left,bottom,right)（前闭后开），含 margin 扩展。
    无前景时返回 None（调用方据此决定降级处理）。"""
    m = (mask_bin > 0)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = m.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    ex = max(int(round(bw * margin)), 1)
    ey = max(int(round(bh * margin)), 1)
    left = max(0, x0 - ex)
    right = min(w, x1 + ex + 1)
    top = max(0, y0 - ey)
    bottom = min(h, y1 + ey + 1)
    if right - left < min_size:
        cx = (left + right) // 2
        left = max(0, cx - min_size // 2)
        right = min(w, left + min_size)
    if bottom - top < min_size:
        cy = (top + bottom) // 2
        top = max(0, cy - min_size // 2)
        bottom = min(h, top + min_size)
    return (top, left, bottom, right)


def crop_patch(img, bbox, patch_size=224):
    """img: HxWx3 uint8（letterbox 后的原始图）；bbox=(top,left,bottom,right)。
    裁剪并 resize 到 patch_size×patch_size，返回 HxWx3 uint8。"""
    top, left, bottom, right = bbox
    patch = img[top:bottom, left:right, :]
    if patch.size == 0:
        patch = img
    patch = np.array(Image.fromarray(patch).resize((patch_size, patch_size), Image.BILINEAR))
    return patch
