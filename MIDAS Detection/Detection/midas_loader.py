import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiDaS'))
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet



def load_midas_model(model_path, device):
    model = DPTDepthModel(
    path=model_path,
    backbone="vitb_rn50_384",  # DPT_Hybrid backbone
    non_negative=True, # No negative depth predictions
)

    model.eval() # Not training mode
    model.to(device) # Put model on CPU/GPU
    return model

def midas_transform(frame):
    transform = transforms.Compose([
        Resize(
            384, 384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    # ✅ Geef een dict mee zoals MiDaS het verwacht
    return transform({"image": frame})["image"]


def midas_predict(frame, model, device):
    img_input = midas_transform(frame)
    sample = torch.from_numpy(img_input).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model.forward(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

def normalize_depth(depth, bits=2):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2 ** (8 * bits)) - 1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)
    return out.astype("uint8") if bits == 1 else out.astype("uint16")
