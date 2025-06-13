import cv2
import torch
import scipy
import numpy as np

def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        if padding > 0:
            img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        if padding > 0:
            img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        if pad_y * 2 < img.shape[0] and pad_x * 2 < img.shape[1]:
            img = img[pad_y:img.shape[0]-pad_y, pad_x:img.shape[1]-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)
    return img

def process_video_path(path):
    frames_to_take = 32
    frame_stride = 2
    video_size = 224

    mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
    std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

    mat_data = scipy.io.loadmat(path)
    volume = np.array(mat_data['cropped'])
    volume = crop_and_scale(volume)
    volume = np.repeat(volume[..., None], 3, axis=3)
    volume = np.transpose(volume, (3, 2, 0, 1))
    x = torch.as_tensor(volume, dtype=torch.float)
    x.sub_(mean).div_(std)

    if x.shape[1] < frames_to_take:
        padding = torch.zeros((3, frames_to_take - x.shape[1], video_size, video_size), dtype=torch.float)
        x = torch.cat((x, padding), dim=1)

    return x[:, :frames_to_take:frame_stride, :, :]