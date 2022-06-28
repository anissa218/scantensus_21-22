from SDY_file_processing import SDY_File
import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
import imageio
import io
import timm
from torch import nn
import requests

# import raw flow spectrum

def sdy_to_pixels(sdy_path):
    sdy_fl = Path(sdy_path)
    file = SDY_File(sdy_fl=sdy_fl)
    pixel_array = file.spectrum
    pixel_array = torch.from_numpy(pixel_array.astype("float32"))

    pixel_array = torch.sqrt(pixel_array * 2) * 1.5

    pixel_array = torch.clip(pixel_array, 0, 255) #1 instead of 255
    pixel_array = torch.flip(pixel_array, [0])
    pixel_array = pixel_array.squeeze(0).squeeze(0).repeat([1 ,3 ,1 ,1])

    pixel_array = pixel_array.div(255.0) # removed the add(-0.5)

    width = pixel_array.shape[3]

    pixel_array = torch.nn.functional.interpolate(pixel_array, (512,width)) # resize so image height is 512

    return pixel_array

# To do inference from png instead of sdy

class FlowImage():
    def __init__(self,image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = imageio.imread(image_path)
        max = image.max()
        image = 255 * (image / max) #image_t = image_t / 255.0
        image = image.astype(np.uint8)
        image_width = image.shape[1]
        center_image = int(image_width / 2)
        image = image[:, center_image - 128: center_image + 128]

        convert_tensor = transforms.ToTensor()
        image = convert_tensor(image)

        return image

# Inference class

# need to upload weights to server

WEIGHTS_LOCATION = '/Users/anissaalloula/Desktop/final_flow_quality_model.pt'

class FlowInfer:
    def __init__(self, device='cpu'):
        self.device = device

        model = timm.create_model('resnet34', num_classes=1)
        model.default_cfg['input_size'] = (1, 512, 256)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.init_weights()
        model.load_state_dict(torch.load(WEIGHTS_LOCATION, map_location="cpu"))
        #model.to(device)
        model.eval()
        self.model = model

    def infer(self, img) -> list:
        with torch.no_grad():

            result = self.model(img) # might be image_t[None, ...]

            result = result.cpu().flatten().tolist()

        return result

# Tests

# test inference on images

img_list = ['/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking/32/0d/fa/32-0dfa5c633ddfba53b77d5c6bbd4d8a27e2c0781dca047bb3413145cd1393bb17-0008.png',
            '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking/32/0d/fa/32-0dfa5c633ddfba53b77d5c6bbd4d8a27e2c0781dca047bb3413145cd1393bb17-0015.png']

flownn = FlowInfer()
images = FlowImage(img_list)

preds = []
for i in range(len(img_list)):
    pred = flownn.infer(images[i][None, ...])
    preds.append(pred)

# test inference on a section of sdys

#a = sdy_to_pixels('/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/CMStudy_2021_09_22_120258.sdy')
sdy_pixel_array = sdy_to_pixels('/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/SDY processing/CMStudy_2019_12_04_095301.sdy')
start_col= 10000
sdy_pred = flownn.infer(sdy_pixel_array[0:1,0:1,:,start_col:start_col + 256])

# inference on a whole sdy file

def sdy_inference(pixel_array,overlap):
    array_width = pixel_array.shape[3]
    start = 0
    end = start+256
    quality_preds = []
    while end < array_width:
        pred = flownn.infer(pixel_array[0:1,0:1,:,start:end])
        quality_preds.append(pred)
        start = start + 256 - overlap
        end = start + 256

    return quality_preds

infer_preds = sdy_inference(sdy_pixel_array,0)
