import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
import re
import math
import json
import yaml
from pathlib import Path

from SDY_File import SDY_File

import torchvision.io
import torch

from scantensus_echo.Scantensus.utils.json import get_keypoint_names_and_colors_from_json

from scantensus_echo.ScantensusPT.utils import load_and_fix_state_dict
from scantensus_echo.ScantensusPT.utils.heatmap_to_label import heatmap_to_label
from scantensus_echo.ScantensusPT.utils.heatmaps import gaussian_blur2d_norm
from scantensus_echo.ScantensusPT.utils.image import image_logit_overlay_alpha
from scantensus_echo.ScantensusPT.utils.draw import draw_predictions
from scantensus_echo.Scantensus.utils.labels import labels_upsample, convert_labels_to_csv, convert_labels_to_firebase, labels_convert_str_to_list, labels_convert_list_to_str, labels_calc_len, label_shift

################
NUM_CUDA_DEVICES = torch.cuda.device_count()
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', None)
################

####################
DEBUG = False
RUN = "flow-171"
###################


#### Details ####
PROJECT = 'flow'
EXPERIMENT = 'flow-004'


#### Host ####
HOST = 'laptop-samay'


#### Host Config ####
USE_CUDA = False
DISTRIBUTED_BACKEND = 'gloo'
DDP_PORT = '42424'

if USE_CUDA:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'



#### image ####
IMAGE_CROP_SIZE = [768, 768]
IMAGE_OUT_SIZE = [512, 512]


### heatmaps ###
DOT_SD = 2
CURVE_SD = 4

DOT_WEIGHT_SD = 10
CURVE_WEIGHT_SD = 20

DOT_WEIGHT = 80
CURVE_WEIGHT = 20

SUBPIXEL = True

#### model ####
MODEL = 'HRNetV2M9'
MODEL_CLASS = 'HigherHRNet'
MODEL_CONFIG_NAME = 'HigherHRNet32'

with open(Path.cwd() / 'config' / 'model' / 'HigherHRNet32-blurpool-bw-prepost.yaml') as yaml_f:
    MODEL_CONFIG = yaml.full_load(yaml_f)

MODEL_CONFIG_DICT = MODEL_CONFIG['model_extra_config']
PRE_POST = False

if MODEL is None:
    raise Exception
elif MODEL == "HRNetV2M7":
    from ScantensusPT.nets.HRNetV2M7 import get_seg_model
elif MODEL == "HRNetV2M8":
    from ScantensusPT.nets.HRNetV2M8 import get_seg_model
elif MODEL == "HRNetV2M9":
    from ScantensusPT.nets.HRNetV2M8 import get_seg_model
elif MODEL == "HRNetV2M10":
    from ScantensusPT.nets.HRNetV2M10 import get_seg_model
else:
    raise Exception


#### inference ####
EPOCH = 400
SINGLE_INFER_BATCH_SIZE = 8

#### output folders ####

CHECKPOINT_DIR = Path.cwd() / 'Model'

CHECKPOINT_KEYS_PATH = CHECKPOINT_DIR / "keys.json"
CHECKPOINT_PATH = CHECKPOINT_DIR / f'weights-{EPOCH}.pt'

#### process #####
LABELS_TO_PROCESS = [
    'curve-flow'
]
LABELS_TO_PROCESS_CURVE_POINTS = [
    55
]

OUT_IMAGE_DIR = Path(r'C:\Users\samay\out')
#########

def main():

    sdy_fl = Path("/home/matthew/CMStudy_2016_12_16_064019.sdy")
    sdy_fl = Path(r"C:\Users\samay\Dropbox\Samay Mehta\BSC 2022\Contrast Flow Samay\Data\Patient Con5\CMStudy_2021_09_22_120258.sdy") #"C:\Users\samay\Dropbox\BSC 2022\Contrast Flow Samay\Data\Patient Con5\CMStudy_2021_09_22_120258.sdy"
    file = SDY_File(sdy_fl=sdy_fl)
    pixel_array = file.spectrum
    pixel_array = torch.from_numpy(pixel_array.astype("float32"))


    pixel_array = torch.sqrt(pixel_array * 2) * 1.5
    pixel_array = torch.clip(pixel_array, 0, 255)
    pixel_array = torch.flip(pixel_array, [0])
    pixel_array = pixel_array.squeeze(0).squeeze(0).repeat([1,3,1,1])


    keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(CHECKPOINT_KEYS_PATH)

    keypoint_sd = [CURVE_SD if 'curve' in keypoint_name else DOT_SD for keypoint_name in keypoint_names]
    keypoint_sd = torch.tensor(keypoint_sd, dtype=torch.float, device=DEVICE)
    keypoint_sd = keypoint_sd.unsqueeze(1).expand(-1, 2)

    net_cfg = {}
    net_cfg['MODEL'] = {}
    net_cfg['MODEL']['PRETRAINED'] = False
    net_cfg['MODEL']['EXTRA'] = MODEL_CONFIG_DICT
    net_cfg['DATASET'] = {}
    net_cfg['DATASET']['NUM_CLASSES'] = len(keypoint_names)

    if PRE_POST:
        net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 3 * 3
    else:
        net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 1 * 3


    single_model = get_seg_model(cfg=net_cfg)

    single_model.init_weights()
    state_dict = load_and_fix_state_dict(CHECKPOINT_PATH, device=DEVICE)
    single_model.load_state_dict(state_dict)

    print(f"Model Loading onto: {DEVICE}")

    model = single_model.to(DEVICE)

    model.eval()

    out_ys = torch.zeros(pixel_array.shape[-1], dtype=torch.float32, device="cpu")

    def get_shards(shard_width, shard_overlap, total_width):
        source_list = []
        destination_list = []

        source_start = 0
        source_end = shard_width

        destination_start = 0
        destination_end = shard_width - shard_overlap

        while destination_end <= total_width:
            source = (source_start, source_end)
            destination = (destination_start, destination_end)

            source_list.append(source)
            destination_list.append(destination)
            #can change 2*shard_overlap to 1*shard_overlap for example
            source_start = source_start + (shard_width - 2*shard_overlap)
            source_end = source_start + shard_width

            destination_end = source_end - shard_overlap
            destination_start = destination_end

            if source_end > total_width:
                source_end = total_width
                source_start = total_width - shard_width
                destination_end = total_width
                destination_start = total_width - shard_width + shard_overlap

                source = (source_start, source_end)
                destination = (destination_start, destination_end)

                source_list.append(source)
                destination_list.append(destination)
                break

        return source_list, destination_list

    source_list, destination_list = get_shards(1024, 256, pixel_array.shape[-1])


    source_list = [
       (0, 1024),(1024, 2048)]

    #destination_list = source_list



    for i, (source, destination) in enumerate(zip(source_list, destination_list)):
        try:
            print(i)
            source_start, source_end = source
            destination_start, destination_end = destination

            image_t = pixel_array[..., source_start:source_end]
            image_t = image_t.to(device=DEVICE, dtype=torch.float32, non_blocking=True).div(255.0).add(-0.5)

            with torch.no_grad():
                y_pred_25_clean, y_pred_50_clean = model(image_t)

                y_pred_25 = torch.nn.functional.interpolate(y_pred_25_clean, scale_factor=4, mode='bilinear',
                                                            align_corners=True)
                y_pred_50 = torch.nn.functional.interpolate(y_pred_50_clean, scale_factor=2, mode='bilinear',
                                                            align_corners=True)

                y_pred = (y_pred_25 + y_pred_50) / 2.0
                # y_pred = gaussian_blur2d_norm(y_pred=y_pred, kernel_size=(25, 25), sigma=keypoint_sd)
                y_pred = torch.clamp(y_pred, 0, 1)

                del y_pred_25, y_pred_50

            if PRE_POST:
                image_t = image_t[:, 3:6, :, :]

            ###
            print("hello")

            out_path = OUT_IMAGE_DIR / "test" / "raw-image" / f"{i}.png"
            print(f"Saving raw-image: {out_path}")
            os.makedirs(Path(out_path).parent, exist_ok=True)
            # input=pixel_array[0, :, :, 1024:4000].to(torch.uint8)
            torchvision.io.write_png(filename=str(out_path),
                                     input=pixel_array[0, :, :, source_start:source_end].to(torch.uint8),
                                     compression_level=7)


            y_pred_raw = image_logit_overlay_alpha(logits=y_pred, images=None, cols=keypoint_cols)
            y_pred_raw = y_pred_raw.mul_(255).type(torch.uint8).cpu()

            out_path = OUT_IMAGE_DIR / "test" / "raw" / f"{i}.png"
            print(f"Saving raw: {out_path}")
            os.makedirs(Path(out_path).parent, exist_ok=True)
            torchvision.io.write_png(filename=str(out_path), input=y_pred_raw[0, ...], compression_level=7)

            del y_pred_raw

            ###
            # 3
            y_pred_mix = image_logit_overlay_alpha(logits=y_pred, images=image_t.add(0.5), cols=keypoint_cols)
            y_pred_mix = y_pred_mix.mul_(255).type(torch.uint8).cpu()

            out_path = OUT_IMAGE_DIR / "test" / "mix" / f"{i}.png"
            print(f"Saving mix: {out_path}")
            os.makedirs(Path(out_path).parent, exist_ok=True)
            torchvision.io.write_png(filename=str(out_path), input=y_pred_mix[0, ...], compression_level=7)

            del y_pred_mix

            ###
            out_labels_dict = {}
            out_labels_dict["SDY"] = {}
            out_labels_dict["SDY"]['labels'] = {}

            label = 'curve-flow'

            ys, xs, confs = heatmap_to_label(y_pred=y_pred[0, ...],
                                             keypoint_names=keypoint_names,
                                             label=label)

            print(ys)
            print(xs)
            print("hello")

            # (ys-file.flow_baseline[source_start:source_end])*/file.flow_scale*X


            df = pd.DataFrame(ys)
            filepath = f'C:/Users/samay/Downloads/fileflowscale_{source_start}_{len(xs)}.xlsx'
            df.to_csv(filepath, index=False)


        except Exception as e:
            print(e)


    print("done")


if __name__ == '__main__':
    main()
