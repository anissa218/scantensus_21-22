## get relevant ids
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import random

import os

from pathlib import Path
import json

## Load label data from json file

#labels_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking/scantensus-imp-coro-seligman-flow-quality-rank-export.json")
#labels_path = Path("/home/anissa/scantensus-ranking/scantensus-imp-coro-seligman-flow-quality-rank-export nov13.json")
#labels_path = Path("/home/anissa/scantensus-ranking/scantensus-imp-coro-seligman-flow-quality-rank-export nov16.json")
#labels_path = Path("/home/anissa/scantensus-ranking/scantensus-imp-coro-seligman-flow-quality-rank-export.json")
#labels_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking/scantensus-imp-coro-seligman-flow-ranking-dev-final-export.json")
labels_path = Path("/scantensus-ranking/ratings_all_users.json")
with open(labels_path, 'r') as json_f:
    rankf = json.load(json_f)

ids = []

data = rankf[0]
for item in data:
    id = item['id']
    ids.append(id)


import pandas as pd

idranking = pd.DataFrame()
idranking['id'] = ids

## file id n1: 01-6de8f0c4f1c4e8632b6e50077538d27a4efce57f1d48e03e690a84e2e98fda29-0080

import logging
import json
import os
import sys
import time
import requests

import multiprocessing
from functools import partial

CONFIG = "laptop"


# This ic address is the best. The computer running it must be on Imperial College Network
# MAGIQUANT_ADDRESS = "http://icch3.ic.shun-shin.com:50601"

# This WG address is second best - must be on Wire Guard network
# MAGIQUANT_ADDRESS = "http://icch2.wg.shun-shin.com:50601"

# This is the worst one to use - use this if not on vpn
MAGIQUANT_ADDRESS = "https://files.magiquant.com"

# This is always the same
# FIREBASE_ADDRESS = "https://storage.googleapis.com/scantensus/fiducial"

if CONFIG == "laptop":
    NUM_PROCESSES = 4
    SOURCE_TYPE = "txt"
    ALL_FRAMES_OVERIDE = False
    # LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/labels/unity/labels-all.txt"
    OUTPUT_ROOT_DIR = "/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking"

elif CONFIG == "thready3":
    NUM_PROCESSES = 48
    SOURCE_TYPE = "txt"
    ALL_FRAMES_OVERIDE = True
    LABELS_PATH = "/zfs-thready3/scantensus-data/labels/unity/labels-all.txt"
    OUTPUT_ROOT_DIR = "/zfs-thready3/scantensus-data/png-cache/unity/"

else:
    print(f"No Source Set")
    sys.exit()


def trim_hash_code(file: str):
    if file.startswith("01-") or file.startswith("02-") or file.startswith("32-"):
        if len(file) == 67:
            hash = file

        elif len(file) == 76:
            hash = file[:-9]

        elif len(file) == 72:
            hash = file[:-5]

    else:
        hash = file

    return hash

def download_hash(file: str, output_root_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f"{file} Starting")

    if file.startswith("01-") or file.startswith("02-") or file.startswith("32-"):
        FILE_SOURCE = "magiquant"
        if len(file) == 67:
            hash = file
            ALL_FRAMES = True
            print(f"{file} All Frames")
        else:
            if len(file) == 76:
                hash = file[:-9]
                ALL_FRAMES = False
                frame_num = int(file[-8:-4])
            elif len(file) == 72:
                hash = file[:-5]
                ALL_FRAMES = False
                frame_num = int(file[-4:])
            else:
                return
            #frame_nums = [frame_num-5, frame_num, frame_num+5]
            frame_nums = [frame_num]

    else:
        FILE_SOURCE = "clusters"
        hash = file
        ALL_FRAMES = False
        frame_nums = [0]

    if ALL_FRAMES:
        frame_nums = range(1000)

    failed_count = 0

    for frame_num in frame_nums:
        if FILE_SOURCE == 'magiquant':
            if failed_count > 5:
                break

            sub_a = hash[:2]
            sub_b = hash[3:5]
            sub_c = hash[5:7]

            file_name = f"{hash}-{frame_num:04}.png"
            location = f"{MAGIQUANT_ADDRESS}/scantensus-database-png-flat/{sub_a}/{sub_b}/{sub_c}/{file_name}"
            output_dir = os.path.join(output_root_dir, sub_a, sub_b, sub_c)
            output_path = os.path.join(output_dir, file_name)
        # elif FILE_SOURCE == 'clusters':
        #     file_name = file
        #     location = f"{FIREBASE_ADDRESS}/{file_name}"
        #     output_dir = os.path.join(output_root_dir)
        #     output_path = os.path.join(output_dir, file_name)

        os.makedirs(output_dir, exist_ok=True)

        logger.warning(f"{location} Downloading")

        try:
            response = requests.get(location)

            if response.status_code == 200:
                with open(output_path, 'wb') as outfile:
                    outfile.write(response.content)
            else:
                failed_count = failed_count + 1
                logger.warning(f"{location} Fail {response.status_code}")
        except Exception as e:
            logger.warning(f"{location} Fail {str(e)}")
            failed_count = failed_count + 1

file = '01-6de8f0c4f1c4e8632b6e50077538d27a4efce57f1d48e03e690a84e2e98fda29-0080' #file n1


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    hashes = idranking['id'].to_list()

    # if SOURCE_TYPE == "json":
    #     with open(LABELS_PATH, 'r') as json_f:
    #         db_raw = json.load(json_f)
    #
    #     hashes = list(db_raw.keys())
    #
    # if SOURCE_TYPE == "txt":
    #     hashes = []
    #     with open(LABELS_PATH, 'r') as labels_f:
    #         for line in labels_f:
    #             hashes.append(line[:-1])
    #
    # if ALL_FRAMES_OVERIDE:
    #     hashes = list(set([trim_hash_code(x) for x in hashes]))

    TOTAL_FILES = len(hashes)

    START_TIME = time.time()

    pool = multiprocessing.Pool(NUM_PROCESSES)

    results = pool.imap_unordered(partial(download_hash, output_root_dir=OUTPUT_ROOT_DIR), hashes)

    for idx, res in enumerate(results):
        elapsed_time = time.time() - START_TIME
        total_time = (elapsed_time / (idx + 1)) * TOTAL_FILES
        remain_time = total_time - elapsed_time
        print(f"Done: {idx}/{TOTAL_FILES} Elapsed:{round(elapsed_time / 60)} min Remain:{round(remain_time / 60)} mins")

    #pool.map(partial(download_hash, output_root_dir=OUTPUT_ROOT_DIR), hashes)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()

