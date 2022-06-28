
## SELECTING 10 images from 21 patients (folders sdy_valid_001 to sdy_valid_021 included)

import os
from os import path
import random
import shutil
from pathlib import Path

data = []
root = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/rank_valid/32/sdy_valid/'
last = 'study_00'
sdyfolders = []
target_folder = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/selected_valid_sdys'

##create a list of all paths to subfolders sdyvalid001 to 21

for i in range(1,10):
    n = i
    foldername = 'sdy_valid_00' + str(n)
    folder_path = os.path.join(root,foldername,last)
    sdyfolders.append(folder_path)

for i in range(10,22):
    n = i
    foldername = 'sdy_valid_0' + str(n)
    folder_path = os.path.join(root,foldername,last)
    sdyfolders.append(folder_path)

import os
from os import path
import random

##randomly select 10 images from each of those 21 subfolders and copy to a new folder
for subfolder in sdyfolders:
    random.seed(4324)
    if len(os.listdir(subfolder)) > 10:
        random_file = random.sample(os.listdir(subfolder),10)
        for file in random_file:
            path_random_file = os.path.join(subfolder,file)
            target_path = os.path.join(target_folder,file)
            shutil.copy(path_random_file, target_path)

## anissa manually filtered through the 210 images to exclude any with a significant discontinuity

# 10 images were excluded of the ids below (3 b/c of discontinuities, and last 7 to get to exactly 200):
# '32-f894da76e89ed31036c15966f9766e4295e0364d2dc20664e62d2b0510686187-0074.png',
#  '32-f894da76e89ed31036c15966f9766e4295e0364d2dc20664e62d2b0510686187-0203.png',
#  '32-b2ee6b1bc659bbd6953491dab1ebebd3cc8bec7e6ddccb3cd639072b02d7e354-0221.png',
#  '32-f894da76e89ed31036c15966f9766e4295e0364d2dc20664e62d2b0510686187-0163.png',
#  '32-88dbc6797da7900b512fb2c45f3042b070103f1b11de099065ff033f393ffeb5-0358.png',
#  '32-f894da76e89ed31036c15966f9766e4295e0364d2dc20664e62d2b0510686187-0219.png',
#  '32-edd69ad6fd8ef0f6f44d1eaa6becf05cdd5bebb1b57a43c1d26fb93a9b8872d5-0154.png',
#  '32-f894da76e89ed31036c15966f9766e4295e0364d2dc20664e62d2b0510686187-0186.png',
#  '32-f894da76e89ed31036c15966f9766e4295e0364d2dc20664e62d2b0510686187-0179.png',
#  '32-f894da76e89ed31036c15966f9766e4295e0364d2dc20664e62d2b0510686187-0178.png'