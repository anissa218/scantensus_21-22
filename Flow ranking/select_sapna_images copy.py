
import os
from os import path
import random
import shutil
from pathlib import Path

import pandas as pd

df = pd.read_excel('/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/Image IDs from tracking validation project.xlsx') # can also index sheet by name or fetch all sheets
image_list = df['filenames'].tolist()

root = '/Volumes/png.database.scantensus.icch.uk/data/32/sdy_valid/'
data = []

target_folder = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/selected_sapna_sdys'


for x,sdyfolder,y in os.walk(root):
    for subfolder in sdyfolder:
        for filename in os.listdir(os.path.join(root,subfolder,'study_00')):
            exist = image_list.count(filename)

            if exist > 0:
                path_to_file = os.path.join(root, subfolder,'study_00', filename)
                target_path = os.path.join(target_folder,filename)
                shutil.copy(path_to_file, target_path)

