
## SELECTING 30 OCTs from 100 patients = 3000 total (folders oct20001 to oct20100 included)
## OCT folders were in data/03/oct_group_2

import os
from os import path
import random
import shutil
from pathlib import Path
#
#
# ##create a list of all paths to subfolders oct20001 to oct20100
#
# data = []
# #root = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/oct/'
# root = '/Volumes/png.database.scantensus.icch.uk/data/03/oct_group_2'
# last = 'study_00/series_00/US000000'
# octfolders = []
#
# for i in range(1,101):
#     n = 20000 + i
#     foldername = 'oct' + str(n)
#     folder_path = os.path.join(root,foldername,last)
#     octfolders.append(folder_path)
#
# target_folder = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/selected_oct'
#
#
# ##randomly select 30 images from each of those 100 subfolders and copy to a new folder
# for subfolder in octfolders:
#     random.seed(4324)
#     if len(os.listdir(subfolder)) > 29:
#         random_file = random.sample(os.listdir(subfolder),30)
#         for file in random_file:
#             path_random_file = os.path.join(subfolder,file)
#             target_path = os.path.join(target_folder,file)
#             shutil.copy(path_random_file, target_path)

import csv
with open('/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/unity_oct_list.csv',) as f:
    reader = csv.reader(f)
    unity_list = list(reader)

#correct first bit of unity_list

unity_list[0][0]='03-0253e04fff5c0f6fa0df620dbd7686c100c58c3be7f5d969939e5eb3abbb36f4-0067'

subfolder = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/selected_oct'
target_folder = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/03'
for filename in unity_list:
    f = filename[0]
    file = f + '.png'
    path_file = os.path.join(subfolder,file)
    target_path = os.path.join(target_folder, file)
    shutil.copy(path_file, target_folder)

