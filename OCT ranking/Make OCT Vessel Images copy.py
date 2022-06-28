import PIL
from PIL import Image
from numpy import asarray
import os
import numpy as np


# make dictionary of all images from one vessel

#path_oct_folder = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/oct/test OCT'
path_oct_folder = '/Volumes/png.database.scantensus.icch.uk/data/03/oct_group_2/oct20011/study_00/series_00/US000000'


image_list = os.listdir(path_oct_folder)

cropped_array_dict = {}
a = 10 # change to choose how big crop is
# for i in range(len(image_list)):

b = 0

for i in range(len(image_list)):
    image_name = image_list[i]
    image = PIL.Image.open(os.path.join(path_oct_folder,image_name))
    data = asarray(image)
    cropped_array_dict[b] = data[:,512-a:512+a,:]
    b = b+1

# merge all cropped images into one big image
sum_array = cropped_array_dict[0]
for i in range(1,len(cropped_array_dict)):
    sum_array = np.concatenate((sum_array,cropped_array_dict[i]),axis=1)

image2 = Image.fromarray(sum_array)