

import sys
import math
import torch
import numpy as np
import timm
import os
import torch.nn as nn
import torch.nn
import torch.nn.functional
import time

import requests
import io
import logging

import ray

import pydicom
import pydicom.pixel_data_handlers

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.widgets import RawImageWidget


WEIGHTS_LOCATION = '/Users/anissaalloula/Desktop/oct_model.pt'
WEIGHTS_ADDRESS = 'https://files.icch.uk/models/oct/oct_model.pt'

OCT_TO_PROCESS_PATH = "/Users/anissaalloula/Desktop/ANISSA OCTs/IMG048"
#OCT_TO_PROCESS_PATH = "/Users/anissaalloula/Desktop/scantensus/IMG001"


# One thing you could do - is directly upload the dicom file (which will be smaller) and let the cluster unpack it and process it

@ray.remote(num_gpus=0.25)
class OCTInfer:
    def __init__(self, device="cpu"):
        self.device = device
        logging.warning("Downloading weights")
        model_weights = requests.get(WEIGHTS_ADDRESS)
        model_weights = io.BytesIO(model_weights.content)
        logging.warning("Weights Downloaded")
        model = timm.create_model('resnet34', num_classes=1)
        model.default_cfg['input_size'] = (3, 1024, 1024)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.init_weights()
        model.load_state_dict(torch.load(model_weights, map_location="cpu"))
        model.to(device)
        model.eval()
        self.model = model
        logging.warning("model init done")

    def infer(self, img) -> list:
        with torch.no_grad():
            image_t = torch.tensor(img, dtype=torch.float32, device=self.device)
            image_t = image_t / 255.0
            image_t = image_t.permute((0, 3, 1, 2))
            image_t = torch.nn.functional.interpolate(image_t, (1024, 1024))
            result = self.model(image_t)

            result = result.cpu().flatten().tolist()

        return result

class OCTFile:
    def __init__(self, file_path):
        dcm = pydicom.dcmread(file_path)
        pixel_array = fix_pixel_data(dcm)

        if True:
            pixel_array = (255.0/65535.0) * pixel_array
            self.pixel_array = pixel_array.astype(np.uint8)

    def get_frame(self, frame) -> np.array:
        return self.pixel_array[frame, ...]

    def get_all_frames(self) -> np.array:

        return self.pixel_array


def main():
    logging.warning(f"Loading file")
    oct_file = OCTFile(OCT_TO_PROCESS_PATH)
    logging.warning(f"Finished loading dicom")
    print(oct_file.get_all_frames().shape)


    ray.init(address="ray://thready3.ic.icch.uk:10001")

    octnn = OCTInfer.remote(device="cuda")

    img = oct_file.get_all_frames()

    num_frames = img.shape[0]
    BATCH_SIZE = 8

    result_refs = []
    results = []

    for batch_num in range(math.ceil(num_frames/BATCH_SIZE)):
        logging.warning(f"Batch {batch_num}")
        start = batch_num * BATCH_SIZE
        end = (1+batch_num) * BATCH_SIZE
        img_temp = img[start:end, ...]

        img_temp_ref = ray.put(img_temp)
        result_ref = octnn.infer.remote(img_temp_ref)
        result_refs.append(result_ref)

    for result_ref in result_refs:
        result_temp = ray.get(result_ref)
        results.extend(result_temp)

    print(results)

    make_gui(oct_file, predictions=results)



def make_gui(oct_file: OCTFile, predictions: list):

    slice_preds_array = np.array(predictions)
    test1 = np.array(slice_preds_array).astype(float)
    test = test1.flatten()

    # Make array of predictions a rolling average to "smooth" curve
    import pandas as pd
    window_size = 5
    test2 = pd.Series(test)
    windows = test2.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    # Remove null entries from the list
    final_list = moving_averages_list[window_size - 1:]
    average_array = np.array(final_list).astype(float)

    # PyQtGraph
    app = pg.mkQApp()
    win = QtGui.QMainWindow()
    pg.setConfigOption('imageAxisOrder', 'row-major')

    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    # plot OCT slices
    plot = pg.plot(name="OCT Slices")
    win.setWindowTitle('OCT vessel ')

    all_frames = oct_file.get_all_frames()
    frames, height, width, channels = all_frames.shape

    xsection = all_frames[:, :, (width//2)-10:(width//2)+10, :]
    xsection = xsection.transpose((1, 0, 2, 3)).reshape((height, -1, channels))

    # xsection_image_viewer = RawImageWidget.RawImageWidget(scaled=True)
    # layout.addWidget(xsection_image_viewer, 0, 4)
    # xsection_image_viewer.setImage(xsection)


    xsection_image_viewer = pg.ImageItem(xsection)
    plot.addItem(xsection_image_viewer)
    layout.addWidget(plot, 0, 4)

    # copy array's columns to make it wider
    stacked_mean_array = np.tile(average_array, (20, 1))

    # Make x axis arrays for both plots
    list = []
    for i in range(len(test)):
        list.append(20 * i)
    list = np.array(list)

    list2 = []
    for i in range(len(average_array)):
        list2.append(20 * i)
    list2 = np.array(list2)

    # color map
    plot_colormap = pg.plot(x=list2, name="colormap")


#Above 0.8, things should look green, with bright green close to 1.
#Between 0.5 and 0.8, between green and yellow.
#Below 0.5 starts to get orange and goes red ish below 0.4 with really red below 0.3.

    # change to rawimage widget??
    img3 = pg.ImageView()
    img3.setImage(stacked_mean_array)

    ## ???? red, orange, yellow, green,
    colors = [(255, 30, 30), (255, 129, 18), (255,215,0), (30, 255, 30)]  # red to green, red = low values, green =high
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 4), color=colors)
    img3.setColorMap(cmap)
    img3.ui.histogram.hide()
    img3.ui.roiBtn.hide()
    img3.ui.menuBtn.hide()
    value = img3.getImageItem()
    plot_colormap.addItem(value)
    layout.addWidget(plot_colormap, 2, 4, )

    # # OCT slices network raw scores - add this to previous plot but hide line
    # plot1 = pg.plot()
    # plot1.plot(x = list, y = test, name="Raw Scores")
    # plot1.setTitle(title="Raw OCT slice scores")
    # plot1.setYRange(0, 1, padding=0)
    # plot1.setXLink('OCT Slices')
    # layout.addWidget(plot1)

    # OCT slices network scores rolling average
    plot2 = pg.plot()
    plot2.plot(x=list2, y=average_array, name="Average Scores")
    plot2.setTitle(title="OCT slice scores rolling average over windows of 5")
    plot2.setXLink('OCT Slices')
    plot2.setYRange(0, 1, padding=0)
    layout.addWidget(plot2, 3, 4)

    # Plot a full OCT image
    plot5 = pg.plot()
    plot5.setAspectLocked(True)
    plot5.setTitle(title="Selected OCT image")
    layout.addWidget(plot5, 0, 0, 4, 4)

    infinite_line = pg.InfiniteLine(pos=5, angle=90, pen='#0FA00F', movable=True)
    plot2.addItem(infinite_line)

    def clicked():

        idx = infinite_line.value()
        print(idx)
        image_idx = int(idx / 20)
        whole_image_array = oct_file.get_frame(image_idx)[::-1, ...]
        im5 = pg.ImageItem(whole_image_array)
        plot5.addItem(im5)

    infinite_line.sigPositionChanged.connect(clicked)

    clicked()
    win.show()
    app.exec_()





def fix_pixel_data(dcm: pydicom.dataset):
    in_TransferSyntaxUID = dcm.file_meta.TransferSyntaxUID
    in_PhotometricInterpretation = dcm.PhotometricInterpretation
    Modality = dcm.get("Modality", None)
    ###

    logging.info(f"In PhotometricInterpretation: {in_PhotometricInterpretation}")

    try:

        if in_PhotometricInterpretation == 'PALETTE COLOR':
            pixel_array = pydicom.pixel_data_handlers.util.apply_color_lut(dcm.pixel_array, dcm)
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        elif "YBR" in in_PhotometricInterpretation:
            pixel_array = pydicom.pixel_data_handlers.util.convert_color_space(dcm.pixel_array, dcm.PhotometricInterpretation, "RGB")
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        elif in_PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = dcm.pixel_array.copy()
            PhotometricInterpretation = "MONOCHROME1"
            SamplesPerPixel = 1
        elif in_PhotometricInterpretation == 'MONOCHROME2':
            if Modality == 'IVOCT':
                pixel_array = pydicom.pixel_data_handlers.util.apply_color_lut(dcm.pixel_array, dcm)
                PhotometricInterpretation = "RGB"
                SamplesPerPixel = 3
            else:
                pixel_array = dcm.pixel_array.copy()
                PhotometricInterpretation = "MONOCHROME2"
                SamplesPerPixel = 1
        elif in_PhotometricInterpretation == "RGB":
            pixel_array = dcm.pixel_array.copy()
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        else:
            logging.error(f"Unrecognised source PhotometricInterpretation: {in_PhotometricInterpretation}")
            return

    except Exception as e:
        print(e)
        logging.warning(f"Error reading images from")
        return

    logging.info(f"PhotometricInterpretation out: {PhotometricInterpretation}")
    return pixel_array



if __name__ == "__main__":
    main()



        if in_PhotometricInterpretation == 'PALETTE COLOR':
            pixel_array = pydicom.pixel_data_handlers.util.apply_color_lut(dcm.pixel_array, dcm)
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        elif "YBR" in in_PhotometricInterpretation:
            pixel_array = pydicom.pixel_data_handlers.util.convert_color_space(dcm.pixel_array, dcm.PhotometricInterpretation, "RGB")
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        elif in_PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = dcm.pixel_array.copy()
            PhotometricInterpretation = "MONOCHROME1"
            SamplesPerPixel = 1
        elif in_PhotometricInterpretation == 'MONOCHROME2':
            if Modality == 'IVOCT':
                pixel_array = pydicom.pixel_data_handlers.util.apply_color_lut(dcm.pixel_array, dcm)
                PhotometricInterpretation = "RGB"
                SamplesPerPixel = 3
            else:
                pixel_array = dcm.pixel_array.copy()
                PhotometricInterpretation = "MONOCHROME2"
                SamplesPerPixel = 1
        elif in_PhotometricInterpretation == "RGB":
            pixel_array = dcm.pixel_array.copy()
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        else:
            logging.error(f"Unrecognised source PhotometricInterpretation: {in_PhotometricInterpretation}")
            return

    except Exception as e:
        print(e)
        logging.warning(f"Error reading images from")
        return

    logging.info(f"PhotometricInterpretation out: {PhotometricInterpretation}")
    return pixel_array



if __name__ == "__main__":
    main()

