import numpy as np

'''
SDY calibration channel.

{2, 3, 4, 6, 8, 10, 12, 16, 22, 30}

2 - highest velocities
30 - lowest velocities

Refers to number of pixels per cm/s
However, have to time the number of pixels by a number between 4.8 and 5 (probably 5) first.

The screen doesn't show the whole spectrum - cuts off the top bit - I think about 30%.

'''

## Can make it some sort of counter where it iterates through the elapsed_time values and adds it to the next one if the delta is suddenly negative (ie it is at a peak)
##look trhough frame to see if there is any other channel which could correspond ton some sort of indication of the time probelms
class SDY_File:

    TOTAL_CHANNELS = 44 + 1079

    details_list = ['last_name',
                    'first_name',
                    'middle_initial',
                    'gender',
                    'patient_id',
                    'physcian',
                    'date_of_birth',
                    'procedure',
                    'procedure_id',
                    'accession',
                    'ffr',
                    'ffr_suid',
                    'refering_physcian',
                    'additional_pt_history',
                    'ivus_suid',
                    'department',
                    'institution',
                    'cathlab_id']

    chan_start = np.array([0])  # Just 3560
    chan_unknown1 = np.array([1])  # Just 96
    chan_blank1 = np.array([2, 3])  # Just 0
    chan_unknown2 = np.array([4, 6, 8, 10])
    chan_pd = np.array([5, 7, 9, 11])
    chan_unknown_pa = np.array([12, 13, 14, 15])  # Sometime pa, sometimes noise
    chan_pa = np.array([16, 17, 18, 19])
    chan_unknown3 = np.array([20, 21, 22, 23])  # ?Noise
    chan_ecg = np.array([24, 26, 28, 30])
    chan_unknown4 = np.array([25])  # Unknown ?Marker
    chan_unknown5 = np.array([27])  # Unknown ?Marker
    chan_unknown6 = np.array([29])  # Unknown ?Marker
    chan_unknown7 = np.array([31])  # Unknown ?Marker
    chan_flow = np.array([32, 33])
    chan_flow_average = np.array([34])  # Flow beat-by-beat average
    chan_blank3 = np.array([35])  # Unknown - high
    chan_calc2 = np.array([36, 37])
    chan_calc3 = np.array([38, 39])
    chan_flow_baseline = np.array([40])  # Baseline Fixed 48 or 114
    chan_blank4 = np.array([41])  # Fixed 1
    chan_calc5 = np.array([42])  # Calc
    chan_flow_scale = np.array([43])  # flow scale for spectrum

    # 43 to 87 inclusive - UNUSED - 0

    # 88:344 - Spectrum A
    # 244:600 - Spectrum B

    # 600 to 1111 inclusive - UNUSED - 0

    # 1112 - ? Part of checksum
    # 1113 - Various discrete values
    # 1114 - Like about 15 minutes. Ascending with occasional drop down
    # 1115 - EVENT 2
    # 1116 - 0
    # 1117 - 0 [ 1116 + 1117 uint32 SPACE]
    chan_ms_loop = np.array([1118])
    chan_ms_loop_counter = np.array([1119])
    # 1118 - Number of seconds per minute 65535 per minutes
    # 1119 - Number of minutes [1118, 1119 - uint32 - elapsed time stamp in packet]
    # 1120 - Event
    # 1121 - CRC
    # 1122 - SPACE - 0
    # End


    def __init__(self, sdy_fl):

        self.sdy_fl = sdy_fl
        self.plot_order = False

        self.sampling_rate = 50  ###Check this is only a guess

        with open(sdy_fl, 'rb') as sdy_f:
            sdy_f.seek(0)
            self.file_header = sdy_f.read(16)
            self.file_patient = sdy_f.read(512*18)
            self.file_raw_data = sdy_f.read(-1)

        self.file_type = np.frombuffer(self.file_header, offset=0, dtype=np.uint32, count=1)
        self.date_time = np.frombuffer(self.file_header, offset=4, dtype=np.uint32, count=2)
        self.exam_type = np.frombuffer(self.file_header, offset=12, dtype=np.int32, count=1)
        # 3=pressure, 4=Flow, 5=Combo

        pt_details = {}

        for i, detail in enumerate(self.details_list):
            start = i * 512
            end = (i+1) * 512
            pt_details[detail] = self.file_patient[start:end].decode('utf-16').replace('\x00', '').strip()

        self.pt_details = pt_details

        raw_data = np.frombuffer(self.file_raw_data, dtype=np.uint16, count=-1)
        data_width = len(raw_data) // self.TOTAL_CHANNELS
        data_len = data_width * self.TOTAL_CHANNELS
        remainder = len(raw_data) % self.TOTAL_CHANNELS

        # The axis to the right is the one which cycles the fastest through the data
        # Note, as from buffers is not-writable
        self.frame = raw_data[:data_len].reshape((data_width, self.TOTAL_CHANNELS))

        # the indexing by [:, [a,b]] leads to a copy so these are mutable
        self.pa = self.frame[:, self.chan_pa].flatten()
        self.pd = self.frame[:, self.chan_pd].flatten()
        self.ecg = self.frame[:, self.chan_ecg].flatten()

        # multiply pa and pd arrays by 4 to get the same values as on study manager
        self.pa_true = self.pa / 4
        self.pd_true = self.pd / 4

        self.flow = self.frame[:, self.chan_flow].flatten().astype(np.float32)
        self.flow_scale = self.frame[:, self.chan_flow_scale].repeat(2).astype(np.float32)
        self.flow_baseline = self.frame[:, self.chan_flow_baseline].repeat(2).astype(np.float32)

        ## need to convert flow pixels to cm/s with scale (check that digitized trace is indeed in pixel units)
        # it looks like phillips does pixels from bottom so don't need to do 256 - y  (256=image height)
        # 48 = baseline height (double it when working with images of height 256)
        self.flow_true = 5 * (self.flow - self.flow_baseline) / self.flow_scale

        self.flow_average = self.frame[:, self.chan_flow_average].flatten()
        self.calc2 = self.frame[:, self.chan_calc2].flatten()
        self.calc3 = self.frame[:, self.chan_calc3].flatten()

        self.spectrum = self.frame[:, 88:600].reshape((-1, 256)).T

        self.elapsed_time_ms = np.zeros(data_len, dtype=np.int64)
        self.ms_loop = self.frame[:, self.chan_ms_loop]
        self.loop_counter = self.frame[:, self.chan_ms_loop_counter]
        self.elapsed_time_ms = self.frame[:, self.chan_ms_loop] + (self.frame[:, self.chan_ms_loop_counter] * 65536)

        ## Downsample ECG, PA, and PD by a factor of 2
        self.target_length = self.flow_true.shape[0]
        self.ecg = self.ecg.reshape(-1,self.target_length, order ='F')
        self.ecg = self.ecg.mean(axis=0)
        self.pa_true = self.pa_true.reshape(-1, self.target_length, order ='F')
        self.pa_true = self.pa_true.mean(axis=0)
        self.pd_true = self.pd_true.reshape(-1,self.target_length, order ='F')
        self.pd_true = self.pd_true.mean(axis=0)

        ## Upsample elapsed time by 2
        self.elapsed_time_ms = (np.transpose(self.elapsed_time_ms)).flatten()
        x = np.linspace(0, 1, self.target_length)
        xp = np.linspace(0, 1, self.elapsed_time_ms.shape[0])
        self.elapsed_time_ms_interp = np.interp(x,xp,self.elapsed_time_ms)
        self.elapsed_time_ms_interp = np.around(self.elapsed_time_ms_interp)


        ## create one array with time, ECG, PA, PD, and FLOW
        # self.array_tuple = (self.elapsed_time_ms, self.ecg, self.pa, self.pd, self.flow_true)
        # self.alldata = np.vstack(self.array_tuple)
        self.rawflow = self.spectrum

        ##input raw flow into trained tracking network
        # model = X ##??
        # self.AIflow = model(self.rawflow)


    # def write_file(self, path):
    #     out_frame = self.frame.copy()
    #     with open(path, "wb") as sdy_out_f:
    #         sdy_out_f.write(self.file_header)
    #         sdy_out_f.write(self.file_patient)
    #         sdy_out_f.write(out_frame.tobytes())
    #
    # def replace_and_write(self, flow, path):
    #
    #     flow = flow.reshape((-1, 2)).astype(np.uint16)
    #
    #     out_frame = self.frame.copy()
    #     out_frame[1000:-1000, 32] = flow[1000:-1000, 0]
    #     out_frame[1000:-1000, 33] = flow[1000:-1000, 1]
    #
    #     with open(path, "wb") as sdy_out_f:
    #         sdy_out_f.write(self.file_header)
    #         sdy_out_f.write(self.file_patient)
    #         sdy_out_f.write(out_frame.tobytes())

## example: name = 'CMStudy_2019_10_29_083353.sdy', path = '/Users/anissaalloula/Downloads/testCM1.sdy'
#
# file_name = input("SDY file name")
# path_to_file = input("Path to SDY file")
processed_sdy = SDY_File("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/CMStudy_2021_09_22_120258.sdy")
#
# pig1 = SDY_File('/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking/CMStudy_2018_11_28_083030.sdy')
#
# pig1_ms_loop_list = pig1.ms_loop.tolist()
# test_time = []
# for i in range(len(pig1_ms_loop_list)):
#     if i == 0:
#         test_time.append(pig1_ms_loop_list[i][0])
#     else:
#         step = pig1_ms_loop_list[i][0] - pig1_ms_loop_list[i-1][0]
#         if step >= 0:
#             test_time.append((step+test_time[i-1]))
#         else:
#             test_time.append((pig1_ms_loop_list[i][0]+test_time[i-1]))
#
# test_time_array = np.array(test_time)
#
# test_time_step = []
# for i in range(len(test_time_array)-1):
#     test_time_step.append((test_time_array[i+1]-test_time_array[i]))
#
# import matplotlib.pyplot as plt
# plt.plot(test_time_step)
# plt.show()
# plt.title('Interval between time points (ms)')
#
# a = np.median(test_time_step)
#
# adjusted_time = []
# for i in range(len(test_time_array)):
#     if i == 0:
#         adjusted_time.append(test_time_array[i])
#     else:
#         step = test_time_array[i] - test_time_array[i-1]
#         # get median step in the +/- 5000ms (approx equivalent to 500 time points)
#         if i < 500:
#             a = np.median(test_time_step[i:i+500])
#         if i > (len(test_time_array) - 250):
#             a = np.median(test_time_step[i-500:i])
#         else:
#             a = np.median(test_time_step[i-250:i+250])
#         ## reduce interval back to median interval in abnormally large intervals
#         if step <= 1000:
#             adjusted_time.append((step + adjusted_time[i-1]))
#         else:
#             adjusted_time.append((a + adjusted_time[i-1]))
#
# adjusted_time_array = np.array(adjusted_time)
#
# plt.plot(adjusted_time_array)
# plt.title('adjusted time')
# plt.show()
#
# x = np.linspace(0, 1, pig1.target_length)
# xp = np.linspace(0, 1, pig1.elapsed_time_ms.shape[0])
# adjusted_time_interp = np.interp(x, xp, adjusted_time_array)
# adjusted_time_interp = np.around(adjusted_time_interp)
#
# test_time_step = []
# for i in range(len(test_time_array)-1):
#     test_time_step.append((adjusted_time_interp[i+1]-adjusted_time_interp[i]))
# plt.plot(test_time_step)
# plt.title('adjusted step')
# plt.show()

# x = np.linspace(0, 1, pig1.target_length)
# xp = np.linspace(0, 1, pig1.elapsed_time_ms.shape[0])
# test_time_interp = np.interp(x, xp, test_time_array)
# test_time_interp = np.around(test_time_interp)
#
# plt.plot(pig1_ms_loop_list)
# plt.title('ms loop')
# plt.show()
#
# plt.plot(pig1.loop_counter)
# plt.title('Loop counter')
# plt.show()
#
# plt.plot(test_time_interp)
# plt.title('Adjusted elapsed time (ms)')
# plt.show()
#
# import pandas as pd
# sdy_df = pd.DataFrame()
# sdy_df['Elapsed Time (ms)'] = pig1.elapsed_time_ms_interp
# sdy_df['Adjusted Elapsed Time (ms)'] = adjusted_time_interp
# sdy_df['Pa'] = pig1.pa_true
# sdy_df['Pd'] = pig1.pd_true
# sdy_df['ECG'] = pig1.ecg
# sdy_df['Flow Velocity'] = pig1.flow_true
#
# #np.savetxt('Processed_CMStudy_2018_11_28_083030.txt',sdy_df,fmt = "%d")
# sdy_df.to_csv('Processed_CMStudy_2018_11_28_083030.csv')
# ## in the future should add a header to index column that is automatically outputed in to csv
#
# ##saves arrays as tsv files in the working directory
# # np.savetxt(str((file_name)+'ECG_pa_pd_flow.tsv'), processed_sdy.alldata, delimiter="\t")
# # np.savetxt(str((file_name)+'rawflow.tsv'), processed_sdy.rawflow, delimiter="\t")
#
# ## + Need to incorporate trained network to get AI flow
#
#
##Pyqt graph
#import pyqtgraph.examples
#pyqtgraph.examples.run()
#

#
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
#
win = pg.GraphicsLayoutWidget(show=True, title="ECG/Pressure/Flow Data")
win.resize(1000,600)
win.setWindowTitle('ECG/Pressure/Flow Data')
# #
p1 = win.addPlot(title="ECG", x = processed_sdy.elapsed_time_ms_interp, y=processed_sdy.ecg)
win.nextRow()
# #
# # p2 = win.addPlot(title="Pa/Pd", pen = pg.mkPen(color=(0, 255, 0)))
# # p2.plot(y=processed_sdy.pa_true, pen = pg.mkPen(color=(0, 0, 255)))
# # p2.plot(y=processed_sdy.pd_true)
# #
# # p2.plot()
#
# win.nextRow()
# #
p3 = win.addPlot(title = "Flow Velocity", x = processed_sdy.elapsed_time_ms_interp, y=processed_sdy.flow_true)
# #
win.nextRow()
p4 = win.addPlot(title = "Elapsed Time", y = processed_sdy.elapsed_time_ms_interp)
# #
# #
# # p5 = win.addPlot(title="ECG w time", x = processed_sdy.elapsed_time_ms_interp, y=processed_sdy.ecg)
# # win.nextRow()
# # p6 = win.addPlot(title="Pa/Pd w time", pen = pg.mkPen(color=(0, 255, 0)))
# # p6.plot(x = processed_sdy.elapsed_time_ms_interp, y=processed_sdy.pa_true, pen = pg.mkPen(color=(0, 0, 255)))
# # p6.plot(x = processed_sdy.elapsed_time_ms_interp, y=processed_sdy.pd_true)
# #
# # p6.plot()
# #
# # #win.nextRow()
# # # p7 = win.addPlot(title="Pa/Pd w time", pen = pg.mkPen(color=(0, 255, 0)))
# # # p7.plot(x = test_time_interp, y=processed_sdy.pa_true, pen = pg.mkPen(color=(0, 0, 255)))
# # # p7.plot(x = test_time_interp, y=processed_sdy.pd_true)
# # #p7 = win.addPlot(title = "Flow Velocity w time", x = processed_sdy.elapsed_time_ms_interp, y=processed_sdy.flow_true)
# #
# win.nextRow()
# p8 = win.addPlot(title="Pa/Pd w adjusted time", pen = pg.mkPen(color=(0, 255, 0)))
# p8.plot(x = adjusted_time_interp, y=processed_sdy.pa_true, pen = pg.mkPen(color=(0, 0, 255)))
# p8.plot(x = adjusted_time_interp, y=processed_sdy.pd_true)
# # #p7 = win.addPlot(title = "Flow Velocity w time", x = processed_sdy.elapsed_time_ms_interp, y=processed_sdy.flow_true)
# #
app = QtGui.QApplication([])
# #
# # ## Create window with ImageView widget
# # # win = QtGui.QMainWindow()
# # # win.resize(800,800)
# # # imv = pg.ImageView()
# # # win.setCentralWidget(imv)
# # # win.show()
# # # win.setWindowTitle('pyqtgraph example: ImageView')
# # # imv.setImage(processed_sdy.rawflow)

if __name__ == '__main__':
     import sys
     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
         QtGui.QApplication.instance().exec_()