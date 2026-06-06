from mvtxm_pal import *
from skimage import io
import numpy as np
import os , h5py
import glob
import pandas as pd
import time
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.widgets import Slider, Button
from scipy.signal import savgol_filter

class thresviewer(object):
    def __init__(self, img_stack, thres):
        self.img_stack = img_stack
        self.nframes = len(img_stack)
        self.thres = thres


        # Setup the axes.
        self.fig, self.ax = plt.subplots()
        self.slider_ax = self.fig.add_axes([0.2, 0.03, 0.65, 0.03])
        self.register_ax = self.fig.add_axes([0.85, 0.72, 0.1, 0.04])
        # Make the slider
        self.slider = Slider(self.slider_ax, 'Frame', 1, self.nframes,
                             valinit=1, valfmt='%1d/{}'.format(self.nframes))
        self.slider.on_changed(self.update)
        # Make the buttons
        self.reg_button = Button(self.register_ax, 'Register')
        self.reg_button.on_clicked(self.img_update)
        # Plot the first slice of the image
        self.im = self.ax.imshow(np.array(img_stack[0]), cmap='gray')

        plt.show()

    def update(self, value):
        self.frame = int(np.round(value - 1))

        # Update the image data
        dat = np.array(self.img_stack[self.frame])
        self.im.set_data(dat)

        # Reset the image scaling bounds (this may not be necessary for you)
        self.im.set_clim([dat.min(), dat.max()])

        # Redraw the plot
        self.fig.canvas.draw()

    def img_update(self, event):
        print(f"Selected threshold: {self.thres[self.frame]}")

def txm_io(kernel=1):
    txm_file = iotools_pal()
    txm_file.load_processed_h5(index = None)
    txm_file.median_filt(kernel)
    return txm_file

def img_crop_single(img):
    itb = imgtoolbox(img)
    itb.crop_img()
    cropped = itb.cropped
    return cropped
    
def fit(img_aligned, eng, peakref, fit_num, savgol_flag = 1):
    if savgol_flag == 1:
        img_aligned = savgol_filter(img_aligned,window_length=3,polyorder=2,axis=0)
    fitting = fit_xanes(img_aligned, eng, peakref, color_flag='rg')
    fitting.set_thickness_pnts(len(eng))
    fitting.set_thickness()
    fitting.thres_stack_generation()
    viewer = thresviewer(fitting.thres_stack,fitting.thres_list)
    fitting.threshold(fitting.thres_list[viewer.frame])
    #fitting.edge50(pre_edge_range = slice(0, 3), post_edge_range = slice(-3, None), window_size=5, polyorder=2)
    fitting.polynomial_second_fit_separate(fit_num,ev_step=1)
    #fitting.polynomial_multi_fit_whole_wl(deg = 5, bounds = None, ev_step=1)
    fitting.cal_stdev()
    fitting.hsvcolormap()
    fitting.relcolormap()
    return fitting

def saving_data(fitting, fn):
    data = data_mng(fitting,fn)
    data.save_id(fn)
    data.create_dir()
    data.save_color()
    data.histogram(bins=100, range=[8350, 8360])
    data.weighted_histogram(bins=100, range=[8350, 8360])
    data.create_h5()
    return data

    
def main():
    txm_file = txm_io()
    cropped = img_crop_single(txm_file.img)
    fitting = fit(cropped, txm_file.eng, fit_num=3, peakref=[8351.5, 8354])  # [8354, 8356.4] [8352, 8355]
    data = saving_data(fitting, fn=txm_file.fn)

    print(f'Scan_id : {data.scan_id}_{data.scan_pos}', f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite!=0])}',
          f"\nStd : {fitting.stdev}")


if __name__ == "__main__":
    main()
