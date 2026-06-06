from mvtxm_pal import *
from skimage import io
import numpy as np
import os
import glob
import pandas as pd
import time
from matplotlib.widgets import Slider, Button
import matplotlib
matplotlib.use('Qt5Agg')


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

def txm_io(kernel=3):
    txm_file = iotools_pal()
    txm_file.select_directory()
    txm_file.img_avg(avg=3)
    txm_file.load_tif()
    txm_file.fn_PAL()
    txm_file.energy_list_extraction()
    txm_file.rmv_bkg()
    txm_file.median_filt(kernel)
    return txm_file

def img_crop_single(medfilt):
    itb = imgtoolbox(medfilt)
    itb.crop_img()
    cropped = itb.cropped
    return cropped

def img_crop_coord(medfilt):

    cropped = medfilt[:, 300:1800, 1:medfilt.shape[2]-1]

    '''
    itb = imgtoolbox(medfilt)
    itb.get_roicoord(scan_note='sungjae')
    itb.cropfromrois()
    cropped = itb.cropped
    '''
    return cropped

def img_reg(cropped, eng, type = 'stackreg'):
    if type == 'mrtv':
        reg = regist(cropped)
        reg.set_ref_mode('single')
        reg.set_indices(0, cropped.shape[0], cropped.shape[0] // 2)
        reg.set_eng(eng)
        reg.compose_dicts()
        reg.set_reg_options()
        reg._alignment_scheduler()
        reg.set_method('MRTV')
        reg.set_shift()
        reg.reg_xanes2D_chunk()
        reg.set_shift_dict()
        reg._sort_absolute_shift(optional_shift_dict=reg.shift_dict)
        reg.apply_xanes2D_chunk_shift(reg.abs_shift_dict)
        reg.crop_residual()
        img_aligned = reg.shift_img_cropped
    else:
        sr = StackReg(StackReg.TRANSLATION)
        def show_progress(current_iteration, end_iteration):
            print(f"Registering {current_iteration} of {end_iteration} images")
        reg = sr.register_transform_stack(cropped,axis=0,progress_callback=show_progress)
        img_aligned = np.array(reg, dtype=np.float32)
    return img_aligned

def fit(img_aligned, eng, peakref, fit_num):
    fitting = fit_xanes(img_aligned, eng, peakref, color_flag='rg')
    #fitting.savgol_filter(window_size = 5, polyorder = 2)
    fitting.set_thickness_pnts(len(eng))
    fitting.set_thickness()
    fitting.thres_stack_generation()
    viewer = thresviewer(fitting.thres_stack, fitting.thres_list)
    fitting.threshold(fitting.thres_list[viewer.frame])
    fitting.polynomial_second_fit_separate(fit_num,ev_step=1)
    #fitting.polynomial_multi_fit_whole_wl(deg = 5, bounds=[15,30],ev_step=1)
    #fitting.edge50(window_size=5, polyorder=2)
    #fitting.ff_xanes_calibration()
    fitting.cal_stdev()
    fitting.hsvcolormap()
    fitting.relcolormap()
    return fitting

def saving_data(fitting, data_dir, fn):
    data = data_mng_pal(fitting, data_dir, fn)
    data.save_id_PAL(fn)
    data.scan_pos = '00'
    data.create_dir()
    data.save_color()
    data.histogram(bins=500, range=[8365, 8375])
    data.weighted_histogram(bins=500, range=[8365, 8375])
    data.create_h5()
    return data


def main():
    txm_file = txm_io()
    cropped = img_crop_single(txm_file.medfilt)
    img_aligned = img_reg(cropped, txm_file.eng, type = 'stackreg')
    fitting = fit(img_aligned, txm_file.eng, fit_num=3, peakref=[8368.5, 8371])  # [6556, 6566], [6568, 6572])
    data = saving_data(fitting, data_dir=txm_file.proj_dir, fn=txm_file.fn)
    print(f'Scan_id : {data.scan_id}', f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite!=0])}',
          f"\nStd : {fitting.stdev}")


if __name__ == "__main__":
    main()
