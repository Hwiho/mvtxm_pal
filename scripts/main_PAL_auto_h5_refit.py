from mvtxm_pal import *
from skimage import io
import numpy as np
import os
import glob
import pandas as pd
import time


def txm_io(fn, kernel=1):
    txm_file = iotools_pal()
    txm_file.load_processed_h5(fn = fn, index= None)
    txm_file.median_filt(kernel)
    return txm_file

def img_crop_coord(medfilt):
    cropped = medfilt[:, :, :]

    '''
    itb = imgtoolbox(medfilt)
    itb.get_roicoord(scan_note='sungjae')
    itb.cropfromrois()
    cropped = itb.cropped
    '''
    return cropped

def fit(img_aligned, eng, polynomial, fit_num, peakref):
    fitting = fit_xanes(img_aligned, eng, peakref,color_flag='rg')
    fitting.set_thickness_pnts(len(eng))
    fitting.set_thickness()
    fitting.threshold(0.03)
    if polynomial == 2:
        fitting.polynomial_second_fit_separate(fit_num, ev_step=1)
    elif polynomial == 'max':
        fitting.polynomial_second_fit(fit_num, maxpoint= None)
    else:
        fitting.polynomial_multi_fit_whole_wl(polynomial, ev_step= 1, bounds = None)
    fitting.ff_xanes_calibration()
    fitting.cal_stdev()
    fitting.hsvcolormap(value=3)
    fitting.relcolormap(value=3)
    return fitting

def saving_data(fitting, polynomial, fit_num, data_dir, fn):
    data = data_mng_pal(fitting, data_dir, fn)
    data.save_id_PAL(fn)
    data.create_dir_all_in_one(polynomial, fit_num)
    data.save_into_separate_folders_all_in_one(bins = 100, range = [8350,8360])
    return data

def image_info(data, fitting):
    File = open(data.all_dir + "/img_info.txt", "a+")
    File.write(f'Scan_id : {data.scan_id}_{data.scan_pos}'+
               f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}'+
               f"\nStd : {fitting.stdev}\n")
    File.close()

def main(dirpath, h5file, peakref, polynomial, fit_num, multi_fit_flag, fit_num_list = None, poly_list = None):
    txm_file = txm_io(h5file)
    img_aligned = img_crop_coord(txm_file.medfilt)

    if multi_fit_flag == 0:
        try:
            fitting = fit(img_aligned, txm_file.eng, polynomial, fit_num, peakref)  # [8354.4, 8356.8], [8352, 8355]
            #[8352, 8355] #[8354, 8356.4])
        except ValueError:
            with open(dirpath + "/error_list.txt", "a+") as f:
                f.write(txm_file.fn + '\n')
        else:
            data = saving_data(fitting, polynomial, fit_num, data_dir=dirpath, fn=txm_file.fn)
            image_info(data,fitting)

            print(f'Scan_id : {data.scan_id}_{data.scan_pos}', f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}',
                f"\nStd : {fitting.stdev}")
    elif multi_fit_flag == 1:
        if fit_num_list is not None and polynomial == 2:
            for fit_num in fit_num_list:
                try:
                    fitting = fit(img_aligned, txm_file.eng, polynomial, fit_num,
                                  peakref)  # [8354.4, 8356.8], [8352, 8355]
                    # [8352, 8355] #[8354, 8356.4])
                except ValueError:
                    File = open(fdir + "/error_list.txt", "a+")
                    File.write(fn + '\n')
                    File.close()
                    pass
                else:
                    data = saving_data(fitting, polynomial, fit_num, data_dir=dirpath, fn=txm_file.fn)
                    image_info(data, fitting)

                    print(f'Scan_id : {data.scan_id}_{data.scan_pos}',
                          f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}',
                          f"\nStd : {fitting.stdev}")
        elif poly_list is not None:
            fit_num = 3
            for polynomial in poly_list:
                try:
                    fitting = fit(img_aligned, txm_file.eng, polynomial, fit_num,
                                  peakref)  # [8354.4, 8356.8], [8352, 8355]
                    # [8352, 8355] #[8354, 8356.4])
                except ValueError:
                    File = open(fdir + "/error_list.txt", "a+")
                    File.write(fn + '/n')
                    File.close()
                    pass
                else:
                    data = saving_data(fitting, polynomial, fit_num, data_dir = dirpath, fn=txm_file.fn)
                    image_info(data, fitting)

                    print(f'Scan_id : {data.scan_id}_{data.scan_pos}',
                          f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}',
                          f"\nStd : {fitting.stdev}")


if __name__ == "__main__":
    dirpath = r"C:\Users\hwiho\Documents\HT_paper\PXM\RT\Data_folder_poly2_fit3\h5files"
    h5_list = sorted(glob.glob(dirpath + '/*.h5'))[29:]
    for index, h5file in enumerate(h5_list):
        print(index, ':', h5file)
        main(dirpath, h5file, peakref=[8367.6, 8369.6], polynomial=2, fit_num=3, multi_fit_flag=1,
                 fit_num_list=[3], poly_list=[])

        #peakref = [8368.1, 8371]


