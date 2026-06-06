from mvtxm_pal import *
from skimage import io
import numpy as np
import os
import glob
import pandas as pd
import time


def txm_io(proj,bkg, kernel = 3):
    txm_file = iotools_pal()
    txm_file.proj_dir = proj
    txm_file.bkg_dir = bkg
    txm_file.fn_PAL()
    txm_file.img_avg(avg=3)
    txm_file.load_tif()
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
    cropped = medfilt[:, 500:1500, :]

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
    data.scan_pos = '00'
    data.create_dir_all_in_one(polynomial, fit_num)
    data.save_into_separate_folders_all_in_one(bins = 100, range = [8365,8375])
    return data

def image_info(data, fitting):
    File = open(data.all_dir + "/img_info.txt", "a+")
    File.write(f'Scan_id : {data.scan_id}_{data.scan_pos}'+
               f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}'+
               f"\nStd : {fitting.stdev}\n")
    File.close()

def main(dirpath, proj, bkg, peakref, polynomial, fit_num, multi_fit_flag, fit_num_list = None, poly_list = None):
    txm_file = txm_io(proj, bkg)
    cropped = img_crop_coord(txm_file.medfilt)
    img_aligned = img_reg(cropped, txm_file.eng,type='stackreg')
    if multi_fit_flag == 0:
        try:
            fitting = fit(img_aligned, txm_file.eng, polynomial, fit_num, peakref)
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
                    fitting = fit(img_aligned, txm_file.eng, polynomial, fit_num, peakref)
                except ValueError:
                    with open(dirpath + "/error_list.txt", "a+") as f:
                        f.write(txm_file.fn + '\n')
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
                    fitting = fit(img_aligned, txm_file.eng, polynomial, fit_num, peakref)
                except ValueError:
                    with open(dirpath + "/error_list.txt", "a+") as f:
                        f.write(txm_file.fn + '\n')
                else:
                    data = saving_data(fitting, polynomial, fit_num, data_dir = dirpath, fn=txm_file.fn)
                    image_info(data, fitting)

                    print(f'Scan_id : {data.scan_id}_{data.scan_pos}',
                          f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}',
                          f"\nStd : {fitting.stdev}")


if __name__ == "__main__":
    dirpath = r"C:\Users\hwiho\Documents\HT_paper\PXM\In-situ\2nd_cycle_4MPa"
    proj_list = []
    bkg_list = []
    for it in os.scandir(dirpath):
        if it.is_dir():
            if it.path[-4:] == 'proj':
                proj_list.append(it.path)
            elif it.path[-4:] == 'back':
                bkg_list.append(it.path)

    for proj,bkg in zip (sorted(proj_list), sorted(bkg_list)):
        print(proj,'\n',bkg)
        if proj[:-4] != bkg[:-4]:
            pass
            print("Directory error")
        else:
            main(dirpath, proj,bkg, peakref=[8370.1, 8372.5], polynomial=2, fit_num=3, multi_fit_flag=0,
                 fit_num_list=[3], poly_list=[])


