from mvtxm_pal import *
from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import numpy as np
import os
import glob
import pandas as pd
import time



def txm_io(dir_path, kernel=1):
    txm_file = iotools_pal()
    txm_file.proj_dir = dir_path
    txm_file.fn_PAL()
    txm_file.load_stack_and_energy(dir_path)
    txm_file.rmv_bkg()
    txm_file.median_filt(kernel)

    return txm_file

def img_crop_single(medfilt):
    itb = imgtoolbox(medfilt)
    itb.crop_img()
    cropped = itb.cropped
    return cropped

def img_crop_coord(medfilt):
    cropped = medfilt[:, :, :]

    '''
    itb = imgtoolbox(medfilt)
    itb.get_roicoord(scan_note='sungjae')
    itb.cropfromrois()
    cropped = itb.cropped
    '''
    return cropped


def img_reg(cropped, eng, type='stackreg'):
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

    elif type == 'fast':  # 새로 추가된 초고속 변환 (FFT 기반)
        print(f"Registering {len(cropped)} images using fast Phase Correlation...")
        img_aligned = np.zeros_like(cropped, dtype=np.float32)
        img_aligned[0] = cropped[0]

        for i in range(1, len(cropped)):
            # 이전 이미지를 기준으로 현재 이미지의 이동량(shift) 계산
            shift_val, error, diffphase = phase_cross_correlation(
                cropped[i - 1], cropped[i], upsample_factor=10
            )
            # 계산된 이동량만큼 이미지 평행 이동2
            img_aligned[i] = shift(cropped[i], shift_val, order=1, mode='constant', cval=0.0)

            if i % 10 == 0 or i == len(cropped) - 1:
                print(f"Processed {i}/{len(cropped) - 1} images")

    else:  # 기존 stackreg (print 속도 최적화 적용)
        sr = StackReg(StackReg.TRANSLATION)

        def show_progress(current_iteration, end_iteration):
            # 10장 단위로만 출력하여 I/O 병목 현상 완화
            if current_iteration % 10 == 0 or current_iteration == end_iteration:
                print(f"Registering {current_iteration} of {end_iteration} images")

        reg = sr.register_transform_stack(cropped, axis=0, progress_callback=show_progress)
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


def main(base_dirpath, target_folder, peakref, polynomial, fit_num, multi_fit_flag, fit_num_list=None, poly_list=None):
    # 수정 1: target_folder 경로 하나만 전달
    txm_file = txm_io(target_folder)

    cropped = img_crop_coord(txm_file.medfilt)
    img_aligned = img_reg(cropped, txm_file.eng, type='stackreg')

    if multi_fit_flag == 0:
        try:
            fitting = fit(img_aligned, txm_file.eng, polynomial, fit_num, peakref)
        except ValueError:
            # 수정 2: 정의되지 않은 fdir, fn 변수 수정
            with open(base_dirpath + "/error_list.txt", "a+") as File:
                File.write(txm_file.fn + '\n')
            pass
        else:
            data = saving_data(fitting, polynomial, fit_num, data_dir=base_dirpath, fn=txm_file.fn)
            image_info(data, fitting)
            print(f'Scan_id : {data.scan_id}_{data.scan_pos}',
                  f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}',
                  f"\nStd : {fitting.stdev}")

    elif multi_fit_flag == 1:
        if fit_num_list is not None and polynomial == 2:
            for fn_item in fit_num_list:  # 변수명 중복 방지(fit_num -> fn_item)
                try:
                    fitting = fit(img_aligned, txm_file.eng, polynomial, fn_item, peakref)
                except ValueError:
                    with open(base_dirpath + "/error_list.txt", "a+") as File:
                        File.write(txm_file.fn + '\n')
                    pass
                else:
                    data = saving_data(fitting, polynomial, fn_item, data_dir=base_dirpath, fn=txm_file.fn)
                    image_info(data, fitting)
                    print(f'Scan_id : {data.scan_id}_{data.scan_pos}',
                          f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}',
                          f"\nStd : {fitting.stdev}")

        elif poly_list is not None:
            fn_item = 3
            for poly_item in poly_list:  # 변수명 중복 방지(polynomial -> poly_item)
                try:
                    fitting = fit(img_aligned, txm_file.eng, poly_item, fn_item, peakref)
                except ValueError:
                    with open(base_dirpath + "/error_list.txt", "a+") as File:
                        File.write(txm_file.fn + '\n')
                    pass
                else:
                    data = saving_data(fitting, poly_item, fn_item, data_dir=base_dirpath, fn=txm_file.fn)
                    image_info(data, fitting)
                    print(f'Scan_id : {data.scan_id}_{data.scan_pos}',
                          f'\nMean : {np.mean(fitting.peaksite[fitting.peaksite != 0])}',
                          f"\nStd : {fitting.stdev}")


if __name__ == "__main__":
    base_dirpath = "/Users/hwiho/Documents/VS_code/PXM"

    target_folders = []
    for it in os.scandir(base_dirpath):
        if it.is_dir():
            target_folders.append(it.path)

    # 폴더를 이름순으로 정렬
    target_folders = sorted(target_folders)

    # 폴더 목록을 번호와 함께 출력
    print("="*40)
    print("📂 [처리 가능한 폴더 목록]")
    for i, folder in enumerate(target_folders):
        print(f"[{i}] {os.path.basename(folder)}")
    print("="*40)

    # 시작할 폴더 번호 하나만 입력받기
    user_input = input("시작할 폴더의 번호를 입력하세요 (입력한 번호부터 끝까지 실행 / 전체 실행은 엔터): ")

    if user_input.strip() == "":
        start_idx = 0  # 엔터만 쳤을 때는 0번(처음)부터
    else:
        try:
            start_idx = int(user_input.strip())
        except ValueError:
            print("올바른 숫자를 입력해 주세요. 프로그램을 종료합니다.")
            exit()

    # 핵심: 입력받은 인덱스(start_idx)부터 리스트의 끝까지 가져오기
    selected_folders = target_folders[start_idx:]

    print(f"\n🚀 [{start_idx}]번 폴더부터 마지막까지 총 {len(selected_folders)}개의 데이터 처리를 시작합니다...\n")

    for folder in selected_folders:
        print(f"\nProcessing directory: {folder}")
        main(base_dirpath, folder, peakref=[8368.2, 8370.6], polynomial=2, fit_num=3, multi_fit_flag=0,
             fit_num_list=[3], poly_list=[])
