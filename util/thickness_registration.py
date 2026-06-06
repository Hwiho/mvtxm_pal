import os
import glob
import numpy as np
import tifffile as tf
import cv2
import tkinter as tk
from tkinter import filedialog
import csv
from pystackreg import StackReg
from skimage.registration import phase_cross_correlation
from skimage.transform import warp

# ==========================================
# 알고리즘 선택 설정: 'stackreg', 'orb', 'phase' 중 하나를 입력하세요.
# ==========================================
ALGO_CHOICE = 'phase'


def get_transform_matrix(ref_mask, curr_mask, algo='stackreg'):
    if algo == 'stackreg':
        sr = StackReg(StackReg.TRANSLATION)
        return sr.register(ref_mask, curr_mask)
    elif algo == 'phase':
        shift, error, diffphase = phase_cross_correlation(ref_mask, curr_mask, upsample_factor=10)
        ty, tx = shift
        return np.float32([[1, 0, -tx], [0, 1, -ty]])
    else:
        raise ValueError("알 수 없는 알고리즘 이름입니다.")


def batch_registration_combined(peaksite_folder, thickness_folder, ref_file_path, output_base, threshold=500,
                                algo='phase'):
    peaksite_list = sorted(glob.glob(os.path.join(peaksite_folder, "*peaksite.tif")))
    if not peaksite_list:
        print("선택한 폴더에서 peaksite.tif 파일을 찾을 수 없습니다.")
        return

    # 사용자가 선택한 Reference 파일을 리스트의 맨 앞으로 이동 (기준점 설정)
    if ref_file_path in peaksite_list:
        peaksite_list.remove(ref_file_path)
    peaksite_list.insert(0, ref_file_path)

    reg_peak_folder = output_base + '_peaksite'
    reg_thick_folder = output_base + '_thickness'
    mask_folder = output_base + '_masks'

    for folder in [reg_peak_folder, reg_thick_folder, mask_folder]:
        if not os.path.exists(folder): os.makedirs(folder)

    # ==========================================
    # 1. Reference(기준) 이미지 설정 및 저장
    # ==========================================
    ref_peak_file = peaksite_list[31]
    ref_peak_img = tf.imread(ref_peak_file)
    ref_mask = (ref_peak_img > threshold).astype(float)

    first_peak_name = os.path.basename(ref_peak_file)
    first_thick_name = first_peak_name.replace('peaksite', 'thickness')
    ref_thick_file = os.path.join(thickness_folder, first_thick_name)

    tf.imwrite(os.path.join(reg_peak_folder, f"reg_{first_peak_name}"), ref_peak_img.astype(np.float32), imagej=True)
    tf.imwrite(os.path.join(mask_folder, f"mask_{first_peak_name}"), (ref_mask * 255).astype(np.uint8), imagej=True)

    if os.path.exists(ref_thick_file):
        ref_thick_img = tf.imread(ref_thick_file)
        tf.imwrite(os.path.join(reg_thick_folder, f"reg_{first_thick_name}"), ref_thick_img.astype(np.float32),
                   imagej=True)

    status_log = [[first_peak_name, True, "Reference"]]
    print(f"\n▶ [{algo}] 알고리즘으로 총 {len(peaksite_list)}쌍의 파일 정합 처리를 시작합니다...")
    print(f"▶ 기준(Reference) 파일: {first_peak_name}")

    # ==========================================
    # [Crop 초정밀 로직] 원본 데이터 영역 추적
    # ==========================================
    canvas_h, canvas_w = ref_peak_img.shape

    # 1. Reference 이미지의 '진짜 데이터(입자들)'가 있는 경계선(Bounding Box)을 먼저 찾습니다.
    ref_data_coords = cv2.findNonZero((ref_peak_img > 0).astype(np.uint8))
    if ref_data_coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(ref_data_coords)
        global_x_min, global_y_min = rx, ry
        global_x_max, global_y_max = rx + rw, ry + rh
    else:
        global_x_min, global_y_min, global_x_max, global_y_max = 0, 0, canvas_w, canvas_h

    # 2. 이 영역만큼을 1로 채운 '가상 마스크'를 만듭니다.
    tracking_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    tracking_canvas[global_y_min:global_y_max, global_x_min:global_x_max] = 1

    # ==========================================
    # 2. 나머지 파일들 순회하며 정합 수행
    # ==========================================
    for i in range(1, len(peaksite_list)):
        curr_peak_file = peaksite_list[i]
        peak_name = os.path.basename(curr_peak_file)
        thick_name = peak_name.replace('peaksite', 'thickness')
        curr_thick_file = os.path.join(thickness_folder, thick_name)

        curr_peak_img = tf.imread(curr_peak_file)
        curr_mask = (curr_peak_img > threshold).astype(float)

        try:
            tmat = get_transform_matrix(ref_mask, curr_mask, algo=algo)
            if tmat.shape == (2, 3):
                tmat = np.vstack([tmat, [0, 0, 1]])

            reg_peak = warp(curr_peak_img, tmat, order=0, mode='constant', cval=0.0, preserve_range=True)
            tf.imwrite(os.path.join(reg_peak_folder, f"reg_{peak_name}"), reg_peak.astype(np.float32), imagej=True)

            reg_mask_warped = warp((curr_mask * 255).astype(np.uint8), tmat, order=0, mode='constant', cval=0.0,
                                   preserve_range=True)
            tf.imwrite(os.path.join(mask_folder, f"mask_{peak_name}"), reg_mask_warped.astype(np.uint8), imagej=True)

            if os.path.exists(curr_thick_file):
                curr_thick_img = tf.imread(curr_thick_file)
                reg_thick = warp(curr_thick_img, tmat, order=0, mode='constant', cval=0.0, preserve_range=True)
                tf.imwrite(os.path.join(reg_thick_folder, f"reg_{thick_name}"), reg_thick.astype(np.float32),
                           imagej=True)

            # ==========================================
            # 가상 마스크를 함께 이동시켜 교집합(공통 영역) 축소
            # ==========================================
            warped_tracking = warp(tracking_canvas, tmat, order=0, mode='constant', cval=0.0,
                                   preserve_range=True).astype(np.uint8)
            coords = cv2.findNonZero(warped_tracking)

            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                # 이동할 때마다 가장 타이트하게 공통 영역을 좁혀 나갑니다.
                global_x_min = max(global_x_min, x)
                global_y_min = max(global_y_min, y)
                global_x_max = min(global_x_max, x + w)
                global_y_max = min(global_y_max, y + h)

            print(f"[{i}/{len(peaksite_list) - 1}] 정합 완료: {peak_name}")
            status_log.append([peak_name, True, "Success"])

        except Exception as e:
            print(f"[{i}] 실패: {peak_name} - 사유: {e}")
            status_log.append([peak_name, False, str(e)])

    with open(os.path.join(output_base + f"_{algo}_log.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "success", "notes"])
        writer.writerows(status_log)

    print(f"\n✅ 1단계: 정합 작업 완료!")

    # ==========================================
    # 3. 산출된 공통 영역으로 3개 폴더 일괄 자르기
    # ==========================================
    if global_x_min < global_x_max and global_y_min < global_y_max:
        print(f"\n▶ 2단계: 빈 여백 완전 제거(Crop) 시작")
        print(f"   - 타이트 컷팅 범위 X: {global_x_min} ~ {global_x_max}, Y: {global_y_min} ~ {global_y_max}")

        for target_folder in [reg_peak_folder, reg_thick_folder, mask_folder]:
            for fname in os.listdir(target_folder):
                if fname.endswith(('.tif', '.tiff')):
                    fpath = os.path.join(target_folder, fname)
                    img_to_crop = tf.imread(fpath)

                    cropped_img = img_to_crop[global_y_min:global_y_max, global_x_min:global_x_max]
                    tf.imwrite(fpath, cropped_img, imagej=True)

        print("✅ 2단계: 불필요한 검은 여백 제거 완료!\n")
    else:
        print("\n⚠️ 경고: 공통 영역을 찾을 수 없습니다. 자르기를 생략합니다.\n")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("1. [peaksite.tif] 폴더를 선택해 주세요.")
    dir_peaksite = filedialog.askdirectory(title="1. peaksite 폴더 선택")
    if not dir_peaksite: exit()

    print("2. [thickness.tif] 폴더를 선택해 주세요.")
    dir_thickness = filedialog.askdirectory(title="2. thickness 폴더 선택")
    if not dir_thickness: exit()

    # ==========================================
    # [신규 추가] Reference 직접 선택 기능
    # ==========================================
    print("3. 기준(Reference)이 될 특정 peaksite 이미지를 직접 선택해 주세요.")
    ref_file = filedialog.askopenfilename(
        title="3. 기준(Reference) peaksite 이미지 선택",
        initialdir=dir_peaksite,
        filetypes=[("TIFF 파일", "*.tif *.tiff")]
    )
    if not ref_file:
        print("Reference 파일 선택이 취소되었습니다.")
        exit()

    print(f"\n- Peaksite 경로: {dir_peaksite}")
    print(f"- Thickness 경로: {dir_thickness}")
    print(f"- 기준(Ref) 파일: {os.path.basename(ref_file)}\n")

    output_base_path = os.path.join(dir_peaksite, f'Registered_{ALGO_CHOICE}')

    batch_registration_combined(
        peaksite_folder=dir_peaksite,
        thickness_folder=dir_thickness,
        ref_file_path=ref_file,  # 선택한 Ref 파일 경로 전달
        output_base=output_base_path,
        threshold=500,
        algo=ALGO_CHOICE
    )