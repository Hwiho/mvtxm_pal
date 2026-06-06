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
ALGO_CHOICE = 'phase'  # <--- 여기서 변경하세요.


def get_transform_matrix(ref_mask, curr_mask, algo='stackreg'):
    """선택된 알고리즘에 따라 변환 행렬(Translation)을 계산합니다."""

    if algo == 'stackreg':
        sr = StackReg(StackReg.TRANSLATION)
        return sr.register(ref_mask, curr_mask)

    elif algo == 'orb':
        ref_8bit = (ref_mask * 255).astype(np.uint8)
        curr_8bit = (curr_mask * 255).astype(np.uint8)

        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(ref_8bit, None)
        kp2, des2 = orb.detectAndCompute(curr_8bit, None)

        if des1 is None or des2 is None:
            raise ValueError("특징점을 찾을 수 없습니다.")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            raise ValueError("매칭점이 너무 적습니다.")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)

        shifts = src_pts - dst_pts
        mean_shift = np.mean(shifts, axis=0)[0]
        return np.float32([[1, 0, mean_shift[0]], [0, 1, mean_shift[1]]])

    elif algo == 'phase':
        shift, error, diffphase = phase_cross_correlation(ref_mask, curr_mask, upsample_factor=10)
        ty, tx = shift
        return np.float32([[1, 0, -tx], [0, 1, -ty]])

    else:
        raise ValueError("알 수 없는 알고리즘 이름입니다.")


def batch_registration_multi(input_folder, reg_folder, mask_folder, threshold=500, algo='stackreg'):
    file_list = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    if not file_list:
        file_list = sorted(glob.glob(os.path.join(input_folder, "*.tiff")))

    if not file_list:
        print("파일을 찾을 수 없습니다.")
        return

    for folder in [reg_folder, mask_folder]:
        if not os.path.exists(folder): os.makedirs(folder)

    # 기준 이미지 설정
    ref_img = tf.imread(file_list[0])
    ref_mask = (ref_img > threshold).astype(float)

    status_log = []
    first_name = os.path.basename(file_list[0])

    # 첫 이미지 저장
    tf.imwrite(os.path.join(reg_folder, f"reg_{first_name}"), ref_img.astype(np.float32), imagej=True)
    tf.imwrite(os.path.join(mask_folder, f"mask_{first_name}"), (ref_mask * 255).astype(np.uint8), imagej=True)
    status_log.append([first_name, True])

    print(f"[{algo}] 알고리즘으로 {len(file_list)}개 파일 처리를 시작합니다...")

    # ==========================================
    # [Crop 추가 1] 공통 유효 영역(Bounding Box) 추적용 변수 초기화
    # ==========================================
    canvas_h, canvas_w = ref_img.shape
    global_x_min, global_y_min = 0, 0
    global_x_max, global_y_max = canvas_w, canvas_h

    for i in range(1, len(file_list)):
        current_file = file_list[i]
        file_name = os.path.basename(current_file)
        curr_img = tf.imread(current_file)
        curr_mask = (curr_img > threshold).astype(float)

        try:
            # 1. 행렬 계산
            tmat = get_transform_matrix(ref_mask, curr_mask, algo=algo)
            if tmat.shape == (2, 3):
                tmat = np.vstack([tmat, [0, 0, 1]])

            # 2. 이미지 & 마스크 변환
            reg_img_warped = warp(curr_img, tmat, order=0, mode='constant', cval=0.0, preserve_range=True)
            reg_img_float = reg_img_warped.astype(np.float32)
            tf.imwrite(os.path.join(reg_folder, f"reg_{file_name}"), reg_img_float, imagej=True)

            curr_mask_uint8 = (curr_mask * 255).astype(np.uint8)
            reg_mask_warped = warp(curr_mask_uint8, tmat, order=0, mode='constant', cval=0.0, preserve_range=True)
            reg_mask = reg_mask_warped.astype(np.uint8)
            tf.imwrite(os.path.join(mask_folder, f"mask_{file_name}"), reg_mask, imagej=True)

            # ==========================================
            # [Crop 추가 2] 가상 캔버스를 통한 공통 여백(교집합) 계산
            # ==========================================
            # 화면 전체가 1로 채워진 가상 캔버스를 똑같이 이동시킵니다.
            dummy_canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8)
            warped_canvas = warp(dummy_canvas, tmat, order=0, mode='constant', cval=0.0, preserve_range=True).astype(
                np.uint8)

            # 이동 후 남아있는 유효 영역(1인 부분)의 경계 좌표를 찾습니다.
            coords = cv2.findNonZero(warped_canvas)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                global_x_min = max(global_x_min, x)
                global_y_min = max(global_y_min, y)
                global_x_max = min(global_x_max, x + w)
                global_y_max = min(global_y_max, y + h)

            print(f"[{i}/{len(file_list) - 1}] 성공: {file_name}")
            status_log.append([file_name, True])

        except Exception as e:
            print(f"[{i}] 실패: {file_name} - 사유: {e}")
            status_log.append([file_name, False])

    # 상태 로그 CSV 저장
    with open(os.path.join(reg_folder, "registration_status.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "image_registered"])
        writer.writerows(status_log)

    print(f"\n✅ 1단계: 정합 작업 완료! (알고리즘: {algo})")

    # ==========================================
    # [Crop 추가 3] 산출된 공통 영역으로 모든 결과물 일괄 자르기
    # ==========================================
    if global_x_min < global_x_max and global_y_min < global_y_max:
        print(f"\n▶ 2단계: 공통 여백 자르기(Crop) 시작")
        print(f"   - 공통 영역 X: {global_x_min} ~ {global_x_max}, Y: {global_y_min} ~ {global_y_max}")

        # 저장된 reg_folder와 mask_folder의 파일들을 다시 읽어서 잘라내고 덮어씁니다.
        for folder in [reg_folder, mask_folder]:
            for fname in os.listdir(folder):
                if fname.endswith(('.tif', '.tiff')):
                    fpath = os.path.join(folder, fname)
                    img_to_crop = tf.imread(fpath)

                    # 공통 영역만큼 자르기
                    cropped_img = img_to_crop[global_y_min:global_y_max, global_x_min:global_x_max]

                    # 덮어쓰기 저장
                    tf.imwrite(fpath, cropped_img, imagej=True)

        print("✅ 2단계: Crop 및 덮어쓰기 완료! 모든 검은 여백이 제거되었습니다.\n")
    else:
        print("\n⚠️ 경고: 이미지들의 이동 폭이 너무 커서 겹치는 공통 영역이 없습니다. Crop을 생략합니다.\n")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    input_dir = filedialog.askdirectory(title="원본 폴더 선택")

    if input_dir:
        batch_registration_multi(
            input_dir,
            input_dir + f'_reg_{ALGO_CHOICE}',
            input_dir + f'_masks_{ALGO_CHOICE}',
            threshold=500,
            algo=ALGO_CHOICE
        )