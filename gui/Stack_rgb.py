import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from skimage import color
from skimage.filters import gaussian, median
from skimage.morphology import disk
import tkinter as tk
from tkinter import filedialog


def apply_filter(image, filter_type, **kwargs):
    if filter_type == 'gaussian':
        return gaussian(image, **kwargs)
    elif filter_type == 'median':
        return median(image, **kwargs)
    else:
        return image


def hsvcolormap(peaksite, peakref, hsvconstant, thickness, mask, value=None):
    hsv = np.zeros((peaksite.shape[0], peaksite.shape[1], 3), dtype="float32")

    conc = (peaksite - peakref[0]) / (peakref[1] - peakref[0])
    conc[conc < 0] = 0
    conc[conc > 1] = 1

    conc = conc * mask

    if hsvconstant[0] != 0:
        hsv[:, :, 0] = conc * hsvconstant[0]
    else:
        maxc = np.max(conc)
        hsv[:, :, 0] = conc / maxc if maxc > 0 else conc

    hsv[:, :, 1] = np.ones_like(peaksite)

    if value is not None:
        hsvconstant[2] = value

    if hsvconstant[2] != 0:
        hsv[:, :, 2] = thickness * hsvconstant[2]
    else:
        max_t = np.max(thickness)
        hsv[:, :, 2] = thickness / max_t if max_t > 0 else thickness

    hsv[:, :, 2] = hsv[:, :, 2] * mask

    rgb = color.hsv2rgb(hsv)
    rgb = rgb * mask[..., np.newaxis]

    if np.sum(mask) > 0:
        brightness = np.mean(hsv[:, :, 2][mask > 0])
    else:
        brightness = 0

    return hsv, rgb, brightness


def batch_rgb_conversion(base_folder, peakref, hsvconstant):
    # ==========================================
    # 수정된 부분: 선택한 폴더 내부에서 3개의 결과 폴더를 확실하게 찾음
    # ==========================================
    peaksite_dir = glob.glob(os.path.join(base_folder, "*_peaksite"))
    thickness_dir = glob.glob(os.path.join(base_folder, "*_thickness"))
    mask_dir = glob.glob(os.path.join(base_folder, "*_masks"))

    if not (peaksite_dir and thickness_dir and mask_dir):
        print("❌ 선택하신 폴더 내부에 정합된 하위 폴더(_peaksite, _thickness, _masks)들이 없습니다.")
        print("경로를 다시 확인해 주세요.")
        return

    # 리스트에서 첫 번째 문자열 추출
    peaksite_dir = peaksite_dir[0]
    thickness_dir = thickness_dir[0]
    mask_dir = mask_dir[0]

    # 출력 폴더 생성
    output_dir = os.path.join(base_folder, "RGB_Output")
    os.makedirs(output_dir, exist_ok=True)

    peaksite_files = sorted(glob.glob(os.path.join(peaksite_dir, "*.tif")))

    if not peaksite_files:
        print("처리할 파일이 없습니다.")
        return

    rgb_images = []

    filter_type1 = 'median'
    filter_type2 = 'no'
    filter_params1 = {'footprint': disk(3)}
    filter_params2 = {}

    print(f"총 {len(peaksite_files)}개의 이미지 변환을 시작합니다...")

    for i, peak_file in enumerate(peaksite_files):
        peak_filename = os.path.basename(peak_file)

        thick_filename = peak_filename.replace('peaksite', 'thickness')
        mask_filename = peak_filename.replace('reg_', 'mask_')

        thick_file = os.path.join(thickness_dir, thick_filename)
        mask_file = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(thick_file) or not os.path.exists(mask_file):
            print(f"⚠️ 건너뜀 [{i + 1}]: {peak_filename} (짝꿍 thickness 또는 mask 파일이 없음)")
            continue

        peaksite = imread(peak_file)
        thickness = imread(thick_file)
        mask_img = imread(mask_file)

        if mask_img.max() > 1:
            mask_img = (mask_img / 255.0).astype(float)

        peaksite_filtered = apply_filter(peaksite, filter_type1, **filter_params1)
        thickness_filtered = apply_filter(thickness, filter_type2, **filter_params2)

        _, rgb, _ = hsvcolormap(
            peaksite_filtered,
            peakref,
            hsvconstant,
            thickness_filtered,
            mask_img
        )

        rgb = np.clip(rgb, 0, 1)

        individual_output_path = os.path.join(output_dir, f"rgb_{peak_filename.replace('.tif', '.png')}")
        plt.imsave(individual_output_path, rgb)

        rgb_images.append((rgb * 255).astype(np.uint8))
        print(f"[{i + 1}/{len(peaksite_files)}] 완료: {peak_filename}")

    if rgb_images:
        output_stack_path = os.path.join(output_dir, "Combined_RGB_Stack.tif")
        imwrite(output_stack_path, np.array(rgb_images), photometric='rgb')
        print(f"\n🎉 작업 완료! \n- 결과물 저장 폴더: {output_dir}")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("정합된 3개의 폴더(_peaksite, _thickness, _masks)가 들어있는 [원본 폴더]를 선택해 주세요.")
    base_folder = filedialog.askdirectory(title="원본 폴더 선택 (결과 폴더들이 있는 곳)")

    if base_folder:
        # 사용자 설정 값
        USER_PEAKREF = [8368.2, 8370.6]
        USER_HSVCONSTANT = np.array((1 / 3, 1, 0))

        batch_rgb_conversion(base_folder, USER_PEAKREF, USER_HSVCONSTANT)