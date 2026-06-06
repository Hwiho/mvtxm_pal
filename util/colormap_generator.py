import numpy as np
from skimage import io, color
import os
from glob import glob


def hsvcolormap_from_image(image_path, peakref=(0, 1), hsvconstant=(1, 1, 1), thickness=None):
    # 이미지 로드
    peaksite = io.imread(image_path).astype(np.float32)
    img_shape = peaksite.shape

    # HSV 칼라맵 배열 초기화
    hsv = np.zeros((img_shape[0], img_shape[1], 3), dtype="float32")

    # 농도 계산
    conc = (peaksite - peakref[0]) / (peakref[1] - peakref[0])
    conc = np.clip(conc, 0, 1)  # 0과 1 사이로 제한

    # Hue 설정
    if hsvconstant[0] != 0:
        hsv[:, :, 0] = conc * hsvconstant[0]
    else:
        hsv[:, :, 0] = conc / np.max(conc)

    # Saturation 설정
    hsv[:, :, 1] = np.ones((img_shape[0], img_shape[1]))

    # Value 설정
    if thickness is None:
        thickness = np.ones((img_shape[0], img_shape[1]), dtype="float32")
    if hsvconstant[2] != 0:
        hsv[:, :, 2] = thickness * hsvconstant[2]
    else:
        hsv[:, :, 2] = thickness / np.max(thickness)
    hsv[:, :, 2] *= 1  # 밝기 스케일 조정 (옵션)

    # HSV를 RGB로 변환
    rgb = color.hsv2rgb(hsv)

    # 평균 밝기 계산
    brightness = np.mean(hsv[:, :, 2])

    # 이미지 파일과 동일한 폴더에 RGB 이미지 저장
    save_path = os.path.join(os.path.dirname(image_path),
                             f"{os.path.splitext(os.path.basename(image_path))[0]}_colormap_rgb.png")
    io.imsave(save_path, (rgb * 255).astype(np.uint8))  # RGB 이미지를 0-255 범위로 저장

    print(f"RGB 컬러맵이 {save_path}에 저장되었습니다.")
    return save_path, brightness


def process_all_tif_in_folder(folder_path, peakref=(0, 1), hsvconstant=(1, 1, 1), thickness=None):
    # 폴더 내 모든 .tif 파일 검색
    tif_files = glob(os.path.join(folder_path, "*.tif"))

    for tif_file in tif_files:
        hsvcolormap_from_image(tif_file, peakref, hsvconstant, thickness)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_all_tif_in_folder(sys.argv[1])
