import os
import numpy as np
from skimage import io, img_as_float, img_as_ubyte, color
from skimage.exposure import match_histograms
from tqdm import tqdm

def load_image(path):
    return img_as_float(io.imread(path))  # RGB [0,1]

def save_image(image, path):
    io.imsave(path, img_as_ubyte(np.clip(image, 0, 1)))

def histogram_match_luminance(folder_path, reference_filename, output_folder=None):
    filenames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])

    # Load reference and convert to YUV
    reference_rgb = load_image(os.path.join(folder_path, reference_filename))
    reference_yuv = color.rgb2yuv(reference_rgb)
    reference_y = reference_yuv[:, :, 0]  # Luminance only

    # Output folder
    if output_folder is None:
        output_folder = os.path.join(folder_path, "matched_luminance")
    os.makedirs(output_folder, exist_ok=True)

    print(f"[INFO] Matching luminance to reference: {reference_filename}")
    for fname in tqdm(filenames):
        img_rgb = load_image(os.path.join(folder_path, fname))
        img_yuv = color.rgb2yuv(img_rgb)

        # Match luminance (Y channel only)
        matched_y = match_histograms(img_yuv[:, :, 0], reference_y, channel_axis=None)
        img_yuv[:, :, 0] = matched_y

        # Convert back to RGB
        matched_rgb = color.yuv2rgb(img_yuv)
        matched_rgb = np.clip(matched_rgb, 0, 1)

        save_path = os.path.join(output_folder, fname)
        save_image(matched_rgb, save_path)

    print(f"[DONE] Luminance-matched images saved to: {output_folder}")


folder_path = r"D:\colormap_wo"      # PNG 이미지들이 있는 폴더
reference_filename = r"D:\colormap_wo\colormap_id_pouc9_insitu_033C_exp25_acc7_wopressMULTI003_pos_00.png"    # 기준 이미지
histogram_match_luminance(folder_path, reference_filename)