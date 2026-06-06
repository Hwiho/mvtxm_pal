import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patheffects as patheffects


class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Master Mask Creator (v4.0 - ROI Target Mode)")
        self.root.geometry("1200x850")

        self.image = None
        self.master_mask = None
        self.roi = None  # (x, y, w, h) 형태로 선택된 영역 저장

        # ==========================================
        # UI 레이아웃 구성 (왼쪽 컨트롤 패널)
        # ==========================================
        control_frame = tk.Frame(root, width=320, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.btn_open = tk.Button(control_frame, text="TIFF 이미지 불러오기", command=self.load_image, height=2, bg="#e0e0e0")
        self.btn_open.pack(fill=tk.X, pady=(0, 15))

        # ==========================================
        # [신규] 부분 영역(ROI) 선택 컨트롤
        # ==========================================
        tk.Label(control_frame, text="[ 타겟 영역 설정 ]", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        roi_btn_frame = tk.Frame(control_frame)
        roi_btn_frame.pack(fill=tk.X, pady=(0, 15))

        self.btn_roi = tk.Button(roi_btn_frame, text="🎯 특정 영역 지정", command=self.select_roi, bg="#FF9800", fg="white")
        self.btn_roi.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.btn_clear_roi = tk.Button(roi_btn_frame, text="전체 이미지로 복귀", command=self.clear_roi)
        self.btn_clear_roi.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 전처리 파라미터
        tk.Label(control_frame, text="[ 전처리 (Pre-processing) ]", font=("Arial", 10, "bold")).pack(anchor=tk.W,
                                                                                                  pady=(0, 5))
        tk.Label(control_frame, text="1. CLAHE 대조 제한치 (1.0 ~ 10.0):").pack(anchor=tk.W)
        self.entry_clahe = tk.Entry(control_frame)
        self.entry_clahe.insert(0, "2.0")
        self.entry_clahe.pack(fill=tk.X, pady=(0, 5))

        tk.Label(control_frame, text="2. 블러 크기 (3, 5, 7... 홀수):").pack(anchor=tk.W)
        self.entry_blur = tk.Entry(control_frame)
        self.entry_blur.insert(0, "5")
        self.entry_blur.pack(fill=tk.X, pady=(0, 5))

        tk.Label(control_frame, text="3. 이진화 보정치 (-50 ~ +50):").pack(anchor=tk.W)
        self.entry_thresh_offset = tk.Entry(control_frame)
        self.entry_thresh_offset.insert(0, "-10")
        self.entry_thresh_offset.pack(fill=tk.X, pady=(0, 15))

        # 분할 파라미터
        tk.Label(control_frame, text="[ 분할 (Segmentation) ]", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        tk.Label(control_frame, text="4. 워터쉐드 최소 거리 (min_distance):").pack(anchor=tk.W)
        self.entry_min_dist = tk.Entry(control_frame)
        self.entry_min_dist.insert(0, "15")
        self.entry_min_dist.pack(fill=tk.X, pady=(0, 5))

        tk.Label(control_frame, text="5. 노이즈 필터 최소 면적 (Area):").pack(anchor=tk.W)
        self.entry_min_area = tk.Entry(control_frame)
        self.entry_min_area.insert(0, "30")
        self.entry_min_area.pack(fill=tk.X, pady=(0, 15))

        self.btn_apply = tk.Button(control_frame, text="▶ 설정 적용 및 분석 실행 (Enter)", command=self.update_view, height=2,
                                   bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.btn_apply.pack(fill=tk.X, pady=(0, 10))
        self.root.bind('<Return>', lambda event: self.update_view())

        self.btn_save = tk.Button(control_frame, text="💾 마스터 마스크 저장 (.tif)", command=self.save_master_mask, height=2,
                                  bg="#2196F3", fg="white")
        self.btn_save.pack(fill=tk.X, pady=(0, 15))

        self.result_label = tk.Label(control_frame, text="탐지된 입자 수: 0개", font=("Arial", 12, "bold"), fg="blue")
        self.result_label.pack(anchor=tk.W, pady=5)

        # ==========================================
        # 오른쪽 이미지 뷰어 패널
        # ==========================================
        self.view_frame = tk.Frame(root)
        self.view_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.view_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="TIFF 이미지 선택",
                                               filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
        if file_path:
            try:
                import tifffile as tf
                raw_img = tf.imread(file_path)
                img_min, img_max = np.min(raw_img), np.max(raw_img)
                if img_max > img_min:
                    normalized_img = (raw_img - img_min) / (img_max - img_min) * 255.0
                else:
                    normalized_img = raw_img
                self.image = normalized_img.astype(np.uint8)
                self.roi = None  # 이미지 새로 불러오면 ROI 초기화
                self.update_view()
            except Exception as e:
                messagebox.showerror("파일 읽기 에러", f"이미지를 불러오는 중 문제가 발생했습니다:\n{e}")

    # ==========================================
    # [신규] 영역 지정 함수 (OpenCV 활용)
    # ==========================================
    def select_roi(self):
        if self.image is None:
            messagebox.showwarning("경고", "먼저 이미지를 불러오세요.")
            return

        messagebox.showinfo("안내", "마우스로 원하는 영역을 드래그하여 박스를 그리고,\n키보드의 [Enter] 또는 [Space]를 누르면 확정됩니다.\n취소하려면 [C]를 누르세요.")

        # OpenCV 내장 ROI 선택 창 띄우기
        window_name = "Select ROI (Enter: Confirm, C: Cancel)"
        r = cv2.selectROI(window_name, self.image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        # width와 height가 0보다 클 때만 ROI 저장
        if r[2] > 0 and r[3] > 0:
            self.roi = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            self.update_view()

    def clear_roi(self):
        self.roi = None
        self.update_view()

    def update_view(self):
        if self.image is None: return

        try:
            clahe_clip = float(self.entry_clahe.get())
            blur_size = int(self.entry_blur.get())
            if blur_size % 2 == 0:
                blur_size += 1
                self.entry_blur.delete(0, tk.END)
                self.entry_blur.insert(0, str(blur_size))
            thresh_offset = int(self.entry_thresh_offset.get())
            min_dist = int(self.entry_min_dist.get())
            min_area = int(self.entry_min_area.get())
        except ValueError:
            messagebox.showerror("Error", "올바른 숫자를 입력해주세요.")
            return

        # ==========================================
        # [핵심] 처리할 타겟 이미지 분리
        # ==========================================
        if self.roi is not None:
            x, y, w, h = self.roi
            target_img = self.image[y:y + h, x:x + w]
        else:
            target_img = self.image

        # 타겟 이미지에 대해서만 연산 수행
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        clahe_img = clahe.apply(target_img)
        blurred_img = cv2.GaussianBlur(clahe_img, (blur_size, blur_size), 0)

        otsu_thresh, _ = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_thresh = max(0, min(255, int(otsu_thresh + thresh_offset)))
        _, binary = cv2.threshold(blurred_img, final_thresh, 255, cv2.THRESH_BINARY)

        distance = ndimage.distance_transform_edt(binary)
        local_max_coords = peak_local_max(distance, min_distance=min_dist, labels=binary)
        markers = np.zeros(distance.shape, dtype=bool)
        markers[tuple(local_max_coords.T)] = True
        markers, _ = ndimage.label(markers)
        labels = watershed(-distance, markers, mask=binary)

        props = regionprops(labels)
        valid_labels = [p.label for p in props if p.area >= min_area]

        remap_array = np.zeros(labels.max() + 1, dtype=np.int32)
        for new_idx, old_label in enumerate(valid_labels, start=1):
            remap_array[old_label] = new_idx

        local_mask = remap_array[labels]
        particle_count = len(valid_labels)

        # ==========================================
        # [핵심] 연산된 작은 마스크를 원본 사이즈의 빈 캔버스에 붙여넣기
        # ==========================================
        self.master_mask = np.zeros_like(self.image, dtype=np.int32)

        # 화면 출력을 위한 원본 사이즈 CLAHE 이미지 (배경은 원본, 박스 안은 전처리된 이미지)
        display_bg = self.image.copy()

        if self.roi is not None:
            x, y, w, h = self.roi
            self.master_mask[y:y + h, x:x + w] = local_mask
            display_bg[y:y + h, x:x + w] = clahe_img
        else:
            self.master_mask = local_mask
            display_bg = clahe_img

        # 6. 결과 시각화
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.imshow(display_bg, cmap='gray')
        self.ax1.set_title(f"Target Mode (Thresh: {final_thresh})")
        self.ax1.axis('off')

        # ROI 박스 그려주기 (시각적 피드백)
        if self.roi is not None:
            x, y, w, h = self.roi
            import matplotlib.patches as patches
            rect1 = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='orange', facecolor='none', linestyle='--')
            rect2 = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='orange', facecolor='none', linestyle='--')
            self.ax1.add_patch(rect1)
            self.ax2.add_patch(rect2)

        masked_res = np.ma.masked_where(self.master_mask == 0, self.master_mask)
        self.ax2.imshow(display_bg, cmap='gray', alpha=0.5)
        self.ax2.imshow(masked_res, cmap='nipy_spectral', alpha=0.8)
        self.ax2.set_title("Segmented Result")
        self.ax2.axis('off')

        if particle_count > 0:
            # 좌표 보정 (ROI 내부에서 찾은 좌표를 전체 캔버스 좌표로 변환)
            offset_y = self.roi[1] if self.roi is not None else 0
            offset_x = self.roi[0] if self.roi is not None else 0

            centers = [p.centroid for p in props if p.area >= min_area]
            for new_idx, (cy, cx) in enumerate(centers, start=1):
                y_pos, x_pos = cy + offset_y, cx + offset_x
                text = self.ax2.text(x_pos, y_pos, str(new_idx), color='white', fontsize=7, ha='center',
                                     fontweight='bold')
                text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])

        self.canvas.draw()
        self.result_label.config(text=f"탐지된 입자 수: {particle_count}개")

    def save_master_mask(self):
        if self.master_mask is None: return
        file_path = filedialog.asksaveasfilename(defaultextension=".tif",
                                                 filetypes=[("TIFF 파일", "*.tif"), ("Numpy", "*.npy")])
        if file_path:
            if file_path.endswith('.npy'):
                np.save(file_path, self.master_mask)
            else:
                cv2.imwrite(file_path, self.master_mask.astype(np.uint16))
            messagebox.showinfo("저장 완료", f"마스터 마스크가 저장되었습니다.\n인덱스 범위: 1 ~ {self.master_mask.max()}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()