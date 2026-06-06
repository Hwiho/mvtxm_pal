import h5py
import tifffile
import os
import glob
import tkinter as tk
from tkinter import filedialog

# 1. tkinter 창 숨기기 (메인 빈 창이 뜨는 것을 방지)
root = tk.Tk()
root.withdraw()

# 2. 폴더 선택 GUI 팝업 띄우기
print("폴더 선택 창을 띄웁니다. 변환할 HDF5 파일들이 있는 폴더를 선택해 주세요...")
input_dir = filedialog.askdirectory(title="HDF5 파일들이 있는 폴더를 선택하세요")

# 사용자가 취소를 누르거나 폴더를 선택하지 않은 경우 프로그램 종료
if not input_dir:
    print("폴더 선택이 취소되었습니다. 프로그램을 종료합니다.")
    exit()

print(f"\n선택된 경로: {input_dir}")

# 3. TIF 파일을 저장할 별도의 하위 폴더 생성 ('tif_output' 폴더)
output_dir = os.path.join(input_dir, 'tif_output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 4. 입력된 폴더 내의 모든 .h5 및 .hdf5 파일 찾기
h5_files = glob.glob(os.path.join(input_dir, '*.h5')) + glob.glob(os.path.join(input_dir, '*.hdf5'))

if not h5_files:
    print("선택한 폴더에 HDF5 파일이 존재하지 않습니다.")
else:
    print(f"총 {len(h5_files)}개의 HDF5 파일을 찾았습니다. 변환을 시작합니다...\n")

    # 5. 각 파일에 대해 변환 작업 반복 수행
    for file_path in h5_files:
        # 순수 파일 이름만 추출 (경로 제외)
        filename = os.path.basename(file_path)

        # 출력할 TIF 파일의 경로와 이름 지정
        output_filename = os.path.splitext(filename)[0] + '_peaksite.tif'
        output_file_path = os.path.join(output_dir, output_filename)

        try:
            # HDF5 파일 열기
            with h5py.File(file_path, 'r') as h5file:
                # 'Peaksite' 데이터가 있는지 확인
                if 'Peaksite' in h5file:
                    peaksite_array = h5file['Peaksite'][:]

                    # TIFF 파일로 저장
                    tifffile.imwrite(output_file_path, peaksite_array)
                    print(f"✅ 저장 완료: {output_filename}")
                else:
                    print(f"⚠️ 건너뜀: '{filename}' 내에 'Peaksite' 데이터가 없습니다.")

        except Exception as e:
            print(f"❌ 에러 발생 ({filename}): {e}")

    print(f"\n🎉 모든 작업이 완료되었습니다! 파일들은 아래 경로에 저장되었습니다:\n{output_dir}")