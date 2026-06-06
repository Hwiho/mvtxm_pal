'''
fn: h5 filename
'''
import numpy as np
import pandas as pd
def bulk_spectrum(fn):
    import h5py

    with h5py.File(fn,'r') as f:
        img = np.array(f["Img_aligned"])
    mean = np.mean(np.mean(img, axis=1), axis=1)
    return mean

'''
해당 h5 있는 폴더의 englist값 얻기 위해서 하나 해야함.. 앞으로는 h5 파일안에 eng 값 넣어 놓겠음 아직까지 필요없어서 안넣어놓았는데;;
'''
def energy_list_extraction(proj_dir):
    import glob
    fn = glob.glob(proj_dir + '/*.tif')
    energy_sort_dict = {}
    energy = []
    for key in range(len(fn)):
        energy_sort_dict[str(key)] = fn[key].split('_')
        energy.append(float(energy_sort_dict[str(key)][energy_sort_dict[str(key)].index('eV') - 1]))
    eng = np.array(energy)
    return eng

def to_csv(proj_dir, fn, file_path):
    eng = energy_list_extraction(proj_dir)
    mean = bulk_spectrum(fn)
    df = pd.DataFrame()
    df['Energy'] = eng
    df['Spectrum'] = mean
    df.to_csv(file_path,sep=',',na_rep="NaN")

if __name__ == "__main__":
    proj_dir = 'F:/insitu/0.2C_Insitu/ASSBs_Insitu_from_17MULTI000_back'
    fn = 'F:/0.2C_new/new_histogram/Image_dataset_id_17MULTI000_pos_00.h5'
    to_csv(proj_dir, fn, file_path="C:/Users/Hwiho/cff.csv")
