import imagej
import h5py
import numpy as np
from skimage import io
import os
from scyjava import jimport
import pandas as pd
from tqdm import trange

class imgtoolbox():

    def __init__(self, img):
        self.img = img
        self.ij = imagej.init(mode='interactive')
        self.ij.ui().showUI()
        self.ip = jimport('ij.ImagePlus')
        self.path = os.getcwd().replace('\\','/')

    def crop_img(self):
        self.ij.ui().show(self.ij.py.to_java(self.img))
        macro = """#@output cropped_id
                setTool("rectangle");
                run("Brightness/Contrast...");
                run("Plot Z-axis Profile");
                waitForUser("Draw ROI, then hit OK");
                run("Duplicate...", "title=Cropped duplicate");
                cropped_id = getImageID();
                """
        output = self.ij.py.run_script('ijm', macro).getOutput("cropped_id")
        imp = self.ij.convert().convert(output, self.ip.class_)
        self.cropped = np.array(self.ij.py.from_java(imp)).transpose((2, 0, 1))
        self.ij.dispose()

    def get_roicoord(self, scan_note, roi_mode='rectangle'):
        '''

        :param scan_note: 'str' : name of images
        :param roi_mode: 'rectangle' , 'oval', "polygon", "freehand"
        :return:
        '''
        self.roi_mode = roi_mode
        self.ij.ui().show(self.ij.py.to_java(self.img))
        macro1 = '''
                 waitForUser("Add one or more multipoint ROIs Press <T>");
                 setTool("%s")
                 numROIs = roiManager("count");
                 rename("ROI_Selection");
                 title = getInfo("image.filename");
                 for(i=0; i<numROIs;i++) {// loop through ROIs
	                 roiManager("Select", i);
	                 getSelectionBounds(x, y, w, h);//if the ROI is a multipoint ROIs x and y are arrays
	                    }
	             roiManager("List");
	             saveAs("Results", "%s/ROI_Selection_%s.csv");
	             run("From ROI Manager");
	             run("Overlay Options...", "stroke=none width=0 fill=none set show");
	             run("Flatten", "stack");
	             run("Duplicate...", "title=ROI_Selection_Image")
	             selectWindow("ROI_Selection_Image")
	             saveAs("tif", "%s/ROI_Selection_Image_%s.tif");
                 '''%(roi_mode, self.path, scan_note, self.path, scan_note)
        self.ij.py.run_script('ijm', macro1)
        self.ij.dispose()
        # Using pandas dataframe for coordinates: X, Y, Width, Height
        raw_excel = pd.read_csv(self.path +"/ROI_Selection_" + scan_note + ".csv")
        self.excel_df = raw_excel.loc[:, ['X', 'Y', 'Width', 'Height']]

    def load_roi_coord(self, csv_path):
        raw_excel = pd.read_csv(csv_path)
        self.excel_df = raw_excel.loc[:, ['X', 'Y', 'Width', 'Height']]

    def cropfromrois(self):
        crop_list = []
        for i in trange(len(self.excel_df)):
            img_crop = self.img[:, self.excel_df.loc[i][1]:self.excel_df.loc[i][1] + self.excel_df.loc[i][3],
                                   self.excel_df.loc[i][0]:self.excel_df.loc[i][0] + self.excel_df.loc[i][2]]
            crop_list.append(img_crop)
        self.cropped = crop_list

    def particle_analysis(self):
        self.ij.ui().show(self.ij.py.to_java(self.img))
        macro1 ='''
        
        
                '''
        self.ij.py.run_script('ijm', macro1)
        self.ij.dispose()