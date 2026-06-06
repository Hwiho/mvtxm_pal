import h5py
import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from tqdm import trange, tqdm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib.widgets import Button
from skimage import io, color
import re
import glob

class InteractiveEdgeSelector:
    def __init__(self, energy, mu):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.energy = energy
        self.mu = mu
        self.edges = []
        self.lines = []
        self.normalized_mu = None
        self.normalized_plot = None
        self.pre_edge_range = None
        self.post_edge_range = None
        self.ax.plot(energy, mu, 'o-', label='Original XANES Spectrum')
        self.ax.set_title('Click to set pre-edge start')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)

        # Setup for 'Display Normalization' button
        self.button_ax = self.fig.add_axes([0.7, 0.05, 0.25, 0.075])
        self.button = Button(self.button_ax, 'Display Normalization')
        self.button.on_clicked(self.display_normalization)

        # Setup for 'Reset' button
        self.reset_button_ax = self.fig.add_axes([0.1, 0.05, 0.25, 0.075])
        self.reset_button = Button(self.reset_button_ax, 'Reset')
        self.reset_button.on_clicked(self.reset)

    def __call__(self, event):
        if event.inaxes != self.ax:
            return
        idx = (np.abs(self.energy - event.xdata)).argmin()
        self.edges.append(self.energy[idx])
        self.update_edges_and_lines(idx)

    def get_edge_ranges(self):
        """Returns the indices in the energy array for the pre-edge and post-edge ranges."""
        if len(self.edges) < 4:
            raise ValueError("Not all edge points have been selected yet.")

        # Find indices of the selected edge points
        pre_start_idx = np.searchsorted(self.energy, self.edges[0])
        pre_end_idx = np.searchsorted(self.energy, self.edges[1])
        post_start_idx = np.searchsorted(self.energy, self.edges[2])
        post_end_idx = np.searchsorted(self.energy, self.edges[3])

        # Return tuples of indices
        return (pre_start_idx, pre_end_idx), (post_start_idx, post_end_idx)

    def update_edges_and_lines(self, idx):
        line = self.ax.axvline(x=self.energy[idx], color='r' if len(self.edges) < 3 else 'g', linestyle='--')
        if len(self.edges) == 4:
            # Once all edges are selected, set the ranges
            self.pre_edge_range = (self.edges[0], self.edges[1])
            self.post_edge_range = (self.edges[2], self.edges[3])
            self.ax.set_title('Selection complete: Adjust lines if necessary')
            self.fit_lines()
        self.fig.canvas.draw()

    def update_title(self):
        titles = ['Set pre-edge end', 'Set post-edge start', 'Set post-edge end', 'Adjust or accept the extrapolation']
        if len(self.edges) <= 4:
            self.ax.set_title(titles[len(self.edges) - 1])

    def fit_lines(self):
        # Fit and display extrapolated lines
        self.display_regions()
        self.ax.legend()

    def display_regions(self):
        # Fit lines to the selected regions and extrapolate
        if len(self.edges) == 4:
            pre_start_idx, pre_end_idx = np.searchsorted(self.energy, (self.edges[0], self.edges[1]))
            post_start_idx, post_end_idx = np.searchsorted(self.energy, (self.edges[2], self.edges[3]))
            pre_coeffs = np.polyfit(self.energy[pre_start_idx:pre_end_idx + 1],
                                    self.mu[pre_start_idx:pre_end_idx + 1], 1)
            post_coeffs = np.polyfit(self.energy[post_start_idx:post_end_idx + 1],
                                     self.mu[post_start_idx:post_end_idx + 1], 1)
            pre_line = np.polyval(pre_coeffs, self.energy)
            post_line = np.polyval(post_coeffs, self.energy)
            self.ax.plot(self.energy, pre_line, 'r--', label='Extrapolated Pre-edge Fit')
            self.ax.plot(self.energy, post_line, 'g--', label='Extrapolated Post-edge Fit')

    def display_normalization(self, event):
        # Normalize and display normalized data if not done yet
        if self.normalized_mu is None:
            self.normalize_spectrum()
        if self.normalized_plot:
            self.normalized_plot.remove()
        self.normalized_plot = self.ax.plot(self.energy, self.normalized_mu, 'm-', label='Normalized Spectrum', alpha=0.7)
        self.ax.legend()
        self.fig.canvas.draw()

    def normalize_spectrum(self):
        if self.normalized_plot:
            self.normalized_plot.remove()
        # Get indices for normalization
        pre_start_idx, pre_end_idx = np.searchsorted(self.energy, (self.edges[0], self.edges[1]))
        post_start_idx, post_end_idx = np.searchsorted(self.energy, (self.edges[2], self.edges[3]))

        # Fit pre-edge baseline
        pre_edge_fit = np.polyfit(self.energy[pre_start_idx:pre_end_idx + 1], self.mu[pre_start_idx:pre_end_idx + 1], 1)
        pre_edge_baseline = np.polyval(pre_edge_fit, self.energy)

        # Step height based on pre-edge and post-edge means
        step_height = np.mean(self.mu[post_start_idx:post_end_idx + 1]) - np.mean(
            pre_edge_baseline[pre_start_idx:pre_end_idx + 1])

        # Normalized mu data
        self.normalized_mu = (self.mu - pre_edge_baseline) / step_height

    def reset(self, event):
        # Reset the plot and clear all data and lines
        for line in self.lines:
            line.remove()
        if self.normalized_plot:
            self.normalized_plot.remove()
        self.lines.clear()
        self.edges.clear()
        self.normalized_mu = None
        self.ax.set_title('Click to set pre-edge start')
        self.fig.canvas.draw()

class fit_xanes():

    def __init__(self, img, eng, peakref, color_flag):
        self.img = img
        self.eng = eng
        self.fit_num = None
        self.peaksite = None
        self.peakref = peakref
        self.pre_no = None
        self.post_no = None
        self.pre_thick = np.zeros(self.img[0].shape)
        self.post_thick = np.zeros(self.img[0].shape)
        self.thickness = None
        if color_flag == 'rgb':
            self.hsvconstant = np.array((2 / 3, 1, 0))
            self.relconstant = np.array((2 / 3, 1, 0))
        else:
            self.hsvconstant = np.array((1 / 3, 1, 0))
            self.relconstant = np.array((1 / 3, 1, 0))
        self.hsv = None
        self.relhsv = None
        self.rgb = None
        self.relrgb = None
        self.binary_mask = np.zeros(self.img[0].shape)
        self.binary_mask_hsv = np.zeros((self.img.shape[1],self.img.shape[2],3))
        self.stdev = None
        self.mean = None
        self.relref = [-0.5,0.5]
        self.norm_img = None

    def set_thickness_pnts(self, eng_len):
        """
            101 fitting: preedge: 30, postedge:91
            63 fitting: preedge: 5, postedge: 57
            21 fitting: preedge: 1. postedge: 20
        """
        if eng_len == 101:
            self.pre_no = 30
            self.post_no = 91
        elif eng_len == 63:
            self.pre_no = 8
            self.post_no = 54
        elif eng_len == 21:
            self.pre_no = 2
            self.post_no = 18
        else:
            self.pre_no = eng_len//5
            self.post_no = eng_len - 5

    def set_thickness(self):
        for i in range(self.pre_no):
            self.pre_thick += self.img[i]
        for i in range(self.post_no, len(self.img)):
            self.post_thick += self.img[i]
        self.post_mean = self.post_thick / (len(self.img) - self.post_no)
        self.pre_mean = self.pre_thick / self.pre_no
        self.thickness = self.post_mean - self.pre_mean

    def threshold(self, threshold):
        self.binary_mask = self.thickness >= threshold
        self.thickness = np.nan_to_num(self.thickness * self.binary_mask)

    def thres_stack_generation(self):
        lower = np.linspace(0.001,0.01,20)
        upper = np.linspace(0.015,0.2,30)
        self.thres_list = list(np.hstack((lower, upper)))
        self.thres_stack = []
        for thres in self.thres_list:
            stack = self.thickness >= thres
            thick_stack = self.thickness * stack
            self.thres_stack.append(thick_stack)

    def polynomial_second_fit(self, fit_num, maxpoint):
        if maxpoint is not None:
            mp = maxpoint
        else:
            mean = np.mean(np.mean(self.img, axis=1), axis=1)
            mp = np.argmax(mean)
        self.fit_num = fit_num
        image_slice = self.img[mp - fit_num:mp + fit_num + 1]
        y = image_slice.reshape(len(image_slice), -1)
        x = np.linspace(0, len(image_slice) - 1, len(image_slice))
        coefs = poly.polyfit(x, y, 2)
        x0 = -(coefs[1] / 2 / coefs[2]) + self.eng[mp - fit_num]
        # Boundary Condition
        peaksite = x0.reshape(image_slice.shape[1], image_slice.shape[2])
        peaksite = peaksite * self.binary_mask
        peaksite[peaksite - self.peakref[1] > 1.5 * np.std(peaksite[self.binary_mask!=0])] = self.peakref[1]
        peaksite[peaksite - self.peakref[0] < -1.5 * np.std(peaksite[self.binary_mask!=0])] = self.peakref[0]
        peaksite[self.binary_mask==0] = 0
        self.peaksite = peaksite

    def edge50(self, pre_edge_range, post_edge_range, window_size=5, polyorder=2):
        num_energy_points, height, width = self.img.shape

        # Initialize output arrays
        normalized_stack = np.zeros_like(self.img)
        smoothed_stack = np.zeros_like(self.img)
        zero_point_5_positions = np.zeros((height, width))

        # Process each pixel in the stack
        progress = tqdm(total=height * width, desc='Processing Pixels', unit='pixel')
        for i in range(height):
            for j in range(width):
                mu = self.img[:, i, j]

                # Normalization
                pre_edge_fit = np.polyfit(self.eng[pre_edge_range], mu[pre_edge_range], 1)
                pre_edge_baseline = np.polyval(pre_edge_fit, self.eng)
                step_height = np.mean(mu[post_edge_range]) - np.mean(pre_edge_baseline[pre_edge_range])
                normalized_mu = (mu - pre_edge_baseline) / step_height
                normalized_stack[:, i, j] = normalized_mu

                # Smoothing
                smoothed_mu = savgol_filter(normalized_mu, window_size, polyorder)
                smoothed_stack[:, i, j] = smoothed_mu

                # Interpolate to find the 0.5 position
                try:
                    max_index = np.argmax(smoothed_mu)
                    range_start = max(pre_edge_range.stop, 5)
                    range_end = max_index + 1

                    if range_end > range_start + 1:
                        interpolator = interp1d(self.eng[range_start:range_end], smoothed_mu[range_start:range_end], kind='linear', bounds_error=False, fill_value='extrapolate')
                        energy_range = np.linspace(self.eng[range_start], self.eng[range_end-1], num=1000)
                        interpolated_mu = interpolator(energy_range)
                        mask = np.isclose(interpolated_mu, 0.5, atol=0.01)
                        if np.any(mask):
                            zero_point_5_positions[i, j] = energy_range[mask][0]
                        else:
                            zero_point_5_positions[i, j] = energy_range[-1]
                except Exception as e:
                    print(f"Error at pixel ({i}, {j}): {e}")
                    zero_point_5_positions[i, j] = self.eng[max(range_start, range_end-1)]

                progress.update(1)

        progress.close()
        self.norm_img = normalized_stack
        self.smth_img = smoothed_stack
        peaksite = zero_point_5_positions
        peaksite = peaksite * self.binary_mask
        peaksite[peaksite > self.peakref[1] + 0.5] = self.peakref[1] + 0.5
        peaksite[peaksite < self.peakref[0] - 0.5] = self.peakref[0] - 0.5
        peaksite[self.binary_mask == 0] = 0
        self.peaksite = peaksite

    def polynomial_second_fit_separate(self, fit_num, ev_step):
        self.fit_num = fit_num
        y = self.img.reshape(len(self.img), -1)
        y_new = np.zeros((fit_num * 2 + 1, y.shape[1]))
        mp_list = []
        for i in range(y.shape[1]):
            a = np.argmax(y[:, i])
            if a + fit_num + 1 > len(y):
                y_new[:, i] = y[len(y) - 2 * fit_num - 1:len(y) + 1, i]
            elif a - fit_num < 0:
                y_new[:, i] = y[0: 2 * fit_num + 1, i]
            else:
                y_new[:, i] = y[a - fit_num:a + fit_num + 1, i]
            mp_list.append(a)

        x = np.linspace(0, ev_step*(len(y_new) - 1), len(y_new))
        coefs, res = poly.polyfit(x, y_new, 2, full=True)
        ind = [a - fit_num for a in mp_list]
        x0 = -(coefs[1] / 2 / coefs[2]) + self.eng[ind]
        peaksite = x0.reshape(self.img.shape[1], self.img.shape[2])

        # Boundary Condition

        peaksite = peaksite * self.binary_mask
        peaksite[peaksite > self.peakref[1] + 0.5] = self.peakref[1] + 0.5
        peaksite[peaksite < self.peakref[0] - 0.5] = self.peakref[0] - 0.5
        peaksite[self.binary_mask == 0] = 0
        self.peaksite = peaksite

    def polynomial_multi_fit_whole_wl(self, deg, bounds, ev_step):
        if bounds is None:
            image_slice = self.img
        else:
            image_slice = self.img[bounds[0]:bounds[1] + 1]

        # Ensure the image slice is a 3D array for consistency
        if len(image_slice.shape) != 3:
            raise ValueError("Image slice must be a 3D array.")

        # Precompute x values (match with the first dimension of the image slice)
        x = np.linspace(0, ev_step*(image_slice.shape[0] - 1), image_slice.shape[0])
        X = np.vander(x, N=deg+1)  # N=9 for an 8th-degree polynomial

        # Reshape image slice for polynomial fitting
        reshaped_slice = image_slice.reshape(image_slice.shape[0], -1)

        # Ensure correct dimensions for least squares fitting
        if X.shape[0] != reshaped_slice.shape[0]:
            raise ValueError("Dimension mismatch between X and reshaped image slice.")

        # Calculate polynomial coefficients
        coefficients = np.linalg.lstsq(X, reshaped_slice, rcond=None)[0]

        # Approximate maxima finding (this is a simplification and might not be accurate)
        sampled_x = np.linspace(x[0], x[-1], 300)
        sampled_X = np.vander(sampled_x, N=deg+1)
        polynomial_values = sampled_X @ coefficients
        maxima_indices = np.argmax(polynomial_values, axis=0)

        # Reshape and adjust results
        if bounds is None:
            peaksite = sampled_x[maxima_indices].reshape(image_slice.shape[1], image_slice.shape[2]) + self.eng[0]
        else:
            peaksite = sampled_x[maxima_indices].reshape(image_slice.shape[1], image_slice.shape[2]) + self.eng[bounds[0]]

        peaksite = peaksite * self.binary_mask
        std_dev = np.std(peaksite[self.binary_mask != 0])
        peaksite[peaksite - self.peakref[1] > 1.5 * std_dev] = self.peakref[1]
        peaksite[peaksite - self.peakref[0] < -1.5 * std_dev] = self.peakref[0]
        peaksite[self.binary_mask == 0] = 0

        self.peaksite = peaksite


    def ff_xanes_calibration(self):
        ref_shape = (len(self.img), 2048, 2048)
        shape = self.img[0].shape
        nx, ny = shape
        calib_parm = (shape[0] / ref_shape[1], shape[1] / ref_shape[2])
        x = np.linspace(0.15 * calib_parm[0], -0.15 * calib_parm[0], nx)
        y = np.linspace(0.15 * calib_parm[1], -0.15 * calib_parm[1], ny)
        xv, yv = np.meshgrid(y, x)
        self.calgrid = yv
        self.calgrid[self.binary_mask == 0] = 0
        self.peaksite = self.peaksite + self.calgrid

    def ff_xanes_calibration_inverse(self):
        ref_shape = (len(self.img), 2048, 2048)
        shape = self.img[0].shape
        nx, ny = shape
        calib_parm = (shape[0] / ref_shape[1], shape[1] / ref_shape[2])
        x = np.linspace(-0.15 * calib_parm[0], 0.15 * calib_parm[0], nx)
        y = np.linspace(-0.15 * calib_parm[1], 0.15 * calib_parm[1], ny)
        xv, yv = np.meshgrid(y, x)
        self.calgrid = yv
        self.calgrid[self.binary_mask == 0] = 0
        self.peaksite = self.peaksite + self.calgrid

    def cal_stdev (self):
        self.std_img = self.peaksite[self.binary_mask!=0]
        self.std_img[self.std_img > self.peakref[1] + 0.5] = self.peakref[1] + 0.5
        self.std_img[self.std_img < self.peakref[0] - 0.5] = self.peakref[0] - 0.5
        self.stdev = np.std(self.std_img)
        self.mean = np.mean(self.std_img)

    def hsvcolormap (self, value = None):
        hsv = np.zeros((self.img.shape[1], self.img.shape[2], 3),dtype="float32")
        conc = (self.peaksite - self.peakref[0])/ (self.peakref[1] - self.peakref[0])
        conc[conc < 0] = 0
        conc[conc > 1] = 1
        if self.hsvconstant[0] != 0:
            hsv[:, :, 0] = conc * self.hsvconstant[0]
        else:
            hsv[:, :, 0] = conc/np.max(np.max(conc))

        hsv[:, :, 1]= np.ones((self.img.shape[1], self.img.shape[2]))
        if value is not None:
            self.hsvconstant[2] = value

        if self.hsvconstant[2] != 0:
            hsv[:, :, 2] = self.thickness * self.hsvconstant[2];
        else:
            hsv[:, :, 2] = self.thickness/np.max(self.thickness)
        hsv[:, :, 2] = (hsv[:, :, 2]*1)
        self.hsv = hsv
        self.rgb = color.hsv2rgb(hsv)
        self.brightness = np.mean(np.mean(hsv[:,:,2]))


    def relcolormap(self, value=None):
        rel = np.zeros((self.img.shape[1], self.img.shape[2], 3), dtype="float32")

        mean_sub = self.peaksite - np.mean(self.peaksite[self.binary_mask != 0])
        rel_conc = (mean_sub - self.relref[0]) / (self.relref[1] - self.relref[0])
        rel_conc[rel_conc < 0] = 0
        rel_conc[rel_conc > 1] = 1
        if self.relconstant[0] != 0:
            rel[:, :, 0] = rel_conc * self.relconstant[0]
        else:
            rel[:, :, 0] = rel_conc / np.max(np.max(rel_conc))

        rel[:, :, 1] = np.ones((self.img.shape[1], self.img.shape[2]))
        if value is not None:
            self.relconstant[2] = value

        if self.relconstant[2] != 0:
            rel[:, :, 2] = self.thickness * self.relconstant[2];
        else:
            rel[:, :, 2] = self.thickness / np.max(self.thickness)
        rel[:, :, 2] = (rel[:, :, 2] * 1)
        self.relhsv = rel
        self.relrgb = color.hsv2rgb(rel)
        self.brightness = np.mean(np.mean(rel[:, :, 2]))

    def misc_relcolormap(self, color_type='coolwarm', region=1.0):
        """
        Applies a colormap to the peaksite data using thickness for normalization.

        Parameters:
        - color_type (str): Name of the matplotlib colormap to apply.
        - region (float): Factor to adjust concentration range.
        """
        # Normalize concentration based on peakref
        mean_sub = self.peaksite - np.mean(self.peaksite[self.binary_mask != 0])
        rel_conc = np.clip((mean_sub - self.relref[0]) / (self.relref[1] - self.relref[0]), 0, 1) * region

        # Normalize thickness
        t_normalized = np.clip(self.thickness / np.max(self.thickness), 0, 1)

        # Initialize colormap
        norm = Normalize(vmin=0, vmax=1)
        #cmap = cc.cm[color_type]
        cmap = plt.get_cmap(color_type)

        # Apply colormap to create RGB image
        rgba = cmap(norm(rel_conc))
        rgb = np.where(self.binary_mask[..., None] == 0, 0, rgba[..., :3] * t_normalized[..., None])

        # Convert to uint8 for image display or storage
        self.relrgb = rgb

    def misc_colormap(self, color_type='coolwarm', region=1.0, cmap_range=(0.0, 1.0)):
        """
        Applies a colormap to the peaksite data without brightness adjustment.

        Parameters:
        - color_type (str): Name of the matplotlib colormap to apply.
        - region (float): Factor to adjust concentration range.
        - cmap_range (tuple): The range of the colormap to use, as a fraction (start, end) between 0 and 1.
                              For example, (0.2, 0.8) would use the middle 60% of the colormap.
        """
        # Calculate concentration based on peakref
        conc = (self.peaksite - self.peakref[0]) / (self.peakref[1] - self.peakref[0])
        conc = np.clip(conc, 0, 1) * region

        # Initialize original colormap and create a truncated colormap based on cmap_range
        #cmap = cc.cm[color_type]
        cmap = plt.get_cmap(color_type)
        start, end = cmap_range
        truncated_cmap = LinearSegmentedColormap.from_list(
            f'truncated({color_type},{start:.2f},{end:.2f})',
            cmap(np.linspace(start, end, 256))
        )

        # Apply truncated colormap to create RGBA values
        norm = Normalize(vmin=0, vmax=1)
        rgba = truncated_cmap(norm(conc))
        rgb_values = rgba[..., :3]

        # Ensure the binary_mask is valid
        if self.binary_mask is not None:
            # Apply mask to RGB values without adjusting brightness
            rgb = np.where(self.binary_mask[..., None] == 0, 0, rgb_values)
        else:
            # If no mask is provided, use RGB values directly
            rgb = rgb_values

        self.rgb = rgb


