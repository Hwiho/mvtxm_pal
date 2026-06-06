import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Slider, Button , TextBox
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
from skimage import io
from matplotlib.ticker import MaxNLocator



class Post_process_basic:

    def __init__(self):

        self.fit= None
        self.image_stack = None
        self.eng = None
        self.save_directory = None
        self.fn = None
        self.index = 0  # current image index
        self.fig = plt.figure(figsize=(15, 6))
        self.select_h5_file()

        # Positions
        self.ax = []  # List to hold the axes
        self.ax.append(self.fig.add_axes([0.035, 0.2, 0.28, 0.7]))  # self.image_stack
        self.ax.append(self.fig.add_axes([0.35, 0.2, 0.28, 0.7]))  # self.fit
        self.ax.append(self.fig.add_axes([0.7, 0.25, 0.28, 0.60]))  # intensity plot
        ax_slider = plt.axes([0.035, 0.175, 0.28, 0.03], facecolor='lightgoldenrodyellow')
        ax_select_dir = plt.axes([0.05, 0.025, 0.1, 0.05])
        self.fig.text(0.01, 0.92, "Left Click : Draw ROI\nRight Click : Reset ROI\nSelect a directory before you save "
                                  "files", fontsize=8, color='r')
        ax_crop_save = plt.axes([0.2, 0.025, 0.1, 0.05])  # Adjust these values as needed
        ax_autoscale = plt.axes([0.05, 0.1, 0.1, 0.05])  # Adjust these values as needed
        ax_reset = plt.axes([0.2, 0.1, 0.1, 0.05])
        ax_save_intensity = plt.axes([0.8, 0.1, 0.1, 0.05])
        ax_min = plt.axes([0.375, 0.175, 0.08, 0.03], facecolor='lightgoldenrodyellow')
        ax_max = plt.axes([0.375, 0.125, 0.08, 0.03], facecolor='lightgoldenrodyellow')
        # Create axes for text boxes
        ax_min_box = plt.axes([0.55, 0.175, 0.05, 0.03])
        ax_max_box = plt.axes([0.55, 0.125, 0.05, 0.03])

        # Initial display
        img_height, img_width = self.image_stack[0].shape
        aspect_ratio = img_width / img_height
        self.image_display = self.ax[0].imshow(self.image_stack[self.index], cmap='gray', aspect=aspect_ratio)
        self.fit_display= self.ax[1].imshow(self.fit, cmap='turbo', aspect=aspect_ratio)
        self.ax[0].set_title(f"Energy : {self.eng[self.index]:.1f}eV", fontsize=15)
        self.ax[1].set_title("Fitted value", fontsize=15)
        self.ax[2].set_title("Z-axis profile", fontsize=15)
        self.cbar_stack = self.fig.colorbar(self.image_display, ax=self.ax[0], orientation='vertical', fraction=0.046, pad=0.02)
        self.cbar_fit = self.fig.colorbar(self.fit_display, ax=self.ax[1], orientation='vertical', fraction=0.046, pad=0.02)
        for i in [0,1]:
            self.ax[i].set_xticks([])
            self.ax[i].set_yticks([])

        # Intensity plot (initialized with zeros)
        self.intensity_line, = self.ax[2].plot(self.eng, np.zeros(len(self.image_stack)),marker='.')  # Set x values to eng here
        self.ax[2].set_xlim(min(self.eng), max(self.eng))  # Set limits to match eng values
        self.ax[2].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

        # Rectangle patch
        self.rect = patches.Rectangle((0, 0), 0, 0, fill=False, edgecolor='blue')
        self.ax[0].add_patch(self.rect)
        self.is_pressed = False
        self.active_corner = None
        self.corner_threshold = 5  # pixels
        self.rect_fit = patches.Rectangle((0, 0), 0, 0, fill=False, edgecolor='red')  # you can choose another color
        self.ax[1].add_patch(self.rect_fit)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Index slider
        self.slider = Slider(ax_slider, 'Index', 0, len(self.image_stack) - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_image_index)

        # Dragging
        self.dragging = False  # Indicates if the rectangle is being dragged
        self.drag_offset_x = 0  # The x-offset between the cursor and the rectangle's origin while dragging
        self.drag_offset_y = 0

        # Button to select save directory
        self.save_directory = "C:/Users/hwiho/Documents/"
          # Adjust these values as needed
        self.btn_select_dir = Button(ax_select_dir, 'Select Directory', hovercolor='0.975')
        self.btn_select_dir.on_clicked(self.select_directory)

        # Adding tips below the index slider


        # Button to crop and save image
        self.btn_crop_save = Button(ax_crop_save, 'Crop & Save', hovercolor='0.975')
        self.btn_crop_save.on_clicked(self.crop_and_save)
        self.btn_crop_save_click_count = 0

        # Button to autoscale image
        self.btn_autoscale = Button(ax_autoscale, 'Autoscale', hovercolor='0.975')
        self.btn_autoscale.on_clicked(self.autoscale_image)
        self.reset_button = Button(ax_reset, 'Reset')  # Make sure to define 'self.ax_reset'
        self.reset_button.on_clicked(self.reset_autoscale)

        # Button to save intensity line plot
        self.btn_save_intensity = Button(ax_save_intensity, 'Save Intensity', hovercolor='0.975')
        self.btn_save_intensity.on_clicked(self.save_intensity_data)
        self.save_intensity_click_count = 0

        self.slider_min = Slider(ax_min, 'Min ', 8340, 8380, valinit=self.fit.min())  # Adjust range if needed
        self.slider_max = Slider(ax_max, 'Max ', 8340, 8380, valinit=self.fit.max())  # Adjust range if needed
        self.slider_min.on_changed(self.update_min)
        self.slider_max.on_changed(self.update_max)



        self.textbox_min = TextBox(ax_min_box, 'Min ', initial=str(int(self.fit_display.get_clim()[0])))  # using the initial display range
        self.textbox_max = TextBox(ax_max_box, 'Max ', initial=str(int(self.fit_display.get_clim()[1]))) # using the initial display range

        self.textbox_min.on_submit(self.submit_min)
        self.textbox_max.on_submit(self.submit_max)

        plt.show()
    def select_h5_file(self):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()

        self.fn = filedialog.askopenfilename(initialdir="/", title="Select h5 file",
                                              filetypes=(("h5 files", "*.h5"),
                                                         ("all files", "*.*")))

        with h5py.File(self.fn, 'r') as h5f:
            self.image_stack = np.array(h5f["Img_aligned"])
            self.fit = np.array(h5f["Peaksite"])
            if "Energy" in h5f:
                self.eng = np.array(h5f["Energy"])
            else:
                print("Energy does not exist in the HDF5 file.")
                if len(self.image_stack) == 21:
                    self.eng = np.arange(8340, 8361)
                print(self.eng)

    def on_press(self, event):
        if event.inaxes != self.ax[0]:
            return

        if event.inaxes != self.ax[0]:
            return

        # Reset the rectangle on right-click
        if event.button == 3:
            self.reset_rectangle()
            self.update_intensity_line()
            return

        x, y = event.xdata, event.ydata
        x0, y0 = self.rect.get_xy()
        width, height = self.rect.get_width(), self.rect.get_height()

        if x0 <= x <= x0 + width and y0 <= y <= y0 + height:
            self.is_pressed = True
            self.dragging = True
            self.drag_offset_x = x - x0
            self.drag_offset_y = y - y0
            return

        x, y = event.xdata, event.ydata
        x0, y0 = self.rect.get_xy()
        width, height = self.rect.get_width(), self.rect.get_height()

        corners = [(x0, y0), (x0 + width, y0), (x0, y0 + height), (x0 + width, y0 + height)]
        for i, (cx, cy) in enumerate(corners):
            if abs(x - cx) < self.corner_threshold and abs(y - cy) < self.corner_threshold:
                self.active_corner = i
                self.is_pressed = True
                return

        self.active_corner = None
        self.is_pressed = True
        self.x0 = x
        self.y0 = y
        self.rect.set_width(0)
        self.rect.set_height(0)
        self.rect.set_xy((self.x0, self.y0))

    def on_motion(self, event):
        if not self.is_pressed or event.inaxes != self.ax[0] or not event.xdata or not event.ydata:
            return

        x, y = event.xdata, event.ydata
        x0, y0 = self.rect.get_xy()
        width, height = self.rect.get_width(), self.rect.get_height()

        if self.dragging:
            # New position is current mouse position minus the drag offset
            new_x0 = x - self.drag_offset_x
            new_y0 = y - self.drag_offset_y
            self.rect.set_xy((new_x0, new_y0))

            # Also update rect_fit
            self.rect_fit.set_xy((new_x0, new_y0))
        else:
            if self.active_corner is not None:
                # Resize existing rectangle
                if self.active_corner == 0:
                    new_width = x0 + width - x
                    new_height = y0 + height - y
                    self.rect.set_xy((x, y))
                    self.rect.set_width(new_width)
                    self.rect.set_height(new_height)

                    # Also update rect_fit
                    self.rect_fit.set_xy((x, y))
                    self.rect_fit.set_width(new_width)
                    self.rect_fit.set_height(new_height)
                elif self.active_corner == 1:
                    new_width = x - x0
                    new_height = y0 + height - y
                    self.rect.set_xy((x0, y))
                    self.rect.set_width(new_width)
                    self.rect.set_height(new_height)

                    # Also update rect_fit
                    self.rect_fit.set_xy((x0, y))
                    self.rect_fit.set_width(new_width)
                    self.rect_fit.set_height(new_height)
                elif self.active_corner == 2:
                    new_width = x0 + width - x
                    new_height = y - y0
                    self.rect.set_xy((x, y0))
                    self.rect.set_width(new_width)
                    self.rect.set_height(new_height)

                    # Also update rect_fit
                    self.rect_fit.set_xy((x, y0))
                    self.rect_fit.set_width(new_width)
                    self.rect_fit.set_height(new_height)
                elif self.active_corner == 3:
                    new_width = x - x0
                    new_height = y - y0
                    self.rect.set_xy((x0, y0))
                    self.rect.set_width(new_width)
                    self.rect.set_height(new_height)

                    # Also update rect_fit
                    self.rect_fit.set_xy((x0, y0))
                    self.rect_fit.set_width(new_width)
                    self.rect_fit.set_height(new_height)
            else:
                # Draw new rectangle
                dx = x - self.x0
                dy = y - self.y0
                self.rect.set_width(dx)
                self.rect.set_height(dy)

                # Also update rect_fit
                self.rect_fit.set_width(dx)
                self.rect_fit.set_height(dy)
                self.rect_fit.set_xy(
                    (self.x0, self.y0))  # Assuming self.x0 and self.y0 are your initial mouse click coordinates

        self.fig.canvas.draw()


    def on_release(self, event):
        self.is_pressed = False
        self.dragging = False
        self.active_corner = None
        self.update_intensity_line()

    def line_select_callback(self, eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.current_rect_coords = [(x1, y1), (x2, y2)]
        self.update_intensity_line()

    def update_image_index(self, val):
        self.index = int(self.slider.val)
        self.image_display.set_data(self.image_stack[self.index])
        # Update energy value in the title
        energy_val = f"Energy : {self.eng[self.index]:.1f}eV"  # format as needed
        self.ax[0].set_title(energy_val, fontsize=15)
        self.cbar_stack.update_normal(self.image_display)
        self.fig.canvas.draw_idle()

    def update_intensity_line(self):
        x0, y0 = self.rect.get_xy()
        width = self.rect.get_width()
        height = self.rect.get_height()

        if width < 0:
            x0 += width
            width *= -1
        if height < 0:
            y0 += height
            height *= -1

        x0, y0, x1, y1 = int(x0), int(y0), int(x0 + width), int(y0 + height)

        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(x1, self.image_stack[self.index].shape[1]), min(y1, self.image_stack[self.index].shape[0])

        if x0 == x1 or y0 == y1:
            return

        intensity = [np.mean(img[y0:y1, x0:x1]) if len(img[y0:y1, x0:x1]) else 0 for img in self.image_stack]

        self.intensity_line.set_xdata(self.eng)  # Update the x values to eng here
        self.intensity_line.set_ydata(intensity)
        self.ax[2].autoscale(enable=True, axis='y')
        self.ax[2].relim()
        self.ax[2].autoscale_view()

        # Redraw the canvas to reflect changes
        self.ax[2].figure.canvas.draw()

    def reset_rectangle(self):
        # Reset rectangle attributes
        self.rect.set_width(0)
        self.rect.set_height(0)
        self.rect.set_xy((0, 0))
        self.fig.canvas.draw()

    def autoscale_image(self, event):
        # Get current image data
        img_data = self.image_stack[self.index]

        # Calculate histogram and bin edges
        histogram, bin_edges = np.histogram(img_data, bins=256)

        # Define thresholds
        limit = histogram.sum() // 10
        if not hasattr(self, 'auto_threshold') or self.auto_threshold < 10:
            self.auto_threshold = 5000  # AUTO_THRESHOLD equivalent
        else:
            self.auto_threshold //= 2

        threshold = histogram.sum() // self.auto_threshold

        # Find minimum bin
        hmin = np.argmax(histogram > threshold)
        while histogram[hmin] > limit and hmin < histogram.size - 1:
            hmin += 1

        # Find maximum bin
        hmax = histogram.size - np.argmax(histogram[::-1] > threshold) - 1
        while histogram[hmax] > limit and hmax > 0:
            hmax -= 1

        if hmax >= hmin:
            # Calculate min and max values
            min_val = bin_edges[hmin]
            max_val = bin_edges[hmax + 1]  # +1 because bin_edges is 1 element longer than histogram

            if min_val == max_val:
                min_val = img_data.min()
                max_val = img_data.max()

            # Update image display
            self.image_display.set_clim(min_val, max_val)
        else:
            # Reset to default scale if min and max aren't valid
            self.image_display.set_clim(img_data.min(), img_data.max())

        self.fig.canvas.draw_idle()

    def reset_autoscale(self, event):
        # Get the original image data
        original_image_data = self.image_stack[self.index]

        # Reset the displayed image with the original data
        self.image_display.set_data(original_image_data)

        # Reset the color limit to the original range if needed
        self.image_display.set_clim(original_image_data.min(), original_image_data.max())

        # Redraw the canvas to reflect the updated image display
        self.fig.canvas.draw()
    def update_min(self, val):
        min_val = self.slider_min.val
        max_val = self.slider_max.val
        if min_val < max_val:  # Ensure the minimum is less than the maximum
            self.fit_display.set_clim(min_val, max_val)
        else:
            self.slider_min.set_val(max_val - 0.01)  # Prevent min from being greater than or equal to max

    def update_max(self, val):
        min_val = self.slider_min.val
        max_val = self.slider_max.val
        if max_val > min_val:  # Ensure the maximum is greater than the minimum
            self.fit_display.set_clim(min_val, max_val)
        else:
            self.slider_max.set_val(min_val + 0.01)
    def crop_and_save(self, event):
        # Get the current rectangle coordinates
        x0, y0 = self.rect.get_xy()
        width = self.rect.get_width()
        height = self.rect.get_height()

        # Convert to integer for indexing
        x0, y0, x1, y1 = int(x0), int(y0), int(x0 + width), int(y0 + height)

        # Ensure coordinates are within bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(x1, self.image_stack[self.index].shape[1]), min(y1, self.image_stack[self.index].shape[0])

        # If no area is selected, return without saving
        if x0 == x1 or y0 == y1:
            print("No area selected, no image saved.")
            return

        # Crop the image
        cropped_image = self.image_stack[:, y0:y1, x0:x1]
        save_path = f"{self.save_directory}/cropped_image_stack_{self.btn_crop_save_click_count}.tif"
        io.imsave(save_path, cropped_image)

        print(f"Cropped image saved at {save_path}")
        self.btn_crop_save_click_count += 1

    def save_intensity_data(self, event):
        # Extract the x and y data from the intensity plot
        x_data = self.eng
        y_data = self.intensity_line.get_ydata()

        # Create an array combining x and y data
        data = np.vstack((x_data, y_data)).T
        save_path = f"{self.save_directory}/intensity_data_{self.save_intensity_click_count}.txt"
        np.savetxt(save_path, data, header="", comments='')

        print(f"Intensity data saved at {save_path}")
        self.save_intensity_click_count += 1

    def select_directory(self, event):
        # Hide the main Tkinter window
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()

        # Open directory dialog
        self.save_directory = filedialog.askdirectory()  # This method returns the selected directory path as a string
        if not self.save_directory:  # If user closes the dialog without selection, revert to default path
            self.save_directory = "C:/Users/hwiho/Documents/"

        print(f"Selected directory: {self.save_directory}")

    def submit_min(self, val):
        try:
            min_val = float(val)
            current_max_val = self.fit_display.get_clim()[1]
            if min_val < current_max_val:  # Ensure the minimum is less than the maximum
                self.fit_display.set_clim(min_val, current_max_val)
                self.fig.canvas.draw_idle()  # Redraw the figure
            else:
                print("Min should be less than Max value")  # Error message in the console
        except ValueError:
            print("Invalid input for Min value")  # Error message in the console

    def submit_max(self, val):
        try:
            max_val = float(val)
            current_min_val = self.fit_display.get_clim()[0]
            if max_val > current_min_val:  # Ensure the maximum is greater than the minimum
                self.fit_display.set_clim(current_min_val, max_val)
                self.fig.canvas.draw_idle()  # Redraw the figure
            else:
                print("Max should be greater than Min value")  # Error message in the console
        except ValueError:
            print("Invalid input for Max value")  # Error message in the console


def main():
    gui = Post_process_basic()


if __name__ == "__main__":
    main()
