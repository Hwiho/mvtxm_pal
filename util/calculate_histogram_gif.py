import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, square
import matplotlib.pyplot as plt
import os
import math
from PIL import Image
from scipy.ndimage import gaussian_filter1d


def load_image(file_path):
    """Load an image from a file path."""
    return io.imread(file_path)


def process_image(image, shell_range=(1, 3), core_start=5):
    """Apply erosion to separate core and shell based on specific iteration ranges."""
    threshold_value = threshold_otsu(image)
    binary_image = image > threshold_value
    eroded_image = binary_image
    cumulative_shell = np.zeros_like(binary_image)
    cumulative_core = np.zeros_like(binary_image)

    for i in range(1, core_start + 1):  # Run erosion up to core_start
        previous_image = eroded_image
        eroded_image = erosion(eroded_image, square(3))
        deleted_parts = previous_image & ~eroded_image

        # Define shell in the specified range
        if shell_range[0] <= i <= shell_range[1]:
            cumulative_shell |= deleted_parts

    # After core_start, the remaining parts are defined as core
    cumulative_core = eroded_image

    core = image * cumulative_core
    shell = image * cumulative_shell
    return core, shell


def calculate_normalized_histogram(region, bins=100, smooth_sigma=1, range=None):
    """Calculate smoothed normalized histogram for non-zero values in a region within a fixed range."""
    values = region[region != 0]
    hist, bin_edges = np.histogram(values, bins=bins, density=True, range=range)
    hist_smooth = gaussian_filter1d(hist, sigma=smooth_sigma)  # Apply Gaussian smoothing
    return hist_smooth, bin_edges[:-1]


def export_histogram(hist, bins, filename):
    """Save histogram data to a text file."""
    data = np.column_stack((bins, hist))
    np.savetxt(filename, data, fmt='%f', header="Bin Edges\tFrequency")


def get_global_intensity_range(folder_path):
    """Determine the global min (non-zero) and max intensity values across all images in the folder."""
    min_intensity, max_intensity = float('inf'), float('-inf')
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image = io.imread(image_path)

        # Get non-zero values only
        non_zero_values = image[image > 0]

        # Update global min and max based on non-zero values
        if non_zero_values.size > 0:
            min_intensity = min(min_intensity, non_zero_values.min())
            max_intensity = max(max_intensity, non_zero_values.max())

    return min_intensity, max_intensity


def process_and_plot_histograms(folder_path, iterations=20, bins=100, cols=2):
    """Process each image to get core and shell histograms, normalize, export, and plot them in a grid."""
    # Get global intensity range, ignoring zeros
    global_min, global_max = get_global_intensity_range(folder_path)

    # Get list of images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])
    rows = math.ceil(len(image_files) / cols)

    # Create a figure for all histograms
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig.suptitle("Overlayed Normalized Core and Shell Histograms Over Time", fontsize=16)

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        image = io.imread(image_path)

        # Process image to separate core and shell
        core, shell = process_image(image, shell_range=(2, 7), core_start=15)

        # Calculate normalized histograms for core and shell with fixed range
        core_hist, core_bins = calculate_normalized_histogram(core, bins, range=(global_min, global_max))
        shell_hist, shell_bins = calculate_normalized_histogram(shell, bins, range=(global_min, global_max))

        # Export core and shell histograms
        core_filename = f"core_histogram_time_{idx}.txt"
        shell_filename = f"shell_histogram_time_{idx}.txt"
        export_histogram(core_hist, core_bins, core_filename)
        export_histogram(shell_hist, shell_bins, shell_filename)

        # Calculate row and column index for subplot
        row = idx // cols
        col = idx % cols

        # Plot core and overlay shell histogram
        ax = axes[row, col]
        ax.bar(core_bins, core_hist, width=(core_bins[1] - core_bins[0]), color='green', alpha=0.5, label="Core")
        ax.bar(shell_bins, shell_hist, width=(shell_bins[1] - shell_bins[0]), color='blue', alpha=0.5, label="Shell")

        ax.set_title(f"Normalized Histogram Overlay (Time {idx})")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Normalized Frequency")
        ax.set_xlim(global_min, global_max)  # Set fixed x-axis limit
        ax.legend()

    # Adjust layout for readability
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def get_max_y_value(folder_path, iterations=20, bins=100, smooth_sigma=1):
    """Determine the maximum y-axis value across all histograms to fix the y-axis range."""
    max_y = 0
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])

    global_min, global_max = get_global_intensity_range(folder_path)

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image = io.imread(image_path)

        # Process image to separate core and shell
        core, shell = process_image(image, shell_range=(2, 7), core_start=15)

        # Calculate smoothed normalized histograms for core and shell with the fixed range
        core_hist, _ = calculate_normalized_histogram(core, bins, smooth_sigma, range=(global_min, global_max))
        shell_hist, _ = calculate_normalized_histogram(shell, bins, smooth_sigma, range=(global_min, global_max))

        # Update max_y if current histogram has higher values
        max_y = max(max_y, core_hist.max(), shell_hist.max())

    return max_y


def process_image_histograms_for_gif(folder_path, iterations=20, bins=100, smooth_sigma=1, output_gif="histogram_animation.gif"):
    """Process each image, save overlayed core and shell histograms in the folder, and create a GIF animation, with fixed x-axis and y-axis."""
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])

    # Get global intensity range for x-axis (ignoring zeros) and max y-value for y-axis
    global_min, global_max = get_global_intensity_range(folder_path)
    max_y = get_max_y_value(folder_path, iterations, bins, smooth_sigma)
    frames = []

    for idx, filename in enumerate(image_files):
        if idx == 8 or idx == 9:  # Skip indexes 8 and 9
            continue

        image_path = os.path.join(folder_path, filename)
        image = io.imread(image_path)

        # Calculate the elapsed time for the current image in minutes
        elapsed_time = idx * 7 * 5  # Each measurement took 7 minutes

        # Process image to separate core and shell
        core, shell = process_image(image, shell_range=(2, 7), core_start=15)

        # Calculate smoothed normalized histograms for core and shell with fixed range
        core_hist, core_bins = calculate_normalized_histogram(core, bins, smooth_sigma, range=(global_min, global_max))
        shell_hist, shell_bins = calculate_normalized_histogram(shell, bins, smooth_sigma, range=(global_min, global_max))

        # Plot overlayed histogram with fixed x and y-axis
        plt.figure(figsize=(8, 6))
        plt.bar(core_bins, core_hist, width=(core_bins[1] - core_bins[0]), color='green', alpha=0.5, label="Core")
        plt.bar(shell_bins, shell_hist, width=(shell_bins[1] - shell_bins[0]), color='blue', alpha=0.5, label="Shell")

        plt.xlim(global_min, global_max)  # Set fixed x-axis limit
        plt.ylim(0, max_y)  # Set fixed y-axis limit
        plt.title(f"Overlayed Smoothed Core and Shell Histogram (Elapsed Time: {elapsed_time} mins)")
        plt.xlabel("Intensity")
        plt.ylabel("Normalized Frequency")
        plt.legend()

        # Save the plot in the same folder with a unique name
        hist_filename = os.path.join(folder_path, f"histogram_time_{idx}.png")
        plt.savefig(hist_filename)
        plt.close()

        # Open the saved image and append it to the frames list for GIF creation
        frames.append(Image.open(hist_filename))

    # Create GIF
    gif_path = os.path.join(folder_path, output_gif)
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=1)

    print(f"Individual histogram images saved in folder and GIF saved as {gif_path}")


# Main Execution
folder_path = r'D:\Insitu_PXM\pouch15_rest\Selected'
iteration = 10
process_and_plot_histograms(folder_path, iterations=iteration)
process_image_histograms_for_gif(folder_path, iterations=iteration)
