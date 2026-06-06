import numpy as np
import pandas as pd
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, square
import matplotlib.pyplot as plt
import os

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

def calculate_statistics(region):
    """Calculate mean and standard deviation for non-zero values in a region."""
    values = region[region != 0]
    return values.mean(), values.std()

def analyze_images_in_folder(folder_path, shell_range=(2, 5), core_start=5):
    """Process all images in a folder and save statistics over time to a CSV file."""
    stats_data = []

    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith('.tif'):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)

            core, shell = process_image(image, shell_range, core_start)

            core_mean, core_std = calculate_statistics(core)
            shell_mean, shell_std = calculate_statistics(shell)

            stats_data.append({
                "Time Point": idx,
                "Core Mean": core_mean,
                "Core Std": core_std,
                "Shell Mean": shell_mean,
                "Shell Std": shell_std
            })

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_csv_path = os.path.join(folder_path, "core_shell_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Statistics saved to {stats_csv_path}")

    return stats_df

def save_histogram_data(hist_data, bin_edges, file_path):
    """Save histogram data (normalized) to a CSV file."""
    hist_df = pd.DataFrame({"Bin Edges": bin_edges[:-1], "Normalized Frequency": hist_data})
    hist_df.to_csv(file_path, index=False)

def plot_histograms(folder_path, core, shell, time_point):
    """Plot, normalize, and save histograms for core and shell at each time point."""
    histogram_folder = os.path.join(folder_path, "histogram_csv")
    os.makedirs(histogram_folder, exist_ok=True)

    core_values = core[core != 0].flatten()
    shell_values = shell[shell != 0].flatten()

    core_hist, core_bins = np.histogram(core_values, bins=100, density=True)
    shell_hist, shell_bins = np.histogram(shell_values, bins=100, density=True)

    core_hist_file = os.path.join(histogram_folder, f"core_histogram_time_{time_point}.csv")
    shell_hist_file = os.path.join(histogram_folder, f"shell_histogram_time_{time_point}.csv")
    save_histogram_data(core_hist, core_bins, core_hist_file)
    save_histogram_data(shell_hist, shell_bins, shell_hist_file)

    plt.figure(figsize=(10, 6))
    plt.bar(core_bins[:-1], core_hist, width=(core_bins[1] - core_bins[0]), color='green', alpha=0.5, label="Core")
    plt.bar(shell_bins[:-1], shell_hist, width=(shell_bins[1] - shell_bins[0]), color='blue', alpha=0.5, label="Shell")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.title(f"Normalized Histogram at Time Point {time_point}")
    plt.legend()

    histogram_plot_path = os.path.join(histogram_folder, f"histogram_time_{time_point}.png")
    plt.savefig(histogram_plot_path)
    plt.close()

def plot_statistics(time_points, core_means, core_stds, shell_means, shell_stds):
    """Plot mean and standard deviation of core and shell values over time."""
    plt.figure(figsize=(12, 6))
    plt.errorbar(time_points, core_means, yerr=core_stds, label='Core', marker='o', color='green', capsize=3)
    plt.errorbar(time_points, shell_means, yerr=shell_stds, label='Shell', marker='x', color='blue', capsize=3)
    plt.xlabel('Time Point')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity and Std Dev of Core and Shell Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(folder_path, shell_range=(2, 5), core_start=5):
    stats_df = analyze_images_in_folder(folder_path, shell_range, core_start)
    time_points = np.arange(len(stats_df))

    plot_statistics(time_points, stats_df["Core Mean"], stats_df["Core Std"], stats_df["Shell Mean"], stats_df["Shell Std"])

    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith('.tif'):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)
            core, shell = process_image(image, shell_range, core_start)
            plot_histograms(folder_path, core, shell, idx)

# Folder path with images
folder_path = r"D:\pressure_comparison"  # Update with your folder path
main(folder_path, shell_range=(2, 7), core_start=15)
