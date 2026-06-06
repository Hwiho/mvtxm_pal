import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, square
import matplotlib.pyplot as plt
import os

def load_image(file_path):
    """Load an image from a file path."""
    return io.imread(file_path)

def process_image(image, iterations=2):
    """Apply erosion to separate core and shell, returning the core and shell regions."""
    threshold_value = threshold_otsu(image)
    binary_image = image > threshold_value
    eroded_image = binary_image
    cumulative_deleted_parts = np.zeros_like(binary_image)

    for _ in range(iterations):
        previous_image = eroded_image
        eroded_image = erosion(eroded_image, square(3))
        deleted_parts = previous_image & ~eroded_image
        cumulative_deleted_parts |= deleted_parts

    core = image * eroded_image
    shell = image * cumulative_deleted_parts
    return core, shell

def calculate_statistics(region):
    """Calculate mean and standard deviation for non-zero values in a region."""
    values = region[region != 0]
    return values.mean(), values.std()

def analyze_images_in_folder(folder_path, iterations=20):
    """Process all images in a folder and calculate statistics over time."""
    core_means, core_stds = [], []
    shell_means, shell_stds = [], []
    file_names = []

    # Prepare a file to save statistics
    stats_file_path = os.path.join(folder_path, "core_shell_statistics.txt")
    with open(stats_file_path, "w") as stats_file:
        stats_file.write("File Name\tCore Mean\tCore Std\tShell Mean\tShell Std\n")

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.tif'):
                image_path = os.path.join(folder_path, filename)
                image = load_image(image_path)

                # Process image to separate core and shell
                core, shell = process_image(image, iterations)

                # Calculate statistics for core and shell
                core_mean, core_std = calculate_statistics(core)
                shell_mean, shell_std = calculate_statistics(shell)

                core_means.append(core_mean)
                core_stds.append(core_std)
                shell_means.append(shell_mean)
                shell_stds.append(shell_std)
                file_names.append(filename)

                # Write statistics to the file
                stats_file.write(f"{filename}\t{core_mean:.3f}\t{core_std:.3f}\t{shell_mean:.3f}\t{shell_std:.3f}\n")

    return core_means, core_stds, shell_means, shell_stds, file_names

def save_histogram_data(hist_data, bin_edges, file_path):
    """Save histogram data (normalized) to a text file."""
    data = np.column_stack((bin_edges, hist_data))
    np.savetxt(file_path, data, fmt='%f', header="Bin Edges\tNormalized Frequency")

def plot_histograms(folder_path, core, shell, filename):
    """Plot, normalize, and save histograms for core and shell at each time point."""
    histogram_folder = os.path.join(folder_path, "histogram")
    os.makedirs(histogram_folder, exist_ok=True)

    # Calculate normalized histograms
    core_values = core[core != 0].flatten()
    shell_values = shell[shell != 0].flatten()

    core_hist, core_bins = np.histogram(core_values, bins=100, density=True)
    shell_hist, shell_bins = np.histogram(shell_values, bins=100, density=True)

    # Save normalized histogram data to text files
    core_hist_file = os.path.join(histogram_folder, f"core_histogram_{filename}.txt")
    shell_hist_file = os.path.join(histogram_folder, f"shell_histogram_{filename}.txt")
    save_histogram_data(core_hist, core_bins[:-1], core_hist_file)
    save_histogram_data(shell_hist, shell_bins[:-1], shell_hist_file)

    # Plot normalized histograms
    plt.figure(figsize=(10, 6))
    plt.bar(core_bins[:-1], core_hist, width=(core_bins[1] - core_bins[0]), color='green', alpha=0.5, label="Core")
    plt.bar(shell_bins[:-1], shell_hist, width=(shell_bins[1] - shell_bins[0]), color='blue', alpha=0.5, label="Shell")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.title(f"Normalized Histogram: {filename}")
    plt.legend()

    # Save histogram plot
    histogram_plot_path = os.path.join(histogram_folder, f"histogram_{filename}.png")
    plt.savefig(histogram_plot_path)
    plt.close()

def plot_statistics(filenames, core_means, core_stds, shell_means, shell_stds):
    """Plot mean and standard deviation of core and shell values over time (by filename)."""
    x = np.arange(len(filenames))
    plt.figure(figsize=(12, 6))
    plt.errorbar(x, core_means, yerr=core_stds, label='Core', marker='o', color='green', capsize=3)
    plt.errorbar(x, shell_means, yerr=shell_stds, label='Shell', marker='x', color='blue', capsize=3)
    plt.xticks(x, filenames, rotation=90)
    plt.xlabel('Image Filename')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity and Std Dev of Core and Shell')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def main(folder_path, iterations=20):
    core_means, core_stds, shell_means, shell_stds, file_names = analyze_images_in_folder(folder_path, iterations)

    # Plot statistics over time using filenames
    plot_statistics(file_names, core_means, core_stds, shell_means, shell_stds)

    # Generate and save histograms for each image
    for filename in file_names:
        image_path = os.path.join(folder_path, filename)
        image = load_image(image_path)
        core, shell = process_image(image, iterations)
        plot_histograms(folder_path, core, shell, filename)

# Example usage
folder_path = r"D:\pressure_comparison"  # 수정 필요
main(folder_path, iterations=20)
