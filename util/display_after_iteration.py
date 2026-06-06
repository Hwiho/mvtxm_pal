import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, square
import matplotlib.pyplot as plt


def load_image(file_path):
    """Load an image from a file path."""
    return io.imread(file_path)


def process_image(image, shell_range=(1, 3), core_start=20, total_iterations=30):
    """Apply erosion to separate core and shell based on specific iteration ranges."""
    threshold_value = threshold_otsu(image)
    binary_image = image > threshold_value
    eroded_image = binary_image
    cumulative_shell = np.zeros_like(binary_image)
    cumulative_core = np.zeros_like(binary_image)

    for i in range(1, total_iterations + 1):  # Run erosion up to the total iterations
        previous_image = eroded_image
        eroded_image = erosion(eroded_image, square(3))
        deleted_parts = previous_image & ~eroded_image

        # Define shell in the specified range
        if shell_range[0] <= i <= shell_range[1]:
            cumulative_shell |= deleted_parts
        # Define core from core_start and onwards, including all remaining parts at the end
        elif i >= core_start:
            cumulative_core |= deleted_parts

    # Final core includes remaining eroded image after all iterations
    cumulative_core |= eroded_image

    core = image * cumulative_core
    shell = image * cumulative_shell
    return core, shell


def display_core_shell(image_path, shell_range=(1, 3), core_start=5, total_iterations=30):
    """Load an image, process it for core and shell, and display them."""
    image = load_image(image_path)
    core, shell = process_image(image, shell_range, core_start, total_iterations)

    # Display the original image, core, and shell
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(core, cmap='gray')
    plt.title("Core after 30 Iterations")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(shell, cmap='gray')
    plt.title("Shell after 30 Iterations")
    plt.axis("off")

    plt.show()


# Path to one image for display
image_path = r'D:\Insitu_PXM\pouch15_rest\rest_Selected\Image_dataset_id_pouch15_insiturest_033C_exp25_acc7_wopressMULTI000_pos_00_peaksite.tif'  # Replace with the actual path to your image
display_core_shell(image_path, shell_range=(0, 3), core_start=10, total_iterations=25)