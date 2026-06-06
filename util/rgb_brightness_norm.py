import numpy as np
from skimage import io, color
from skimage.exposure import rescale_intensity


def normalize_brightness(image, target_brightness=0.05):
    """Normalize the brightness of a single RGB image in HSV space to retain color information."""
    # Convert RGB to HSV
    hsv_image = color.rgb2hsv(image)

    # Calculate current brightness as the mean of the Value channel
    current_brightness = np.mean(hsv_image[:, :, 2])

    # Calculate the adjustment ratio
    brightness_ratio = target_brightness / current_brightness

    # Adjust the Value channel
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_ratio, 0, 1)

    # Convert back to RGB
    normalized_image = color.hsv2rgb(hsv_image)
    return normalized_image


def process_rgb_stack_for_brightness(stack_path, target_brightness=0.5, output_path="normalized_stack.tif"):
    """Process each frame in an RGB stack to normalize brightness while retaining color information."""
    # Read the stack
    stack = io.imread(stack_path).astype(np.float32) / 255  # Normalize pixel values to [0, 1]

    # Prepare an array to store the normalized frames
    normalized_stack = np.zeros_like(stack)

    # Process each frame
    for idx in range(stack.shape[0]):
        frame = stack[idx]

        # Normalize brightness for the current frame
        normalized_frame = normalize_brightness(frame, target_brightness)

        # Store the normalized frame
        normalized_stack[idx] = normalized_frame
        print(f"Normalized brightness for frame {idx + 1}/{stack.shape[0]}")

    # Save the normalized stack
    io.imsave(output_path, (normalized_stack * 255).astype(np.uint8))
    print(f"Normalized stack saved as {output_path}")


# Path to the RGB stack
stack_path = r'D:\Insitu_PXM\pouch15_rest\Selected\colormap\rgb_stack.tif'  # Update with the path to your RGB stack
process_rgb_stack_for_brightness(stack_path, output_path= r'D:\Insitu_PXM\pouch15_rest\Selected\colormap\normalized_stack.tif')