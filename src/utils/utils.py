import math

import numpy as np
import torch


def exponential_decrease(t):
    """
    Returns an integer between 15 and 1 that decreases exponentially.
    At t=0, it returns 15.
    At t=300, it returns 1.
    """
    if t < 0:
        t = 0
    elif t > 300:
        t = 300
    return round(
        1 + (15 - 1) * math.exp(-t / 50)
    )  # Adjust the divisor for controlling steepness


def linear_decrease(t):
    """
    Returns an integer between 15 and 1 that decreases linearly.
    At t=0, it returns 15.
    At t=300, it returns 1.
    """
    if t < 0:
        t = 0
    elif t > 300:
        t = 300
    return round(15 - (14 / 300) * t)


def noise_image(image, noise_std_dev, noise_mean=0):
    """
    Adds Gaussian noise to the input image.

    Args:
        image (torch.Tensor): Input image in the shape (C, H, W).
        noise_std_dev (float): Standard deviation of the Gaussian noise.
        mean (float): Mean of the Gaussian noise.

    Returns:
        torch.Tensor: The noisy image as a PyTorch tensor in the shape (C, H, W),
    """
    # Generate Gaussian noise
    noise = torch.normal(
        mean=noise_mean,
        std=noise_std_dev,
        size=image.shape,
        device=image.device,
    )

    # Add noise to the image
    noisy_image = image + noise
    return noisy_image


def patch_image(image, patch_size, device, position):
    """
    Applies a black patch of size `patch_size` x `patch_size` to a random location in the input image,
    normalizes the image pixel values to the range [0, 1].

    Args:
        image (torch.Tensor): Input image in the shape (H, W, C) with pixel values in [0, 255].
        patch_size (int): The size of the black patch to be applied.
        device (torch.device): The target device for the output tensor.

    Returns:
        torch.Tensor: The processed image as a PyTorch tensor in the shape (C, H, W) with pixel values in [0, 1],
    """
    top_left_x, top_left_y = position[0], position[1]

    # Place the black patch (set the pixel values to 0 in the patch area)
    if patch_size > 0:
        image[
            top_left_x : top_left_x + patch_size,
            top_left_y : top_left_y + patch_size,
            :,
        ] = 0
    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)
    # Send data to device
    image = image.to(device, non_blocking=True)

    return image


def generate_random_position(width, height, patch_size):
    """
    Generate a random position for the patch.
    """
    return (
        np.random.randint(0, height - patch_size),
        np.random.randint(0, width - patch_size),
    )


def generate_brownian_step(current_position, width, height, step_std_dev=1.0):
    """
    Update the position of the patch following a brownian movement of std `step_std_dev`
    """
    x, y = current_position
    # Generate random displacements from a Gaussian distribution
    dx = np.random.normal(0, step_std_dev)
    dy = np.random.normal(0, step_std_dev)
    # Compute the new position
    new_x = x + dx
    new_y = y + dy
    # Clamp the position within the bounds
    new_x = max(0, min(int(new_x), int(width)))
    new_y = max(0, min(int(new_y), int(height)))
    return (new_x, new_y)
