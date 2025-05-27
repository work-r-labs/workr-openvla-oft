import pickle
import numpy as np
import cv2

# Load the pickle file to examine the image data
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

print("Keys in observation:", observation.keys())

# Check the full_image
full_image = observation["full_image"]
print(f"Full image shape: {full_image.shape}")
print(f"Full image dtype: {full_image.dtype}")
print(f"Full image min: {full_image.min()}")
print(f"Full image max: {full_image.max()}")
print(f"Full image mean: {full_image.mean()}")

# Check if there's a wrist image
if "wrist_image" in observation:
    wrist_image = observation["wrist_image"]
    print(f"Wrist image shape: {wrist_image.shape}")
    print(f"Wrist image dtype: {wrist_image.dtype}")
    print(f"Wrist image min: {wrist_image.min()}")
    print(f"Wrist image max: {wrist_image.max()}")
    print(f"Wrist image mean: {wrist_image.mean()}")

# Save the raw images to see what they look like
cv2.imwrite("debug_full_image_raw.jpg", full_image)

# Try different normalizations
if full_image.max() <= 1.0:
    # Image is in [0,1] range, scale to [0,255]
    full_image_scaled = (full_image * 255).astype(np.uint8)
    cv2.imwrite("debug_full_image_scaled.jpg", full_image_scaled)
    print("Saved scaled version (0-1 to 0-255)")

if full_image.min() < 0:
    # Image might be normalized to [-1,1], scale to [0,255]
    full_image_norm = ((full_image + 1) * 127.5).astype(np.uint8)
    cv2.imwrite("debug_full_image_normalized.jpg", full_image_norm)
    print("Saved normalized version (-1,1 to 0-255)")

print("\nFirst few pixel values:")
print(full_image[:5, :5, 0])  # Show first 5x5 pixels of first channel 