import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

CLASS_NAMES = ["Highway", "Street Road", "Rural Road", "Dirt Road", "Flyover Road"]

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Standard preprocessor:
    1. Loads an image using OpenCV in BGR format.
    2. Converts it to RGB color space.
    3. Resizes it to target size (224x224) using bilinear interpolation.
    4. Normalizes pixel values to [0.0, 1.0].
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    image = image.astype("float32") / 255.0
    return image

# ====================================================
# ADVANCED DATA AUGMENTATION MODULE (CV2 & NUMPY)
# ====================================================

def augment_rotate(image, max_angle=15):
    """Rotate image by a random angle between -max_angle and max_angle."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def augment_flip(image):
    """Randomly apply horizontal or vertical flips."""
    flip_type = random.choice([-1, 0, 1, 999])  # -1: both, 0: vertical, 1: horizontal, 999: no flip
    if flip_type == 999:
        return image
    return cv2.flip(image, flip_type)

def augment_zoom(image, zoom_factor_range=(0.9, 1.1)):
    """Zoom in or out of the image randomly."""
    zoom = random.uniform(zoom_factor_range[0], zoom_factor_range[1])
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom), int(w * zoom)
    
    if zoom < 1.0:  # Zoom out: shrink image and pad boundaries
        shrunk = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        return cv2.copyMakeBorder(shrunk, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    else:  # Zoom in: enlarge and crop center
        enlarged = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return enlarged[start_y:start_y+h, start_x:start_x+w]

def augment_brightness(image, max_delta=0.15):
    """Adjust image brightness randomly."""
    delta = random.uniform(-max_delta, max_delta)
    return np.clip(image + delta, 0.0, 1.0)

def augment_contrast(image, factor_range=(0.85, 1.15)):
    """Adjust image contrast randomly."""
    factor = random.uniform(factor_range[0], factor_range[1])
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0.0, 1.0)

def augment_shifts(image, max_shift_pct=0.1):
    """Apply height and width translation shifts."""
    h, w = image.shape[:2]
    shift_x = int(random.uniform(-max_shift_pct, max_shift_pct) * w)
    shift_y = int(random.uniform(-max_shift_pct, max_shift_pct) * h)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def augment_shear(image, max_shear=0.08):
    """Apply shear mapping transformations."""
    h, w = image.shape[:2]
    shear_x = random.uniform(-max_shear, max_shear)
    shear_y = random.uniform(-max_shear, max_shear)
    matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def augment_crop(image, crop_size=(192, 192)):
    """Randomly crop a subregion and resize back to original dimensions."""
    h, w = image.shape[:2]
    ch, cw = crop_size
    max_y = h - ch
    max_x = w - cw
    start_y = random.randint(0, max_y)
    start_x = random.randint(0, max_x)
    cropped = image[start_y:start_y+ch, start_x:start_x+cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def apply_full_augmentation(image):
    """Apply a comprehensive combined augmentation pipeline randomly."""
    img = image.copy()
    if random.random() > 0.5:
        img = augment_rotate(img)
    if random.random() > 0.5:
        img = augment_flip(img)
    if random.random() > 0.5:
        img = augment_zoom(img)
    if random.random() > 0.5:
        img = augment_brightness(img)
    if random.random() > 0.5:
        img = augment_contrast(img)
    if random.random() > 0.5:
        img = augment_shifts(img)
    if random.random() > 0.5:
        img = augment_shear(img)
    if random.random() > 0.6:
        img = augment_crop(img)
    return img

def generate_augmentation_demo(sample_path="sample_highway.jpg", output_path="models/augmentation_demo.png"):
    """
    Generate a 3x3 grid visualizing distinct augmentations applied to a sample image.
    This serves as an excellent visual asset for faculty evaluations.
    """
    if not os.path.exists(sample_path):
        print(f"Demo image {sample_path} not found. Skipping grid generation.")
        return
        
    img = load_and_preprocess_image(sample_path)
    
    titles = [
        "Original", "Rotated", "Flipped", 
        "Zoomed", "Brightness Adjust", "Contrast Shift",
        "Translation Shifts", "Shear Map", "Random Crop + Resize"
    ]
    
    images = [
        img,
        augment_rotate(img, 25),
        augment_flip(img),
        augment_zoom(img, (1.2, 1.2)),
        augment_brightness(img, 0.25),
        augment_contrast(img, (0.7, 0.7)),
        augment_shifts(img, 0.15),
        augment_shear(img, 0.15),
        augment_crop(img, (160, 160))
    ]
    
    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=12, fontweight='bold', color='#1e293b')
        plt.axis('off')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Successfully generated visual augmentation demo at: {output_path}")

if __name__ == "__main__":
    print("Testing data preprocessing and advanced augmentation pipeline...")
    # Check if a sample image is available to construct the demo chart
    generate_augmentation_demo()
    print("Pre-processing checks complete!")
