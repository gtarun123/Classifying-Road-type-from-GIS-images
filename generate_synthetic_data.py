import os
import cv2
import numpy as np
import random

CLASS_NAMES = ["Highway", "Street Road", "Rural Road", "Dirt Road", "Flyover Road"]

def create_dirs(base_dir, classes):
    # Clean up existing classes if needed, or make sure we don't have old ones
    for c in classes:
        os.makedirs(os.path.join(base_dir, c), exist_ok=True)

def generate_highway(width=224, height=224):
    # Gray background (asphalt)
    img = np.full((height, width, 3), 100, dtype=np.uint8)
    # White lines on the sides
    cv2.line(img, (20, 0), (20, height), (255, 255, 255), 3)
    cv2.line(img, (width - 20, 0), (width - 20, height), (255, 255, 255), 3)
    # Dashed yellow line in center
    for y in range(0, height, 30):
        cv2.line(img, (width // 2, y), (width // 2, y + 15), (0, 255, 255), 4)
    # Add noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def generate_street_road(width=224, height=224):
    # Dark gray background
    img = np.full((height, width, 3), 50, dtype=np.uint8)
    # Draw building blocks (light gray/blue-gray rectangles)
    for _ in range(15):
        x1 = random.randint(10, width - 40)
        y1 = random.randint(10, height - 40)
        w = random.randint(20, 45)
        h = random.randint(20, 45)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (140, 140, 140), -1)
    # Grid lines representing street roads (thick gray lines)
    for x in range(30, width, 60):
        cv2.line(img, (x, 0), (x, height), (90, 90, 90), 10)
    for y in range(30, height, 60):
        cv2.line(img, (0, y), (width, y), (90, 90, 90), 10)
    # Add noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def generate_rural_road(width=224, height=224):
    # Green background (fields/vegetation) - BGR [34, 139, 34]
    img = np.full((height, width, 3), [34, 139, 34], dtype=np.uint8)
    # Winding narrow grey line
    points = []
    for y in range(0, height + 10, 15):
        x = int(width // 2 + 40 * np.sin(y / 40.0))
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (128, 128, 128), 6)
    # Add some dark green patches (trees)
    for _ in range(8):
        cx = random.randint(10, width - 10)
        cy = random.randint(10, height - 10)
        cv2.circle(img, (cx, cy), random.randint(10, 18), (20, 80, 20), -1)
    # Add noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def generate_dirt_road(width=224, height=224):
    # Brown background (dirt/sand) - BGR [19, 69, 139] (RGB [139, 69, 19])
    img = np.full((height, width, 3), [19, 69, 139], dtype=np.uint8)
    # Winding dirt path (darker brown/beige) - BGR [10, 40, 80]
    points = []
    for y in range(0, height + 10, 15):
        x = int(width // 2 + 50 * np.cos(y / 50.0))
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (10, 40, 80), 16)
    # Add some random stone/gravel circles (light brown)
    for _ in range(12):
        cx = random.randint(10, width - 10)
        cy = random.randint(10, height - 10)
        cv2.circle(img, (cx, cy), random.randint(2, 5), (40, 100, 170), -1)
    # High noise for gravel texture
    noise = np.random.normal(0, 18, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def generate_flyover_road(width=224, height=224):
    # Green/grey ground landscape
    img = np.full((height, width, 3), [40, 90, 40], dtype=np.uint8)
    
    # Ground road passing underneath (horizontal dark grey street)
    cv2.line(img, (0, height // 2), (width, height // 2), (60, 60, 60), 20)
    
    # Elevated flyover bridge passing vertically
    # Concrete deck structure (light grey bridge body)
    bridge_left = width // 2 - 35
    bridge_right = width // 2 + 35
    cv2.rectangle(img, (bridge_left, 0), (bridge_right, height), (160, 160, 160), -1)
    
    # Draw drop shadow borders on the sides of the bridge (solid black/dark grey thin lines)
    cv2.line(img, (bridge_left - 3, 0), (bridge_left - 3, height), (20, 20, 20), 4)
    cv2.line(img, (bridge_right + 3, 0), (bridge_right + 3, height), (20, 20, 20), 4)
    
    # Bridge deck lane dividers (white dashed line)
    for y in range(0, height, 25):
        cv2.line(img, (width // 2, y), (width // 2, y + 12), (255, 255, 255), 3)
        
    # Bridge concrete safety barriers (side walls)
    cv2.line(img, (bridge_left, 0), (bridge_left, height), (200, 200, 200), 2)
    cv2.line(img, (bridge_right, 0), (bridge_right, height), (200, 200, 200), 2)
    
    # Add noise
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def save_dataset(base_dir, num_images):
    create_dirs(base_dir, CLASS_NAMES)
    for c in CLASS_NAMES:
        print(f"Generating {num_images} images for {base_dir}/{c}...")
        for i in range(num_images):
            if c == "Highway":
                img = generate_highway()
            elif c == "Street Road":
                img = generate_street_road()
            elif c == "Rural Road":
                img = generate_rural_road()
            elif c == "Dirt Road":
                img = generate_dirt_road()
            else:
                img = generate_flyover_road()
            
            filename = f"{c.lower().replace(' ', '_')}_{i}.jpg"
            filepath = os.path.join(base_dir, c, filename)
            cv2.imwrite(filepath, img)

if __name__ == "__main__":
    # Generate train and validation source (50 images per class)
    save_dataset("dataset/train", 50)
    # Generate test source (10 images per class)
    save_dataset("dataset/test", 10)
    
    # Save 5 nice high-res samples to the root workspace directory for user demo/upload
    cv2.imwrite("sample_highway.jpg", generate_highway(400, 400))
    cv2.imwrite("sample_street.jpg", generate_street_road(400, 400))
    cv2.imwrite("sample_rural.jpg", generate_rural_road(400, 400))
    cv2.imwrite("sample_dirt.jpg", generate_dirt_road(400, 400))
    cv2.imwrite("sample_flyover.jpg", generate_flyover_road(400, 400))
    
    print("Synthetic dataset generation complete with 5 classes!")
