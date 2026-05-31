Team Members    Name                   USN
    1           Tarun Kumar G          KUB23CSE146
    2           B Basavaraj Reddy      KUB23CSE015
    3           D M Isaq               KUB23CSE033
    
# Road Type Classification from GIS Images using Deep Learning

This academic project implements a state-of-the-art Deep Learning classification pipeline that categorizes satellite, GIS, and aerial road segments into five main road types: **Highway**, **Street Road**, **Rural Road**, **Dirt Road**, and **Flyover Road**. 
The backend utilizes **Transfer Learning with MobileNetV2** (via TensorFlow 2.x and Keras), and the frontend is a beautifully designed, responsive **Flask Web Application** styled with modern dark-mode glassmorphic aesthetics.

---

## 🚀 Key Features
- **5-Class Classifier**: highway, urban street grids, countryside rural roads, unpaved dirt tracks, and concrete elevated flyovers.
- **Transfer Learning Backbone**: Uses pre-trained MobileNetV2 features for resource-efficient training and high evaluation scores.
- **Data Augmentation**: Robust real-time image augmentation (flips, rotations, contrast adjustments, zooms) to combat dataset scarcity.
- **Premium Web Dashboard**: Beautiful Glassmorphic CSS frontend allowing files to be dragged and dropped, showing glowing progress meters and structured printable reports.
- **Comprehensive Docs**: Includes full academic report, system diagrams, viva question bank, and slideshow presentation outline under the `docs/` folder.

---

## 📁 Project Folder Structure
```
ML/
├── flask_app.py               # Flask Main Web Server
├── app.py                     # Streamlit Frontend (Alternate)
├── train.py                   # Model Training Pipeline (MobileNetV2 / CNN)
├── predict.py                 # Core CLI Prediction Script
├── generate_synthetic_data.py # Synthetic GIS Dataset Generator
├── requirements.txt           # Python Project Dependencies
├── README.md                  # Project Documentation
├── dataset/                   # Dataset Directory
│   ├── train/                 # Training set directories (80% split)
│   │   ├── Highway/
│   │   ├── Street Road/
│   │   ├── Rural Road/
│   │   ├── Dirt Road/
│   │   └── Flyover Road/
│   └── test/                  # Test set directories (10 samples per class)
├── models/                    # Model Output Directory
│   ├── road_type_classifier.h5  # Peak Trained Model Weights
│   ├── training_history.png   # Accuracy/Loss Plots
│   ├── confusion_matrix.png   # Validation Confusion Matrix
│   └── test_results.txt       # Unseen Test Set Metrics
├── templates/                 # Flask UI Templates
│   ├── index.html             # Glassmorphic Landing & Upload Page
│   └── result.html            # Glowing Prediction Report Page
└── docs/                      # Comprehensive Academic Material
    ├── report.md              # 14-Section Academic Report
    ├── viva_questions.md      # Faculty Viva Q&A Guide
    ├── presentation_points.md # Slide Outlines for presentation
    └── diagrams.md            # Workflow & System Diagrams
```

---

## 📖 Theoretical Explanations (For Viva & Faculty Presentation)

### 1. How a Convolutional Neural Network (CNN) Works (In Simple Language)
Think of a standard CNN as a series of visual magnifying glasses. When you look at a satellite image of a road, you don't instantly see a "Highway." Instead, your eyes first spot small elements: straight lines, concrete colors, yellow dash marks, and trees.
- **Convolutional Layers**: These act as filters. They slide across the image pixel-by-pixel, identifying local shapes like straight lines (highways), grids (street networks), curves (rural paths), or concrete borders (flyovers).
- **Pooling Layers (MaxPooling)**: These act as consolidators. They shrink the image dimension while preserving the most prominent features. For instance, if a white marker line is spotted, pooling summarizes its existence into a smaller footprint.
- **Dense Layer (Softmax Classifier)**: This is the decision-maker. It gathers all the processed abstract visual cues (e.g., straight gray strip + center yellow dash + high contrast lines) and outputs a probability score for each of the 5 road classes.

### 2. What is Transfer Learning and Why is it Better?
In traditional machine learning, models are trained from scratch. This requires thousands of high-quality images and hours of expensive GPU computation. If the training dataset is small, the model simply memorizes the images (overfitting) and fails on new images.
**Transfer Learning** solves this by reusing a pre-trained network (like MobileNetV2) that has already spent weeks training on 1.4 million images (ImageNet dataset). It already knows how to detect curves, shadows, gradients, and textures perfectly. 
We "freeze" these base layers and only train a small customized classifier head at the end. 
- *Why it is better:*
  - Achieves near 100% accuracy in just 5 epochs.
  - Requires only a small local dataset.
  - Reduces training time from hours to seconds on a standard laptop CPU.

### 3. Why MobileNetV2 is Used?
MobileNetV2 is specifically designed for mobile devices and embedded systems with limited processing power.
- **Depthwise Separable Convolutions**: Traditional convolutions process spatial details and colors together, causing massive parameter sizes. MobileNetV2 splits this process into a Depthwise step (spatial details per channel) and a Pointwise step (linear channel combinations).
- This unique approach reduces the computational mathematical workload by **8 to 9 times** compared to older architectures like VGG16, making it extremely lightweight (~2.2M parameters vs VGG16's ~138M) and perfect for rapid server responses in web applications.

---

## 🛠️ Step-by-Step Installation & Running Guide

### 1. Setup Environment & Install Dependencies
Initialize a Python virtual environment and install the required modules:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate the Synthetic Dataset
Run the data generator to instantly generate a realistic simulated dataset with 50 images per class for training and 10 images per class for testing, as well as 5 high-resolution demo images:
```bash
python generate_synthetic_data.py
```

### 3. Train the MobileNetV2 Transfer Learning Model
Execute the training script. This script automatically splits the training directory into 80% training / 20% validation, runs transfer learning using MobileNetV2, saves the model checkpoint, evaluates the test set, and saves training diagnostic plots:
```bash
python train.py --data_dir dataset/train --test_dir dataset/test --epochs 5 --transfer
```

### 4. Run the Flask Web Application
Launch the web interface locally:
```bash
python flask_app.py
```
Open your web browser and navigate to: **[http://localhost:5000](http://localhost:5000)**.
Upload the generated high-resolution samples in the project root directory (e.g., `sample_highway.jpg`, `sample_flyover.jpg`) and verify the live predictions!

---

## 💡 Academic Success Pack

### Accuracy Improvement Tips
- **Increase Dataset Size**: Gather real satellite images from Google Earth Engine or open Kaggle sets.
- **Fine-Tuning**: After training the classifier head, unfreeze the top layers of MobileNetV2 (`base_model.trainable = True`) and train at a very low learning rate (e.g., `1e-5`) to specialize features for GIS textures.
- **Data Resolution**: Change target resolution from `224x224` to `384x384` (note that this increases training execution time on CPUs).

### Common Errors & Solutions
- **Error: `Model file not found`**: Ensure you have successfully run `train.py` before starting `flask_app.py`.
- **Error: `OutOfMemoryError`**: If running on a system with very low RAM, reduce `--batch_size` in the training execution command to `16` or `8`.
- **Error: `Double Rescaling / Off-Range predictions`**: Ensure you do not divide the image pixels by 255.0 twice. The model architecture already contains a `Rescaling(1.0/255)` layer. In prediction scripts, keep the pixel range in `[0, 255]`.
