# Comprehensive System & UML Diagrams

This document contains detailed text-based ASCII block diagrams of the project workflow, architectures, and standard software engineering UML charts.

---

## 1. Project Workflow Diagram

```
+---------------------------------------------------------------+
|                       Dataset Generation                      |
| - generate_synthetic_data.py draws 5 structural classes       |
+-------------------------------v-------------------------------+
                                |
+-------------------------------v-------------------------------+
|                   Preprocessing & Normalization               |
| - preprocessing.py resizes, normalizes, & applies augmentations|
+-------------------------------v-------------------------------+
                                |
+-------------------------------v-------------------------------+
|                  Deep Learning Classifier Core                |
| - MobileNetV2 (ImageNet backbone) extracts generic spatial features|
| - Global Average Pooling + 40% Dropout regularizes tensors     |
| - Softmax Dense Classifier maps features to 5 infrastructure target categories|
+-------------------------------v-------------------------------+
                                |
+-------------------------------v-------------------------------+
|                      Diagnostic Evaluations                   |
| - plot_training_history() -> accuracy/loss curves             |
| - test.py -> detailed classification reports & confusion matrix heatmaps |
+-------------------------------v-------------------------------+
                                |
+-------------------------------v-------------------------------+
|                      Production Flask Server                  |
| - Loads pre-trained model in-memory at server boot            |
| - Receives browser single/batch image uploads                 |
| - Preprocesses, performs inference, caches query histories    |
| - Renders beautiful UI with Leaflet Maps, Chart.js, smart analysis|
+---------------------------------------------------------------+
```

---

## 2. Detailed CNN Layer Architecture

```
[ Input: 224x224x3 Image ]
           |
           v
[ Data Augmentation Layer ] ---> (Rotation, Flip, Zoom, Contrast)
           |
           v
[ Rescaling Layer (1/255) ] ---> Scales pixels to [0, 1] range
           |
           v
[ MobileNetV2 Feature Extractor ] ---> (Frozen Convolution layers, Depthwise Separable)
           |
           v
[ Feature Maps Output: 7x7x1280 ]
           |
           v
[ Global Average Pooling 2D ] ---> Shrinks dimensions to 1D vector (1280 parameters)
           |
           v
[ Dropout Regularization (40%) ] ---> Randomly deactivates 512 connections
           |
           v
[ Dense Classifier (Softmax) ] ---> Outputs 5 infrastructure probabilities (Highway, etc.)
```

---

## 3. Depthwise Separable Convolution Workflow (MobileNetV2)

Traditional 3D Convolution filter calculations are split into:

```
  Traditional 3D Filter                     Depthwise Separable filter splits
-------------------------                 -------------------------------------
   [3x3xChannel Filter]                    [3x3 Depthwise Filter]   [1x1 Pointwise]
  Processes space & color                 Processes spatial details  Combines channels
      simultaneously                         per channel separately      linearly
            |                                         |                      |
            v                                         +----------+-----------+
    (Heavy parameter weight)                                     |
                                                                 v
                                                     (8.5x computational savings!)
```

---

## 4. Data Flow Diagram (DFD)

### Level 0 DFD:
```
               +----------------------------------------+
               |                 User                   |
               +---+--------------------------------+---+
                   |                                ^
        Uploads    |                                |   Renders Prediction,
        GIS segment|                                |   Maps & PDF report
        Image      v                                |
               +---+--------------------------------+---+
               |  GIS Road Classification System (Flask) |
               +----------------------------------------+
```

### Level 1 DFD:
```
           +--------+   Upload GIS file   +------------------+
           |  User  | ==================> | 1.0 Upload Save  |
           +--------+                     +--------+---------+
                                                   |
                                            In-memory file
                                                   v
           +--------+   HTML report page  +--------+---------+
           |  User  | <================== | 2.0 Preprocess   |
           +--------+                     +--------+---------+
               ^                                   |
               |                            Normalized Tensor
               |                                   v
           +---+----+   Returns Class &   +--------+---------+
           | Render | <================== | 3.0 DL Inference |
           +--------+     Confidence      +------------------+
```

---

## 5. UML Use Case Diagram

```
                 +-----------------------------------------------+
                 |             GIS Classifier System             |
                 |                                               |
                 |   +---------------------------------------+   |
                 |   |       1. Upload GIS Satellite Image   |   |
                 |   +-------------------+-------------------+   |
                 |                       ^                       |
                 |                       | (Extends)             |
                 |   +-------------------+-------------------+   |
                 |   |   2. Run Multi-Image Batch Uploads    |   |
                 |   +-------------------+-------------------+   |
                 |                       ^                       |
                 |                       | (Includes)            |
                 |   +-------------------+-------------------+   |
                 |   |   3. View Leaflet Geospatial Map      |   |
                 |   +---------------------------------------+   |
  (Actor)        |                                               |
+----------+     |   +---------------------------------------+   |
|   User   | ===>|   |   4. View Chart.js Probability Graph  |   |
+----------+     |   +---------------------------------------+   |
  Student        |                                               |
  Faculty        |   +---------------------------------------+   |
  Examiner       |   |   5. Inspect AI Smart Road Analysis   |   |
                 |   +---------------------------------------+   |
                 |                                               |
                 |   +---------------------------------------+   |
                 |   |   6. Download / Print Academic PDF    |   |
                 |   +---------------------------------------+   |
                 |                                               |
                 |   +---------------------------------------+   |
                 |   |   7. Clear Local Analysis History Log |   |
                 |   +---------------------------------------+   |
                 +-----------------------------------------------+
```

---

## 6. UML Activity Flow Diagram

```
+-----------+     +--------------+     +---------------+     +---------------+
|   Start   | --> | Upload Image | --> | Preprocessing | --> | MobileNetV2   |
| (Web App) |     |  from Portal |     | (Bilinear/224)|     | Feature Eval  |
+-----------+     +--------------+     +---------------+     +-------+-------+
                                                                     |
                                                                     v
+-----------+     +--------------+     +---------------+     +-------+-------+
|    End    | <-- | Render HTML  | <-- | Cache History | <-- | Dense Softmax |
| (Dashboard|     | result.html  |     |   JSON Logs   |     | Probability   |
+-----------+     +--------------+     +---------------+     +---------------+
```

---

## 7. UML Sequence Flow Diagram

```
User             Browser            Flask Server          Neural Model
 |                  |                     |                     |
 |---Upload Image-->|                     |                     |
 |                  |----POST File------->|                     |
 |                  |                     |---predict_tensor--->|
 |                  |                     |<--Class/confidence--|
 |                  |                     |---write_history()-->|
 |                  |<--Returns HTML------|                     |
 |<--Print PDF------|                     |                     |
```

---

## 8. Deployment Topology Diagram

```
+--------------------------------------------------------------+
|                     Client Web Browser                       |
|   - Triggers bilinar photo previews, Chart.js & Leaflet JS   |
+------------------------------+-------------------------------+
                               |
                        Internet connection (HTTP / WSGI)
                               |
                               v
+------------------------------+-------------------------------+
|               Flask Server Environment (WSGI / App)          |
|   - Executes Python 3.10 virtual environment modules         |
|   - Coordinates Flask server uploads, static directories      |
+------------------------------+-------------------------------+
                               |
                       Executes in-memory predictions
                               |
                               v
+------------------------------+-------------------------------+
|                 Deep Learning Inference Core                 |
|   - TensorFlow / Keras executing road_type_classifier.h5     |
+--------------------------------------------------------------+
```
