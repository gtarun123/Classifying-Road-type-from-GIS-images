import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["Highway", "Street Road", "Rural Road", "Dirt Road", "Flyover Road"]
MODEL_PATH = "models/road_type_classifier.h5"
TEST_DIR = "dataset/test"

def evaluate_classifier_model():
    """
    Evaluates the trained model on the test dataset and outputs:
    1. A detailed classification report (Precision, Recall, F1-Score) to models/test_results.txt.
    2. An interactive confusion matrix heatmap using Seaborn saved to models/confusion_matrix.png.
    """
    print("Starting Deep Learning Model Evaluation...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run training first.")
        
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found at {TEST_DIR}. Ensure generate_synthetic_data.py has been run.")

    # Load Model
    print("Loading TensorFlow Keras Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load Test Dataset
    print("Loading Test Dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        shuffle=False,
        image_size=(224, 224),
        batch_size=32
    )

    # Perform Predictions
    print("Generating predictions on unseen test dataset...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        # tf.keras.utils.image_dataset_from_directory already loads images in range [0, 255]
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predicted_labels.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Classification Report
    print("Computing evaluation metrics...")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    print("\n================ CLASSIFICATION REPORT ================")
    print(report)
    print("=======================================================\n")
    
    # Save Report to file
    report_path = "models/test_results.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("ROAD TYPE CLASSIFICATION SYSTEM - TEST RESULTS\n")
        f.write("=======================================================\n\n")
        f.write(report)
    print(f"Successfully saved text report to: {report_path}")

    # 2. Confusion Matrix Heatmap
    print("Plotting Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    # Elegant custom purple/blue palette for premium feel
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Purples", 
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES,
        cbar=True,
        square=True,
        annot_kws={"size": 12, "weight": "bold"}
    )
    
    plt.title("Evaluation Confusion Matrix - GIS Road Classification", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Predicted Road Categories", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("Actual Road Categories", fontsize=12, fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    matrix_path = "models/confusion_matrix.png"
    plt.savefig(matrix_path, dpi=150)
    plt.close()
    print(f"Successfully saved confusion matrix heatmap image to: {matrix_path}")
    print("Evaluation Pipeline Completed successfully!")

if __name__ == "__main__":
    evaluate_classifier_model()
