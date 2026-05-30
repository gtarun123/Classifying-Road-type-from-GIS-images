import os
import argparse
import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["Highway", "Street Road", "Rural Road", "Dirt Road", "Flyover Road"]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load an image using OpenCV, resize and normalize it for prediction."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    # Multiply by 255.0 because the model contains a rescaling layer Rescaling(1.0/255)
    image = image * 255.0
    image = np.expand_dims(image, axis=0)
    return image

def load_model(model_path="models/road_type_classifier.h5"):
    """Load the trained Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
    return tf.keras.models.load_model(model_path)

def predict_road_type(model, image_path):
    """Predict the road type for a single GIS image."""
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    predicted_label = CLASS_NAMES[predicted_index]
    return predicted_label, confidence

def predict_batch(model, folder_path, output_path="models/batch_predictions.txt"):
    """Predict road types for all images in a folder and write a report."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Provided path is not a folder: {folder_path}")

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not files:
        print(f"No valid images found in folder: {folder_path}")
        return
        
    print(f"Starting batch prediction on {len(files)} files in folder: {folder_path}")
    results = []
    
    print("\n-----------------------------------------------------------")
    print(f"{'Filename':<30} | {'Prediction':<15} | {'Confidence':<10}")
    print("-----------------------------------------------------------")
    
    for filename in files:
        image_path = os.path.join(folder_path, filename)
        label, conf = predict_road_type(model, image_path)
        print(f"{filename:<30} | {label:<15} | {conf * 100:.2f}%")
        results.append((filename, label, conf))
        
    print("-----------------------------------------------------------\n")
    
    # Save batch predictions report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("ROAD TYPE BATCH PREDICTION REPORT\n")
        f.write("===================================\n\n")
        f.write(f"{'Filename':<35} | {'Prediction':<18} | {'Confidence':<12}\n")
        f.write("-" * 72 + "\n")
        for filename, label, conf in results:
            f.write(f"{filename:<35} | {label:<18} | {conf * 100:.2f}%\n")
    print(f"Batch report successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict road type from GIS image(s) using MobileNetV2.")
    parser.add_argument("input_path", help="Path to a single GIS image or a folder containing multiple images.")
    parser.add_argument(
        "--model_path",
        default="models/road_type_classifier.h5",
        help="Path to the trained model file.",
    )
    args = parser.parse_args()

    print("Loading pre-trained Deep Learning model...")
    model = load_model(args.model_path)

    if os.path.isdir(args.input_path):
        predict_batch(model, args.input_path)
    else:
        label, confidence = predict_road_type(model, args.input_path)
        print("\n================ INFERENCE OUTPUT ================")
        print(f"File Classified     : {os.path.basename(args.input_path)}")
        print(f"Predicted Road Type : {label}")
        print(f"Confidence Score    : {confidence * 100:.2f}%")
        print("==================================================\n")
