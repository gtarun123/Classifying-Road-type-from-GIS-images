import os
import predict

TEST_DIR = "dataset/test"
OUTPUT_PATH = "models/batch_predictions.txt"

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

if __name__ == "__main__":
    print("Loading model...")
    model = predict.load_model()

    results = []
    for root, dirs, files in os.walk(TEST_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                image_path = os.path.join(root, f)
                try:
                    label, conf = predict.predict_road_type(model, image_path)
                    rel_path = os.path.relpath(image_path)
                    print(f"{rel_path} -> {label} ({conf*100:.2f}%)")
                    results.append((rel_path, label, conf))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as out:
        out.write("ROAD TYPE BATCH PREDICTION REPORT\n")
        out.write("===================================\n\n")
        out.write(f"{'Filename':<60} | {'Prediction':<18} | {'Confidence':<12}\n")
        out.write("-" * 100 + "\n")
        for filename, label, conf in results:
            out.write(f"{filename:<60} | {label:<18} | {conf * 100:.2f}%\n")

    print(f"Saved combined batch predictions to: {OUTPUT_PATH}")
