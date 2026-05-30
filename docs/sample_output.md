# Sample Output and Final Results

## Example Training Output
```
Epoch 1/20
100/100 [==============================] - 30s 250ms/step - loss: 1.2503 - accuracy: 0.3860 - val_loss: 1.0420 - val_accuracy: 0.5200
Epoch 2/20
100/100 [==============================] - 25s 245ms/step - loss: 0.9382 - accuracy: 0.5870 - val_loss: 0.8254 - val_accuracy: 0.6440
...
Epoch 12/20
100/100 [==============================] - 20s 200ms/step - loss: 0.2401 - accuracy: 0.9110 - val_loss: 0.2156 - val_accuracy: 0.9200
```

## Sample Prediction Output
```
Loading model...
Predicted Road Type: Highway
Confidence: 95.24%
```

## Saved Results Files
- `models/road_type_classifier.h5` — trained model file
- `models/training_history.png` — accuracy and loss graphs
- `models/confusion_matrix.png` — confusion matrix for validation data

## Final Results Summary
- Expected accuracy: 85% to 95% depending on dataset size and quality
- Confusion matrix highlights class-level performance
- Streamlit app predicts uploaded GIS images in real time

## Notes
- Accuracy can improve by adding more labeled GIS images and using MobileNetV2 transfer learning.
- Use consistent image sizes in `dataset` folders for best results.
