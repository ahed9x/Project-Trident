# Model Training Guide for Vision One AI Features

## Overview
This guide explains how to train custom TensorFlow Lite models for the Vision One plugin's AI features.

## Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Sufficient training data (1000+ images per class minimum)
- GPU recommended for training (not required for inference)

---

## 1. Spaghetti Detection Model

### Dataset Collection

**Class 0: Normal Print**
- Capture images during successful prints
- Various viewing angles
- Different lighting conditions
- Multiple filament colors/materials
- **Target:** 2000+ images

**Class 1: Spaghetti/Failed Print**
- Capture images of failed prints with spaghetti
- Various failure types (adhesion, warping, detachment)
- Different stages of failure
- **Target:** 2000+ images

### Directory Structure
```
dataset/
├── train/
│   ├── normal/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── spaghetti/
│       ├── img_0001.jpg
│       ├── img_0002.jpg
│       └── ...
└── validation/
    ├── normal/
    └── spaghetti/
```

### Training Script

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = './dataset'

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
])

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + '/train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + '/validation',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Build model (MobileNetV2 for efficiency)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(),
             keras.metrics.Recall()]
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Fine-tune (optional)
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization (important for Raspberry Pi)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

# Save
with open('spaghetti_detector.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved: spaghetti_detector.tflite")
print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

# Test inference speed
import time

interpreter = tf.lite.Interpreter(model_path='spaghetti_detector.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dummy input
test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)

# Warmup
for _ in range(5):
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    times.append(time.time() - start)

print(f"Average inference time: {np.mean(times)*1000:.2f} ms")
print(f"Target: <100ms for real-time use on RPi4")
```

### Model Evaluation

```python
# Evaluate on test set
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + '/test',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

results = model.evaluate(test_ds)
print(f"Test accuracy: {results[1]:.4f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((predictions > 0.5).astype(int).flatten())

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Spaghetti Detection - Confusion Matrix')
plt.savefig('confusion_matrix.png')

print(classification_report(y_true, y_pred, 
                           target_names=['Normal', 'Spaghetti']))
```

---

## 2. First Layer Inspection Model

### Dataset Collection

**Class 0: Poor First Layer**
- Under-extrusion (gaps between lines)
- Over-extrusion (blobs, too thick)
- Poor adhesion (lifting corners)
- Inconsistent extrusion
- **Target:** 1500+ images

**Class 1: Good First Layer**
- Smooth, consistent lines
- Perfect adhesion
- Correct z-offset
- No gaps or over-extrusion
- **Target:** 1500+ images

### Training Script (Similar to above)

```python
# Same approach as spaghetti detection
# Key differences:
# - Focus on close-up texture analysis
# - May benefit from higher resolution (299x299)
# - Consider using InceptionV3 or EfficientNet

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Rest of training code similar...
```

---

## 3. Data Collection Tips

### Automated Capture During Prints

Add to your `printer.cfg`:

```gcode
[gcode_macro CAPTURE_TRAINING_IMAGE]
gcode:
    # Capture image for dataset
    {% set class_name = params.CLASS|default("normal") %}
    {% set timestamp = printer.system_stats.cputime %}
    
    # Save current frame
    # (Would need modification to vision_one.py to save with specific filename)
```

### Labeling Tools
- [LabelImg](https://github.com/tzutalin/labelImg) - For bounding boxes
- [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) - For classification
- Custom script to sort images into folders

### Data Augmentation
- Rotation (0-360°)
- Brightness adjustment (±30%)
- Zoom (±20%)
- Horizontal/vertical flips
- Gaussian noise
- Motion blur (simulates movement)

---

## 4. Model Deployment

### Copy to Raspberry Pi

```bash
# From your training machine
scp spaghetti_detector.tflite pi@octopi.local:/tmp/vision_one/models/
scp first_layer_classifier.tflite pi@octopi.local:/tmp/vision_one/models/
```

### Verify Model

```python
import tflite_runtime.interpreter as tflite
import numpy as np

# Load model
interpreter = tflite.Interpreter(model_path='/tmp/vision_one/models/spaghetti_detector.tflite')
interpreter.allocate_tensors()

# Check input/output specs
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# Test inference
test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print("Test output:", output)
```

---

## 5. Continuous Improvement

### Active Learning Pipeline

1. **Initial Model:** Train with curated dataset
2. **Deploy:** Use on real prints
3. **Collect Edge Cases:** Save images where model is uncertain (confidence 0.4-0.6)
4. **Human Review:** Manually label edge cases
5. **Retrain:** Add edge cases to training set
6. **Repeat:** Deploy improved model

### Monitoring Model Performance

Add to `vision_one.py`:

```python
def log_prediction(self, frame, prediction, confidence):
    """Log predictions for model monitoring"""
    if 0.4 < confidence < 0.6:  # Uncertain predictions
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"uncertain_{timestamp}_{confidence:.2f}.jpg"
        cv2.imwrite(str(self.calibration_path / filename), frame)
```

---

## 6. Advanced Techniques

### Ensemble Models
Train multiple models and average predictions for better accuracy:

```python
models = [model1, model2, model3]
predictions = [m.predict(image) for m in models]
final_prediction = np.mean(predictions, axis=0)
```

### Quantization for Speed

```python
# Post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

# Full integer quantization (fastest)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
```

---

## Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNet Paper](https://arxiv.org/abs/1704.04861)
- [Transfer Learning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Image Classification Guide](https://www.tensorflow.org/tutorials/images/classification)

---

## Model Performance Targets

| Metric | Target | Why |
|--------|--------|-----|
| Accuracy | >95% | Minimize false alarms |
| Inference Time | <100ms | Real-time operation on RPi4 |
| Model Size | <10MB | Fast loading, minimal storage |
| False Positives | <2% | Avoid unnecessary pauses |
| False Negatives | <5% | Catch most failures |

---

For questions or assistance with model training, open an issue on the project repository.
