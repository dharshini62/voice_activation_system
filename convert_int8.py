import tensorflow as tf
import numpy as np

# Load trained DS-CNN model
model = tf.keras.models.load_model("dscnn_model.keras")

# Load training data for INT8 calibration
X_train = np.load("X.npy").astype(np.float32)

# Representative dataset
def representative_dataset():
    for i in range(min(40, len(X_train))):
        sample = X_train[i:i+1]
        yield [sample]

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

# Force INT8 input/output
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
tflite_model = converter.convert()

# Save INT8 model
with open("dscnn_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("\nINT8 quantization completed!")
print("Saved as: dscnn_model_int8.tflite")
print("Model size:", len(tflite_model), "bytes")
