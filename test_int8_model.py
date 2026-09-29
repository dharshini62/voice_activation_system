import tensorflow as tf
import numpy as np

# Load INT8 model
interpreter = tf.lite.Interpreter(
    model_path="dscnn_model_int8.tflite"
)

interpreter.allocate_tensors()

# Get input/output information
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]["shape"])
print("Input dtype:", input_details[0]["dtype"])
print("Input quantization:", input_details[0]["quantization"])

print("Output shape:", output_details[0]["shape"])
print("Output dtype:", output_details[0]["dtype"])
print("Output quantization:", output_details[0]["quantization"])

# Load one MFCC sample
mfcc = np.load("mfcc_positive/activate_001.npy")

# Add batch + channel dimensions
mfcc = mfcc[np.newaxis, ..., np.newaxis]

# Convert float MFCC to INT8 using model quantization
scale, zero_point = input_details[0]["quantization"]

mfcc_int8 = np.round(mfcc / scale + zero_point)
mfcc_int8 = np.clip(mfcc_int8, -128, 127).astype(np.int8)

# Give input to model
interpreter.set_tensor(
    input_details[0]["index"],
    mfcc_int8
)

# Run inference
interpreter.invoke()

# Get output
output = interpreter.get_tensor(
    output_details[0]["index"]
)

print("\nRaw INT8 output:", output)

# Convert INT8 output back to float
output_scale, output_zero_point = output_details[0]["quantization"]

prediction = (
    (output.astype(np.float32) - output_zero_point)
    * output_scale
)

print("Prediction:", prediction)

if prediction[0][0] >= 0.5:
    print("Result: ACTIVATE detected")
else:
    print("Result: Negative")