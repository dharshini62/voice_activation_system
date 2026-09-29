input_file = "dscnn_model_int8.tflite"
output_file = "dscnn_model_int8.h"

with open(input_file, "rb") as f:
    data = f.read()

with open(output_file, "w") as f:

    f.write("#ifndef DSCNN_MODEL_INT8_H\n")
    f.write("#define DSCNN_MODEL_INT8_H\n\n")

    f.write("const unsigned char dscnn_model_int8[] = {\n")

    for i, byte in enumerate(data):

        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02x}")

        if i != len(data) - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n\n")
    f.write(f"const unsigned int dscnn_model_int8_len = {len(data)};\n\n")
    f.write("#endif\n")

print("Model converted to C array!")
print("Saved as:", output_file)
print("Model size:", len(data), "bytes")