import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
X = np.load("X.npy")
y = np.load("y.npy")

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# DS-CNN model
model = tf.keras.Sequential([
    
    tf.keras.layers.Input(shape=(13, 51, 1)),

    # First convolution
    tf.keras.layers.Conv2D(
        16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    # Depthwise convolution
    tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    # Pointwise convolution
    tf.keras.layers.Conv2D(
        16,
        kernel_size=(1, 1),
        activation="relu"
    ),

    # Second depthwise-separable block
    tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.Conv2D(
        32,
        kernel_size=(1, 1),
        activation="relu"
    ),

    # Convert feature maps to one vector
    tf.keras.layers.GlobalAveragePooling2D(),

    # Output: 0 = negative, 1 = Activate
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Show model
model.summary()

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)

# Test
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTraining completed!")
print("Test Accuracy:", test_accuracy)

# Save model
model.save("dscnn_model.keras")

print("Model saved as dscnn_model.keras")