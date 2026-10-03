import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)


def build_cnn(input_dim, num_classes):
    """
    Build a 1D Convolutional Neural Network for
    network traffic classification.

    Parameters:
        input_dim (int):
            Number of input features.

        num_classes (int):
            Number of output classes.

            Dataset 1:
                num_classes = 2

            Dataset 2:
                num_classes = 6

    Returns:
        tensorflow.keras.Model
    """

    model = Sequential([

        # Input:
        # (number of features, 1)
        Input(shape=(input_dim, 1)),

        # Convolutional Feature Extraction
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(0.30),

        # Convert feature maps into a compact representation
        GlobalAveragePooling1D(),

        # Classification layers
        Dense(64, activation="relu"),

        Dropout(0.30),

        # Output layer
        Dense(num_classes, activation="softmax")
    ])

    return model


if __name__ == "__main__":

    print("=" * 70)
    print("CNN MODEL TEST")
    print("=" * 70)

    # Dataset 1
    # 19 input features
    # dt is NOT used as a CNN feature.
    model_dataset1 = build_cnn(
        input_dim=19,
        num_classes=2
    )

    print("\nDataset 1 CNN:")
    model_dataset1.summary()

    # Dataset 2
    # 10 input features
    model_dataset2 = build_cnn(
        input_dim=10,
        num_classes=6
    )

    print("\nDataset 2 CNN:")
    model_dataset2.summary()