import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


def build_dnn(input_dim, num_classes):
    """
    Build a Deep Neural Network for traffic classification.

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
        Input(shape=(input_dim,)),

        # Hidden Layer 1
        Dense(128, activation="relu"),

        # Regularization
        Dropout(0.30),

        # Hidden Layer 2
        Dense(64, activation="relu"),

        Dropout(0.30),

        # Hidden Layer 3
        Dense(32, activation="relu"),

        # Output Layer
        Dense(num_classes, activation="softmax")
    ])

    return model


if __name__ == "__main__":

    print("=" * 70)
    print("DNN MODEL TEST")
    print("=" * 70)

    # ================================================================
    # Dataset 1
    # ================================================================
    # Dataset 1 now has 19 model input features.
    #
    # dt is NOT included as a DNN input feature.
    # It is retained separately for future LSTM/GRU sequence creation.
    model_dataset1 = build_dnn(
        input_dim=19,
        num_classes=2
    )

    print("\nDataset 1 DNN:")
    model_dataset1.summary()

    # ================================================================
    # Dataset 2
    # ================================================================
    # Dataset 2 uses 10 selected input features.
    model_dataset2 = build_dnn(
        input_dim=10,
        num_classes=6
    )

    print("\nDataset 2 DNN:")
    model_dataset2.summary()