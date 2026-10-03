import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)


def build_lstm(
    sequence_length,
    input_dim,
    num_classes
):
    """
    Build an LSTM model for temporal network
    traffic classification.

    Parameters:
        sequence_length (int):
            Number of consecutive observations
            in each sequence.

        input_dim (int):
            Number of network traffic features.

        num_classes (int):
            Number of output classes.

            Dataset 1:
                num_classes = 2

    Returns:
        tensorflow.keras.Model
    """

    model = Sequential([

        # Input:
        # (sequence_length, number_of_features)
        Input(
            shape=(
                sequence_length,
                input_dim
            )
        ),

        # First LSTM layer
        LSTM(
            64,
            return_sequences=True
        ),

        Dropout(0.30),

        # Second LSTM layer
        LSTM(
            32,
            return_sequences=False
        ),

        Dropout(0.30),

        # Fully connected layer
        Dense(
            32,
            activation="relu"
        ),

        # Output layer
        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model


if __name__ == "__main__":

    print("=" * 70)
    print("LSTM MODEL TEST")
    print("=" * 70)

    # Dataset 1:
    # 10 consecutive observations
    # 18 network traffic features
    # Pairflow excluded because of severe temporal distribution shift
    model = build_lstm(
        sequence_length=10,
        input_dim=18,
        num_classes=2
    )

    model.summary()