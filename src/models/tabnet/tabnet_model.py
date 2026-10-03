from torch.optim import Adam
from pytorch_tabnet.tab_model import TabNetClassifier


def build_tabnet(
    num_classes,
    seed=42
):
    """
    Build a TabNet classifier for network traffic
    classification.

    Parameters:
        num_classes (int):
            Number of output classes.

        seed (int):
            Random seed for reproducibility.

    Returns:
        TabNetClassifier
    """

    model = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,

        lambda_sparse=1e-4,

        # Adam optimizer
        optimizer_fn=Adam,
        optimizer_params={
            "lr": 0.02
        },

        mask_type="entmax",

        seed=seed,

        verbose=0
    )

    return model


if __name__ == "__main__":

    print("=" * 70)
    print("TABNET MODEL TEST")
    print("=" * 70)

    # Dataset 1
    model_d1 = build_tabnet(
        num_classes=2
    )

    print("\nDataset 1 TabNet created successfully.")
    print("Classes: 2")

    # Dataset 2
    model_d2 = build_tabnet(
        num_classes=6
    )

    print("\nDataset 2 TabNet created successfully.")
    print("Classes: 6")

    print("\nTabNet architecture configuration:")
    print("n_d              :", 32)
    print("n_a              :", 32)
    print("n_steps          :", 5)
    print("gamma            :", 1.5)
    print("lambda_sparse    :", 1e-4)
    print("optimizer        : Adam")
    print("learning rate    :", 0.02)
    print("mask_type        : entmax")

    print("\n" + "=" * 70)
    print("TABNET MODEL TEST COMPLETED")
    print("=" * 70)