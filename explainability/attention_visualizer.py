import matplotlib.pyplot as plt
import numpy as np


class AttentionVisualizer:
    @staticmethod
    def plot(
        attention_matrix,
        title="Attention",
    ):

        attention_matrix = np.asarray(attention_matrix)

        plt.figure(figsize=(8, 6))

        plt.imshow(
            attention_matrix,
            aspect="auto",
            interpolation="nearest",
        )

        plt.colorbar()

        plt.title(title)

        plt.xlabel("Key")

        plt.ylabel("Query")

        plt.tight_layout()

        plt.show()
