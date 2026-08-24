from dataclasses import dataclass


@dataclass
class LSTMConfig:

    input_size: int = 10

    hidden_size: int = 64

    num_layers: int = 2

    output_size: int = 1

    dropout: float = 0.20

    batch_size: int = 64

    learning_rate: float = 0.001

    epochs: int = 25

    sequence_length: int = 30

    device: str = "cuda"
    