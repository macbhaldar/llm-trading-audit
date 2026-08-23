from dataclasses import dataclass


@dataclass
class TransformerConfig:

    sequence_length: int = 30

    input_size: int = 10

    d_model: int = 64

    nhead: int = 4

    num_layers: int = 3

    dropout: float = 0.10

    batch_size: int = 64

    learning_rate: float = 1e-3

    epochs: int = 20

    device: str = "cuda"